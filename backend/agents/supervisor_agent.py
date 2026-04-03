import sys
import os
import json
import logging
import httpx
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    MISTRAL_API_KEY,
    MISTRAL_FINE_TUNED_MODEL,
    HF_TOKEN,
    HF_INFERENCE_URL,
    CONFIDENCE_HIGH_THRESHOLD,
    CONFIDENCE_MEDIUM_THRESHOLD,
)
from agents.prompts import SUPERVISOR_AGENT_SYSTEM, SUPERVISOR_AGENT_QUERY

from mistralai.client import Mistral

logger = logging.getLogger(__name__)

_mistral_client: Optional[Mistral] = None


def _get_client() -> Mistral:
    global _mistral_client
    if _mistral_client is None:
        _mistral_client = Mistral(api_key=MISTRAL_API_KEY)
    return _mistral_client


# ── HuggingFace Inference ─────────────────────────────────────────────────────

def _call_hf_inference(system_prompt: str, user_message: str) -> Optional[str]:
    """
    Call fine-tuned Mistral 7B via HuggingFace Inference API.
    Returns None if unavailable — supervisor falls back to Mistral API.
    """
    if not HF_TOKEN or not HF_INFERENCE_URL:
        return None

    try:
        prompt = f"<s>[INST] {system_prompt}\n\n{user_message} [/INST]"

        response = httpx.post(
            HF_INFERENCE_URL,
            headers={
                "Authorization": f"Bearer {HF_TOKEN}",
                "Content-Type": "application/json",
            },
            json={
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": 1800,
                    "temperature": 0.1,
                    "return_full_text": False,
                }
            },
            timeout=90.0,
        )

        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                generated = result[0].get("generated_text", "")
                logger.info("  Supervisor Agent: using fine-tuned HF model ✓")
                return generated.strip()
        elif response.status_code == 503:
            logger.warning("  HF model cold start — falling back to Mistral API")
            return None
        else:
            logger.warning(f"  HF inference error {response.status_code} — falling back")
            return None

    except Exception as e:
        logger.warning(f"  HF inference failed: {e} — falling back to Mistral API")
        return None


# ── Confidence Scoring ────────────────────────────────────────────────────────

def _calculate_confidence_score(critique: dict, raw_score: Optional[float]) -> tuple[float, str, str]:
    """
    Calculate final confidence score based on critic report.
    Order: apply all penalties first, then ACCEPT boost last.
    HIGH   >= 0.85
    MEDIUM >= 0.60
    LOW    <  0.60
    """
    score = raw_score if isinstance(raw_score, (int, float)) else 0.92
    penalties = []

    # ── Step 1: Apply all downgrades ─────────────────────────────────────────
    if critique.get("source_verified") == "FAIL":
        score = min(score, 0.40)
        penalties.append("source not from official .gov.in domain")

    if critique.get("factual_accuracy") == "FAIL":
        score = min(score, 0.45)
        penalties.append("factual accuracy issue detected")

    if critique.get("recency") == "FLAG":
        score = min(score, 0.72)
        penalties.append(f"circular may be outdated ({critique.get('recency_note', '')})")

    if critique.get("recency") == "FAIL":
        score = min(score, 0.50)
        penalties.append("circular is outdated")

    if critique.get("completeness") == "GAPS_FOUND":
        gaps = critique.get("gaps", [])
        if gaps:
            score -= 0.05 * min(len(gaps), 3)
            score = max(score, 0.30)

    if critique.get("overall_verdict") == "REJECT":
        score = min(score, 0.35)
        penalties.append("answer rejected by verification agent")

    if critique.get("overall_verdict") == "REVISE":
        score = min(score, 0.65)

    # ── Step 2: ACCEPT boost LAST — overrides all penalties ──────────────────
    if critique.get("overall_verdict") == "ACCEPT":
        score = max(score, 0.92)

    score = round(max(0.0, min(1.0, score)), 2)

    # Map to level
    if score >= CONFIDENCE_HIGH_THRESHOLD:
        level = "HIGH"
    elif score >= CONFIDENCE_MEDIUM_THRESHOLD:
        level = "MEDIUM"
    else:
        level = "LOW"

    # Build explanation
    if not penalties:
        explanation = (
            "High confidence: the answer is sourced from an official government circular "
            "with a verified .gov.in URL and circular number. "
            "Verified by our 3-agent critique pipeline using fine-tuned Mistral 7B."
        )
    else:
        explanation = (
            f"Confidence reduced to {level} because: {'; '.join(penalties)}. "
            f"Verify this information directly on the official government portal before acting."
        )

    return score, level, explanation


# ── Main Supervisor Agent ─────────────────────────────────────────────────────

def run_supervisor_agent(
    user_query: str,
    domain: str,
    agent1_answer: dict,
    critic_report: dict,
) -> dict:
    """
    Main function for Supervisor Agent (Agent 3).
    Uses fine-tuned Mistral 7B → falls back to Mistral API.
    """
    logger.info(f"Supervisor Agent | Verdict from Critic: {critic_report.get('overall_verdict')}")

    clean_agent1 = {k: v for k, v in agent1_answer.items() if not k.startswith("_")}

    user_message = SUPERVISOR_AGENT_QUERY.format(
        user_query=user_query,
        domain=domain,
        agent1_answer=json.dumps(clean_agent1, indent=2, default=str),
        critic_report=json.dumps(critic_report, indent=2, default=str),
    )

    # Try fine-tuned HF model first, fall back to Mistral API
    try:
        raw_output = _call_hf_inference(SUPERVISOR_AGENT_SYSTEM, user_message)
        if raw_output is None:
            client = _get_client()
            response = client.chat.complete(
                model=MISTRAL_FINE_TUNED_MODEL,
                messages=[
                    {"role": "system", "content": SUPERVISOR_AGENT_SYSTEM},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.1,
                max_tokens=1800,
            )
            raw_output = response.choices[0].message.content.strip()
            logger.info("  Supervisor Agent: using Mistral API (fallback)")
    except Exception as e:
        logger.error(f"Mistral API error in Supervisor Agent: {e}")
        score, level, explanation = _calculate_confidence_score(critic_report, 0.40)
        return {
            "final_answer": "Unable to generate final answer due to API error. Please try again.",
            "circular_number": agent1_answer.get("circular_number", "Unknown"),
            "source_url": agent1_answer.get("source_url", "Unknown"),
            "published_date": agent1_answer.get("published_date"),
            "is_gov_verified": False,
            "confidence_level": "LOW",
            "confidence_score": 0.30,
            "confidence_explanation": f"API error occurred: {str(e)}",
            "deadlines": [],
            "action_required": "Please retry your query.",
            "domain": domain,
            "gaps_acknowledged": [],
            "error": str(e),
        }

    # Parse supervisor output
    try:
        clean = raw_output.replace("```json", "").replace("```", "").strip()
        final_dict = json.loads(clean)
    except json.JSONDecodeError:
        logger.warning("Supervisor returned non-JSON — using fallback structure")
        final_dict = {
            "final_answer": raw_output,
            "circular_number": agent1_answer.get("circular_number", "Unknown"),
            "source_url": agent1_answer.get("source_url", "Unknown"),
            "published_date": agent1_answer.get("published_date"),
            "is_gov_verified": _check_gov_url(agent1_answer.get("source_url", "")),
            "confidence_level": "MEDIUM",
            "confidence_score": 0.65,
            "confidence_explanation": "Confidence set to MEDIUM due to response parsing issue.",
            "deadlines": [],
            "action_required": agent1_answer.get("action_required", "Verify with official source."),
            "domain": domain,
            "gaps_acknowledged": critic_report.get("gaps", []),
        }

    # Override with rule-based confidence
    raw_score = final_dict.get("confidence_score")
    score, level, explanation = _calculate_confidence_score(critic_report, raw_score)
    final_dict["confidence_score"]       = score
    final_dict["confidence_level"]       = level
    final_dict["confidence_explanation"] = explanation

    logger.info(f"  Final confidence: {level} ({score})")

    return final_dict


def _check_gov_url(url: str) -> bool:
    return bool(url) and (".gov.in" in url.lower() or "rbi.org.in" in url.lower())