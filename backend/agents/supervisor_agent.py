import sys
import os
import json
import logging
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    MISTRAL_API_KEY,
    MISTRAL_FINE_TUNED_MODEL,
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


def _calculate_confidence_score(critique: dict, raw_score: Optional[float]) -> tuple[float, str, str]:
    """
    Calculate final confidence score based on critic report.
    Uses thresholds from config.py:
      HIGH   >= CONFIDENCE_HIGH_THRESHOLD   (0.85)
      MEDIUM >= CONFIDENCE_MEDIUM_THRESHOLD (0.60)
      LOW    <  CONFIDENCE_MEDIUM_THRESHOLD
    Returns (score, level, explanation).
    """
    # Start with LLM-suggested score or default
    score = raw_score if isinstance(raw_score, (int, float)) else 0.75

    # Apply automatic downgrades based on critic findings
    penalties = []

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
            score -= 0.05 * min(len(gaps), 3)   # Small penalty per gap
            score = max(score, 0.30)

    if critique.get("overall_verdict") == "REJECT":
        score = min(score, 0.35)

    score = round(max(0.0, min(1.0, score)), 2)

    # Map to level using thresholds from config
    if score >= CONFIDENCE_HIGH_THRESHOLD:
        level = "HIGH"
    elif score >= CONFIDENCE_MEDIUM_THRESHOLD:
        level = "MEDIUM"
    else:
        level = "LOW"

    # Build plain-English explanation
    if not penalties:
        explanation = (
            f"High confidence: the answer is sourced from an official government circular "
            f"with a verified URL, recent publication date, and no factual gaps detected."
        )
    else:
        explanation = (
            f"Confidence reduced to {level} because: {'; '.join(penalties)}. "
            f"Verify this information directly on the official government portal before acting."
        )

    return score, level, explanation


def run_supervisor_agent(
    user_query: str,
    domain: str,
    agent1_answer: dict,
    critic_report: dict,
) -> dict:
    """
    Main function for Supervisor Agent (Agent 3).
    1. Reads Agent 1 answer + Agent 2 critique
    2. Synthesizes final answer (fixing critic-identified gaps)
    3. Assigns confidence score
    Returns the final structured response shown to the user.
    """
    logger.info(f"Supervisor Agent | Verdict from Critic: {critic_report.get('overall_verdict')}")

    # Clean agent1 answer (remove internal keys)
    clean_agent1 = {k: v for k, v in agent1_answer.items() if not k.startswith("_")}

    user_message = SUPERVISOR_AGENT_QUERY.format(
        user_query=user_query,
        domain=domain,
        agent1_answer=json.dumps(clean_agent1, indent=2, default=str),
        critic_report=json.dumps(critic_report, indent=2, default=str),
    )

    try:
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
    except Exception as e:
        logger.error(f"Mistral API error in Supervisor Agent: {e}")
        # Graceful fallback
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

    # Override confidence score with calculated value (rule-based is more reliable)
    raw_score = final_dict.get("confidence_score")
    score, level, explanation = _calculate_confidence_score(critic_report, raw_score)
    final_dict["confidence_score"]       = score
    final_dict["confidence_level"]       = level
    final_dict["confidence_explanation"] = explanation

    logger.info(f"  Final confidence: {level} ({score})")

    return final_dict


def _check_gov_url(url: str) -> bool:
    """Local helper — avoids circular import with critic_agent."""
    return bool(url) and ".gov.in" in url.lower()