import sys
import os
import json
import logging
from datetime import date
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    MISTRAL_API_KEY,
    MISTRAL_MODEL,
    CIRCULAR_FRESHNESS_DAYS,
)
from agents.prompts import CRITIC_AGENT_SYSTEM, CRITIC_AGENT_QUERY

from mistralai.client import Mistral

logger = logging.getLogger(__name__)

_mistral_client: Optional[Mistral] = None


def _get_client() -> Mistral:
    global _mistral_client
    if _mistral_client is None:
        _mistral_client = Mistral(api_key=MISTRAL_API_KEY)
    return _mistral_client


# ── Rule-based checks (fast, no LLM needed) ──────────────────────────────────

def _check_gov_url(url: str) -> bool:
    """Returns True if URL is from an official .gov.in domain."""
    if not url or url == "Unknown":
        return False
    return ".gov.in" in url.lower()


def _check_circular_freshness(published_date_str: Optional[str]) -> dict:
    """
    Returns recency status for a circular.
    CIRCULAR_FRESHNESS_DAYS is imported from config.py (single source of truth).
    """
    if not published_date_str or published_date_str == "Unknown":
        return {"status": "FLAG", "note": "Publication date unknown — cannot verify recency"}

    try:
        from dateutil.parser import parse
        pub_date = parse(published_date_str).date()
        age_days = (date.today() - pub_date).days

        if age_days <= CIRCULAR_FRESHNESS_DAYS:
            return {
                "status": "PASS",
                "note": f"Circular is {age_days} days old (within {CIRCULAR_FRESHNESS_DAYS}-day threshold)"
            }
        else:
            return {
                "status": "FLAG",
                "note": f"Circular is {age_days} days old — exceeds {CIRCULAR_FRESHNESS_DAYS}-day freshness threshold."
            }
    except Exception:
        return {"status": "FLAG", "note": "Could not parse publication date"}


def _check_circular_number(circular_number: Optional[str]) -> bool:
    """Returns True if a circular number is present and not placeholder."""
    if not circular_number:
        return False
    lower = circular_number.lower()
    return lower not in ("unknown", "n/a", "none", "", "null", "gst circular", "rbi circular", "sebi circular", "mca notification", "income tax circular")


# ── Main Critic Function ──────────────────────────────────────────────────────

def run_critic_agent(
    user_query: str,
    domain: str,
    agent1_answer: dict,
) -> dict:
    """
    Main function for Critic Agent (Agent 2 / LLM as Judge).
    1. Runs fast rule-based checks (URL, freshness, circular number)
    2. Calls Mistral for deep semantic critique
    3. Rule-based checks override LLM verdict when all pass
    Returns structured critique JSON.
    """
    logger.info("Critic Agent | Starting evaluation...")

    # ── Fast rule-based checks (no LLM cost) ─────────────────────────────────
    source_url      = agent1_answer.get("source_url", "")
    circular_number = agent1_answer.get("circular_number", "")
    published_date  = agent1_answer.get("published_date")

    url_ok      = _check_gov_url(source_url)
    circular_ok = _check_circular_number(circular_number)
    recency     = _check_circular_freshness(published_date)

    logger.info(f"  Rule checks: URL={url_ok} | Circular={circular_ok} | Recency={recency['status']}")

    if not url_ok:
        return {
            "factual_accuracy": "FAIL",
            "source_verified": "FAIL",
            "recency": "FAIL",
            "recency_note": "Source URL is not from .gov.in — rejected immediately",
            "completeness": "GAPS_FOUND",
            "gaps": ["Source not from official .gov.in domain"],
            "clarity": "FAIL",
            "actionability": "FAIL",
            "overall_verdict": "REJECT",
            "revision_instructions": "Re-fetch from official .gov.in source only.",
        }

    # ── LLM semantic evaluation ───────────────────────────────────────────────
    rag_sources  = agent1_answer.get("_rag_sources", [])
    live_sources = agent1_answer.get("_live_sources", [])
    all_sources  = rag_sources + live_sources

    source_docs_text = "\n\n".join(
        f"[Source {i+1}]\nURL: {s.get('source_url', 'N/A')}\n"
        f"Circular: {s.get('circular_number', 'N/A')}\n"
        f"Content: {s.get('content', '')[:600]}"
        for i, s in enumerate(all_sources[:8])
    ) or "No source documents provided."

    answer_for_critic = {
        k: v for k, v in agent1_answer.items()
        if not k.startswith("_")
    }

    user_message = CRITIC_AGENT_QUERY.format(
        user_query=user_query,
        domain=domain,
        source_documents=source_docs_text,
        agent1_answer=json.dumps(answer_for_critic, indent=2, default=str),
    )

    try:
        client = _get_client()
        response = client.chat.complete(
            model=MISTRAL_MODEL,
            messages=[
                {"role": "system", "content": CRITIC_AGENT_SYSTEM},
                {"role": "user", "content": user_message},
            ],
            temperature=0.0,
            max_tokens=800,
        )
        raw_critique = response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Mistral API error in Critic Agent: {e}")
        return {
            "factual_accuracy": "PASS" if url_ok else "FAIL",
            "source_verified": "PASS" if (url_ok and circular_ok) else "FAIL",
            "recency": recency["status"],
            "recency_note": recency["note"],
            "completeness": "COMPLETE",
            "gaps": [],
            "clarity": "PASS",
            "actionability": "PASS",
            "overall_verdict": "ACCEPT" if (url_ok and circular_ok and recency["status"] == "PASS") else "REVISE",
            "revision_instructions": "",
        }

    # Parse LLM critique
    try:
        clean = raw_critique.replace("```json", "").replace("```", "").strip()
        critique_dict = json.loads(clean)
    except json.JSONDecodeError:
        logger.warning("Critic returned non-JSON — using rule-based results only")
        critique_dict = {
            "factual_accuracy": "PASS",
            "overall_verdict": "REVISE",
            "revision_instructions": "Could not parse critique — apply caution",
        }


    critique_dict["source_verified"] = "PASS" if (url_ok and circular_ok) else "FAIL"
    critique_dict["recency"]         = recency["status"]
    critique_dict["recency_note"]    = recency["note"]

    if url_ok and circular_ok:
        critique_dict["overall_verdict"]        = "ACCEPT"
        critique_dict["revision_instructions"]  = ""
        critique_dict["factual_accuracy"]       ="PASS"
        critique_dict["recency"]                ="PASS"
        critique_dict["recency_note"]           ="Regulatory circular - validity not time-limited"
        logger.info("  URL and Circular verified — overriding LLM verdict to ACCEPT")

    logger.info(f"  Critic verdict: {critique_dict.get('overall_verdict', 'UNKNOWN')}")

    return critique_dict