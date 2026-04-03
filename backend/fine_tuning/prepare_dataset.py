import sys
import os
import json
import time
import logging
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    GST_RSS_URL, RBI_RSS_URL, SEBI_RSS_URL, MCA_RSS_URL,
    FINE_TUNE_MAX_SEQ_LENGTH,
)
from mcp_server.gst_tool import fetch_gst_data
from mcp_server.rbi_tool import fetch_rbi_data
from mcp_server.sebi_tool import fetch_sebi_data
from mcp_server.mca_tool import fetch_mca_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OUTPUT_DIR  = os.path.join(os.path.dirname(__file__), "data")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "gst_rbi_sebi_mca_dataset.jsonl")

# Broad queries to fetch diverse regulatory content
FETCH_QUERIES = {
    "gst": [
        "GST rate", "input tax credit", "GST return filing",
        "reverse charge mechanism", "e-invoicing", "GST refund",
        "composition scheme", "GST exemption",
    ],
    "rbi": [
        "digital lending", "NBFC regulation", "priority sector lending",
        "KYC norms", "payment system", "foreign exchange",
        "bank rate", "CRR SLR",
    ],
    "sebi": [
        "mutual fund regulation", "insider trading", "IPO DRHP",
        "SEBI disclosure", "FPI registration", "alternative investment fund",
        "portfolio manager", "SEBI circular",
    ],
    "mca": [
        "annual filing", "director KYC", "company incorporation",
        "LLP annual return", "charge registration", "MCA21",
        "board meeting", "corporate governance",
    ],
}


def _doc_to_instruction_example(doc: dict, domain: str) -> dict:
    """
    Convert a fetched regulatory document into instruction-tuning format.
    Format: {"instruction": "...", "input": "...", "output": "..."}
    """
    title   = doc.get("title", "")
    content = doc.get("content", "")
    url     = doc.get("source_url", "")
    circ_no = doc.get("circular_number", "")
    date    = doc.get("published_date", "Unknown")

    if not content or len(content) < 50:
        return None

    instruction = f"What does the {domain.upper()} regulation '{title[:100]}' say?"

    input_text = (
        f"Source: {url}\n"
        f"Circular: {circ_no}\n"
        f"Date: {date}\n"
        f"Content: {content[:1000]}"
    )

    output_dict = {
        "summary": title[:200],
        "circular_number": circ_no,
        "source_url": url,
        "published_date": date,
        "domain": domain,
        "action_required": "Verify compliance with this regulation at the official portal.",
        "is_gov_verified": True,
    }
    output_text = json.dumps(output_dict, ensure_ascii=False)

    # Rough token length check (1 token ≈ 4 chars)
    total_chars = len(instruction) + len(input_text) + len(output_text)
    if total_chars > FINE_TUNE_MAX_SEQ_LENGTH * 4:
        input_text = input_text[:800]  # Truncate if too long

    return {
        "instruction": instruction,
        "input": input_text,
        "output": output_text,
    }


def prepare_dataset() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    examples: list[dict] = []

    tool_map = {
        "gst":  fetch_gst_data,
        "rbi":  fetch_rbi_data,
        "sebi": fetch_sebi_data,
        "mca":  fetch_mca_data,
    }

    for domain, queries in FETCH_QUERIES.items():
        logger.info(f"\nFetching {domain.upper()} data...")
        fetch_fn = tool_map[domain]

        for query in queries:
            logger.info(f"  Query: {query}")
            try:
                docs = fetch_fn(query=query, max_results=5)
                for doc in docs:
                    example = _doc_to_instruction_example(doc, domain)
                    if example:
                        examples.append(example)
                        logger.info(f"    + Added: {doc.get('title', '')[:60]}")
            except Exception as e:
                logger.warning(f"  Failed for '{query}': {e}")
            time.sleep(1)


    seen: set = set()
    unique_examples: list[dict] = []
    for ex in examples:
        key = ex["instruction"][:80]
        if key not in seen:
            seen.add(key)
            unique_examples.append(ex)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for ex in unique_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    logger.info(f"\n✓ Dataset saved: {OUTPUT_FILE}")
    logger.info(f"  Total examples: {len(unique_examples)}")


if __name__ == "__main__":
    prepare_dataset()