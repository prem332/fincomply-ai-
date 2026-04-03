import sys
import os
import json
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.seed_data import get_seed_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

# ── Q&A templates per domain ──────────────────────────────────────────────────

GST_QUESTIONS = [
    "What is the GST rate applicable for {topic}?",
    "Explain the GST compliance requirements for {topic}.",
    "What are the GST rules regarding {topic} in India?",
    "How does GST apply to {topic} under Indian tax law?",
    "What circular governs GST on {topic}?",
]

RBI_QUESTIONS = [
    "What are the RBI guidelines on {topic}?",
    "Explain the RBI regulations for {topic} in India.",
    "What does RBI say about {topic}?",
    "What are the compliance requirements for {topic} as per RBI?",
    "How does RBI regulate {topic}?",
]

SEBI_QUESTIONS = [
    "What are the SEBI regulations on {topic}?",
    "Explain SEBI rules regarding {topic}.",
    "What does SEBI mandate for {topic}?",
    "What are the disclosure requirements for {topic} under SEBI?",
    "How does SEBI regulate {topic} in Indian capital markets?",
]

MCA_QUESTIONS = [
    "What are the MCA requirements for {topic}?",
    "Explain the Companies Act provisions on {topic}.",
    "What does the Companies Act 2013 say about {topic}?",
    "What are the compliance requirements for {topic} under MCA?",
    "What are the filing requirements related to {topic}?",
]

INCOME_TAX_QUESTIONS = [
    "What is the income tax treatment for {topic}?",
    "Explain the tax rules regarding {topic} in India.",
    "What does the Income Tax Act say about {topic}?",
    "How is {topic} taxed under Indian income tax law?",
    "What are the deductions available for {topic} under income tax?",
]

DOMAIN_QUESTIONS = {
    "gst": GST_QUESTIONS,
    "rbi": RBI_QUESTIONS,
    "sebi": SEBI_QUESTIONS,
    "mca": MCA_QUESTIONS,
    "income_tax": INCOME_TAX_QUESTIONS,
}

SYSTEM_PROMPT = """You are FinComply AI, an expert Indian financial regulatory compliance assistant. 
You provide accurate, actionable compliance guidance based on official government circulars from 
GST (CBIC), RBI, SEBI, MCA, and Income Tax Department. 
Always cite circular numbers and source URLs when available.
Be concise, accurate, and always recommend verifying with official government portals."""


def _extract_topic(circular_number: str, content: str) -> str:
    """Extract a short topic from circular number or content."""
    # Try to get topic from first line of content
    first_line = content.strip().split("\n")[0].strip()
    # Remove common prefixes
    for prefix in ["GST rate on", "RBI guidelines on", "SEBI regulations on",
                   "MCA requirements for", "Income Tax on", "Section"]:
        if first_line.startswith(prefix):
            topic = first_line[len(prefix):].strip().rstrip(".")
            if len(topic) > 5 and len(topic) < 60:
                return topic

    words = circular_number.replace("/", " ").replace("-", " ").split()
    meaningful = [w for w in words if len(w) > 3 and not w.isdigit()]
    if meaningful:
        return " ".join(meaningful[:4])

    return "regulatory compliance"


def _generate_answer(item: dict) -> str:
    """Format the seed data content as a proper answer."""
    content = item["content"].strip()
    circular = item.get("circular_number", "")
    url = item.get("source_url", "")
    date = item.get("published_date", "")
    domain = item.get("domain", "").upper()

    lines = [line.strip() for line in content.split("\n") if line.strip()]
    clean_content = " ".join(lines)

    answer = clean_content

    if circular:
        answer += f"\n\nReference: {circular}"
    if url:
        answer += f"\nSource: {url}"
    if date:
        answer += f"\nDate: {date}"

    answer += f"\n\nAlways verify the latest updates directly on the official government portal before taking compliance action."

    return answer


def generate_dataset() -> None:
    """Generate fine-tuning dataset from seed data."""

    seed_data = get_seed_data()
    logger.info(f"Loaded {len(seed_data)} seed facts")

    output_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "data",
        "gst_rbi_sebi_mca_dataset.jsonl"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    examples = []
    domain_counts = {}

    for item in seed_data:
        domain = item.get("domain", "gst")
        questions = DOMAIN_QUESTIONS.get(domain, GST_QUESTIONS)
        topic = _extract_topic(
            item.get("circular_number", ""),
            item.get("content", "")
        )
        answer = _generate_answer(item)

        idx = len(examples) % len(questions)
        question = questions[idx].format(topic=topic)

        example = {
            "messages": [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": question
                },
                {
                    "role": "assistant",
                    "content": answer
                }
            ]
        }

        examples.append(example)
        domain_counts[domain] = domain_counts.get(domain, 0) + 1

    # Write JSONL
    with open(output_path, "w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

    logger.info("=" * 50)
    logger.info(f"Dataset generated: {output_path}")
    logger.info(f"Total examples: {len(examples)}")
    logger.info("Domain breakdown:")
    for domain, count in domain_counts.items():
        logger.info(f"  {domain}: {count} examples")
    logger.info("=" * 50)


if __name__ == "__main__":
    generate_dataset()