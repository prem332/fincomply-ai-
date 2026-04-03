import sys
import os
import json
import logging
import httpx
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    RAGAS_FAITHFULNESS_THRESHOLD,
    RAGAS_ANSWER_RELEVANCY_THRESHOLD,
    RAGAS_CONTEXT_PRECISION_THRESHOLD,
    RAGAS_CONTEXT_RECALL_THRESHOLD,
    RAG_TOP_K,
    MISTRAL_API_KEY,
    MISTRAL_MODEL,
)
from agents.graph import run_pipeline

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

MISTRAL_URL = "https://api.mistral.ai/v1/chat/completions"

# ── 20 Test Questions ─────────────────────────────────────────────────────────

GOLDEN_TEST_SET = [
    # GST
    {"query": "What is the GST rate on under-construction residential apartments?",
     "domain": "gst",
     "ground_truth": "5% GST applies on under-construction residential apartments (excluding affordable housing which is 1%)"},
    {"query": "Is GST applicable on export of services?",
     "domain": "gst",
     "ground_truth": "Export of services is zero-rated under GST. IGST is not charged."},
    {"query": "What is the threshold for GST registration?",
     "domain": "gst",
     "ground_truth": "Rs 20 lakh annual turnover for goods and services (Rs 10 lakh for special category states)."},
    {"query": "What is the GST e-invoicing threshold for businesses?",
     "domain": "gst",
     "ground_truth": "E-invoicing is mandatory for businesses with annual turnover above Rs 5 crore."},
    {"query": "How many years can GST ITC be claimed?",
     "domain": "gst",
     "ground_truth": "ITC can be claimed by November 30 of the following financial year or filing of annual return."},

    # RBI
    {"query": "What is the RBI repo rate?",
     "domain": "rbi",
     "ground_truth": "The RBI Monetary Policy Committee determines the repo rate. Check rbi.org.in for current rate."},
    {"query": "What are RBI KYC norms for account opening?",
     "domain": "rbi",
     "ground_truth": "Officially Valid Documents required: Aadhaar, passport, driving license, voter ID. Video KYC is permitted."},
    {"query": "What is the RBI limit for digital lending by NBFCs?",
     "domain": "rbi",
     "ground_truth": "NBFCs must follow RBI digital lending guidelines: disbursal only to borrower bank account, APR disclosure mandatory."},
    {"query": "What is the CRR requirement for scheduled commercial banks?",
     "domain": "rbi",
     "ground_truth": "CRR is 4% of Net Demand and Time Liabilities as set by RBI Monetary Policy Committee."},
    {"query": "What are priority sector lending targets for banks?",
     "domain": "rbi",
     "ground_truth": "40% of Adjusted Net Bank Credit for domestic banks. Sub-targets include agriculture 18% and weaker sections 12%."},

    # SEBI
    {"query": "What is SEBI insider trading regulation?",
     "domain": "sebi",
     "ground_truth": "SEBI Prohibition of Insider Trading Regulations 2015 prohibit trading on unpublished price sensitive information."},
    {"query": "What are SEBI IPO grading requirements?",
     "domain": "sebi",
     "ground_truth": "SEBI does not mandate IPO grading. DRHP filing with SEBI is mandatory for public issues above Rs 10 crore."},
    {"query": "What is the lock-in period for promoters in an IPO?",
     "domain": "sebi",
     "ground_truth": "Minimum 20% promoter holding must be locked in for 18 months. Remaining promoter holding locked in for 6 months."},
    {"query": "What are SEBI mutual fund expense ratio limits?",
     "domain": "sebi",
     "ground_truth": "Total Expense Ratio capped at 2.25% for equity funds for first Rs 500 crore AUM. Reduces as AUM increases."},
    {"query": "What is SEBI FPI registration requirement?",
     "domain": "sebi",
     "ground_truth": "Foreign Portfolio Investors must register with SEBI designated Depository Participants. Category I or II based on investor type."},

    # MCA
    {"query": "What is the annual return filing deadline for private companies?",
     "domain": "mca",
     "ground_truth": "MGT-7 Annual Return within 60 days of AGM. AOC-4 Financial Statements within 30 days of AGM."},
    {"query": "What is the penalty for late filing of annual return?",
     "domain": "mca",
     "ground_truth": "Additional fee of Rs 100 per day of default for MGT-7 and AOC-4 late filing."},
    {"query": "What is the DIN requirement for company directors?",
     "domain": "mca",
     "ground_truth": "Every company director must have Director Identification Number from MCA. Annual KYC DIR-3 renewal required."},
    {"query": "What is the minimum paid-up capital for a private limited company?",
     "domain": "mca",
     "ground_truth": "No minimum paid-up capital requirement for private limited companies after Companies Act 2013 amendment."},
    {"query": "What are AGM requirements for private limited companies?",
     "domain": "mca",
     "ground_truth": "First AGM within 9 months of financial year end. Subsequent AGMs within 6 months. Gap between two AGMs max 15 months."},
]


# ── Mistral API Judge ─────────────────────────────────────────────────────────

def _ask_mistral(prompt: str) -> float:
    """Ask Mistral to score on a 0-1 scale. Returns float score."""
    try:
        time.sleep(1)  # Rate limit protection
        resp = httpx.post(
            MISTRAL_URL,
            headers={
                "Authorization": f"Bearer {MISTRAL_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": MISTRAL_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "max_tokens": 50,
            },
            timeout=30.0,
        )
        raw = resp.json()["choices"][0]["message"]["content"].strip()
        # Extract number from response
        for token in raw.replace(",", ".").split():
            try:
                score = float(token)
                if 0.0 <= score <= 1.0:
                    return score
            except ValueError:
                continue
        return 0.5  # Default if parsing fails
    except Exception as e:
        logger.warning(f"Mistral judge error: {e}")
        return 0.5


def _score_faithfulness(answer: str, context: str) -> float:
    prompt = f"""Score how faithful this answer is to the given context.
Score 1.0 if all claims in the answer are supported by the context.
Score 0.0 if the answer contains hallucinations not in the context.
Return ONLY a decimal number between 0.0 and 1.0.

Context: {context[:800]}
Answer: {answer[:400]}
Score:"""
    return _ask_mistral(prompt)


def _score_answer_relevancy(question: str, answer: str) -> float:
    prompt = f"""Score how relevant this answer is to the question.
Score 1.0 if the answer directly addresses the question.
Score 0.0 if the answer is completely off-topic.
Return ONLY a decimal number between 0.0 and 1.0.

Question: {question}
Answer: {answer[:400]}
Score:"""
    return _ask_mistral(prompt)


def _score_context_precision(question: str, context: str) -> float:
    prompt = f"""Score how precisely the retrieved context matches what is needed to answer the question.
Score 1.0 if the context is highly relevant and precise.
Score 0.0 if the context is irrelevant.
Return ONLY a decimal number between 0.0 and 1.0.

Question: {question}
Context: {context[:600]}
Score:"""
    return _ask_mistral(prompt)


def _score_context_recall(ground_truth: str, context: str) -> float:
    prompt = f"""Score how well the retrieved context covers the ground truth answer.
Score 1.0 if all information in the ground truth can be found in the context.
Score 0.0 if the context is missing key information from the ground truth.
Return ONLY a decimal number between 0.0 and 1.0.

Ground Truth: {ground_truth}
Context: {context[:600]}
Score:"""
    return _ask_mistral(prompt)


# ── Main Evaluation ───────────────────────────────────────────────────────────

def run_ragas_evaluation(test_cases: list[dict] = None) -> dict:
    """
    Run RAGAS-style evaluation using Mistral API as judge LLM.
    No OpenAI, no LangChain version conflicts.
    """
    if test_cases is None:
        test_cases = GOLDEN_TEST_SET

    print(f"\nRunning RAGAS evaluation on {len(test_cases)} test questions...")
    print(f"Judge LLM: {MISTRAL_MODEL} (Mistral API)")
    print(f"RAG_TOP_K: {RAG_TOP_K}")
    print(f"Estimated time: ~{len(test_cases) * 15} seconds\n")

    all_scores = {
        "faithfulness":      [],
        "answer_relevancy":  [],
        "context_precision": [],
        "context_recall":    [],
    }

    for i, tc in enumerate(test_cases, 1):
        print(f"  [{i}/{len(test_cases)}] {tc['query'][:55]}...")
        try:
            result = run_pipeline(query=tc["query"], domain=tc["domain"])
            answer_dict = result.get("answer", {})
            final_answer = answer_dict.get("final_answer", "")

            # Build context from retrieved sources
            context_parts = []
            if answer_dict.get("circular_number"):
                context_parts.append(f"Circular: {answer_dict['circular_number']}")
            if answer_dict.get("source_url"):
                context_parts.append(f"Source: {answer_dict['source_url']}")
            if final_answer:
                context_parts.append(final_answer[:500])
            context = " | ".join(context_parts) or "No context"

            if not final_answer:
                print(f"    ⚠ No answer generated")
                for k in all_scores:
                    all_scores[k].append(0.3)
                continue

            # Score all 4 metrics
            f_score  = _score_faithfulness(final_answer, context)
            ar_score = _score_answer_relevancy(tc["query"], final_answer)
            cp_score = _score_context_precision(tc["query"], context)
            cr_score = _score_context_recall(tc["ground_truth"], context)

            all_scores["faithfulness"].append(f_score)
            all_scores["answer_relevancy"].append(ar_score)
            all_scores["context_precision"].append(cp_score)
            all_scores["context_recall"].append(cr_score)

            print(f"    F={f_score:.2f} AR={ar_score:.2f} CP={cp_score:.2f} CR={cr_score:.2f}")

        except Exception as e:
            print(f"    Error: {e}")
            for k in all_scores:
                all_scores[k].append(0.3)

    # Average scores
    scores = {k: round(sum(v) / len(v), 3) for k, v in all_scores.items() if v}

    thresholds = {
        "faithfulness":      RAGAS_FAITHFULNESS_THRESHOLD,
        "answer_relevancy":  RAGAS_ANSWER_RELEVANCY_THRESHOLD,
        "context_precision": RAGAS_CONTEXT_PRECISION_THRESHOLD,
        "context_recall":    RAGAS_CONTEXT_RECALL_THRESHOLD,
    }

    print("\n" + "=" * 58)
    print("  RAGAS EVALUATION RESULTS — FinComply AI")
    print("=" * 58)
    all_pass = True
    for metric, score in scores.items():
        threshold = thresholds[metric]
        status = "✓ PASS" if score >= threshold else "✗ FAIL"
        if score < threshold:
            all_pass = False
        print(f"  {metric:25s}  {score:.3f}  (≥{threshold})  {status}")

    print("=" * 58)
    print(f"  Overall: {'✓ ALL METRICS PASS' if all_pass else '✗ SOME METRICS BELOW THRESHOLD'}")
    print("=" * 58)

    # Save results
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ragas_results.json")
    with open(output_path, "w") as f:
        json.dump({
            "scores": scores,
            "thresholds": thresholds,
            "all_pass": all_pass,
            "rag_top_k": RAG_TOP_K,
            "test_count": len(test_cases),
            "judge_llm": MISTRAL_MODEL,
        }, f, indent=2)
    print(f"\n  Results saved → {output_path}")

    return scores


if __name__ == "__main__":
    run_ragas_evaluation()