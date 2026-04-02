import sys
import os
import re
import json
import logging
import time
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    API_HOST, API_PORT,
    MAX_QUERY_LENGTH, ALLOWED_DOMAINS, INJECTION_PATTERNS,
    MISTRAL_API_KEY, MISTRAL_MODEL,
)
from agents.graph import run_pipeline

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from mistralai.client import Mistral

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

_nli_model = None

def _load_nli_model():
    global _nli_model
    try:
        from transformers import pipeline as hf_pipeline
        logger.info("Loading NLI model at startup...")
        _nli_model = hf_pipeline(
            "text-classification",
            model="cross-encoder/nli-deberta-v3-small",
            device=-1,
        )
        logger.info("NLI model loaded successfully")
    except Exception as e:
        logger.warning(f"NLI model failed to load: {e} — hallucination check will be skipped")
        _nli_model = None

_load_nli_model()

_mistral_client: Optional[Mistral] = None

def _get_mistral_client() -> Mistral:
    global _mistral_client
    if _mistral_client is None:
        _mistral_client = Mistral(api_key=MISTRAL_API_KEY)
    return _mistral_client

# ── FastAPI App ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="FinComply AI",
    description="India's Real-Time Financial Regulatory Intelligence Agent",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response Models ─────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str
    domain: str = "all"

    @field_validator("query")
    @classmethod
    def query_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Query cannot be empty")
        if len(v) > MAX_QUERY_LENGTH:
            raise ValueError(f"Query too long. Maximum {MAX_QUERY_LENGTH} characters.")
        return v

    @field_validator("domain")
    @classmethod
    def domain_must_be_valid(cls, v: str) -> str:
        v = v.lower().strip()
        if v not in ALLOWED_DOMAINS:
            raise ValueError(f"domain must be one of: {ALLOWED_DOMAINS}")
        return v


class QueryResponse(BaseModel):
    success:          bool
    query:            str
    domain:           str
    answer:           Optional[dict]
    processing_steps: list[str]
    response_time_ms: float
    rejected:         bool = False
    rejection_reason: Optional[str] = None


# ── Safety Layer 1: Fast Pattern Matching ────────────────────────────────────

def _fast_injection_check(query: str) -> bool:
    """
    Quick check against known injection phrases from config.py.
    Returns True if injection detected.
    Runs before any LLM call — zero cost.
    """
    query_lower = query.lower()
    return any(pattern in query_lower for pattern in INJECTION_PATTERNS)


# ── Safety Layer 2: LLM Injection Classifier ─────────────────────────────────

def _llm_injection_check(query: str) -> bool:
    """
    Second-layer LLM classification for sophisticated injection attempts.
    Only called if fast pattern check passes.
    Returns True if injection detected.
    """
    from agents.prompts import INJECTION_DETECTION_PROMPT
    try:
        client = _get_mistral_client()
        response = client.chat.complete(
            model=MISTRAL_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": INJECTION_DETECTION_PROMPT.format(query=query),
                }
            ],
            temperature=0.0,
            max_tokens=100,
        )
        raw = response.choices[0].message.content.strip()
        clean = raw.replace("```json", "").replace("```", "").strip()
        result = json.loads(clean)
        return result.get("is_injection", False)
    except Exception as e:
        logger.warning(f"LLM injection check failed: {e} — defaulting to safe (allow)")
        return False


# ── Safety Layer 3: Input Guardrail ──────────────────────────────────────────

def _input_guardrail(query: str, domain: str) -> Optional[str]:
    """
    Validates query is relevant to FinComply AI scope.
    Returns None if OK, or a rejection reason string.
    """
    pii_patterns = [
        r"\b\d{10}\b",                                              # Phone numbers
        r"\b\d{12}\b",                                              # Aadhaar
        r"\b[A-Z]{5}\d{4}[A-Z]\b",                                 # PAN card
        r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b",   # Email
    ]
    for pattern in pii_patterns:
        if re.search(pattern, query):
            return "Please do not share personal information (phone, Aadhaar, PAN, email) in your query."
    return None


# ── Hallucination NLI Check ───────────────────────────────────────────────────

def _hallucination_check(answer_text: str, source_text: str) -> dict:
    """
    Uses pre-loaded NLI model to verify answer is entailed by source.
    _nli_model is loaded ONCE at startup — not per request.
    """
    if _nli_model is None:
        return {"hallucinated": False, "label": "SKIPPED", "confidence": 0.0}

    try:
        truncated_source = source_text[:512]
        truncated_answer = answer_text[:256]
        result = _nli_model(f"{truncated_source} [SEP] {truncated_answer}")
        label = result[0]["label"]
        score = result[0]["score"]
        return {
            "hallucinated": label == "CONTRADICTION" or (label == "NEUTRAL" and score > 0.8),
            "label": label,
            "confidence": score,
        }
    except Exception as e:
        logger.warning(f"Hallucination check failed: {e} — skipping")
        return {"hallucinated": False, "label": "UNKNOWN", "confidence": 0.0}


# ── API Routes ────────────────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    """Health check endpoint — used by EC2 load balancer and Docker."""
    return {
        "status": "healthy",
        "service": "FinComply AI",
        "nli_model_loaded": _nli_model is not None,
    }


@app.post("/query", response_model=QueryResponse)
def process_query(request: QueryRequest) -> QueryResponse:
    """
    Main query endpoint.
    Full pipeline: Injection Check → Guardrail → Agents → Hallucination Check → Response
    """
    start_time = time.time()
    query = request.query
    domain = request.domain

    logger.info(f"New query | domain={domain} | query={query[:60]}...")

    # ── Step 1: Fast injection check (pattern matching) ───────────────────────
    if _fast_injection_check(query):
        logger.warning(f"Injection detected (fast check): {query[:60]}")
        return QueryResponse(
            success=False,
            query=query,
            domain=domain,
            answer=None,
            processing_steps=["Injection Detected"],
            response_time_ms=_ms(start_time),
            rejected=True,
            rejection_reason="Your query contains patterns that are not allowed. Please ask a compliance question.",
        )

    # ── Step 2: LLM injection check (second layer) ───────────────────────────
    if _llm_injection_check(query):
        logger.warning(f"Injection detected (LLM check): {query[:60]}")
        return QueryResponse(
            success=False,
            query=query,
            domain=domain,
            answer=None,
            processing_steps=["Security Check Failed"],
            response_time_ms=_ms(start_time),
            rejected=True,
            rejection_reason="Your query was flagged by our security system. Please ask a GST, RBI, SEBI, or MCA compliance question.",
        )

    # ── Step 3: Input guardrail (PII check) ───────────────────────────────────
    guardrail_reason = _input_guardrail(query, domain)
    if guardrail_reason:
        return QueryResponse(
            success=False,
            query=query,
            domain=domain,
            answer=None,
            processing_steps=["Input Guardrail Failed"],
            response_time_ms=_ms(start_time),
            rejected=True,
            rejection_reason=guardrail_reason,
        )

    # ── Step 4: Run 3-agent LangGraph pipeline ────────────────────────────────
    try:
        pipeline_result = run_pipeline(query=query, domain=domain)
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        raise HTTPException(status_code=500, detail=f"Agent pipeline failed: {str(e)}")

    final_answer = pipeline_result.get("answer", {})
    processing_steps = pipeline_result.get("processing_steps", [])

    # ── Step 5: Hallucination NLI check (uses pre-loaded model) ──────────────
    answer_text = final_answer.get("final_answer", "")
    source_text = (
        final_answer.get("source_url", "") + " " +
        str(final_answer.get("circular_number", ""))
    )

    if answer_text and source_text.strip():
        hallucination_result = _hallucination_check(answer_text, source_text)
        if hallucination_result["hallucinated"]:
            logger.warning("Hallucination detected — downgrading confidence to LOW")
            final_answer["confidence_level"] = "LOW"
            final_answer["confidence_score"] = min(
                final_answer.get("confidence_score", 0.5), 0.40
            )
            final_answer["confidence_explanation"] = (
                "Confidence lowered: NLI verification detected a potential mismatch "
                "between the answer and the source document. "
                "Please verify directly at the official government portal."
            )

    elapsed_ms = _ms(start_time)
    logger.info(f"Query complete | {elapsed_ms:.0f}ms | confidence={final_answer.get('confidence_level')}")

    return QueryResponse(
        success=True,
        query=query,
        domain=domain,
        answer=final_answer,
        processing_steps=processing_steps,
        response_time_ms=elapsed_ms,
        rejected=False,
    )


def _ms(start_time: float) -> float:
    return (time.time() - start_time) * 1000


# ── Run server ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=API_HOST,
        port=API_PORT,
        reload=True,
        log_level="info",
    )