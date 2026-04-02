import sys
import os
import json
import logging
import httpx
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    MISTRAL_API_KEY,
    MISTRAL_FINE_TUNED_MODEL,
    DATABASE_URL,
    RAG_TOP_K,
    VECTOR_TABLE_NAME,
    EMBEDDING_MODEL,
    VECTOR_DIMENSION,
    GST_TOOL_URL,
    RBI_TOOL_URL,
    SEBI_TOOL_URL,
    MCA_TOOL_URL,
    ITAX_TOOL_URL,
)
from agents.prompts import RESEARCH_AGENT_SYSTEM, RESEARCH_AGENT_QUERY

import psycopg2
from mistralai.client import Mistral
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# ── Module-level singletons ───────────────────────────────────────────────────
_mistral_client: Optional[Mistral] = None
_embedding_model: Optional[SentenceTransformer] = None


def _get_mistral_client() -> Mistral:
    global _mistral_client
    if _mistral_client is None:
        _mistral_client = Mistral(api_key=MISTRAL_API_KEY)
    return _mistral_client


def _get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    return _embedding_model


# ── Helpers ───────────────────────────────────────────────────────────────────

def _is_valid_gov_url(url: str) -> bool:
    """Returns True if URL is from an official government source."""
    if not url or url in ("Unknown", "N/A", ""):
        return False
    url_lower = url.lower()
    return ".gov.in" in url_lower or "rbi.org.in" in url_lower


def _embed_text(text: str) -> list[float]:
    model = _get_embedding_model()
    return model.encode(text, normalize_embeddings=True).tolist()


def _embedding_to_pg_string(embedding: list[float]) -> str:
    return "[" + ",".join(f"{v:.6f}" for v in embedding) + "]"


# ── Vector DB Retrieval ───────────────────────────────────────────────────────

def retrieve_from_vector_db(query: str, domain: str) -> list[dict]:
    """
    Retrieve top-K relevant regulatory documents using cosine similarity.
    RAG_TOP_K imported from config.py — single source of truth.
    """
    embedding = _embed_text(query)
    embedding_str = _embedding_to_pg_string(embedding)

    try:
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()

        if domain == "all":
            cursor.execute(
                f"""
                SELECT content, domain, circular_number, source_url,
                       published_date, metadata,
                       1 - (embedding <=> %s::vector) AS similarity
                FROM {VECTOR_TABLE_NAME}
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (embedding_str, embedding_str, RAG_TOP_K),
            )
        else:
            cursor.execute(
                f"""
                SELECT content, domain, circular_number, source_url,
                       published_date, metadata,
                       1 - (embedding <=> %s::vector) AS similarity
                FROM {VECTOR_TABLE_NAME}
                WHERE domain = %s
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (embedding_str, domain, embedding_str, RAG_TOP_K),
            )

        rows = cursor.fetchall()
        conn.close()

        return [
            {
                "content": row[0],
                "domain": row[1],
                "circular_number": row[2],
                "source_url": row[3],
                "published_date": str(row[4]) if row[4] else None,
                "metadata": row[5] or {},
                "similarity": float(row[6]),
            }
            for row in rows
        ]

    except Exception as e:
        logger.error(f"Vector DB retrieval error: {e}")
        return []


# ── MCP Tool Calls (Parallel) ─────────────────────────────────────────────────

def _fetch_single_mcp(d: str, url: str, query: str) -> list[dict]:
    """Fetch from a single MCP tool — called in parallel via ThreadPoolExecutor."""
    try:
        resp = httpx.post(
            url,
            json={"query": query, "domain": d},
            timeout=5.0,
        )
        if resp.status_code == 200:
            data = resp.json()
            # Only keep results with valid government URLs
            return [
                r for r in data.get("results", [])
                if _is_valid_gov_url(r.get("source_url", ""))
            ]
    except Exception as e:
        logger.warning(f"MCP tool call failed for {d}: {e}")
    return []


def _call_mcp_tool(domain: str, query: str) -> list[dict]:
    """
    Call MCP Lambda tools in PARALLEL for faster response.
    All domain tools called simultaneously — saves 3-4 seconds vs sequential.
    Only returns documents with valid .gov.in or rbi.org.in URLs.
    """
    url_map = {
        "gst":        GST_TOOL_URL,
        "rbi":        RBI_TOOL_URL,
        "sebi":       SEBI_TOOL_URL,
        "mca":        MCA_TOOL_URL,
        "income_tax": ITAX_TOOL_URL,
    }

    domains_to_call = list(url_map.keys()) if domain == "all" else [domain]
    live_results: list[dict] = []

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {
            executor.submit(_fetch_single_mcp, d, url_map[d], query): d
            for d in domains_to_call
            if url_map.get(d)
        }
        for future in futures:
            try:
                results = future.result()
                live_results.extend(results)
            except Exception as e:
                logger.warning(f"MCP future error: {e}")

    return live_results


# ── Context Formatting ────────────────────────────────────────────────────────

def _format_context(rag_docs: list[dict], live_docs: list[dict]) -> str:
    """Combine RAG and live MCP results into a single context string."""
    parts: list[str] = []

    if live_docs:
        parts.append("=== LIVE GOVERNMENT DATA (fetched now) ===")
        for i, doc in enumerate(live_docs[:5], 1):
            parts.append(
                f"[Live-{i}] Source: {doc.get('source_url', 'Unknown')}\n"
                f"Circular: {doc.get('circular_number', 'N/A')}\n"
                f"Date: {doc.get('published_date', 'Unknown')}\n"
                f"Content: {doc.get('content', '')[:800]}"
            )

    if rag_docs:
        parts.append("=== VECTOR DB (historical regulatory documents) ===")
        for i, doc in enumerate(rag_docs, 1):
            parts.append(
                f"[RAG-{i}] Domain: {doc['domain']} | "
                f"Circular: {doc.get('circular_number', 'N/A')} | "
                f"Similarity: {doc['similarity']:.2f}\n"
                f"Source: {doc.get('source_url', 'Unknown')}\n"
                f"Date: {doc.get('published_date', 'Unknown')}\n"
                f"Content: {doc['content'][:800]}"
            )

    return "\n\n".join(parts) if parts else "No relevant regulatory documents found."


# ── Main Research Agent ───────────────────────────────────────────────────────

def run_research_agent(query: str, domain: str) -> dict:
    """
    Main function for Research Agent (Agent 1).
    1. Retrieves from vector DB (RAG_TOP_K=10 documents)
    2. Calls MCP tools in parallel for live data
    3. Generates answer using Mistral
    Returns structured JSON answer.
    """
    logger.info(f"Research Agent | Query: {query[:60]}... | Domain: {domain}")

    # Step 1: Vector DB retrieval
    rag_docs = retrieve_from_vector_db(query, domain)
    logger.info(f"  Retrieved {len(rag_docs)} documents from vector DB (TOP_K={RAG_TOP_K})")

    # Step 2: Parallel MCP tool calls (valid gov URLs only)
    live_docs = _call_mcp_tool(domain, query)
    logger.info(f"  Fetched {len(live_docs)} live documents from MCP tools")

    # Step 3: Format context
    context = _format_context(rag_docs, live_docs)

    # Step 4: Generate answer with Mistral
    client = _get_mistral_client()
    user_message = RESEARCH_AGENT_QUERY.format(
        context=context,
        query=query,
        domain=domain,
    )

    try:
        response = client.chat.complete(
            model=MISTRAL_FINE_TUNED_MODEL,
            messages=[
                {"role": "system", "content": RESEARCH_AGENT_SYSTEM},
                {"role": "user", "content": user_message},
            ],
            temperature=0.1,
            max_tokens=1500,
        )
        raw_answer = response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Mistral API error in Research Agent: {e}")
        return {
            "error": str(e),
            "summary": "Research failed due to API error.",
            "confidence_level": "LOW",
        }

    # Step 5: Parse JSON response
    try:
        clean = raw_answer.replace("```json", "").replace("```", "").strip()
        answer_dict = json.loads(clean)
    except json.JSONDecodeError:
        logger.warning("Research Agent returned non-JSON — wrapping as plain text")
        answer_dict = {
            "summary": "Regulatory information retrieved.",
            "detailed_answer": raw_answer,
            "circular_number": "Unknown",
            "source_url": "Unknown",
            "published_date": None,
            "deadline": None,
            "action_required": "Verify with official source.",
            "domain": domain,
        }


    answer_dict["_rag_sources"] = rag_docs
    answer_dict["_live_sources"] = live_docs

    return answer_dict