import sys
import os
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    DATABASE_URL,
    VECTOR_TABLE_NAME,
    EMBEDDING_MODEL,
    RAG_TOP_K,
)
from database.seed_data import get_seed_data

import psycopg2
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def _embedding_to_pg_string(embedding: list[float]) -> str:
    """Format embedding as pgvector string."""
    return "[" + ",".join(f"{v:.6f}" for v in embedding) + "]"


def ingest_seed_data() -> None:
    """
    Load all seed regulatory data into pgvector.
    Generates embeddings for each document and stores in RDS.
    """
    logger.info("Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    logger.info(f"  Model loaded: {EMBEDDING_MODEL}")

    logger.info("Connecting to database...")
    conn = psycopg2.connect(DATABASE_URL)
    cursor = conn.cursor()
    logger.info("  Connected successfully")

    # Clear existing seed data (clean reload)
    logger.info("Clearing existing seed data...")
    cursor.execute(f"DELETE FROM {VECTOR_TABLE_NAME} WHERE metadata->>'source' = 'seed'")
    deleted = cursor.rowcount
    conn.commit()
    logger.info(f"  Deleted {deleted} existing seed records")

    # Load seed data
    seed_data = get_seed_data()
    logger.info(f"Loading {len(seed_data)} regulatory facts into vector DB...")
    logger.info(f"  RAG_TOP_K = {RAG_TOP_K} (documents retrieved per query)")

    success_count = 0
    error_count = 0

    for i, item in enumerate(seed_data, 1):
        try:
            # Generate embedding for the content
            embedding = model.encode(item["content"], normalize_embeddings=True).tolist()
            embedding_str = _embedding_to_pg_string(embedding)

            # Insert into vector DB
            cursor.execute(
                f"""
                INSERT INTO {VECTOR_TABLE_NAME}
                (content, domain, circular_number, source_url, published_date, is_gov_verified, metadata, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s::vector)
                """,
                (
                    item["content"],
                    item["domain"],
                    item.get("circular_number", ""),
                    item.get("source_url", ""),
                    item.get("published_date"),
                    True,
                    '{"source": "seed"}',
                    embedding_str,
                )
            )

            success_count += 1

            if i % 10 == 0:
                conn.commit()
                logger.info(f"  Progress: {i}/{len(seed_data)} ({item['domain']})")

        except Exception as e:
            error_count += 1
            logger.error(f"  Error on item {i} ({item.get('domain')}): {e}")
            conn.rollback()

    conn.commit()
    cursor.close()
    conn.close()

    logger.info("")
    logger.info("=" * 50)
    logger.info("Ingestion complete!")
    logger.info(f"  Successfully loaded: {success_count} facts")
    logger.info(f"  Errors: {error_count}")
    logger.info(f"  RAG will retrieve top {RAG_TOP_K} documents per query")
    logger.info("=" * 50)


def verify_ingestion() -> None:
    """Verify data was loaded correctly."""
    conn = psycopg2.connect(DATABASE_URL)
    cursor = conn.cursor()

    cursor.execute(f"SELECT domain, COUNT(*) FROM {VECTOR_TABLE_NAME} GROUP BY domain ORDER BY domain")
    rows = cursor.fetchall()

    logger.info("\nVector DB contents:")
    total = 0
    for domain, count in rows:
        logger.info(f"  {domain}: {count} documents")
        total += count
    logger.info(f"  Total: {total} documents")

    conn.close()


if __name__ == "__main__":
    ingest_seed_data()
    verify_ingestion()
