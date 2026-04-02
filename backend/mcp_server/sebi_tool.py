import sys
import os
import re
import logging
import feedparser
import httpx
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SEBI_RSS_URL

logger = logging.getLogger(__name__)

SEBI_RSS_URLS = [
    "https://www.sebi.gov.in/sebirss.xml",
]

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; FinComplyAI/1.0)"}


def _is_sebi_url(url: str) -> bool:
    return bool(url) and "sebi.gov.in" in url.lower()


def _parse_date(date_str: str) -> Optional[str]:
    if not date_str:
        return None
    try:
        from dateutil.parser import parse
        return parse(date_str).strftime("%Y-%m-%d")
    except Exception:
        return None


def _extract_sebi_circular_number(title: str) -> str:
    patterns = [
        r"SEBI/LAD-NRO/\w+/\d{4}-\d{2}/\d+",
        r"CIR/[A-Z]+/\d+/\d{4}",
        r"SEBI Circular[^\n]{0,50}",
    ]
    for p in patterns:
        m = re.search(p, title, re.IGNORECASE)
        if m:
            return m.group(0)[:100]
    return "SEBI Circular"


def fetch_sebi_data(query: str, max_results: int = 10) -> list[dict]:
    """Fetch SEBI regulatory data from official sebi.gov.in RSS."""
    results: list[dict] = []
    query_terms = query.lower().split()

    for rss_url in SEBI_RSS_URLS:
        try:
            resp = httpx.get(rss_url, headers=HEADERS, timeout=10.0, follow_redirects=True)
            feed = feedparser.parse(resp.text)
            for entry in feed.entries[:30]:
                title = getattr(entry, "title", "")
                summary = getattr(entry, "summary", "")
                link = getattr(entry, "link", "")
                published = getattr(entry, "published", "")

                if not _is_sebi_url(link):
                    continue

                # Relaxed filter — fetch top results even if no exact match
                # Government RSS titles use formal language that may not match query terms


                results.append({
                    "title": title,
                    "content": f"{title}\n\n{summary}",
                    "source_url": link,
                    "published_date": _parse_date(published),
                    "domain": "sebi",
                    "circular_number": _extract_sebi_circular_number(title),
                    "is_gov_verified": True,
                })
                if len(results) >= max_results:
                    break
        except Exception as e:
            logger.error(f"SEBI RSS fetch error: {e}")

    logger.info(f"SEBI Tool: fetched {len(results)} results")
    return results[:max_results]