import sys
import os
import re
import logging
import feedparser
import httpx
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import RBI_RSS_URL

logger = logging.getLogger(__name__)

RBI_RSS_URLS = [
    "https://www.rbi.org.in/scripts/rss.aspx",
    "https://www.rbi.org.in/scripts/NotificationUser.aspx?Id=12500&Mode=0",
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; FinComplyAI/1.0)",
}


def _is_rbi_url(url: str) -> bool:
    return bool(url) and ("rbi.org.in" in url.lower() or "reservebank.gov.in" in url.lower())


def _parse_date(date_str: str) -> Optional[str]:
    if not date_str:
        return None
    try:
        from dateutil.parser import parse
        return parse(date_str).strftime("%Y-%m-%d")
    except Exception:
        return None


def _extract_rbi_circular_number(title: str) -> str:
    patterns = [
        r"RBI/\d{4}-\d{2}/\d+",
        r"DBOD\.\w+\.\w+\.\d+",
        r"Master Direction[^\n]+",
        r"Master Circular[^\n]+",
    ]
    for pattern in patterns:
        match = re.search(pattern, title, re.IGNORECASE)
        if match:
            return match.group(0)[:100]
    return "RBI Circular"


def fetch_rbi_data(query: str, max_results: int = 10) -> list[dict]:
    """Fetch RBI regulatory data from official rbi.org.in RSS feed."""
    results: list[dict] = []
    query_terms = query.lower().split()

    for rss_url in RBI_RSS_URLS:
        try:
            resp = httpx.get(rss_url, headers=HEADERS, timeout=10.0, follow_redirects=True)
            feed = feedparser.parse(resp.text)

            for entry in feed.entries[:30]:
                title = getattr(entry, "title", "")
                summary = getattr(entry, "summary", "")
                link = getattr(entry, "link", "")
                published = getattr(entry, "published", "")

                if not _is_rbi_url(link):
                    continue

                # Relaxed filter — fetch top results even if no exact match
                # Government RSS titles use formal language that may not match query terms


                results.append({
                    "title": title,
                    "content": f"{title}\n\n{summary}",
                    "source_url": link,
                    "published_date": _parse_date(published),
                    "domain": "rbi",
                    "circular_number": _extract_rbi_circular_number(title),
                    "is_gov_verified": True,
                })

                if len(results) >= max_results:
                    break
        except Exception as e:
            logger.error(f"RBI RSS fetch error: {e}")

    logger.info(f"RBI Tool: fetched {len(results)} results")
    return results[:max_results]