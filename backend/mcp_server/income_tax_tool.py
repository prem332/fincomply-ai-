import sys
import os
import re
import logging
import feedparser
import httpx
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ITAX_RSS_URL

logger = logging.getLogger(__name__)

ITAX_URLS = [
    "https://www.incometaxindia.gov.in/pages/whats-new.aspx",
    "https://www.incometaxindia.gov.in/communications/circular/circular.aspx",
]

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; FinComplyAI/1.0)"}


def _is_itax_url(url: str) -> bool:
    return bool(url) and ".gov.in" in url.lower()


def _parse_date(date_str: str) -> Optional[str]:
    if not date_str:
        return None
    try:
        from dateutil.parser import parse
        return parse(date_str).strftime("%Y-%m-%d")
    except Exception:
        return None


def _extract_itax_circular_number(title: str) -> str:
    patterns = [
        r"Circular\s+No\.?\s*\d+/\d{4}",
        r"Notification\s+No\.?\s*\d+/\d{4}",
        r"CBDT\s+Circular[^\n]{0,50}",
        r"Section\s+\d+[A-Z]?",
    ]
    for p in patterns:
        m = re.search(p, title, re.IGNORECASE)
        if m:
            return m.group(0)[:100]
    return "Income Tax Circular"


def fetch_income_tax_data(query: str, max_results: int = 10) -> list[dict]:
    """Fetch Income Tax regulatory data from official incometaxindia.gov.in."""
    results: list[dict] = []

    for url in ITAX_URLS:
        try:
            resp = httpx.get(url, headers=HEADERS, timeout=10.0, follow_redirects=True)
            feed = feedparser.parse(resp.text)

            for entry in feed.entries[:30]:
                title = getattr(entry, "title", "")
                summary = getattr(entry, "summary", "")
                link = getattr(entry, "link", "")
                published = getattr(entry, "published", "")

                if not _is_itax_url(link):
                    continue

                results.append({
                    "title": title,
                    "content": f"{title}\n\n{summary}",
                    "source_url": link,
                    "published_date": _parse_date(published),
                    "domain": "income_tax",
                    "circular_number": _extract_itax_circular_number(title),
                    "is_gov_verified": True,
                })
                if len(results) >= max_results:
                    break
        except Exception as e:
            logger.error(f"Income Tax fetch error: {e}")

    logger.info(f"Income Tax Tool: fetched {len(results)} results")
    return results[:max_results]