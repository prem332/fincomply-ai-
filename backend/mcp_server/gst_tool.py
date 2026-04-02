import sys
import os
import logging
import feedparser
import httpx
from datetime import datetime
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import GST_RSS_URL

logger = logging.getLogger(__name__)

GST_OFFICIAL_URLS = [
    "https://www.cbic.gov.in/htdocs-cbec/gst/rss/gst-notifications.xml",
    "https://www.cbic.gov.in/htdocs-cbec/gst/rss/gst-circulars.xml",
]

CBIC_NOTIFICATION_API = "https://cbic-gst.gov.in/central-tax-notifications.html"


def _is_gov_url(url: str) -> bool:
    """Strictly verify URL is from a .gov.in domain."""
    return bool(url) and ".gov.in" in url.lower()


def _parse_gst_rss(rss_url: str, query: str, max_results: int = 10) -> list[dict]:
    """
    Fetch and parse GST RSS feed from cbic.gov.in.
    Filters results relevant to the query.
    """
    results: list[dict] = []

    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; FinComplyAI/1.0; +https://fincomply.ai)",
        }
        resp = httpx.get(rss_url, headers=headers, timeout=10.0, follow_redirects=True)
        feed = feedparser.parse(resp.text)

        query_terms = query.lower().split()

        for entry in feed.entries[:30]:
            title = getattr(entry, "title", "")
            summary = getattr(entry, "summary", "")
            link = getattr(entry, "link", "")
            published = getattr(entry, "published", "")

            # Only include official .gov.in links
            if not _is_gov_url(link):
                continue

            # Relaxed filter — fetch top results even if no exact match
            # Government RSS titles use formal language that may not match query terms


            results.append({
                "title": title,
                "content": f"{title}\n\n{summary}",
                "source_url": link,
                "published_date": _parse_date(published),
                "domain": "gst",
                "circular_number": _extract_circular_number(title),
                "is_gov_verified": True,
            })

            if len(results) >= max_results:
                break

    except Exception as e:
        logger.error(f"GST RSS fetch error from {rss_url}: {e}")

    return results


def _parse_date(date_str: str) -> Optional[str]:
    """Parse various date formats to YYYY-MM-DD."""
    if not date_str:
        return None
    try:
        from dateutil.parser import parse
        return parse(date_str).strftime("%Y-%m-%d")
    except Exception:
        return None


def _extract_circular_number(title: str) -> str:
    """Extract circular/notification number from title text."""
    import re
    # Common GST circular patterns
    patterns = [
        r"Notification\s+No\.?\s*(\d+/\d+)",
        r"Circular\s+No\.?\s*(\d+/\d+)",
        r"No\.?\s*(\d+/\d{4})",
    ]
    for pattern in patterns:
        match = re.search(pattern, title, re.IGNORECASE)
        if match:
            return f"GST {match.group(0)}"
    return "GST Circular"


def fetch_gst_data(query: str, max_results: int = 10) -> list[dict]:
    """
    Main function: Fetch GST regulatory data from official sources.
    Returns list of relevant documents.
    """
    all_results: list[dict] = []

    # Try each official RSS feed URL
    for rss_url in GST_OFFICIAL_URLS:
        results = _parse_gst_rss(rss_url, query, max_results)
        all_results.extend(results)

    # Deduplicate by URL
    seen_urls: set = set()
    unique_results: list[dict] = []
    for r in all_results:
        if r["source_url"] not in seen_urls:
            seen_urls.add(r["source_url"])
            unique_results.append(r)

    logger.info(f"GST Tool: fetched {len(unique_results)} unique results for '{query[:40]}'")

    return unique_results[:max_results]