"""RSS feed ingestion.

Fetches all configured feeds asynchronously and returns deduplicated NewsItems.
Deduplication is by URL hash (SHA-256 prefix): the same story from two feeds
appears once.
"""

import calendar
import hashlib
import logging
import re
from datetime import datetime, timezone

import feedparser
import httpx

from src.core.config import SentimentConfig
from src.core.models import NewsItem

logger = logging.getLogger(__name__)


async def fetch_feeds(config: SentimentConfig) -> list[NewsItem]:
    """Fetch and deduplicate all configured RSS feeds."""
    items: list[NewsItem] = []
    seen_ids: set[str] = set()

    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
        for url in config.rss_feeds:
            try:
                response = await client.get(url)
                feed = feedparser.parse(response.text)
                for entry in feed.entries:
                    item = _parse_entry(entry, url)
                    if item and item.id not in seen_ids:
                        items.append(item)
                        seen_ids.add(item.id)
            except httpx.HTTPError as exc:
                logger.warning("Error fetching %s: %s", url, exc)

    return items


def _parse_entry(entry: feedparser.util.FeedParserDict, source_url: str) -> NewsItem | None:
    """Convert a feedparser entry into a NewsItem; return None if malformed."""
    url = entry.get("link", "").strip()
    title = entry.get("title", "").strip()
    if not url or not title:
        return None

    news_id = hashlib.sha256(url.encode()).hexdigest()[:16]

    return NewsItem(
        id=news_id,
        title=title,
        source=source_url,
        url=url,
        published_at=_parse_time(entry),
        summary=_strip_html(entry.get("summary", "")),
    )


def _parse_time(entry: feedparser.util.FeedParserDict) -> datetime:
    """Best-effort UTC datetime from a feedparser entry.

    feedparser.published_parsed is a time.struct_time in UTC, so we use
    calendar.timegm (not time.mktime, which would wrongly apply local TZ).
    """
    pt = getattr(entry, "published_parsed", None)
    if pt:
        try:
            return datetime.fromtimestamp(calendar.timegm(pt), tz=timezone.utc)
        except (OverflowError, ValueError):
            pass
    return datetime.now(timezone.utc)


def _strip_html(text: str) -> str:
    """Remove HTML tags for cleaner summary text."""
    return re.sub(r"<[^>]+>", "", text).strip()
