"""RSS feed ingestion.

Fetches all configured feeds concurrently and returns deduplicated NewsItems.
Deduplication is by URL hash (SHA-256 prefix): the same story from two feeds
appears once.

Fetches run in parallel (asyncio.gather): total latency = max(individual feeds),
not sum — so one slow or blocked feed can't stall the others.
"""

import asyncio
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

_USER_AGENT = "Mozilla/5.0 (compatible; NewsBot/1.0)"


async def _fetch_one(
    client: httpx.AsyncClient, url: str
) -> list[NewsItem]:
    """Fetch a single RSS feed; return [] on any error (don't block other feeds)."""
    try:
        response = await client.get(url)
        feed = feedparser.parse(response.text)
        items = []
        for entry in feed.entries:
            item = _parse_entry(entry, url)
            if item:
                items.append(item)
        return items
    except Exception as exc:
        logger.warning("Error fetching %s: %s", url, exc)
        return []


async def fetch_feeds(config: SentimentConfig) -> list[NewsItem]:
    """Fetch and deduplicate all configured RSS feeds (concurrent)."""
    async with httpx.AsyncClient(
        timeout=config.fetch_timeout_seconds,
        follow_redirects=True,
        headers={"User-Agent": _USER_AGENT},
    ) as client:
        batches = await asyncio.gather(
            *[_fetch_one(client, url) for url in config.rss_feeds]
        )

    items: list[NewsItem] = []
    seen_ids: set[str] = set()
    for batch in batches:
        for item in batch:
            if item.id not in seen_ids:
                items.append(item)
                seen_ids.add(item.id)
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
