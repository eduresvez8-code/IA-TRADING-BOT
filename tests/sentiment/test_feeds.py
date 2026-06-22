"""Tests for RSS feed ingestion.

We test _parse_entry (pure function) directly, and mock httpx for fetch_feeds
so tests run without network access.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.config import SentimentConfig
from src.sentiment.feeds import _parse_entry, _parse_time, _strip_html, fetch_feeds

MOCK_RSS = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>CoinDesk</title>
    <item>
      <title>Bitcoin Hits New High</title>
      <link>https://coindesk.com/bitcoin-high</link>
      <pubDate>Fri, 13 Jun 2026 12:00:00 +0000</pubDate>
      <description>BTC surpasses previous record at $112,000.</description>
    </item>
    <item>
      <title>Ethereum Upgrade Live</title>
      <link>https://coindesk.com/eth-upgrade</link>
      <pubDate>Fri, 13 Jun 2026 10:00:00 +0000</pubDate>
      <description>Network throughput improved after mainnet upgrade.</description>
    </item>
  </channel>
</rss>"""

MOCK_RSS_SAME_ARTICLE = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <item>
      <title>Bitcoin Hits New High</title>
      <link>https://coindesk.com/bitcoin-high</link>
      <pubDate>Fri, 13 Jun 2026 12:00:00 +0000</pubDate>
      <description>Duplicate from second feed.</description>
    </item>
  </channel>
</rss>"""


def make_config(feeds: list[str] | None = None) -> SentimentConfig:
    return SentimentConfig(
        enabled=False,
        rss_feeds=feeds or ["https://coindesk.com/rss"],
        poll_interval_seconds=120,
        fetch_timeout_seconds=10,
        claude_model="claude-haiku-4-5-20251001",
        heuristic_weight=0.7,
        escalate_score_threshold=0.3,
        max_news_age_hours=24,
    )


# ---------------------------------------------------------------------------
# _strip_html
# ---------------------------------------------------------------------------


def test_strip_html_removes_tags():
    assert _strip_html("<p>Hello <b>world</b></p>") == "Hello world"


def test_strip_html_plain_text_unchanged():
    assert _strip_html("plain text") == "plain text"


def test_strip_html_empty_string():
    assert _strip_html("") == ""


# ---------------------------------------------------------------------------
# _parse_entry (pure, no network)
# ---------------------------------------------------------------------------

import feedparser


def _make_entry(title: str, link: str, summary: str = "") -> feedparser.util.FeedParserDict:
    """Build a minimal feedparser entry from raw RSS."""
    feed = feedparser.parse(f"""<?xml version="1.0"?>
    <rss version="2.0"><channel>
    <item>
      <title>{title}</title>
      <link>{link}</link>
      <description>{summary}</description>
    </item>
    </channel></rss>""")
    return feed.entries[0]


def test_parse_entry_returns_news_item():
    entry = _make_entry("BTC rally", "https://example.com/btc", "Bitcoin up 10%")
    item = _parse_entry(entry, "https://example.com/rss")
    assert item is not None
    assert item.title == "BTC rally"
    assert item.url == "https://example.com/btc"
    assert "Bitcoin" in item.summary


def test_parse_entry_id_is_deterministic():
    entry = _make_entry("Test", "https://example.com/test")
    item1 = _parse_entry(entry, "https://x.com")
    item2 = _parse_entry(entry, "https://y.com")
    assert item1 is not None and item2 is not None
    assert item1.id == item2.id  # same URL → same id regardless of source


def test_parse_entry_missing_link_returns_none():
    feed = feedparser.parse("""<?xml version="1.0"?>
    <rss version="2.0"><channel>
    <item><title>No link here</title></item>
    </channel></rss>""")
    entry = feed.entries[0]
    result = _parse_entry(entry, "https://example.com")
    assert result is None


def test_parse_entry_missing_title_returns_none():
    feed = feedparser.parse("""<?xml version="1.0"?>
    <rss version="2.0"><channel>
    <item><link>https://example.com/article</link></item>
    </channel></rss>""")
    entry = feed.entries[0]
    result = _parse_entry(entry, "https://example.com")
    assert result is None


def test_parse_entry_published_at_is_utc():
    entry = _make_entry("Test", "https://example.com/t", "")
    item = _parse_entry(entry, "https://example.com")
    assert item is not None
    assert item.published_at.tzinfo is not None


# ---------------------------------------------------------------------------
# fetch_feeds (mocked httpx)
# ---------------------------------------------------------------------------


async def test_fetch_feeds_returns_items():
    config = make_config()

    mock_response = MagicMock()
    mock_response.text = MOCK_RSS

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.get = AsyncMock(return_value=mock_response)

    with patch("src.sentiment.feeds.httpx.AsyncClient", return_value=mock_client):
        items = await fetch_feeds(config)

    assert len(items) == 2
    titles = {i.title for i in items}
    assert "Bitcoin Hits New High" in titles
    assert "Ethereum Upgrade Live" in titles


async def test_fetch_feeds_deduplicates_across_feeds():
    """Same article URL in two feeds → appears once."""
    config = make_config(
        feeds=["https://feed1.com/rss", "https://feed2.com/rss"]
    )

    def side_effect(url):
        response = MagicMock()
        response.text = MOCK_RSS if "feed1" in url else MOCK_RSS_SAME_ARTICLE
        return response

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.get = AsyncMock(side_effect=side_effect)

    with patch("src.sentiment.feeds.httpx.AsyncClient", return_value=mock_client):
        items = await fetch_feeds(config)

    # feed1 has 2 articles, feed2 has 1 that duplicates feed1's first → 2 unique
    assert len(items) == 2


async def test_fetch_feeds_continues_on_http_error():
    """A failing feed should not abort the other feeds."""
    import httpx as _httpx

    config = make_config(
        feeds=["https://broken.com/rss", "https://good.com/rss"]
    )

    def side_effect(url):
        if "broken" in url:
            raise _httpx.ConnectError("Connection refused")
        mock_response = MagicMock()
        mock_response.text = MOCK_RSS
        return mock_response

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.get = AsyncMock(side_effect=side_effect)

    with patch("src.sentiment.feeds.httpx.AsyncClient", return_value=mock_client):
        items = await fetch_feeds(config)

    # Good feed still returns 2 items despite broken feed failing
    assert len(items) == 2
