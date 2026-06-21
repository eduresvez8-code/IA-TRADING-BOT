"""Tests for the Claude Haiku sentiment analyzer.

All Claude API calls are mocked — tests run without a real API key.
"""

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from src.core.config import SentimentConfig, Secrets
from src.core.models import NewsItem, SentimentScore
from src.sentiment.analyzer import analyze


def make_config() -> SentimentConfig:
    return SentimentConfig(
        enabled=False,
        rss_feeds=["https://coindesk.com/rss"],
        poll_interval_seconds=120,
        claude_model="claude-haiku-4-5-20251001",
        heuristic_weight=0.7,
        escalate_score_threshold=0.3,
        max_news_age_hours=24,
    )


def make_secrets(key: str = "sk-test") -> Secrets:
    return Secrets(anthropic_api_key=key)


def make_news(
    title: str = "Bitcoin ETF approved",
    summary: str = "Regulators approve spot Bitcoin ETF.",
) -> NewsItem:
    return NewsItem(
        id="abc123",
        title=title,
        source="https://coindesk.com",
        url="https://coindesk.com/test",
        published_at=datetime.now(timezone.utc),
        summary=summary,
    )


def _mock_claude(json_payload: dict) -> AsyncMock:
    """Build a mock anthropic.AsyncAnthropic that returns json_payload."""
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text=json.dumps(json_payload))]

    mock_messages = AsyncMock()
    mock_messages.create = AsyncMock(return_value=mock_response)

    mock_client = MagicMock()
    mock_client.messages = mock_messages

    return mock_client


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


async def test_analyze_returns_sentiment_score():
    payload = {
        "score": 0.85,
        "confidence": 0.9,
        "high_impact": True,
        "symbol_scope": ["BTC"],
        "rationale": "ETF approval is a major bullish catalyst.",
    }
    with patch("src.sentiment.analyzer.anthropic.AsyncAnthropic", return_value=_mock_claude(payload)):
        result = await analyze(make_news(), make_config(), make_secrets())

    assert isinstance(result, SentimentScore)
    assert result.news_id == "abc123"
    assert result.score == pytest.approx(0.85)
    assert result.confidence == pytest.approx(0.9)
    assert result.high_impact is True
    assert result.symbol_scope == ["BTC"]
    assert "ETF" in result.rationale


async def test_analyze_negative_score():
    payload = {
        "score": -0.9,
        "confidence": 0.95,
        "high_impact": True,
        "symbol_scope": ["*"],
        "rationale": "Exchange hack destroys market confidence.",
    }
    with patch("src.sentiment.analyzer.anthropic.AsyncAnthropic", return_value=_mock_claude(payload)):
        result = await analyze(
            make_news("Exchange hacked, $100M stolen"), make_config(), make_secrets()
        )

    assert result.score < 0
    assert result.high_impact is True


async def test_analyze_neutral_score():
    payload = {
        "score": 0.0,
        "confidence": 0.5,
        "high_impact": False,
        "symbol_scope": ["BTC"],
        "rationale": "Mining difficulty update has no price implication.",
    }
    with patch("src.sentiment.analyzer.anthropic.AsyncAnthropic", return_value=_mock_claude(payload)):
        result = await analyze(
            make_news("Bitcoin mining difficulty stable"), make_config(), make_secrets()
        )

    assert result.score == pytest.approx(0.0)
    assert not result.high_impact


async def test_analyze_whole_market_scope():
    payload = {
        "score": -0.4,
        "confidence": 0.7,
        "high_impact": True,
        "symbol_scope": ["*"],
        "rationale": "Fed rate hike dampens all risk assets.",
    }
    with patch("src.sentiment.analyzer.anthropic.AsyncAnthropic", return_value=_mock_claude(payload)):
        result = await analyze(make_news(), make_config(), make_secrets())

    assert result.symbol_scope == ["*"]


# ---------------------------------------------------------------------------
# Score clamping (Claude might return slightly out-of-range values)
# ---------------------------------------------------------------------------


async def test_score_is_clamped_above_one():
    payload = {
        "score": 1.5,  # out of range
        "confidence": 0.8,
        "high_impact": False,
        "symbol_scope": ["ETH"],
        "rationale": "Test.",
    }
    with patch("src.sentiment.analyzer.anthropic.AsyncAnthropic", return_value=_mock_claude(payload)):
        result = await analyze(make_news(), make_config(), make_secrets())

    assert result.score <= 1.0


async def test_confidence_is_clamped_above_one():
    payload = {
        "score": 0.5,
        "confidence": 1.2,  # out of range
        "high_impact": False,
        "symbol_scope": ["BTC"],
        "rationale": "Test.",
    }
    with patch("src.sentiment.analyzer.anthropic.AsyncAnthropic", return_value=_mock_claude(payload)):
        result = await analyze(make_news(), make_config(), make_secrets())

    assert result.confidence <= 1.0


async def test_score_is_clamped_below_minus_one():
    payload = {
        "score": -1.8,
        "confidence": 0.9,
        "high_impact": True,
        "symbol_scope": ["*"],
        "rationale": "Test.",
    }
    with patch("src.sentiment.analyzer.anthropic.AsyncAnthropic", return_value=_mock_claude(payload)):
        result = await analyze(make_news(), make_config(), make_secrets())

    assert result.score >= -1.0


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def _mock_claude_raw(text: str) -> MagicMock:
    """Mock que devuelve `text` CRUDO (sin json.dumps): para simular fences/preámbulos."""
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text=text)]
    mock_client = MagicMock()
    mock_client.messages = AsyncMock()
    mock_client.messages.create = AsyncMock(return_value=mock_response)
    return mock_client


async def test_analyze_raises_when_no_json_object():
    """Sin ningún objeto JSON (no hay llaves) → ValueError claro."""
    mock_client = _mock_claude_raw("Sorry, I cannot analyze this.")
    with patch("src.sentiment.analyzer.anthropic.AsyncAnthropic", return_value=mock_client):
        with pytest.raises(ValueError, match="no JSON object"):
            await analyze(make_news(), make_config(), make_secrets())


async def test_analyze_raises_on_malformed_json_object():
    """Hay llaves pero el contenido no es JSON válido → ValueError 'not valid JSON'."""
    mock_client = _mock_claude_raw("{ score: not-json, }")
    with patch("src.sentiment.analyzer.anthropic.AsyncAnthropic", return_value=mock_client):
        with pytest.raises(ValueError, match="not valid JSON"):
            await analyze(make_news(), make_config(), make_secrets())


async def test_analyze_strips_markdown_fence():
    """REGRESIÓN (cazado por el demo del Fast Path en vivo): claude-haiku-4-5
    envuelve el JSON en ```json … ``` pese a pedirle 'no markdown'. Debe parsearse
    igual extrayendo el objeto entre la primera { y la última }."""
    payload = {
        "score": -0.85, "confidence": 0.92, "high_impact": True,
        "symbol_scope": ["*"], "rationale": "Major exchange hack.",
    }
    fenced = "```json\n" + json.dumps(payload) + "\n```"
    with patch("src.sentiment.analyzer.anthropic.AsyncAnthropic", return_value=_mock_claude_raw(fenced)):
        result = await analyze(make_news(), make_config(), make_secrets())
    assert result.score == pytest.approx(-0.85)
    assert result.confidence == pytest.approx(0.92)
    assert result.symbol_scope == ["*"]


async def test_analyze_handles_preamble_text():
    """Claude a veces añade un preámbulo antes del JSON ('Here is the analysis:')."""
    payload = {
        "score": 0.6, "confidence": 0.8, "high_impact": False,
        "symbol_scope": ["BTC"], "rationale": "Bullish.",
    }
    noisy = "Here is the analysis:\n" + json.dumps(payload) + "\nHope this helps!"
    with patch("src.sentiment.analyzer.anthropic.AsyncAnthropic", return_value=_mock_claude_raw(noisy)):
        result = await analyze(make_news(), make_config(), make_secrets())
    assert result.score == pytest.approx(0.6)
    assert result.symbol_scope == ["BTC"]


# ---------------------------------------------------------------------------
# analyzed_at timestamp
# ---------------------------------------------------------------------------


async def test_analyzed_at_is_utc_aware():
    payload = {
        "score": 0.5,
        "confidence": 0.8,
        "high_impact": False,
        "symbol_scope": ["BTC"],
        "rationale": "Test.",
    }
    with patch("src.sentiment.analyzer.anthropic.AsyncAnthropic", return_value=_mock_claude(payload)):
        result = await analyze(make_news(), make_config(), make_secrets())

    assert result.analyzed_at.tzinfo is not None
