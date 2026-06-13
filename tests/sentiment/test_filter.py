"""Tests for the heuristic crypto filter.

Pure unit tests — no network, no API calls. The filter is a deterministic
function, so we can check exact sign / impact / bounds.
"""

from datetime import datetime, timezone

import pytest

from src.core.models import NewsItem
from src.sentiment.filter import FilterResult, filter_news

HW = 0.7  # heuristic_weight used in all tests


def make_news(title: str, summary: str = "") -> NewsItem:
    return NewsItem(
        id="test",
        title=title,
        source="https://example.com",
        url="https://example.com/test",
        published_at=datetime.now(timezone.utc),
        summary=summary,
    )


# ---------------------------------------------------------------------------
# Relevance
# ---------------------------------------------------------------------------


def test_non_crypto_news_is_not_relevant():
    item = make_news("Stock market sees gains in the tech sector today")
    result = filter_news(item, heuristic_weight=HW)
    assert not result.is_relevant
    assert result.local_score == 0.0
    assert result.matched_terms == []


def test_crypto_mention_makes_relevant():
    item = make_news("Bitcoin breaks resistance level")
    result = filter_news(item, heuristic_weight=HW)
    assert result.is_relevant


def test_eth_mention_makes_relevant():
    item = make_news("ETH staking rewards reach new high")
    result = filter_news(item, heuristic_weight=HW)
    assert result.is_relevant


# ---------------------------------------------------------------------------
# Sentiment direction
# ---------------------------------------------------------------------------


def test_hack_produces_negative_score():
    item = make_news("Major crypto exchange gets hacked")
    result = filter_news(item, heuristic_weight=HW)
    assert result.is_relevant
    assert result.local_score < 0


def test_etf_approval_produces_positive_score():
    item = make_news("Bitcoin ETF approval granted by regulators")
    result = filter_news(item, heuristic_weight=HW)
    assert result.is_relevant
    assert result.local_score > 0


def test_bullish_terms_produce_positive_score():
    item = make_news("Ethereum bullish rally and institutional adoption")
    result = filter_news(item, heuristic_weight=HW)
    assert result.local_score > 0


def test_bearish_terms_produce_negative_score():
    item = make_news("Bitcoin crash and mass liquidation event")
    result = filter_news(item, heuristic_weight=HW)
    assert result.local_score < 0


def test_rug_pull_is_maximally_negative():
    item = make_news("DeFi protocol suffers rugpull, funds drained")
    result = filter_news(item, heuristic_weight=HW)
    assert result.local_score < -0.5


# ---------------------------------------------------------------------------
# High-impact detection
# ---------------------------------------------------------------------------


def test_hack_is_high_impact():
    item = make_news("Crypto exchange hacked, $100M stolen")
    result = filter_news(item, heuristic_weight=HW)
    assert result.is_high_impact


def test_halving_is_high_impact():
    item = make_news("Bitcoin halving event scheduled next month")
    result = filter_news(item, heuristic_weight=HW)
    assert result.is_high_impact


def test_etf_approval_is_high_impact():
    item = make_news("Bitcoin ETF approval from SEC expected this week")
    result = filter_news(item, heuristic_weight=HW)
    assert result.is_high_impact


def test_fomc_is_high_impact():
    item = make_news("FOMC meeting outcome impacts Bitcoin market")
    result = filter_news(item, heuristic_weight=HW)
    assert result.is_high_impact


def test_crash_is_high_impact():
    item = make_news("Bitcoin crash wipes out $500B in market cap")
    result = filter_news(item, heuristic_weight=HW)
    assert result.is_high_impact


def test_neutral_mining_news_is_not_high_impact():
    item = make_news("Bitcoin mining difficulty hits new record high")
    result = filter_news(item, heuristic_weight=HW)
    assert not result.is_high_impact


# ---------------------------------------------------------------------------
# Bounds and consistency
# ---------------------------------------------------------------------------


def test_score_always_in_unit_range():
    # Pile on extreme terms — the result must still be clamped.
    item = make_news(
        "Bitcoin hack exploit rugpull crash bankruptcy fraud scam depegged",
        summary="BTC ETH crypto",
    )
    result = filter_news(item, heuristic_weight=HW)
    assert -1.0 <= result.local_score <= 1.0


def test_matched_terms_non_empty_when_relevant():
    item = make_news("Bitcoin ETF approved by the SEC")
    result = filter_news(item, heuristic_weight=HW)
    assert len(result.matched_terms) > 0


def test_non_relevant_has_empty_matched_terms():
    item = make_news("S&P 500 closes green amid positive earnings season")
    result = filter_news(item, heuristic_weight=HW)
    assert result.matched_terms == []


def test_summary_text_contributes_to_score():
    """A title without crypto terms but a summary with them → still relevant."""
    item = make_news("Markets update", summary="Bitcoin and Ethereum rally today.")
    result = filter_news(item, heuristic_weight=HW)
    assert result.is_relevant


def test_heuristic_weight_zero_uses_only_vader():
    """heuristic_weight=0 disables the crypto dict; score comes from VADER only."""
    item = make_news("Bitcoin hits a great new high — wonderful news")
    # VADER should detect positive sentiment
    result = filter_news(item, heuristic_weight=0.0)
    assert result.is_relevant  # 'bitcoin' is in RELEVANCE_TERMS
    # Score should be driven by VADER's positive reading
    assert result.local_score >= 0


def test_heuristic_weight_one_ignores_vader():
    """heuristic_weight=1 → only the crypto dict score."""
    # "bitcoin" is in RELEVANCE_TERMS but not in CRYPTO_TERMS with a sentiment
    # Neutral title with no explicit sentiment terms → heuristic_score = 0
    item = make_news("Bitcoin trading volume stable this week")
    result = filter_news(item, heuristic_weight=1.0)
    assert result.is_relevant
    # No crypto sentiment term matched → heuristic_score = 0 → local_score = 0
    # (VADER is disabled; matched_terms won't include 'bitcoin' since it's only in RELEVANCE_TERMS)
    assert result.local_score == 0.0
