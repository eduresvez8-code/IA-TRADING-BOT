"""Heuristic crypto news filter.

Two-stage pipeline before spending API tokens on Claude:

  1. Relevance guard: does the item mention crypto at all? Discard if not.
  2. Scoring: weighted average of (a) crypto dict matches + (b) VADER.
  3. High-impact flag: certain events always escalate regardless of score.

Parameters (thresholds) live in config/settings.yaml → SentimentConfig.
The term dictionaries below are linguistic annotations — each entry is a
(pattern → sentiment) fact, not a tunable number.

Why VADER as complement?
  VADER was trained on social media sentences and catches general sentiment
  ("regulators approve", "firm collapses") for headlines that don't contain
  explicit crypto jargon. It scores poorly on terms like "halving" or "depeg"
  that it has never seen — that's what the crypto dict fixes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from src.core.models import NewsItem

# Lazy singleton: VADER reads ~2 MB of lexicon once at first call.
_vader: SentimentIntensityAnalyzer | None = None


def _get_vader() -> SentimentIntensityAnalyzer:
    global _vader
    if _vader is None:
        _vader = SentimentIntensityAnalyzer()
    return _vader


# ---------------------------------------------------------------------------
# Term dictionaries — linguistic domain knowledge, not magic numbers
# ---------------------------------------------------------------------------

# Sentiment-bearing crypto terms with weights in [-1, +1].
# Scores calibrated by domain knowledge: "rugpull" is -1 (unambiguously fatal),
# "whale" is 0 (context-dependent), "halving" is +0.7 (historically bullish).
CRYPTO_TERMS: dict[str, float] = {
    # Strongly bullish
    "etf approval": 0.9,
    "etf approved": 0.9,
    "etf": 0.5,
    "halving": 0.7,
    "all-time high": 0.8,
    "ath": 0.8,
    "record high": 0.7,
    "bullish": 0.7,
    "breakout": 0.6,
    "rally": 0.6,
    "adoption": 0.6,
    "institutional": 0.5,
    "partnership": 0.5,
    "upgrade": 0.4,
    "mainnet": 0.4,
    "accumulation": 0.5,
    "rate cut": 0.4,
    "regulated": 0.2,
    # Strongly bearish
    "rugpull": -1.0,
    "rug pull": -1.0,
    "hack": -0.9,
    "hacked": -0.9,
    "exploit": -0.9,
    "exploited": -0.9,
    "depeg": -0.9,
    "depegged": -0.9,
    "crash": -0.8,
    "bankruptcy": -0.8,
    "bankrupt": -0.8,
    "bearish": -0.7,
    "dump": -0.7,
    "delisting": -0.7,
    "sec lawsuit": -0.8,
    "sec charges": -0.8,
    "scam": -0.8,
    "fraud": -0.8,
    "insolvent": -0.7,
    "liquidation": -0.6,
    "liquidated": -0.6,
    "collapse": -0.8,
    "ban": -0.6,
    "banned": -0.6,
    "dumping": -0.6,
    "fud": -0.5,
    "shutdown": -0.5,
    "bear": -0.4,
    "rate hike": -0.4,
    "lawsuit": -0.5,
    "inflation": -0.2,
    "recession": -0.3,
    # Context-dependent (weight near 0, but trigger escalation)
    "fomc": 0.0,
    "cpi": 0.0,
    "fed": 0.0,
    "sec": -0.2,
    "regulation": -0.1,
    "whale": 0.0,
}

# Any match here → is_high_impact = True (always call Claude).
# These are events where local scoring is unreliable: "crash" could be a
# correction, a flash-crash, or a total collapse — only Claude can judge.
HIGH_IMPACT_TERMS: frozenset[str] = frozenset({
    "hack", "hacked", "exploit", "exploited",
    "rugpull", "rug pull",
    "depeg", "depegged",
    "etf approval", "etf approved",
    "fomc", "cpi",
    "halving",
    "sec lawsuit", "sec charges",
    "crash",
    "bankruptcy", "bankrupt",
})

# Minimum signal that the item is crypto-related at all.
RELEVANCE_TERMS: frozenset[str] = frozenset({
    "bitcoin", "btc", "ethereum", "eth", "crypto", "cryptocurrency",
    "blockchain", "defi", "nft", "altcoin", "stablecoin", "usdt", "usdc",
    "binance", "coinbase", "exchange", "wallet", "token", "coin",
    "web3", "layer 2", "layer 1", "l2", "l1", "dao", "protocol",
    "staking", "yield", "apy", "apr", "liquidity",
    "solana", "sol", "cardano", "ada", "polkadot", "dot", "ripple", "xrp",
    "trading", "market cap",
    # Macro events that always affect crypto
    "halving", "etf", "fomc", "cpi",
})


@dataclass
class FilterResult:
    is_relevant: bool
    is_high_impact: bool
    local_score: float          # combined score in [-1, 1]
    matched_terms: list[str] = field(default_factory=list)


def filter_news(item: NewsItem, *, heuristic_weight: float) -> FilterResult:
    """Apply heuristic + VADER filter to a NewsItem.

    Args:
        item: The news item to evaluate.
        heuristic_weight: Weight for the crypto dict score.
            (1 - heuristic_weight) is given to VADER.
            Value comes from SentimentConfig.heuristic_weight.
    """
    text = f"{item.title} {item.summary}".lower()

    # 1. Relevance guard — ignore anything unrelated to crypto
    if not any(term in text for term in RELEVANCE_TERMS):
        return FilterResult(is_relevant=False, is_high_impact=False, local_score=0.0)

    # 2. Heuristic score from the crypto dictionary
    matched: list[tuple[str, float]] = [
        (term, weight)
        for term, weight in CRYPTO_TERMS.items()
        if term in text
    ]
    if matched:
        # Average matched weights: multiple signals moderate each other.
        # "bullish rally" → (+0.7 + 0.6) / 2 = +0.65 (not double-counting).
        heuristic_score = sum(w for _, w in matched) / len(matched)
        heuristic_score = max(-1.0, min(1.0, heuristic_score))
    else:
        heuristic_score = 0.0

    # 3. VADER on the headline (shorter text → less noise)
    vader_compound = _get_vader().polarity_scores(item.title)["compound"]

    # 4. Weighted combination — heuristic dominates, VADER fills gaps
    vader_weight = 1.0 - heuristic_weight
    local_score = heuristic_weight * heuristic_score + vader_weight * vader_compound
    local_score = max(-1.0, min(1.0, local_score))

    # 5. High-impact detection
    is_high_impact = any(term in text for term in HIGH_IMPACT_TERMS)

    return FilterResult(
        is_relevant=True,
        is_high_impact=is_high_impact,
        local_score=local_score,
        matched_terms=[t for t, _ in matched],
    )
