"""Sentiment Engine demo — corpus de 20 titulares etiquetados.

Corre el filtro heurístico sobre todos y, si hay ANTHROPIC_API_KEY,
llama a Claude Haiku para los que el filtro escalaría.

Uso:
    uv run python -m src.sentiment.sentiment_demo
"""

import asyncio
import os
import sys
from datetime import datetime, timezone
from typing import NamedTuple

from src.core.config import load_settings, load_secrets
from src.core.models import NewsItem
from src.sentiment.analyzer import analyze
from src.sentiment.filter import filter_news


# ---------------------------------------------------------------------------
# Corpus: 20 titulares etiquetados a mano
# label: "bullish" | "bearish" | "neutral"
# high_impact: esperamos que el filtro lo marque como alto impacto?
# ---------------------------------------------------------------------------
class Headline(NamedTuple):
    title: str
    summary: str
    label: str      # ground truth
    high_impact: bool


CORPUS: list[Headline] = [
    Headline("Bitcoin ETF Approval Granted by SEC",
             "Spot Bitcoin ETF starts trading on NYSE with record volume.",
             "bullish", True),
    Headline("Crypto Exchange Hacked, $120M in Funds Stolen",
             "Attackers exploited a smart contract vulnerability to drain funds.",
             "bearish", True),
    Headline("Ethereum Completes Major Network Upgrade",
             "The upgrade reduces gas fees and boosts transaction throughput.",
             "bullish", False),
    Headline("Bitcoin Halving Event Scheduled in 30 Days",
             "Mining reward will drop from 6.25 to 3.125 BTC per block.",
             "bullish", True),
    Headline("Federal Reserve Raises Interest Rates by 50 Basis Points",
             "FOMC decision puts pressure on risk assets including crypto.",
             "bearish", True),
    Headline("USDC Stablecoin Temporarily Depegs from $1",
             "USDC briefly traded at $0.87 before recovering on liquidity support.",
             "bearish", True),
    Headline("Bitcoin Surpasses Previous All-Time High",
             "BTC reaches $112,000 after institutional buying accelerates.",
             "bullish", False),
    Headline("DeFi Protocol Suffers $50M Exploit",
             "Smart contract bug allowed attacker to drain liquidity pools.",
             "bearish", True),
    Headline("BlackRock and Fidelity Accumulate Ethereum",
             "Institutional investors add ETH to balance sheets amid market dip.",
             "bullish", False),
    Headline("Crypto Market Crashes 22% in 24 Hours",
             "Mass liquidations cascade as BTC drops below $60,000.",
             "bearish", True),
    Headline("Solana Mainnet Upgrade Boosts TPS to 100,000",
             "Network performance improvements attract new developers.",
             "bullish", False),
    Headline("SEC Files Charges Against Major Crypto Exchange for Securities Fraud",
             "Regulator alleges unregistered securities offering to retail investors.",
             "bearish", True),
    Headline("Bitcoin Mining Difficulty Reaches Record High",
             "Hashrate growth signals strong miner confidence in long-term outlook.",
             "bullish", False),
    Headline("Whale Transfers 15,000 BTC to Binance",
             "Large holder moves coins to exchange, raising sell pressure concerns.",
             "bearish", False),
    Headline("Crypto Startup Raises $300M in Series B Funding",
             "Venture capital firm leads round to expand blockchain infrastructure.",
             "bullish", False),
    Headline("CPI Data Shows Inflation Cooling to 2.1%",
             "Lower inflation supports risk appetite across equities and crypto.",
             "bullish", True),
    Headline("JPMorgan Announces Bitcoin Custody Service for Clients",
             "Major US bank to hold Bitcoin on behalf of institutional customers.",
             "bullish", False),
    Headline("Crypto Ponzi Scheme Collapses, $2B Lost",
             "Fraudulent project exits with all investor funds.",
             "bearish", True),
    Headline("Ethereum Gas Fees Drop to Record Low After L2 Migration",
             "Layer-2 adoption reduces mainnet congestion significantly.",
             "bullish", False),
    Headline("Bitcoin Dominance Falls Below 40% as Altcoins Rally",
             "BTC market share drops as traders rotate into altcoins.",
             "neutral", False),
]


def _make_news(h: Headline, idx: int) -> NewsItem:
    return NewsItem(
        id=f"demo_{idx:02d}",
        title=h.title,
        source="demo",
        url=f"https://demo.example.com/{idx}",
        published_at=datetime.now(timezone.utc),
        summary=h.summary,
    )


def _label_from_score(score: float) -> str:
    if score > 0.1:
        return "bullish"
    if score < -0.1:
        return "bearish"
    return "neutral"


async def run_demo() -> None:
    settings = load_settings()
    secrets = load_secrets()
    has_api_key = bool(secrets.anthropic_api_key)

    print("\n" + "=" * 90)
    print("SPRINT 4 — SENTIMENT ENGINE DEMO  (corpus de 20 titulares)")
    print("=" * 90)
    if not has_api_key:
        print("  ⚠  Sin ANTHROPIC_API_KEY — solo se muestra el filtro local (sin Claude).")
    print()

    filter_correct = 0
    claude_correct = 0
    escalated_count = 0
    claude_count = 0

    header = f"{'#':>2}  {'Titular':<45}  {'GT':>7}  {'Filt':>6}  {'HI':>3}"
    if has_api_key:
        header += f"  {'Claude':>6}  {'HI-C':>4}"
    print(header)
    print("-" * (len(header) + 2))

    for idx, h in enumerate(CORPUS):
        item = _make_news(h, idx)
        result = filter_news(item, heuristic_weight=settings.sentiment.heuristic_weight)

        filter_label = _label_from_score(result.local_score) if result.is_relevant else "skip"
        should_escalate = result.is_high_impact or (
            result.is_relevant
            and abs(result.local_score) >= settings.sentiment.escalate_score_threshold
        )
        if should_escalate:
            escalated_count += 1

        filter_match = filter_label == h.label
        if filter_match:
            filter_correct += 1

        row = (
            f"{idx:>2}  {h.title[:45]:<45}  {h.label:>7}  "
            f"{filter_label:>6}  {'Y' if result.is_high_impact else 'N':>3}"
        )

        if has_api_key and should_escalate:
            try:
                sentiment = await analyze(item, settings.sentiment, secrets)
                claude_label = _label_from_score(sentiment.score)
                claude_hi = "Y" if sentiment.high_impact else "N"
                claude_match = claude_label == h.label
                if claude_match:
                    claude_correct += 1
                claude_count += 1
                row += f"  {claude_label:>6}  {claude_hi:>4}"
            except Exception as exc:
                row += f"  {'ERR':>6}  {str(exc)[:10]}"
        elif has_api_key:
            row += f"  {'(skip)':>6}  {'':>4}"

        print(row)

    print()
    n = len(CORPUS)
    print(f"Filtro local:  {filter_correct}/{n} correctos ({100*filter_correct/n:.0f}%)")
    print(f"Escalados a Claude: {escalated_count}/{n}")
    if has_api_key and claude_count:
        print(f"Claude Haiku:  {claude_correct}/{claude_count} correctos ({100*claude_correct/claude_count:.0f}%)")
    print()


if __name__ == "__main__":
    asyncio.run(run_demo())
