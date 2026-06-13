"""Claude Haiku sentiment analyzer.

Called only when the heuristic filter escalates an item (is_high_impact or
|local_score| >= escalate_score_threshold). Relevance was already confirmed
by the filter — we skip that here.

Why a strict JSON schema?
  Free-form responses from Claude require complex parsing and fail silently
  when the model hedges ("it could be bullish OR bearish..."). A fixed schema
  forces an explicit decision and lets Pydantic validate the bounds at the
  model boundary, not deep in the pipeline.
"""

import json
import logging
from datetime import datetime, timezone

import anthropic

from src.core.config import SentimentConfig, Secrets
from src.core.models import NewsItem, SentimentScore

logger = logging.getLogger(__name__)

_SYSTEM = """You are a crypto market sentiment analyst. Given a news headline and summary, assess the likely impact on cryptocurrency prices.

Return ONLY valid JSON with these exact fields (no markdown, no extra text):
{
  "score": <float in [-1.0, 1.0], where -1=very bearish, 0=neutral, 1=very bullish>,
  "confidence": <float in [0.0, 1.0], your certainty given the available context>,
  "high_impact": <boolean, true if this is a major market-moving event>,
  "symbol_scope": <list of affected tickers, e.g. ["BTC", "ETH"], or ["*"] for whole market>,
  "rationale": <one concise sentence explaining your score>
}"""

_USER_TEMPLATE = "Headline: {title}\nSummary: {summary}"


async def analyze(
    item: NewsItem,
    config: SentimentConfig,
    secrets: Secrets,
) -> SentimentScore:
    """Call Claude Haiku to score a news item's market sentiment.

    Returns a SentimentScore validated by Pydantic. Raises ValueError if the
    model response is not parseable JSON, or ValidationError if values are
    out of bounds.
    """
    client = anthropic.AsyncAnthropic(api_key=secrets.anthropic_api_key)

    response = await client.messages.create(
        model=config.claude_model,
        max_tokens=256,
        system=_SYSTEM,
        messages=[
            {
                "role": "user",
                "content": _USER_TEMPLATE.format(
                    title=item.title,
                    summary=item.summary or "(no summary)",
                ),
            }
        ],
    )

    raw = response.content[0].text
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.error("Claude returned non-JSON for news_id=%s: %.200s", item.id, raw)
        raise ValueError(f"Claude response is not valid JSON: {exc}") from exc

    # Clamp before Pydantic so out-of-range values don't surface as cryptic errors.
    score = max(-1.0, min(1.0, float(data["score"])))
    confidence = max(0.0, min(1.0, float(data["confidence"])))

    return SentimentScore(
        news_id=item.id,
        symbol_scope=data["symbol_scope"],
        score=score,
        confidence=confidence,
        high_impact=bool(data["high_impact"]),
        rationale=data.get("rationale", ""),
        analyzed_at=datetime.now(timezone.utc),
    )
