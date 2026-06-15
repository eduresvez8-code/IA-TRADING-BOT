"""Puntuación de una noticia reutilizando el pipeline de dos etapas del Sprint 4.

Mismo principio que en vivo: el filtro local (gratis) decide la escalación; solo
lo escalado va a Claude. La diferencia es que aquí se usa offline sobre el corpus
histórico para construir la serie de SentimentScores que alimentará el backtest.

`analyze_fn` es inyectable: en producción envuelve `analyzer.analyze` (Claude);
en tests, un doble que evita la red.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Awaitable, Callable

from src.core.config import SentimentConfig
from src.core.models import NewsItem, SentimentScore
from src.sentiment.filter import filter_news

AnalyzeFn = Callable[[NewsItem], Awaitable[SentimentScore]]


async def score_item(
    item: NewsItem, config: SentimentConfig, *, analyze_fn: AnalyzeFn
) -> SentimentScore | None:
    """Filtra y puntúa una noticia.

    - No relevante para cripto → None (no entra en la serie).
    - Escalada (high_impact o |local| ≥ umbral) → score de Claude vía analyze_fn.
    - Si no → score local, con confianza = |local_score|. Por construcción la
      confianza local es baja (< umbral de escalación), así el Risk Manager le
      da menos peso al sizing — coherente con que no mereció el análisis caro.
    """
    fr = filter_news(item, heuristic_weight=config.heuristic_weight)
    if not fr.is_relevant:
        return None

    escalate = fr.is_high_impact or abs(fr.local_score) >= config.escalate_score_threshold
    if escalate:
        return await analyze_fn(item)

    return SentimentScore(
        news_id=item.id,
        symbol_scope=["*"],
        score=fr.local_score,
        confidence=abs(fr.local_score),
        high_impact=fr.is_high_impact,
        rationale="score local (sin escalar a Claude)",
        analyzed_at=datetime.now(timezone.utc),
    )
