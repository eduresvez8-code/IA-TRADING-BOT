"""Runner GRATIS: puebla el histórico de sentimiento con el índice Fear & Greed.

    uv run python -m src.sentiment.build_fear_greed

No requiere ninguna clave. Descarga el histórico diario completo y lo guarda en
la tabla `sentiment_scores` (la misma que consume el backtest de confluencia),
como sentimiento de mercado con `symbol_scope=["*"]`. Idempotente: re-ejecutarlo
actualiza, no duplica (PK = news_id por día).
"""

import asyncio

from src.core.config import load_settings
from src.core.models import SentimentScore
from src.data.storage import Storage
from src.sentiment.fear_greed import fear_greed_to_score, fetch_fear_greed


async def build_fear_greed() -> int:
    cfg = load_settings()
    storage = await Storage(cfg.storage.db_path, cfg.storage.candles_dir).init()
    try:
        readings = await fetch_fear_greed(limit=0)
        for ts, value, classification in readings:
            score = SentimentScore(
                news_id=f"fng-{ts.date().isoformat()}",
                symbol_scope=["*"],
                score=fear_greed_to_score(value),
                confidence=1.0,  # es una lectura medida, no una estimación incierta
                high_impact=False,
                rationale=f"Fear & Greed {value} ({classification})",
                analyzed_at=ts,
            )
            await storage.save_sentiment_score(score, ts_ms=int(ts.timestamp() * 1000))
        print(f"✓ {len(readings)} lecturas diarias de Fear & Greed guardadas "
              f"como sentimiento de mercado en sentiment_scores.")
        if readings:
            print(f"  rango: {readings[-1][0].date()} → {readings[0][0].date()}")
    finally:
        await storage.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(build_fear_greed()))
