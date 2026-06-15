"""Runner operativo: construye el corpus histórico de sentimiento.

    uv run python -m src.sentiment.build_history

Flujo: CryptoPanic → guardar noticias → filtro+Claude → guardar scores. El free
tier limita la profundidad, así que ejecútalo periódicamente para ACUMULAR.

⚠️ Capa operativa (red + Claude): requiere CRYPTOPANIC_TOKEN y ANTHROPIC_API_KEY
en .env. Solo los ítems ESCALADOS llaman a Claude (céntimos), el resto se puntúa
localmente gratis. La lógica está cubierta por tests con dobles.
"""

import asyncio
import logging

from src.core.config import load_secrets, load_settings
from src.data.storage import Storage
from src.sentiment.analyzer import analyze
from src.sentiment.cryptopanic import fetch_cryptopanic
from src.sentiment.scoring import score_item

logger = logging.getLogger(__name__)


async def build_history(*, max_pages: int = 20) -> int:
    cfg, secrets = load_settings(), load_secrets()
    if not secrets.cryptopanic_token:
        print("⚠ falta CRYPTOPANIC_TOKEN en .env — no se puede ingerir histórico.")
        return 1

    # Tickers de los pares de trading: BTCUSDT → BTC, ETHUSDT → ETH.
    currencies = ",".join(s.removesuffix("USDT") for s in cfg.market.symbols)
    storage = await Storage(cfg.storage.db_path, cfg.storage.candles_dir).init()

    async def analyze_fn(item):
        return await analyze(item, cfg.sentiment, secrets)

    try:
        news = await fetch_cryptopanic(secrets.cryptopanic_token,
                                       currencies=currencies, max_pages=max_pages)
        for item in news:
            await storage.save_news(item)

        scored = 0
        for item in news:
            score = await score_item(item, cfg.sentiment, analyze_fn=analyze_fn)
            if score is not None:
                await storage.save_sentiment_score(
                    score, ts_ms=int(item.published_at.timestamp() * 1000))
                scored += 1
        print(f"✓ {len(news)} noticias guardadas · {scored} con score "
              f"({len(news) - scored} no relevantes)")
    finally:
        await storage.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(build_history()))
