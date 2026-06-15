"""Índice Fear & Greed de alternative.me como sentimiento de mercado histórico.

Tras la investigación de C, resultó que las noticias cripto históricas gratis ya
no existen (CryptoPanic, NewsData, CoinGecko, LunarCrush movieron el archivo a
pago en 2025-26). El índice Fear & Greed es la ÚNICA fuente $0 con histórico
completo (sin key, desde 2018) — y agrega un componente de sentimiento social
(Twitter/X ~15%) junto a volatilidad, momentum y dominancia.

Endpoint: GET https://api.alternative.me/fng/?limit=0&format=json
limit=0 → todos los valores diarios históricos.

Mapeo a nuestro score [-1,+1] (interpretación MOMENTUM): greed (>50) confirma
tendencia alcista, fear (<50) bajista. La hipótesis CONTRARIA (greed extremo =
techo) es un experimento futuro: basta negar el score (lo dejamos como pregunta
abierta, no como número mágico, hasta saber si la señal aporta algo).
"""

from __future__ import annotations

from datetime import datetime, timezone

import httpx

FNG_URL = "https://api.alternative.me/fng/"

# Una lectura diaria del índice: (instante UTC, valor 0-100, clasificación).
FearGreedReading = tuple[datetime, int, str]


def fear_greed_to_score(value: int) -> float:
    """Mapea el índice [0,100] a nuestro score [-1,+1]. 0→-1, 50→0, 100→+1."""
    return max(-1.0, min(1.0, (value - 50) / 50.0))


async def fetch_fear_greed(
    *, limit: int = 0, client: httpx.AsyncClient | None = None
) -> list[FearGreedReading]:
    """Descarga el histórico del índice (limit=0 = todo). Sin API key.

    Args:
        limit:  nº de lecturas más recientes; 0 = histórico completo.
        client: inyectable en tests; por defecto crea un httpx.AsyncClient.
    """
    own_client = client is None
    if own_client:
        client = httpx.AsyncClient(timeout=30.0)
    try:
        resp = await client.get(FNG_URL, params={"limit": str(limit), "format": "json"})
        data = resp.json()
    finally:
        if own_client:
            await client.aclose()

    readings: list[FearGreedReading] = []
    for row in data.get("data", []):
        try:
            ts = datetime.fromtimestamp(int(row["timestamp"]), tz=timezone.utc)
            value = int(row["value"])
        except (KeyError, ValueError, TypeError):
            continue  # fila malformada: se ignora, no rompe la ingesta
        readings.append((ts, value, row.get("value_classification", "")))
    return readings
