"""Ingesta de noticias de CryptoPanic (para construir el corpus HISTÓRICO).

A diferencia de `feeds.py` (RSS, solo los últimos titulares), CryptoPanic expone
una API paginada de la que se puede recuperar histórico para backtestear la señal
de sentimiento. Mapea cada post al mismo `NewsItem` y usa el MISMO hash de URL
como id, así una noticia que llega por RSS y por CryptoPanic se deduplica igual.

El free tier limita peticiones (HTTP 429): se respeta con backoff exponencial.
La profundidad de histórico del free tier es limitada — la estrategia es
ACUMULAR el corpus en SQLite ejecutando esto periódicamente.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from datetime import datetime, timezone

import httpx

from src.core.models import NewsItem

logger = logging.getLogger(__name__)

CRYPTOPANIC_URL = "https://cryptopanic.com/api/v1/posts/"


async def fetch_cryptopanic(
    token: str,
    *,
    currencies: str = "BTC,ETH",
    max_pages: int = 10,
    client: httpx.AsyncClient | None = None,
    sleep=asyncio.sleep,
) -> list[NewsItem]:
    """Descarga noticias de CryptoPanic siguiendo la paginación por cursor `next`.

    Args:
        token:      auth_token del free tier (secreto, de .env).
        currencies: tickers separados por coma, ej. "BTC,ETH".
        max_pages:  tope de páginas a seguir (cada una ~20 ítems).
        client:     inyectable en tests; por defecto crea un httpx.AsyncClient.
        sleep:      inyectable en tests (para no esperar de verdad en el backoff).

    Returns:
        NewsItems deduplicados por hash de URL, en el orden recibido.
    """
    own_client = client is None
    if own_client:
        client = httpx.AsyncClient(timeout=30.0, follow_redirects=True)

    items: list[NewsItem] = []
    seen: set[str] = set()
    url: str | None = CRYPTOPANIC_URL
    params: dict | None = {
        "auth_token": token, "currencies": currencies,
        "kind": "news", "public": "true",
    }
    try:
        for _ in range(max_pages):
            if url is None:
                break
            resp = await _get_with_backoff(client, url, params, sleep)
            data = resp.json()
            for post in data.get("results", []):
                item = _parse_post(post)
                if item and item.id not in seen:
                    items.append(item)
                    seen.add(item.id)
            url = data.get("next")     # el cursor `next` es una URL completa…
            params = None              # …que ya lleva su propia query
    finally:
        if own_client:
            await client.aclose()
    return items


async def _get_with_backoff(client, url, params, sleep, *, max_retries: int = 5):
    """GET reintentando solo ante 429 (rate limit del free tier), backoff exponencial."""
    delay = 1.0
    resp = None
    for attempt in range(max_retries + 1):
        resp = await client.get(url, params=params)
        if getattr(resp, "status_code", 200) != 429 or attempt >= max_retries:
            return resp
        logger.warning("CryptoPanic 429; reintento %d en %.1fs", attempt + 1, delay)
        await sleep(delay)
        delay *= 2
    return resp


def _parse_post(post: dict) -> NewsItem | None:
    """Convierte un post de CryptoPanic en NewsItem; None si está malformado."""
    url = (post.get("url") or "").strip()
    title = (post.get("title") or "").strip()
    if not url or not title:
        return None
    news_id = hashlib.sha256(url.encode()).hexdigest()[:16]
    src = (post.get("source") or {}).get("domain") or post.get("domain") or "cryptopanic"
    return NewsItem(
        id=news_id, title=title, source=src, url=url,
        published_at=_parse_time(post), summary="",
    )


def _parse_time(post: dict) -> datetime:
    """published_at en ISO8601 (…Z) → datetime UTC consciente."""
    raw = post.get("published_at") or post.get("created_at") or ""
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except (ValueError, AttributeError):
        return datetime.now(timezone.utc)
