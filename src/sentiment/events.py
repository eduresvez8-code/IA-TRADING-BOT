"""Productor real del Fast Path: RSS → detección de shock → Claude (Plan V2 §2.5(ii)).

`fetch_events` es la fuente que el orquestador inyecta en `run(event_fetch=...)`:
sondea los feeds, se queda SOLO con los `event_kind=="shock"` (los únicos que
originan en el Fast Path; `scheduled` bloquea, `none` es ruido), los analiza con
Claude y devuelve `list[SentimentScore]` lista para encolar.

Tres guardias, en orden de coste creciente (lo barato primero, Claude al final):
    1. `seen`  — dedup por news_id: los RSS repiten un titular durante horas; sin
       esto se llamaría a Claude cada poll sobre lo mismo (inaceptable a $0/mes) y
       se inundaría la cola. Se purga por la ventana de frescura (un id más viejo
       que `max_age_seconds` ya no pasaría frescura → olvidarlo es seguro y acota
       la memoria del set).
    2. frescura — descarta titulares con `published_at` más viejo que
       `max_age_seconds`: §0(A) del plan, el edge es el drift POST-evento, no
       perseguir noticias rancias. Defensa válida aun con el gate de impulso ablado.
    3. `filter_news` (VADER) → `analyze_fn` (Claude) — solo para shocks frescos no
       vistos. Los shocks son high-impact: SIEMPRE escalan (la rama local de
       `score_item` nunca aplicaría), así que llamamos a `analyze_fn` directo y
       fijamos `event_kind="shock"` (etiqueta determinista del filtro, no de Claude).

Todo inyectable (`analyze_fn`, `fetch_feeds_fn`, `now`) → unit-testeable sin red.
El builder `build_event_fetch` arma la versión de producción (Claude real + `seen`
persistente entre polls); es capa operativa y se valida en testnet.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Awaitable, Callable

from src.core.config import Secrets, SentimentConfig, Settings
from src.core.models import NewsItem, SentimentScore
from src.sentiment.analyzer import analyze
from src.sentiment.feeds import fetch_feeds
from src.sentiment.filter import filter_news
from src.sentiment.scoring import AnalyzeFn

logger = logging.getLogger(__name__)

FetchFeedsFn = Callable[[SentimentConfig], Awaitable[list[NewsItem]]]


async def fetch_events(
    config: SentimentConfig,
    *,
    analyze_fn: AnalyzeFn,
    seen: dict[str, datetime],
    max_age_seconds: int,
    fetch_feeds_fn: FetchFeedsFn = fetch_feeds,
    now: datetime | None = None,
) -> list[SentimentScore]:
    """Sondea los feeds y devuelve los shocks FRESCOS y NO VISTOS, ya analizados.

    Args:
        config:          SentimentConfig (feeds RSS + heuristic_weight).
        analyze_fn:      escala un NewsItem a SentimentScore (Claude en prod).
        seen:            estado del caller: news_id → instante en que se emitió.
                         Se muta in-place (dedup persistente entre polls).
        max_age_seconds: ventana de frescura por published_at (y de purga de `seen`).
        fetch_feeds_fn:  fuente de NewsItems (inyectable para tests).
        now:             reloj (inyectable); por defecto utcnow.

    Returns:
        Lista de SentimentScore con event_kind="shock", uno por titular nuevo.
    """
    now = now or datetime.now(timezone.utc)

    # Purga de `seen`: ids más viejos que la ventana ya no pasarían frescura, así
    # que olvidarlos es seguro y evita que el set crezca sin límite.
    stale_cutoff = now - timedelta(seconds=max_age_seconds)
    for nid in [nid for nid, ts in seen.items() if ts < stale_cutoff]:
        del seen[nid]

    items = await fetch_feeds_fn(config)
    out: list[SentimentScore] = []
    for item in items:
        if item.id in seen:
            continue  # ya emitido: no re-llamar a Claude ni re-encolar
        if (now - item.published_at).total_seconds() > max_age_seconds:
            continue  # titular rancio: no es un shock operable (no chasing)
        fr = filter_news(item, heuristic_weight=config.heuristic_weight)
        if fr.event_kind != "shock":
            continue  # solo los shocks originan en el Fast Path
        try:
            score = await analyze_fn(item)  # un shock SIEMPRE escala a Claude
        except Exception:
            # Resiliencia operativa: un fallo puntual de Claude no tumba el batch ni
            # marca el id como visto → se reintenta en el siguiente poll.
            logger.warning("analyze falló para news_id=%s; se reintentará", item.id,
                           exc_info=True)
            continue
        # event_kind es etiqueta determinista del filtro, no juicio de Claude.
        out.append(score.model_copy(update={"event_kind": "shock"}))
        seen[item.id] = now
    return out


def build_event_fetch(
    settings: Settings, secrets: Secrets
) -> Callable[[], Awaitable[list[SentimentScore]]]:
    """Arma el `event_fetch` de PRODUCCIÓN (Claude real + `seen` persistente).

    Capa operativa (red + Claude): se valida en testnet, no en unit-tests. El
    `seen` se crea una vez y se cierra sobre el closure, así persiste entre polls
    (y entre reinicios de `_event_loop` bajo `_supervise`, que reusa este mismo
    callable). El wiring en `main.py` se hará en el hardening operativo (igual que
    `sentiment_fetch`); aquí solo se construye la función.
    """
    seen: dict[str, datetime] = {}

    async def _analyze(item: NewsItem) -> SentimentScore:
        return await analyze(item, settings.sentiment, secrets)

    async def _event_fetch() -> list[SentimentScore]:
        return await fetch_events(
            settings.sentiment,
            analyze_fn=_analyze,
            seen=seen,
            max_age_seconds=settings.event.max_headline_age_seconds,
        )

    return _event_fetch
