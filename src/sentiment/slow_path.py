"""Productor real del Slow Path: RSS → scoring → store por símbolo (Plan V2 §1/§3).

Espejo de `events.py` (Fast Path), pero para el overlay de sentimiento del Slow
Path. Mientras `fetch_events` se queda SOLO con los shocks (originan trades) y
devuelve una `list` para la cola, aquí puntuamos TODO lo relevante (`score_item`)
y devolvemos un `dict[symbol → SentimentScore]` que el `_sentiment_loop` vuelca al
`sentiment_store`. Ese store lo lee `decide()` por la confluencia: el sentimiento
NO origina en el Slow Path, confirma/modula la señal quant (y `scheduled` bloquea).

Tres guardias, en el mismo orden de coste creciente que el Fast Path (lo barato
primero, Claude al final):
    1. `seen`  — dedup por news_id: los RSS repiten un titular durante horas; sin
       esto se llamaría a `score_item`→Claude cada poll sobre lo mismo (inaceptable
       a $0/mes). Se purga por la ventana de frescura (un id más viejo que
       `max_news_age_hours` ya no pasaría frescura → olvidarlo es seguro y acota la
       memoria del set). Se marca visto AUNQUE el score sea None (irrelevante): no
       merece re-evaluarse. Si `score_item` LANZA (fallo de Claude), NO se marca
       → se reintenta en el siguiente poll.
    2. frescura — descarta titulares con `published_at` más viejo que
       `max_news_age_hours`: el mismo horizonte con el que el backtest caduca las
       noticias. Defiende el presupuesto (no escala titulares rancios a Claude).
    3. `score_item` (filtro local VADER → Claude solo si escala). Reutiliza la
       pieza del Sprint 4: irrelevante→None; escalado→Claude; si no→score local.

Resolución de scope (`resolve_scope`): misma semántica que `_resolve_scope` del
engine — "*" (mercado) → todos los símbolos que operamos; en otro caso, la
intersección exacta con `market.symbols`. ⚠️ Limitación compartida con el Fast
Path: Claude devuelve tickers como ["BTC"]; nuestros símbolos son ["BTCUSDT"], así
que un scope NO-wildcard rara vez machea — en la práctica el overlay entra por "*"
(market-wide, como el viejo Fear&Greed). Arreglarlo (normalizar tickers) tocaría
AMBOS paths + el prompt de Claude → módulo aparte.

Cuando varias noticias mapean al MISMO símbolo en un poll, gana la de `published_at`
más reciente (last-write-wins): procesamos en orden ascendente y la última escritura
sobre la clave es la más nueva. Determinista, sin umbral.

Todo inyectable (`analyze_fn`, `fetch_feeds_fn`, `now`) → unit-testeable sin red.
El builder `build_sentiment_fetch` arma la versión de producción (Claude real +
`seen` persistente entre polls); es capa operativa y se valida en testnet.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Awaitable, Callable

from src.core.config import Secrets, SentimentConfig, Settings
from src.core.models import NewsItem, SentimentScore
from src.sentiment.analyzer import analyze
from src.sentiment.events import FetchFeedsFn
from src.sentiment.feeds import fetch_feeds
from src.sentiment.scoring import AnalyzeFn, score_item

logger = logging.getLogger(__name__)


def resolve_scope(scope: list[str], symbols: list[str]) -> list[str]:
    """Resuelve el symbol_scope de una noticia a los símbolos que operamos.

    "*" (todo el mercado) → todos los configurados; en otro caso, la intersección
    EXACTA con `symbols` (ignoramos tickers que no seguimos). Misma semántica que
    `Orchestrator._resolve_scope` para que Fast y Slow Path traten el scope igual.

    TODO: DEUDA_TICKER — Claude devuelve tickers como ["BTC"], pero `symbols` son
    ["BTCUSDT"], así que un scope NO-wildcard rara vez machea y casi todo el overlay
    entra por "*". Normalizar (BTC→BTCUSDT) debe hacerse de forma consistente en
    AMBOS paths (este `resolve_scope` y `Orchestrator._resolve_scope`) y alineado con
    el prompt de `analyze` → es un módulo aparte, fuera del alcance de esta sesión.
    """
    if "*" in scope:
        return list(symbols)
    return [s for s in scope if s in symbols]


async def fetch_sentiment(
    config: SentimentConfig,
    symbols: list[str],
    *,
    analyze_fn: AnalyzeFn,
    seen: dict[str, datetime],
    fetch_feeds_fn: FetchFeedsFn = fetch_feeds,
    now: datetime | None = None,
) -> dict[str, SentimentScore]:
    """Sondea los feeds y devuelve el sentimiento FRESCO y NO VISTO por símbolo.

    Args:
        config:     SentimentConfig (feeds RSS + heuristic_weight + max_news_age_hours).
        symbols:    los símbolos que operamos (`settings.market.symbols`), para
                    resolver el scope de cada noticia.
        analyze_fn: escala un NewsItem a SentimentScore (Claude en prod).
        seen:       estado del caller: news_id → instante en que se emitió. Se muta
                    in-place (dedup persistente entre polls).
        fetch_feeds_fn: fuente de NewsItems (inyectable para tests).
        now:        reloj (inyectable); por defecto utcnow.

    Returns:
        dict[symbol → SentimentScore] con los símbolos para los que hubo noticia
        nueva y fresca. Los símbolos ausentes conservan su score previo en el store
        (que el engine caduca por TTL); por eso devolvemos solo lo NUEVO, no None.
    """
    now = now or datetime.now(timezone.utc)
    max_age_seconds = config.max_news_age_hours * 3600

    # Purga de `seen`: ids más viejos que la ventana ya no pasarían frescura, así
    # que olvidarlos es seguro y evita que el set crezca sin límite.
    stale_cutoff = now - timedelta(seconds=max_age_seconds)
    for nid in [nid for nid, ts in seen.items() if ts < stale_cutoff]:
        del seen[nid]

    items = await fetch_feeds_fn(config)
    out: dict[str, SentimentScore] = {}
    # Orden ascendente por published_at: la última escritura sobre cada símbolo es
    # la noticia más reciente (last-write-wins determinista).
    for item in sorted(items, key=lambda it: it.published_at):
        if item.id in seen:
            continue  # ya puntuado: no re-llamar a Claude ni re-emitir
        if (now - item.published_at).total_seconds() > max_age_seconds:
            continue  # titular rancio: fuera del horizonte del overlay
        try:
            score = await score_item(item, config, analyze_fn=analyze_fn)
        except Exception:
            # Resiliencia operativa: un fallo puntual de Claude no tumba el batch ni
            # marca el id como visto → se reintenta en el siguiente poll.
            logger.warning("score_item falló para news_id=%s; se reintentará", item.id,
                           exc_info=True)
            continue
        seen[item.id] = now  # visto aunque sea None (irrelevante no merece re-evaluación)
        if score is None:
            continue
        for sym in resolve_scope(score.symbol_scope, symbols):
            out[sym] = score
    return out


def build_sentiment_fetch(
    settings: Settings, secrets: Secrets
) -> Callable[[], Awaitable[dict[str, SentimentScore]]]:
    """Arma el `sentiment_fetch` de PRODUCCIÓN (Claude real + `seen` persistente).

    Capa operativa (red + Claude): se valida en testnet, no en unit-tests. El `seen`
    se crea una vez y se cierra sobre el closure, así persiste entre polls (y entre
    reinicios de `_sentiment_loop` bajo `_supervise`, que reusa este mismo callable).

    ⚠️ A diferencia de `build_event_fetch` (inerte tras el gate `event.enabled`),
    ESTE callable se invoca en cuanto `run()` arranca el `_sentiment_loop` —sin gate—
    así que ACTIVAR el overlay en `main.py` empieza a gastar tokens (Claude Haiku,
    solo en los titulares que escalen). Requiere `ANTHROPIC_API_KEY`.
    """
    seen: dict[str, datetime] = {}

    async def _analyze(item: NewsItem) -> SentimentScore:
        return await analyze(item, settings.sentiment, secrets)

    async def _sentiment_fetch() -> dict[str, SentimentScore]:
        return await fetch_sentiment(
            settings.sentiment,
            settings.market.symbols,
            analyze_fn=_analyze,
            seen=seen,
        )

    return _sentiment_fetch
