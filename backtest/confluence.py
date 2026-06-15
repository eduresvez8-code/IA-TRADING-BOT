"""Backtest de la matriz de confluencia (quant × sentimiento) + walk-forward.

El motor del Sprint 3 decidía por umbrales de quant. Aquí lo conducimos con la
MISMA `confluence.decide` que usa el bot en vivo, alimentada con una serie
histórica de sentimiento alineada a las velas. Así el A/B es honesto: misma ruta
de decisión, sentimiento ON vs OFF.

Anti look-ahead: cada vela t solo ve el sentimiento más reciente con
`published_at <= t` y dentro de la ventana de antigüedad (`max_news_age_hours`).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd

from src.core.config import Settings, load_settings
from src.core.models import Action, SentimentScore, Signal
from src.decision.confluence import decide as confluence_decide

from backtest.engine import BacktestEngine, BacktestResult

# Una observación de sentimiento situada en el tiempo.
SentimentEvent = tuple[datetime, SentimentScore]


def events_from_rows(rows: list[dict]) -> list[SentimentEvent]:
    """Convierte filas de storage.get_sentiment_scores() en eventos temporales."""
    events: list[SentimentEvent] = []
    for r in rows:
        ts = datetime.fromtimestamp(r["ts"] / 1000, tz=timezone.utc)
        score = SentimentScore(
            news_id=r["news_id"], symbol_scope=r["symbol_scope"], score=r["score"],
            confidence=r["confidence"], high_impact=r["high_impact"],
            rationale=r.get("rationale", ""), analyzed_at=ts,
        )
        events.append((ts, score))
    return events


def align_sentiment(
    times: list[datetime], events: list[SentimentEvent], max_age_hours: int
) -> list[SentimentScore | None]:
    """Para cada vela, el SentimentScore vigente (más reciente y no caduco).

    Merge de dos series ordenadas en el tiempo, O(n+m). Un score "expira" tras
    `max_age_hours`: una noticia vieja no debe seguir moviendo trades para siempre.
    """
    max_age = timedelta(hours=max_age_hours)
    ordered = sorted(events, key=lambda e: e[0])
    out: list[SentimentScore | None] = []
    j = 0
    latest: SentimentEvent | None = None
    for t in times:
        while j < len(ordered) and ordered[j][0] <= t:
            latest = ordered[j]
            j += 1
        if latest is not None and (t - latest[0]) <= max_age:
            out.append(latest[1])
        else:
            out.append(None)
    return out


def make_confluence_decider(symbol, sentiments, settings, *, allow_short):
    """Decider para BacktestEngine.run que corre la confluencia por vela.

    Aplica además la reducción de tamaño por baja confianza (espejo del Risk
    Manager en vivo), para que el sizing del backtest no sea más optimista.
    """
    conf = settings.confluence
    rk = settings.risk

    def decide(i, position_side, score, ts):
        timestamp = ts if isinstance(ts, datetime) else datetime.now(timezone.utc)
        sig = Signal(symbol=symbol, score=float(score),
                     strategy="ema_cross_rsi", timestamp=timestamp)
        sent = sentiments[i]
        d = confluence_decide(sig, sent, settings)

        size_factor = d.size_factor
        if sent is not None and sent.confidence < rk.low_confidence_threshold:
            size_factor *= rk.low_confidence_size_factor

        if position_side is None:
            if d.action == Action.LONG:
                return ("enter", "LONG", size_factor)
            if d.action == Action.SHORT and allow_short:
                return ("enter", "SHORT", size_factor)
            return None
        # Salida cuando la confluencia deja de respaldar la pierna abierta.
        if position_side == "LONG" and d.action != Action.LONG:
            return ("exit",)
        if position_side == "SHORT" and d.action != Action.SHORT:
            return ("exit",)
        return None

    # `conf` referenciado para dejar claro de dónde salen los umbrales (auditoría).
    _ = conf
    return decide


def run_confluence(
    df: pd.DataFrame, symbol: str, timeframe: str,
    sentiment_events: list[SentimentEvent] | None = None,
    settings: Settings | None = None,
) -> BacktestResult:
    """Corre el backtest conducido por la confluencia (sentimiento ON si se pasa)."""
    settings = settings or load_settings()
    df = df.reset_index(drop=True)
    times = df["open_time"].tolist()
    sentiments = align_sentiment(
        times, sentiment_events or [], settings.sentiment.max_news_age_hours)
    decider = make_confluence_decider(
        symbol, sentiments, settings, allow_short=settings.backtest.allow_short)
    return BacktestEngine(settings).run(df, symbol, timeframe, decider=decider)


def walk_forward(
    df: pd.DataFrame, symbol: str, timeframe: str, *, n_folds: int,
    sentiment_events: list[SentimentEvent] | None = None,
    settings: Settings | None = None,
) -> list[tuple[int, int, BacktestResult]]:
    """Divide el histórico en `n_folds` tramos contiguos y backtestea cada uno.

    Sin optimización de parámetros (aún no hay perillas que ajustar), es un test
    de ROBUSTEZ: ¿el resultado es consistente entre periodos o un artefacto de un
    tramo afortunado? Devuelve [(lo, hi, result)] por tramo.
    """
    settings = settings or load_settings()
    df = df.reset_index(drop=True)
    n = len(df)
    size = n // n_folds
    out = []
    for k in range(n_folds):
        lo = k * size
        hi = (k + 1) * size if k < n_folds - 1 else n
        fold = df.iloc[lo:hi].reset_index(drop=True)
        if sentiment_events is not None:
            res = run_confluence(fold, symbol, timeframe, sentiment_events, settings)
        else:
            res = BacktestEngine(settings).run(fold, symbol, timeframe)
        out.append((lo, hi, res))
    return out
