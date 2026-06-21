"""Tests del backtest de confluencia: alineación anti-look-ahead, A/B del
sentimiento (ON vs OFF) sobre datos sintéticos, y walk-forward."""

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from src.core.config import load_settings
from src.core.models import SentimentScore
from backtest.confluence import (
    align_sentiment,
    events_from_rows,
    run_confluence,
    walk_forward,
)

CFG = load_settings()
T0 = datetime(2025, 6, 1, 0, 0, tzinfo=timezone.utc)


def make_uptrend_df(n: int = 90) -> pd.DataFrame:
    # Tendencia alcista clara → el quant da score fuerte LONG en las últimas velas.
    close = [1000.0 + 5 * i for i in range(n)]
    return pd.DataFrame({
        "open_time": pd.to_datetime([T0 + timedelta(minutes=5 * i) for i in range(n)], utc=True),
        "open": [c - 1 for c in close],
        "high": [c + 5 for c in close],
        "low": [c - 5 for c in close],
        "close": close,
        "volume": [10.0] * n,
    })


def event(score: float, *, conf: float = 0.8, at: datetime = T0) -> tuple:
    return (at, SentimentScore(news_id="e", symbol_scope=["*"], score=score,
                               confidence=conf, high_impact=False, analyzed_at=at))


# ------------------------------ align_sentiment ------------------------------

def test_alineacion_usa_el_score_mas_reciente_no_futuro():
    times = [T0, T0 + timedelta(hours=1), T0 + timedelta(hours=2)]
    evs = [event(0.5, at=T0 + timedelta(hours=1))]  # noticia llega en t1
    out = align_sentiment(times, evs, max_age_hours=24)
    assert out[0] is None                 # t0: la noticia aún no existía
    assert out[1].score == 0.5 and out[2].score == 0.5


def test_alineacion_expira_noticias_viejas():
    times = [T0, T0 + timedelta(hours=5)]
    evs = [event(0.7, at=T0)]
    out = align_sentiment(times, evs, max_age_hours=2)
    assert out[0].score == 0.7
    assert out[1] is None                 # a las 5h ya caducó (ventana 2h)


def test_events_from_rows():
    rows = [{"news_id": "n", "ts": int(T0.timestamp() * 1000), "score": -0.9,
             "confidence": 0.8, "high_impact": True, "symbol_scope": ["BTC"],
             "rationale": "hack"}]
    evs = events_from_rows(rows)
    assert evs[0][1].score == -0.9 and evs[0][1].high_impact is True


# ------------------------------ A/B del sentimiento ------------------------------

def test_sin_noticias_no_origina():
    # Opción 2 (inversión): sin noticias el quant ya NO origina → 0 trades. El
    # backtest llama al `decide` en vivo, así que refleja la nueva causalidad: el
    # EMA-cross en solitario (que perdía dinero) deja de meter entradas.
    res = run_confluence(make_uptrend_df(), "BTCUSDT", "5m", None, CFG)
    assert len(res.trades) == 0


def test_noticia_opuesta_al_regimen_veta_las_entradas():
    # Régimen (quant) alcista fuerte + noticia bajista significativa → la noticia
    # pediría SHORT pero el régimen la veta (regime_conflict) → 0 trades.
    res = run_confluence(make_uptrend_df(), "BTCUSDT", "5m", [event(-0.8)], CFG)
    assert len(res.trades) == 0


def test_noticia_alineada_con_regimen_origina():
    # Noticia alcista + régimen (quant) alcista fuerte → regime_confirms → opera.
    # (El tamaño pleno vs reducido por régimen/confianza se cubre en los tests
    # unitarios de `decide` y en test_baja_confianza_reduce_aunque_confirme.)
    res = run_confluence(make_uptrend_df(), "BTCUSDT", "5m", [event(0.6)], CFG)
    assert len(res.trades) > 0


def test_baja_confianza_reduce_aunque_confirme():
    # Sentimiento que confirma pero con confianza baja → tamaño recortado.
    full = run_confluence(make_uptrend_df(), "BTCUSDT", "5m", [event(0.6, conf=0.9)], CFG)
    low = run_confluence(make_uptrend_df(), "BTCUSDT", "5m", [event(0.6, conf=0.1)], CFG)
    assert low.trades[0].quantity < full.trades[0].quantity


# ------------------------------ walk-forward ------------------------------

def test_walk_forward_divide_en_tramos():
    folds = walk_forward(make_uptrend_df(120), "BTCUSDT", "5m", n_folds=3, settings=CFG)
    assert len(folds) == 3
    assert folds[0][0] == 0 and folds[-1][1] == 120   # cubren todo el histórico
    assert all(f[2].metrics is not None for f in folds)
