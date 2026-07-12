"""Tests de la parte PURA del paper trading RSI-2 (sin red, sin disco).

Reusa el MISMO escenario ya congelado en
tests/backtest/test_sp500_families.py::test_rsi_reversion_compra_dip_dentro_de_tendencia
— la señal cruda debe coincidir exactamente con la ya probada del backtest,
porque `compute_daily_rows` reutiliza `rsi_reversion_daily_position` sin
reimplementar nada.
"""

import numpy as np
import pandas as pd

from src.paper_trading.rsi2 import compute_daily_rows


def _daily_df(dates, closes):
    return pd.DataFrame({
        "open_time": pd.to_datetime(dates, utc=True),
        "open": np.asarray(closes, dtype=float),
        "high": np.asarray(closes, dtype=float) * 1.01,
        "low": np.asarray(closes, dtype=float) * 0.99,
        "close": np.asarray(closes, dtype=float),
        "volume": np.full(len(dates), 1e6),
    })


def _scenario():
    # MISMO escenario que test_rsi_reversion_compra_dip_dentro_de_tendencia:
    # tendencia alcista + dip suave (entra) + recuperación fuerte (sale).
    up = list(np.linspace(100, 160, 80))
    dip = [155.0, 150.0]
    rec = [158.0, 166.0, 172.0]
    closes = up + dip + rec
    dates = pd.date_range("2020-01-01", periods=len(closes), freq="B", tz="UTC")
    return _daily_df(dates, closes), dates


def test_compute_daily_rows_detecta_enter_y_exit_una_sola_vez():
    df, dates = _scenario()
    since = pd.Timestamp(dates[0]).tz_convert("UTC").tz_localize(None)
    out = compute_daily_rows(df, rsi_period=2, entry_below=10.0, exit_above=70.0,
                             trend_sma_days=50, since=since)
    enters = out.index[out["action"] == "ENTER"]
    exits = out.index[out["action"] == "EXIT"]
    assert len(enters) == 1
    assert len(exits) == 1
    assert enters[0] < exits[0]
    # La entrada cae dentro de la ventana del dip (índices 80-81 del escenario).
    assert dates[80] <= enters[0].tz_localize("UTC") <= dates[81]
    # position=1.0 desde la entrada hasta justo antes de la salida.
    span = out.loc[enters[0]:exits[0]]
    assert (span["position"].iloc[:-1] == 1.0).all()
    assert span["position"].iloc[-1] == 0.0


def test_compute_daily_rows_since_no_pierde_la_transicion_en_el_borde():
    # Pedir SOLO desde el día de la entrada (no desde el inicio de la serie)
    # debe seguir viendo el ENTER ahí — la posición se calcula sobre la serie
    # COMPLETA antes de recortar, así una corrida que retoma tras días
    # perdidos no pierde la transición real.
    df, dates = _scenario()
    full = compute_daily_rows(df, rsi_period=2, entry_below=10.0, exit_above=70.0,
                              trend_sma_days=50,
                              since=pd.Timestamp(dates[0]).tz_convert("UTC").tz_localize(None))
    entry_date = full.index[full["action"] == "ENTER"][0]

    trimmed = compute_daily_rows(df, rsi_period=2, entry_below=10.0, exit_above=70.0,
                                 trend_sma_days=50, since=entry_date)
    assert trimmed.index.min() == entry_date
    assert trimmed.loc[entry_date, "action"] == "ENTER"


def test_compute_daily_rows_sin_transicion_columna_accion_vacia():
    # 30 días planos: RSI nunca entra en pánico → nunca hay acción.
    closes = [100.0 + 0.01 * (i % 2) for i in range(30)]
    dates = pd.date_range("2020-01-01", periods=len(closes), freq="B", tz="UTC")
    df = _daily_df(dates, closes)
    out = compute_daily_rows(df, rsi_period=2, entry_below=10.0, exit_above=70.0,
                             trend_sma_days=20,
                             since=pd.Timestamp(dates[0]).tz_convert("UTC").tz_localize(None))
    assert (out["action"] == "").all()
    assert (out["position"] == 0.0).all()
