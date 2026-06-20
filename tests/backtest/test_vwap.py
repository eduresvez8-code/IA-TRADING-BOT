"""Tests del simulador de reversión a VWAP (Familia C) — lógica pura, sintética.

Verificamos la mecánica con series construidas a mano: VWAP anclado, z-score,
dirección de la posición, robustez al bounce (señal shift-1), cierre forzado en
el límite del día, costos, e IC con corrección n_eff. Los datos reales (5m de los
5 activos) se ejercitan en run_quant_matrix.py.
"""

import math

import numpy as np
import pandas as pd
import pytest

from src.core.config import load_settings
from backtest.vwap import (
    BARS_PER_YEAR_5M,
    VwapStats,
    _simulate_bars,
    _spearman_r,
    atr_over_price,
    daily_anchored_vwap,
    simulate_vwap,
    vwap_deviation_z,
    vwap_ic,
)
from backtest.quant_matrix import run_family_volume

QM = load_settings().quant_matrix


def _s(arr) -> pd.Series:
    return pd.Series(np.asarray(arr, dtype=float))


def _cfg(**kw):
    return QM.model_copy(update=kw)


# Config base para los tests de la máquina de estados: sin slippage (aislar la
# mecánica del P&L de la fricción dinámica), umbrales explícitos.
def _bars_cfg(**kw):
    base = dict(vwap_z_entry=2.0, vwap_z_exit=0.5,
                slippage_pct=0.0, slippage_atr_mult=0.0)
    base.update(kw)
    return _cfg(**base)


def _run_bars(dev_z, close, *, day_key=None, atr_px=None, **cfg_kw):
    n = len(dev_z)
    day = _s([0] * n) if day_key is None else pd.Series(day_key)
    atr = _s([0.0] * n) if atr_px is None else _s(atr_px)
    return _simulate_bars(_s(dev_z), _s(close), atr, day, _bars_cfg(**cfg_kw))


# ---------------------------------------------------------------------------
# Construcción de la señal
# ---------------------------------------------------------------------------

def test_vwap_anclado_resetea_cada_dia():
    # Precio constante → VWAP == precio. Dos días, mismo valor.
    n = 6
    high = _s([100] * n); low = _s([100] * n); close = _s([100] * n)
    vol = _s([10] * n)
    day = pd.Series([0, 0, 0, 1, 1, 1])
    vwap = daily_anchored_vwap(high, low, close, vol, day)
    assert np.allclose(vwap.to_numpy(), 100.0)


def test_vwap_pondera_por_volumen():
    # Día único, dos barras: tp=[100,110], vol=[1,3] → VWAP final = (100·1+110·3)/4 = 107.5
    high = _s([100, 110]); low = _s([100, 110]); close = _s([100, 110])
    vol = _s([1, 3]); day = pd.Series([0, 0])
    vwap = daily_anchored_vwap(high, low, close, vol, day)
    assert vwap.iloc[0] == pytest.approx(100.0)
    assert vwap.iloc[1] == pytest.approx(107.5)


def test_deviation_z_estandariza():
    # Con desviaciones simétricas, el z-score tiene media ~0.
    close = _s([100, 102, 98, 101, 99, 100, 103, 97] * 50)
    vwap = _s([100] * len(close))
    z = vwap_deviation_z(close, vwap, window=12).dropna()
    assert abs(z.mean()) < 1.0   # centrado


def test_atr_over_price_positivo():
    high = _s([101, 102, 103, 104, 105])
    low = _s([99, 100, 101, 102, 103])
    close = _s([100, 101, 102, 103, 104])
    atr = atr_over_price(high, low, close, period=2).dropna()
    assert (atr > 0).all()


# ---------------------------------------------------------------------------
# Máquina de estados — dirección y P&L
# ---------------------------------------------------------------------------

def test_sin_trades_si_z_bajo_umbral():
    # |z| = 1.5 < z_entry = 2.0 → nunca entra.
    bars, trades, holds = _run_bars([1.5] * 20, [100] * 20)
    assert len(trades) == 0
    assert np.all(bars == 0.0)


def test_short_gana_si_precio_cae():
    # z alto (sobre VWAP) → SHORT; el precio cae → beneficio.
    # Señal en t actúa en t+1 (shift-1): dev_z[2]=3 → entra en close[3].
    dev_z = [np.nan, np.nan, 3.0, 1.5, 1.5, 0.3, 0.0, 0.0]
    close = [100, 100, 100, 100, 99, 98, 97, 97]
    bars, trades, holds = _run_bars(dev_z, close)
    assert len(trades) == 1
    assert trades[0] > 0          # short con precio cayendo gana
    assert holds[0] == 3          # entró en i=3, salió en i=6


def test_long_gana_si_precio_sube():
    # z muy negativo (bajo VWAP) → LONG; el precio sube → beneficio.
    dev_z = [np.nan, np.nan, -3.0, -1.5, -1.5, -0.3, 0.0, 0.0]
    close = [100, 100, 100, 100, 101, 102, 103, 103]
    bars, trades, holds = _run_bars(dev_z, close)
    assert len(trades) == 1
    assert trades[0] > 0          # long con precio subiendo gana


def test_short_pierde_si_precio_sube():
    # z alto → SHORT, pero el precio SUBE → pérdida.
    dev_z = [np.nan, np.nan, 3.0, 1.5, 1.5, 0.3]
    close = [100, 100, 100, 101, 102, 103]
    bars, trades, holds = _run_bars(dev_z, close)
    assert len(trades) == 1
    assert trades[0] < 0


def test_senal_es_shift1_no_lookahead():
    # Un pico de z SOLO en la última barra no puede abrir trade (se actuaría en
    # t+1, fuera de la serie). Esto verifica el shift-1 (sin lookahead).
    dev_z = [np.nan, np.nan, np.nan, 3.0]
    close = [100, 100, 100, 100]
    bars, trades, holds = _run_bars(dev_z, close)
    assert len(trades) == 0


def test_cierre_forzado_en_limite_de_dia():
    # Posición abierta y cambia el día → cierre forzado aunque |z| > z_exit.
    dev_z = [np.nan, 3.0, 1.5, 1.5, 1.5]
    close = [100, 100, 100, 100, 100]
    day = [0, 0, 0, 1, 1]    # cambio de día en i=3
    bars, trades, holds = _run_bars(dev_z, close, day_key=day)
    assert len(trades) == 1
    assert holds[0] == 1     # entró en i=2, forzado a salir en i=3 (nuevo día)


def test_mayor_comision_reduce_neto():
    dev_z = [np.nan, np.nan, 3.0, 1.5, 0.3]
    close = [100, 100, 100, 99, 98]
    _, t_low, _ = _run_bars(dev_z, close, taker_commission_pct=0.01)
    _, t_high, _ = _run_bars(dev_z, close, taker_commission_pct=0.30)
    assert t_high[0] < t_low[0]


def test_slippage_dinamico_reduce_neto():
    # Con ATR>0 y k>0, el slippage extra recorta el neto.
    dev_z = [np.nan, np.nan, 3.0, 1.5, 0.3]
    close = [100, 100, 100, 99, 98]
    atr = [0.0, 0.0, 0.01, 0.01, 0.01]
    _, t_sin, _ = _run_bars(dev_z, close, atr_px=atr, slippage_atr_mult=0.0)
    _, t_con, _ = _run_bars(dev_z, close, atr_px=atr, slippage_atr_mult=1.0)
    assert t_con[0] < t_sin[0]


def test_equity_neta_incluye_costos():
    # La suma de bar_returns (equity neta) debe ser < gross (porque resta costos).
    dev_z = [np.nan, np.nan, 3.0, 1.5, 1.5, 0.3]
    close = [100, 100, 100, 99, 98, 97]
    bars, trades, _ = _run_bars(dev_z, close, taker_commission_pct=0.10)
    # gross del trade = neto + costo > neto; la equity (sum bars) iguala el neto
    assert float(bars.sum()) == pytest.approx(trades[0], rel=1e-9)


# ---------------------------------------------------------------------------
# IC gate (Etapa 1)
# ---------------------------------------------------------------------------

def test_spearman_perfecto_anticorr():
    x = np.arange(10, dtype=float)
    assert _spearman_r(x, -x) == pytest.approx(-1.0, abs=1e-9)


def test_ic_negativo_en_reversion_sintetica():
    # Construimos close tal que un z alto precede a una caída → IC < 0.
    n = 2000
    rng = np.random.default_rng(7)
    z = np.sin(np.linspace(0, 60 * np.pi, n)) + rng.normal(0, 0.05, n)
    # ret_{t+1} = -0.001 * z_t  → precio baja tras z alto
    ret = -0.001 * z
    close = 100.0 * np.exp(np.cumsum(np.concatenate([[0.0], ret[:-1]])))
    ic, t = vwap_ic(_s(z), _s(close), horizon=1)
    assert ic < 0
    assert t < 0


def test_ic_cero_con_pocas_observaciones():
    ic, t = vwap_ic(_s([1.0] * 10), _s([100.0] * 10), horizon=1)
    assert ic == 0.0
    assert t == 0.0


# ---------------------------------------------------------------------------
# simulate_vwap end-to-end + integración
# ---------------------------------------------------------------------------

def _synth_df(n=3000, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.001, n)))
    high = close * (1 + rng.uniform(0, 0.002, n))
    low = close * (1 - rng.uniform(0, 0.002, n))
    vol = rng.uniform(1, 10, n)
    t0 = pd.Timestamp("2024-01-01", tz="UTC")
    open_time = pd.date_range(t0, periods=n, freq="5min")
    return pd.DataFrame({
        "open_time": open_time, "high": high, "low": low,
        "close": close, "volume": vol,
    })


def test_simulate_vwap_devuelve_stats():
    df = _synth_df()
    stats = simulate_vwap(df, _cfg(vwap_z_window=288), symbol="SYNTH")
    assert isinstance(stats, VwapStats)
    assert stats.symbol == "SYNTH"
    assert stats.n_periods > 0


def test_ruido_puro_no_pasa_la_regla_de_oro():
    # Un random walk sin reversión estructural no debe coronar.
    df = _synth_df(seed=123)
    stats = simulate_vwap(df, _cfg(vwap_z_window=288), symbol="NOISE")
    assert stats.passes_golden is False


def test_run_family_volume_ya_no_es_stub():
    df = _synth_df(n=1000)
    result = run_family_volume(df, _cfg(vwap_z_window=288), symbol="X")
    assert isinstance(result, VwapStats)


def test_bars_per_year_5m_correcto():
    assert BARS_PER_YEAR_5M == 365 * 24 * 12
