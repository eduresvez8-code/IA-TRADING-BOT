"""Tests del simulador de squeeze→ruptura (Familia D) — lógica pura, sintética.

Verificamos con series construidas a mano: detección de squeeze (BB dentro de
Keltner), dirección de la ruptura, continuación (LONG si rompe arriba), holding
time-based, robustez al shift-1 (sin lookahead), costos, descomposición gross/net
e IC con corrección n_eff. Los datos reales (1h de los 5 activos) se ejercitan en
run_quant_matrix.py.
"""

import math

import numpy as np
import pandas as pd
import pytest

from src.core.config import load_settings
from backtest.squeeze import (
    PERIODS_PER_YEAR,
    SqueezeStats,
    _atr,
    _simulate_bars,
    _spearman_r,
    simulate_squeeze,
    squeeze_ic,
    squeeze_signal,
)
from backtest.quant_matrix import run_family_squeeze

QM = load_settings().quant_matrix


def _s(arr) -> pd.Series:
    return pd.Series(np.asarray(arr, dtype=float))


def _cfg(**kw):
    return QM.model_copy(update=kw)


# Config base para la máquina de estados: sin slippage (aislar la mecánica del
# P&L de la fricción dinámica), holding corto y explícito.
def _bars_cfg(**kw):
    base = dict(squeeze_forward_horizon=3, taker_commission_pct=0.0,
                slippage_pct=0.0, slippage_atr_mult=0.0)
    base.update(kw)
    return _cfg(**base)


def _run_bars(breakout, fire, close, *, atr_px=None, **cfg_kw):
    n = len(close)
    atr = _s([0.0] * n) if atr_px is None else _s(atr_px)
    return _simulate_bars(_s(breakout), pd.Series(fire), _s(close), atr, _bars_cfg(**cfg_kw))


# ---------------------------------------------------------------------------
# Construcción de la señal: squeeze y dirección de ruptura
# ---------------------------------------------------------------------------

def test_atr_positivo():
    high = _s([101, 102, 103, 104, 105])
    low = _s([99, 100, 101, 102, 103])
    close = _s([100, 101, 102, 103, 104])
    atr = _atr(high, low, close, period=2).dropna()
    assert (atr > 0).all()


def test_squeeze_on_cuando_banda_estrecha():
    # Precio plano con micro-ruido → σ del cierre ≈ 0 pero el ATR (high−low) es
    # apreciable → band < keltner_k·ATR → squeeze ON en la cola de la serie.
    n = 60
    rng = np.random.default_rng(0)
    close = _s(100 + rng.normal(0, 0.001, n))          # dispersión del cierre ínfima
    high = close + 1.0                                  # rango intrabar amplio → ATR alto
    low = close - 1.0
    sq, breakout, fire = squeeze_signal(high, low, close, _cfg(squeeze_bb_period=20))
    assert sq.iloc[-1]                                  # comprimido: BB dentro de Keltner


def test_no_squeeze_cuando_cierre_muy_volatil():
    # Cierre muy disperso vs rango intrabar pequeño → band > keltner·ATR → sin squeeze.
    n = 60
    rng = np.random.default_rng(1)
    close = _s(100 + np.cumsum(rng.normal(0, 2.0, n)))  # cierre con gran dispersión
    high = close + 0.01
    low = close - 0.01
    sq, _, _ = squeeze_signal(high, low, close, _cfg(squeeze_bb_period=20))
    assert not sq.iloc[-1]


def test_breakout_firmado_y_direccion():
    # Construye un squeeze y luego un salto del cierre por encima de la banda.
    n = 40
    close = _s([100.0] * (n - 1) + [105.0])             # último bar rompe hacia arriba
    high = close + 1.0
    low = close - 1.0
    sq, breakout, fire = squeeze_signal(high, low, close, _cfg(squeeze_bb_period=20))
    assert breakout.iloc[-1] > 0                        # rompe ARRIBA → breakout positivo
    assert fire.iloc[-1]                                # squeeze previo + ruptura → fire


# ---------------------------------------------------------------------------
# Máquina de estados — continuación, holding, P&L
# ---------------------------------------------------------------------------

def test_long_gana_si_continua_subiendo():
    # fire en t=2 con breakout>0 → señal shift-1 → entra LONG en close[3]; sube → gana.
    breakout = [np.nan, np.nan, 1.2, 0.0, 0.0, 0.0]
    fire = [False, False, True, False, False, False]
    close = [100, 100, 100, 100, 101, 102]
    g, net, trades, holds = _run_bars(breakout, fire, close)
    assert len(trades) == 1
    assert trades[0] > 0                                # continuación alcista gana


def test_short_gana_si_continua_bajando():
    # fire con breakout<0 → entra SHORT; el precio cae → gana.
    breakout = [np.nan, np.nan, -1.2, 0.0, 0.0, 0.0]
    fire = [False, False, True, False, False, False]
    close = [100, 100, 100, 100, 99, 98]
    g, net, trades, holds = _run_bars(breakout, fire, close)
    assert len(trades) == 1
    assert trades[0] > 0                                # continuación bajista gana


def test_long_pierde_si_revierte():
    # Rompe arriba (LONG) pero el precio CAE (falso breakout) → pérdida.
    breakout = [np.nan, np.nan, 1.2, 0.0, 0.0, 0.0]
    fire = [False, False, True, False, False, False]
    close = [100, 100, 100, 100, 99, 98]
    g, net, trades, holds = _run_bars(breakout, fire, close)
    assert len(trades) == 1
    assert trades[0] < 0


def test_holding_time_based():
    # horizon=3 → entra en i=3, sale en i=6 (3 barras). Verifica el holding fijo.
    breakout = [np.nan, np.nan, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    fire = [False, False, True, False, False, False, False, False]
    close = [100, 100, 100, 100, 101, 102, 103, 104]
    g, net, trades, holds = _run_bars(breakout, fire, close, squeeze_forward_horizon=3)
    assert len(trades) == 1
    assert holds[0] == 3                                # entró i=3, salió i=6


def test_senal_es_shift1_no_lookahead():
    # Una fire SOLO en la última barra no puede abrir trade (se actuaría en t+1,
    # fuera de la serie). Verifica el shift-1 (sin lookahead).
    breakout = [np.nan, np.nan, np.nan, 1.5]
    fire = [False, False, False, True]
    close = [100, 100, 100, 100]
    g, net, trades, holds = _run_bars(breakout, fire, close)
    assert len(trades) == 0


def test_no_piramidea():
    # Una segunda fire mientras hay posición abierta NO abre un segundo trade.
    breakout = [np.nan, np.nan, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    fire = [False, False, True, True, False, False, False, False]
    close = [100, 100, 100, 100, 101, 102, 103, 104]
    g, net, trades, holds = _run_bars(breakout, fire, close, squeeze_forward_horizon=3)
    assert len(trades) == 1                             # solo el primero


def test_mayor_comision_reduce_neto():
    breakout = [np.nan, np.nan, 1.0, 0.0, 0.0, 0.0]
    fire = [False, False, True, False, False, False]
    close = [100, 100, 100, 100, 101, 102]
    _, _, t_low, _ = _run_bars(breakout, fire, close, taker_commission_pct=0.01)
    _, _, t_high, _ = _run_bars(breakout, fire, close, taker_commission_pct=0.30)
    assert t_high[0] < t_low[0]


def test_slippage_dinamico_reduce_neto():
    breakout = [np.nan, np.nan, 1.0, 0.0, 0.0, 0.0]
    fire = [False, False, True, False, False, False]
    close = [100, 100, 100, 100, 101, 102]
    atr = [0.0, 0.0, 0.01, 0.01, 0.01, 0.01]
    _, _, t_sin, _ = _run_bars(breakout, fire, close, atr_px=atr, slippage_atr_mult=0.0)
    _, _, t_con, _ = _run_bars(breakout, fire, close, atr_px=atr, slippage_atr_mult=1.0)
    assert t_con[0] < t_sin[0]


def test_gross_mayor_que_net_con_costos():
    # La equity GROSS (sin costos) debe superar a la NETA cuando hay comisión.
    breakout = [np.nan, np.nan, 1.0, 0.0, 0.0, 0.0]
    fire = [False, False, True, False, False, False]
    close = [100, 100, 100, 100, 101, 102]
    g, net, trades, _ = _run_bars(breakout, fire, close, taker_commission_pct=0.10)
    assert float(g.sum()) > float(net.sum())            # costos separan gross de neto
    assert float(net.sum()) == pytest.approx(trades[0], rel=1e-9)  # neto = trade neto


# ---------------------------------------------------------------------------
# IC gate (Etapa 1) — continuación: IC > 0
# ---------------------------------------------------------------------------

def test_spearman_perfecto_corr():
    x = np.arange(10, dtype=float)
    assert _spearman_r(x, x) == pytest.approx(1.0, abs=1e-9)


def test_ic_positivo_en_continuacion_sintetica():
    # Construimos close tal que un breakout grande precede a una subida (continúa)
    # → IC(breakout, ret_fwd) > 0, medido SOLO sobre las fires.
    n = 1500
    rng = np.random.default_rng(3)
    breakout = rng.normal(0, 1.0, n)
    fire = np.ones(n, dtype=bool)                       # toda barra es fire (test del IC)
    # ret_{t+1} proporcional al breakout_t (continuación) + ruido
    ret = 0.002 * breakout + rng.normal(0, 0.0005, n)
    close = 100.0 * np.exp(np.cumsum(np.concatenate([[0.0], ret[:-1]])))
    ic, t = squeeze_ic(_s(breakout), pd.Series(fire), _s(close), horizon=1)
    assert ic > 0
    assert t > 0


def test_ic_cero_si_pocas_fires():
    # Menos de 30 fires → IC neutral (no se arriesga un t-stat con n diminuto).
    n = 200
    breakout = np.linspace(-1, 1, n)
    fire = np.zeros(n, dtype=bool)
    fire[:10] = True                                   # solo 10 fires
    close = _s(100 + np.arange(n) * 0.0)
    ic, t = squeeze_ic(_s(breakout), pd.Series(fire), close, horizon=1)
    assert ic == 0.0 and t == 0.0


# ---------------------------------------------------------------------------
# simulate_squeeze end-to-end + integración
# ---------------------------------------------------------------------------

def _synth_df(n=2000, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.002, n)))
    high = close * (1 + rng.uniform(0, 0.003, n))
    low = close * (1 - rng.uniform(0, 0.003, n))
    return pd.DataFrame({
        "high": high, "low": low, "close": close,
        "volume": rng.uniform(1, 10, n),
    })


def test_simulate_squeeze_devuelve_stats():
    stats = simulate_squeeze(_synth_df(), _cfg(squeeze_bb_period=20), symbol="SYNTH")
    assert isinstance(stats, SqueezeStats)
    assert stats.symbol == "SYNTH"
    assert stats.n_periods > 0
    assert 0.0 <= stats.pct_squeeze <= 100.0


def test_ruido_puro_no_pasa_la_regla_de_oro():
    # Un random walk sin continuación estructural no debe coronar.
    stats = simulate_squeeze(_synth_df(seed=123), _cfg(squeeze_bb_period=20), symbol="NOISE")
    assert stats.passes_golden is False


def test_run_family_squeeze_ya_no_es_stub():
    result = run_family_squeeze(_synth_df(n=800), _cfg(squeeze_bb_period=20), symbol="X")
    assert isinstance(result, SqueezeStats)


def test_periods_per_year_1h_correcto():
    assert PERIODS_PER_YEAR == 365 * 24
