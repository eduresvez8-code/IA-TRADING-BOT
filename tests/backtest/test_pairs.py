"""Tests del simulador de cointegración de pares (Familia B) — lógica pura.

Todos los tests usan datos sintéticos para verificar la mecánica:
    - Dirección de la posición (long/short spread)
    - Rentabilidad bruta y neta de costos
    - Efecto del multiplicador de capital (β)
    - IC Spearman y corrección n_eff por autocorrelación
    - Regla de Oro (IC negativo + t-stat significativo + PF)
    - Casos degenerados: sin trades, pocos datos

Los datos reales (BTCUSDT, ETHUSDT, etc.) se ejercitan en run_quant_matrix.py.
"""

import math

import numpy as np
import pandas as pd
import pytest

from src.core.config import load_settings
from backtest.pairs import (
    PairsStats,
    _simulate_bars,
    _spearman_r,
    pairs_ic,
    run_pairs_all,
    simulate_pairs,
)
from backtest.quant_matrix import run_family_pairs

QM = load_settings().quant_matrix


def _s(arr) -> pd.Series:
    return pd.Series(np.asarray(arr, dtype=float))


def _cfg(**kw):
    return QM.model_copy(update=kw)


def _bars(z_vals, spread_vals, beta_vals=None, **kw):
    """Shortcut: llama a _simulate_bars con arrays."""
    n = len(z_vals)
    beta = np.ones(n) if beta_vals is None else np.asarray(beta_vals, dtype=float)
    cfg = _cfg(**kw)
    return _simulate_bars(_s(z_vals), _s(spread_vals), _s(beta), cfg)


# ---------------------------------------------------------------------------
# Tests de _simulate_bars
# ---------------------------------------------------------------------------

def test_z_bajo_umbral_no_genera_trades():
    # |z| = 1.5 nunca supera z_entry = 2.0 → 0 trades, bar_returns = 0
    bars, trades = _bars(np.full(100, 1.5), np.zeros(100))
    assert len(trades) == 0
    assert np.all(bars == 0.0)


def test_long_spread_gana_con_spread_creciente():
    # z=-3 en bar 0 (long spread), luego z=1.5 (holding), z=0.3 en bar 19 (exit)
    n = 20
    z = np.full(n, 1.5)
    z[0] = -3.0
    z[n - 1] = 0.3   # < pairs_z_exit = 0.5 → cierra
    spread = np.linspace(0.0, 0.2, n)   # spread sube 0.2 total

    bars, trades = _bars(z, spread)
    assert len(trades) == 1
    # gross = (spread[19] - spread[0]) / cap_mult_beta1 = 0.2 / 2.0 = 0.1
    # cost = 4 * 0.0005 / 2.0 = 0.001
    gross = 0.2 / 2.0
    cost = (4.0 * QM.taker_commission_pct / 100.0) / 2.0
    assert trades[0] == pytest.approx(gross - cost, rel=1e-6)
    assert trades[0] > 0


def test_short_spread_gana_con_spread_decreciente():
    # z=+3 en bar 0 (short spread), spread cae → profitable
    n = 20
    z = np.full(n, 1.5)
    z[0] = 3.0
    z[n - 1] = 0.3
    spread = np.linspace(0.2, 0.0, n)   # spread baja 0.2

    bars, trades = _bars(z, spread)
    assert len(trades) == 1
    gross = 0.2 / 2.0   # position=-1, spread baja → short_spread gana
    cost = (4.0 * QM.taker_commission_pct / 100.0) / 2.0
    assert trades[0] == pytest.approx(gross - cost, rel=1e-6)
    assert trades[0] > 0


def test_long_spread_pierde_si_spread_cae():
    # z=-3 (long spread) pero el spread BAJA → pérdida neta
    n = 20
    z = np.full(n, 1.5)
    z[0] = -3.0
    z[n - 1] = 0.3
    spread = np.linspace(0.2, 0.0, n)   # spread baja: long spread pierde

    bars, trades = _bars(z, spread)
    assert len(trades) == 1
    assert trades[0] < 0   # pérdida neta (la caída supera el costo)


def test_beta_mayor_reduce_retorno_sobre_capital():
    # beta=2 → cap_mult=3 → misma ganancia bruta de spread / cap_mult más grande
    n = 20
    z = np.full(n, 1.5)
    z[0] = -3.0
    z[n - 1] = 0.3
    spread = np.linspace(0.0, 0.3, n)

    _, trades_b1 = _bars(z, spread, np.ones(n))       # beta=1, cap=2
    _, trades_b2 = _bars(z, spread, np.full(n, 2.0))  # beta=2, cap=3

    # El retorno neto con beta=2 es menor que con beta=1
    assert trades_b2[0] < trades_b1[0]
    # cap_mult_b2 / cap_mult_b1 = 3 / 2 → gross_b2 = gross_b1 * (2/3)
    gross_b1 = spread[-1] / 2.0
    gross_b2 = spread[-1] / 3.0
    cost_b1 = 4 * QM.taker_commission_pct / 100.0 / 2.0
    cost_b2 = 4 * QM.taker_commission_pct / 100.0 / 3.0
    assert trades_b1[0] == pytest.approx(gross_b1 - cost_b1, rel=1e-6)
    assert trades_b2[0] == pytest.approx(gross_b2 - cost_b2, rel=1e-6)


def test_mayor_comision_reduce_retorno_neto():
    # Misma señal; comisión alta → net más bajo
    n = 20
    z = np.full(n, 1.5)
    z[0] = -3.0
    z[n - 1] = 0.3
    spread = np.linspace(0.0, 0.2, n)

    _, trades_low = _bars(z, spread, taker_commission_pct=0.01)
    _, trades_high = _bars(z, spread, taker_commission_pct=0.20)
    assert trades_high[0] < trades_low[0]


def test_bar_returns_suma_igual_a_gross_del_trade():
    # Los bar_returns acumulan el M2M; su suma debe igualar el gross del trade.
    n = 30
    z = np.full(n, 1.5)
    z[0] = -3.0
    z[n - 1] = 0.3
    spread = np.linspace(0.0, 0.6, n)

    bars, trades = _bars(z, spread)
    cap_mult = 2.0  # beta=1 por defecto
    gross_from_bars = float(bars.sum())
    gross_from_trade = trades[0] + (4.0 * QM.taker_commission_pct / 100.0) / cap_mult
    assert gross_from_bars == pytest.approx(gross_from_trade, rel=1e-6)


# ---------------------------------------------------------------------------
# Tests de _spearman_r y pairs_ic
# ---------------------------------------------------------------------------

def test_spearman_perfecto_anticorr():
    x = np.arange(10, dtype=float)
    y = -x
    assert _spearman_r(x, y) == pytest.approx(-1.0, abs=1e-9)


def test_ic_negativo_en_spread_perfectamente_reversivo():
    # Construimos z y spread tales que Δspread_{t+1} = -0.5 * z_t
    # → IC teórico = -1.0
    n = 500
    z_arr = np.sin(np.linspace(0, 20 * np.pi, n))   # muchos ciclos, oscila
    future_delta = -0.5 * z_arr                       # anticorrelación perfecta
    spread_arr = np.concatenate([[0.0], np.cumsum(future_delta[:-1])])

    ic, t = pairs_ic(_s(z_arr), _s(spread_arr))
    assert ic < -0.9    # IC muy negativo
    assert t < 0        # t también negativo (misma dirección)


def test_n_eff_reduce_tstat_en_z_muy_autocorrelado():
    # Con z de baja frecuencia (1 ciclo) la autocorr es muy alta → n_eff mínimo
    # → |t_corr| << |t_naive|
    n = 500
    z_arr = np.sin(np.linspace(0, 2 * np.pi, n))    # UN ciclo → autocorr ≈ 1
    future_delta = -0.5 * z_arr
    spread_arr = np.concatenate([[0.0], np.cumsum(future_delta[:-1])])

    ic, t = pairs_ic(_s(z_arr), _s(spread_arr))
    # t_naive (sin corregir) sería ~ IC × √(n-2) ~ enorme
    t_naive = abs(ic) * math.sqrt(n - 2) / math.sqrt(max(1 - ic ** 2, 1e-9))
    assert abs(t) < t_naive   # la corrección n_eff reduce el t-stat


def test_ic_cero_con_menos_de_30_observaciones():
    n = 25
    z = _s(np.zeros(n))
    spread = _s(np.zeros(n))
    ic, t = pairs_ic(z, spread)
    assert ic == 0.0
    assert t == 0.0


# ---------------------------------------------------------------------------
# Tests de simulate_pairs y run_pairs_all
# ---------------------------------------------------------------------------

def test_regla_de_oro_falla_si_spread_no_es_estacionario():
    # Spread = random walk (I(1), no estacionario, par NO cointegrado).
    # La innovación en t+1 es independiente de z_t → IC ≈ 0.
    # Además, el random walk genera un z muy autocorrelado (rho1≈0.99) →
    # n_eff≈19 → |t_corr| ≈ 0 aunque IC sea ligeramente ≠0 → falla el gate.
    # (Contraste: spread I(0) da rho1≈0.01 → n_eff≈2700 → t≈−49 → PASA.)
    rng = np.random.default_rng(42)
    n = 3000
    log_b = pd.Series(np.cumsum(rng.normal(0, 0.01, n)))
    nonstat_spread = pd.Series(np.cumsum(rng.normal(0, 0.001, n)))  # I(1)
    log_a = log_b + nonstat_spread

    stats = simulate_pairs(
        log_a, log_b,
        _cfg(pairs_lookback_hours=200, pairs_z_entry=2.0, pairs_z_exit=0.5),
        pair_name="TEST",
    )
    assert stats.passes_golden is False


def test_run_pairs_all_devuelve_c52_pares():
    # 5 activos → C(5,2) = 10 pares
    rng = np.random.default_rng(0)
    n = 100
    cols = ["A", "B", "C", "D", "E"]
    prices = pd.DataFrame(
        {c: np.cumsum(rng.normal(0, 0.01, n)) for c in cols}
    )
    cfg = _cfg(pairs_lookback_hours=24)
    results = run_pairs_all(prices, cfg)
    assert len(results) == 10
    assert all(isinstance(r, PairsStats) for r in results)


def test_run_family_pairs_ya_no_es_stub():
    # El stub fue reemplazado; llamar a run_family_pairs NO lanza NotImplementedError
    rng = np.random.default_rng(1)
    n = 60
    log_prices = pd.DataFrame({
        "X": np.cumsum(rng.normal(0, 0.01, n)),
        "Y": np.cumsum(rng.normal(0, 0.01, n)),
    })
    cfg = _cfg(pairs_lookback_hours=24)
    result = run_family_pairs(log_prices, cfg)
    assert isinstance(result, list)
