"""Tests de backtest/positioning.py: funciones puras con valores a mano."""

import math

import numpy as np
import pandas as pd
import pytest

from backtest.positioning import (
    annualized_sharpe,
    net_strategy_returns,
    rolling_zscore,
    split_by_date,
    taker_imbalance,
    threshold_positions,
)


def _idx(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")


# ---------- taker_imbalance ----------

def test_taker_imbalance_extremos_y_equilibrio():
    vol = pd.Series([100.0, 100.0, 100.0], index=_idx(3))
    buy = pd.Series([100.0, 0.0, 50.0], index=_idx(3))
    imb = taker_imbalance(vol, buy)
    # Todo compra agresiva → +1; todo venta → −1; mitad y mitad → 0.
    assert imb.tolist() == [1.0, -1.0, 0.0]


def test_taker_imbalance_volumen_cero_es_nan():
    vol = pd.Series([0.0, 10.0], index=_idx(2))
    buy = pd.Series([0.0, 10.0], index=_idx(2))
    imb = taker_imbalance(vol, buy)
    # Sin volumen no hay información: NaN, no un 0 que se colaría en la media.
    assert math.isnan(imb.iloc[0])
    assert imb.iloc[1] == 1.0


# ---------- rolling_zscore ----------

def test_rolling_zscore_valor_a_mano():
    s = pd.Series([1.0, 2.0, 3.0, 6.0], index=_idx(4))
    z = rolling_zscore(s, 3)
    # Ventana [2,3,6]: media 11/3, std muestral ≈ 2.0817 → z = (6-11/3)/2.0817
    assert z.iloc[3] == pytest.approx((6 - 11 / 3) / 2.081665999, rel=1e-6)
    # Warmup: las primeras window-1 barras no tienen z (causal, sin inventar).
    assert z.iloc[:2].isna().all()


def test_rolling_zscore_sin_varianza_es_nan():
    s = pd.Series([5.0] * 6, index=_idx(6))
    z = rolling_zscore(s, 3)
    # std=0 → NaN (no ±inf): una serie constante no tiene "sorpresa" medible.
    assert z.iloc[2:].isna().all()


# ---------- threshold_positions ----------

def test_threshold_positions_momentum_y_contrarian():
    z = pd.Series([2.0, -2.0, 0.3, np.nan], index=_idx(4))
    mom = threshold_positions(z, 1.0, +1)
    con = threshold_positions(z, 1.0, -1)
    # momentum sigue el signo del z; contrarian lo invierte; |z|<=umbral → plano.
    assert mom.tolist() == [1.0, -1.0, 0.0, 0.0]
    assert con.tolist() == [-1.0, 1.0, 0.0, 0.0]


def test_threshold_positions_direction_invalida():
    z = pd.Series([1.0], index=_idx(1))
    with pytest.raises(ValueError):
        threshold_positions(z, 0.5, 0)


# ---------- net_strategy_returns ----------

def test_net_strategy_returns_timing_y_costos_a_mano():
    # pos decidida al cierre de t gana el retorno de t+1, nunca el de t.
    pos = pd.Series([0.0, 1.0, 1.0, 0.0], index=_idx(4))
    ret = pd.Series([0.10, 0.20, 0.05, -0.30], index=_idx(4))
    cost = 0.01
    pnl = net_strategy_returns(pos, ret, cost)
    # t0: plano y sin cambio → 0. t1: entra (turnover 1 → paga 0.01) y gana el
    # retorno de la barra SIGUIENTE (0.05, no el 0.20 de su propia barra) → 0.04.
    # t2: mantiene (turnover 0) y se come el −0.30 de t3 sin costo → −0.30.
    # t3: la salida cae en la última barra, que no tiene forward → se descarta.
    assert pnl.iloc[0] == pytest.approx(0.0)
    assert pnl.iloc[1] == pytest.approx(1.0 * 0.05 - 0.01)
    assert pnl.iloc[2] == pytest.approx(1.0 * (-0.30))
    assert len(pnl) == 3


def test_net_strategy_returns_flip_paga_dos_lados():
    pos = pd.Series([1.0, -1.0], index=_idx(2))
    ret = pd.Series([0.0, 0.0], index=_idx(2))
    pnl = net_strategy_returns(pos, ret, 0.01)
    # Primera barra: entrar de 0→1 paga 1 lado. Voltear 1→−1 paga 2 lados,
    # pero cae en la última barra (sin forward) → solo queda la primera.
    assert pnl.iloc[0] == pytest.approx(-0.01)


# ---------- annualized_sharpe ----------

def test_annualized_sharpe_valor_a_mano():
    r = pd.Series([0.01, -0.01, 0.01, -0.01] * 25, index=_idx(100))
    # media 0 → Sharpe 0.
    assert annualized_sharpe(r, 8760) == pytest.approx(0.0)
    r2 = pd.Series([0.01] * 100, index=_idx(100))
    # sin varianza → 0 por convención (misma que metrics.sharpe_ratio).
    assert annualized_sharpe(r2, 8760) == 0.0


# ---------- split_by_date ----------

def test_split_by_date_sin_solape_ni_hueco():
    s = pd.Series(range(10), index=_idx(10))
    cut = s.index[6]
    tr, te = split_by_date(s, cut)
    assert len(tr) + len(te) == len(s)
    assert tr.index.max() < cut
    assert te.index.min() == cut  # el corte cae en TEST (>=), jamás en ambos
