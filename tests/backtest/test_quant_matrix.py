"""Tests del simulador de carry (Familia E) — lógica pura, sin red ni datos reales.

Construimos series de funding sintéticas para verificar la mecánica: signo del
yield, costos one-time, fricción de capital, mantenimiento, walk-forward y Regla
de Oro. Los datos reales se ejercitan en el runner (no en unit-tests).
"""

import math

import numpy as np
import pandas as pd
import pytest

from src.core.config import load_settings
from backtest.quant_matrix import (
    PERIODS_PER_YEAR,
    carry_period_returns,
    simulate_carry,
    run_family_pairs,
    run_family_volume,
    run_family_squeeze,
)

QM = load_settings().quant_matrix


def _funding(values) -> pd.Series:
    return pd.Series(np.asarray(values, dtype=float))


def test_funding_positivo_constante_da_yield_positivo():
    # Funding fijo +0.01%/8h, sin mantenimiento → yield neto positivo, t enorme.
    fr = _funding([0.0001] * 1095)  # un año de periodos
    stats = simulate_carry(fr, QM, symbol="TEST")
    assert stats.net_yield_ann_pct > 0
    assert stats.t_stat > QM.golden_min_tstat
    assert stats.pct_negative_periods == 0.0


def test_capital_multiplier_parte_el_yield():
    # El yield sobre capital = yield sobre notional / capital_multiplier.
    fr = _funding([0.0001] * 1095)
    qm1 = QM.model_copy(update={"carry_capital_multiplier": 1.0})
    qm2 = QM.model_copy(update={"carry_capital_multiplier": 2.0})
    y1 = simulate_carry(fr, qm1).net_yield_ann_pct
    y2 = simulate_carry(fr, qm2).net_yield_ann_pct
    assert y2 == pytest.approx(y1 / 2.0, rel=1e-6)


def test_gross_yield_es_independiente_del_capital():
    # El yield BRUTO es sobre notional → no lo toca el capital_multiplier.
    fr = _funding([0.00008] * 1095)
    g1 = simulate_carry(fr, QM.model_copy(update={"carry_capital_multiplier": 1.0})).gross_yield_ann_pct
    g2 = simulate_carry(fr, QM.model_copy(update={"carry_capital_multiplier": 5.0})).gross_yield_ann_pct
    assert g1 == pytest.approx(g2, rel=1e-9)
    assert g1 == pytest.approx(0.00008 * PERIODS_PER_YEAR * 100, rel=1e-9)


def test_mantenimiento_resta_del_retorno():
    # Un haircut de mantenimiento reduce el yield neto.
    fr = _funding([0.0001] * 1095)
    sin = simulate_carry(fr, QM.model_copy(update={"carry_maintenance_bps_per_period": 0.0}))
    con = simulate_carry(fr, QM.model_copy(update={"carry_maintenance_bps_per_period": 0.5}))
    assert con.net_yield_ann_pct < sin.net_yield_ann_pct


def test_funding_negativo_se_descuenta():
    # Mezcla con periodos negativos: PF finito y % negativos > 0.
    fr = _funding(([0.0002] * 700) + ([-0.0001] * 395))
    stats = simulate_carry(fr, QM)
    assert stats.pct_negative_periods == pytest.approx(395 / 1095 * 100, rel=1e-3)
    assert math.isfinite(stats.profit_factor)
    assert stats.worst_period_pct == pytest.approx(-0.0001 * 100, rel=1e-9)


def test_costos_entrada_salida_reducen_el_neto_vs_bruto():
    # Con comisión > 0, el yield neto sobre notional (corrigiendo capital) queda
    # por debajo del bruto por los 4 lados taker one-time.
    fr = _funding([0.0001] * 1095)
    qm = QM.model_copy(update={"carry_capital_multiplier": 1.0})  # aísla la fricción de capital
    stats = simulate_carry(fr, qm)
    assert stats.net_yield_ann_pct < stats.gross_yield_ann_pct


def test_walk_forward_signo_consistente():
    # Funding siempre positivo → 4/4 folds mismo signo → puede coronar.
    fr = _funding([0.0001] * 1200)
    stats = simulate_carry(fr, QM)
    assert stats.folds_same_sign == 4


def test_regla_de_oro_falla_si_un_fold_invierte():
    # Si el último tramo se vuelve fuertemente negativo, NO debe pasar (4/4 roto).
    fr = _funding(([0.0001] * 900) + ([-0.0005] * 300))
    stats = simulate_carry(fr, QM)
    assert stats.folds_same_sign < 4
    assert stats.passes_golden is False


def test_serie_vacia_lanza():
    with pytest.raises(ValueError):
        simulate_carry(_funding([]), QM)


def test_period_returns_sobre_capital():
    fr = _funding([0.0001, 0.0002])
    r = carry_period_returns(fr, QM.model_copy(update={
        "carry_capital_multiplier": 2.0, "carry_maintenance_bps_per_period": 0.0}))
    assert r[0] == pytest.approx(0.0001 / 2.0)
    assert r[1] == pytest.approx(0.0002 / 2.0)


def test_matriz_quant_sin_stubs_pendientes():
    # Las 5 familias están implementadas: B (pairs), C (vwap), D (squeeze), E (carry).
    # run_family_squeeze ya NO es un stub que lanza NotImplementedError; delega en
    # backtest/squeeze.py y devuelve un SqueezeStats sobre un OHLCV de 1h.
    from backtest.squeeze import SqueezeStats
    n = 600
    rng = np.random.default_rng(0)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.003, n)))
    df = pd.DataFrame({
        "high": close * 1.001, "low": close * 0.999, "close": close,
        "volume": rng.uniform(1, 10, n),
    })
    result = run_family_squeeze(df, QM, symbol="X")
    assert isinstance(result, SqueezeStats)
