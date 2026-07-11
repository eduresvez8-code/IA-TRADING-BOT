"""Tests de los diagnósticos anti-sobreajuste (valores a mano, sin I/O).

Estos son los guardianes del protocolo: si una de estas funciones mintiera,
todo el listón de validación quedaría podrido. Por eso se testean contra
construcciones donde la respuesta correcta se conoce por diseño.
"""

import numpy as np
import pandas as pd
import pytest

from backtest.diagnostics import (
    bootstrap_sharpe_ci,
    concentration_top_decile,
    evaluate_gate,
    halves_stability,
    sharpe,
)


# ---------- sharpe ----------

def test_sharpe_valor_a_mano():
    # mean=0.01, std=0.01 (poblacional) → 1.0 · √252 ≈ 15.87
    r = [0.0, 0.02, 0.0, 0.02]
    assert sharpe(r, 252) == pytest.approx(1.0 * np.sqrt(252))


def test_sharpe_sin_varianza_es_cero():
    assert sharpe([0.01, 0.01, 0.01], 252) == 0.0


def test_sharpe_vacio_es_cero():
    assert sharpe([], 252) == 0.0
    assert sharpe([0.01], 252) == 0.0


def test_sharpe_ignora_nans():
    r = [0.0, 0.02, np.nan, 0.0, 0.02]
    assert sharpe(r, 252) == pytest.approx(1.0 * np.sqrt(252))


# ---------- bootstrap ----------

def test_bootstrap_ci_positivo_con_senal_fuerte():
    # 200 retornos consistentemente positivos → el CI entero sobre cero.
    rng = np.random.default_rng(42)
    r = rng.normal(0.01, 0.005, 200)
    lo, hi = bootstrap_sharpe_ci(r, 252, iterations=2000, ci=0.90)
    assert lo > 0 and hi > lo


def test_bootstrap_ci_cruza_cero_con_ruido():
    # Ruido con media EXACTAMENTE cero (simétrico por construcción): el CI
    # debe incluir el cero. Con una normal muestreada la media muestral puede
    # alejarse de 0 por azar y el test parpadearía.
    rng = np.random.default_rng(7)
    half = rng.normal(0.01, 0.01, 100)
    r = np.concatenate([half, -half])
    lo, hi = bootstrap_sharpe_ci(r, 252, iterations=2000, ci=0.90)
    assert lo < 0 < hi


def test_bootstrap_es_reproducible():
    r = np.random.default_rng(1).normal(0.001, 0.01, 100)
    a = bootstrap_sharpe_ci(r, 252, iterations=500, ci=0.90, seed=3)
    b = bootstrap_sharpe_ci(r, 252, iterations=500, ci=0.90, seed=3)
    assert a == b


def test_bootstrap_detecta_dependencia_de_colas():
    # 99 trades neutros + 1 jackpot: el Sharpe "medio" es positivo pero el
    # bootstrap debe revelar que sin el jackpot no hay nada (lo <= 0).
    r = np.array([0.0001] * 99 + [0.50])
    lo, _ = bootstrap_sharpe_ci(r, 252, iterations=2000, ci=0.90)
    assert lo <= 0.05  # la cola inferior no sostiene un edge


# ---------- concentración ----------

def test_concentracion_uniforme_es_el_decil():
    # 100 trades idénticos → el top 10% aporta exactamente el 10%.
    assert concentration_top_decile([1.0] * 100) == pytest.approx(0.10)


def test_concentracion_jackpot_cerca_de_uno():
    # 99 trades de nada + 1 que es toda la ganancia → ~1.0 (cola de suerte).
    p = [0.001] * 99 + [10.0]
    assert concentration_top_decile(p) > 0.9


def test_concentracion_sin_ganancia_es_nan():
    assert np.isnan(concentration_top_decile([-1.0, -2.0]))
    assert np.isnan(concentration_top_decile([]))


# ---------- mitades ----------

def test_mitades_por_calendario_no_por_conteo():
    # 4 retornos: 3 en enero (buenos) y 1 en diciembre (malo). El corte por
    # CALENDARIO (jun 30) deja 3|1; por conteo dejaría 2|2. La segunda mitad
    # debe reflejar el diciembre malo.
    ts = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03", "2020-12-31"])
    r = [0.01, 0.012, 0.011, -0.05]
    h1, h2 = halves_stability(r, ts.values, 252)
    assert h1 > 0
    assert h2 == 0.0  # una sola observación en la 2ª mitad → sin varianza → 0


def test_mitades_estables_ambas_positivas():
    ts = pd.date_range("2020-01-01", periods=100, freq="D")
    rng = np.random.default_rng(5)
    r = rng.normal(0.01, 0.005, 100)
    h1, h2 = halves_stability(r, ts.values, 252)
    assert h1 > 0 and h2 > 0


# ---------- el gate completo ----------

def _gate(returns, ts, *, bh=0.5):
    return evaluate_gate(
        returns, ts, returns, 12,
        sharpe_min=0.5, iterations=1000, ci=0.90,
        concentration_max=0.60, sharpe_buyhold=bh,
    )


def test_gate_pasa_con_edge_limpio():
    ts = pd.date_range("2015-01-31", periods=120, freq="ME").values
    rng = np.random.default_rng(11)
    r = rng.normal(0.02, 0.02, 120)          # Sharpe mensual ~1 anualizado ~3.4
    g = _gate(r, ts, bh=0.5)
    assert g.passes_all


def test_gate_falla_por_buyhold_aunque_gane():
    # Estrategia decente pero con B&H mejor: criterio 5 la tumba.
    ts = pd.date_range("2015-01-31", periods=120, freq="ME").values
    rng = np.random.default_rng(11)
    r = rng.normal(0.02, 0.02, 120)
    g = _gate(r, ts, bh=99.0)
    assert g.passes_sharpe and not g.passes_vs_buyhold and not g.passes_all


def test_gate_falla_con_ruido():
    ts = pd.date_range("2015-01-31", periods=120, freq="ME").values
    rng = np.random.default_rng(13)
    r = rng.normal(0.0, 0.03, 120)
    g = _gate(r, ts)
    assert not g.passes_all


def test_gate_concentracion_nan_no_pasa():
    # Pérdida neta → concentración NaN → el criterio 3 no puede darse por pasado.
    ts = pd.date_range("2015-01-31", periods=24, freq="ME").values
    r = np.full(24, -0.01)
    g = _gate(r, ts)
    assert not g.passes_concentration
