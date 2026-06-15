"""Tests del edge test: funciones puras con valores de referencia a mano.

Como en test_metrics.py, las funciones reciben Series/arrays y devuelven números,
así que se verifican con entradas pequeñas de resultado conocido. Construimos
señales perfectamente (anti)correlacionadas con el retorno futuro para fijar el
comportamiento esperado de cada métrica.
"""

import math

import numpy as np
import pandas as pd
import pytest

from src.core.config import load_settings
from backtest.edge import (
    analyze_edge,
    base_up_rate,
    corr_tstat,
    directional_hit_rate,
    forward_returns,
    horizon_stats,
    pearson_ic,
    quantile_forward_means,
    spearman_ic,
)


class TestForwardReturns:
    def test_horizonte_1(self):
        close = pd.Series([100.0, 110.0, 121.0])
        fr = forward_returns(close, 1)
        assert fr.iloc[0] == pytest.approx(0.10)
        assert fr.iloc[1] == pytest.approx(0.10)
        assert math.isnan(fr.iloc[2])  # la última vela no tiene futuro

    def test_horizonte_2(self):
        close = pd.Series([100.0, 110.0, 121.0, 133.1])
        fr = forward_returns(close, 2)
        assert fr.iloc[0] == pytest.approx(0.21)        # 121/100 - 1
        assert math.isnan(fr.iloc[2]) and math.isnan(fr.iloc[3])


class TestIC:
    def test_spearman_perfecto_positivo(self):
        s = pd.Series([1.0, 2, 3, 4, 5])
        f = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5])
        assert spearman_ic(s, f) == pytest.approx(1.0)

    def test_spearman_perfecto_negativo(self):
        s = pd.Series([1.0, 2, 3, 4, 5])
        f = pd.Series([0.5, 0.4, 0.3, 0.2, 0.1])
        assert spearman_ic(s, f) == pytest.approx(-1.0)

    def test_spearman_robusto_a_no_linealidad(self):
        # monótono pero no lineal: Spearman=1, Pearson<1.
        s = pd.Series([1.0, 2, 3, 4, 5])
        f = pd.Series([1.0, 4, 9, 16, 25])
        assert spearman_ic(s, f) == pytest.approx(1.0)
        assert pearson_ic(s, f) < 1.0

    def test_ignora_pares_con_nan(self):
        # solo (1,.1),(2,.2),(3,.3) válidos → monótono → 1.0
        s = pd.Series([1.0, 2, 3, np.nan, 5])
        f = pd.Series([0.1, 0.2, 0.3, 0.4, np.nan])
        assert spearman_ic(s, f) == pytest.approx(1.0)

    def test_muestra_minima_devuelve_cero(self):
        assert spearman_ic(pd.Series([1.0, 2]), pd.Series([0.1, 0.2])) == 0.0


class TestCorrTstat:
    def test_ic_cero(self):
        assert corr_tstat(0.0, 100) == 0.0

    def test_formula(self):
        # r=0.2, n_eff=102 → t = 0.2*sqrt(100/0.96) ≈ 2.0412
        assert corr_tstat(0.2, 102) == pytest.approx(2.0412, abs=1e-3)

    def test_muestra_insuficiente(self):
        assert corr_tstat(0.5, 2) == 0.0

    def test_n_eff_descuenta_solape(self):
        # mismo IC, más solape (horizonte mayor) ⇒ menos t (menos confianza).
        assert abs(corr_tstat(0.1, 1000)) > abs(corr_tstat(0.1, 100))


class TestDirectionalHitRate:
    def test_acierto_perfecto(self):
        s = pd.Series([0.8, -0.8, 0.9])
        f = pd.Series([0.01, -0.02, 0.03])
        hr, n = directional_hit_rate(s, f, threshold=0.5)
        assert n == 3 and hr == pytest.approx(1.0)

    def test_umbral_filtra_senales_debiles(self):
        s = pd.Series([0.1, 0.2, 0.8])
        f = pd.Series([0.01, -0.01, 0.02])
        hr, n = directional_hit_rate(s, f, threshold=0.5)
        assert n == 1 and hr == pytest.approx(1.0)  # solo |0.8|≥0.5, acierta

    def test_sin_senales_fuertes(self):
        s = pd.Series([0.1, 0.2, 0.3])
        f = pd.Series([0.01, 0.02, 0.03])
        assert directional_hit_rate(s, f, threshold=0.5) == (0.0, 0)


class TestBaseUpRate:
    def test_mitad_arriba(self):
        f = pd.Series([0.1, -0.1, 0.2, -0.2])
        assert base_up_rate(f) == pytest.approx(0.5)


class TestQuantileMeans:
    def test_monotonico_creciente(self):
        s = pd.Series(np.arange(100, dtype=float))
        f = pd.Series(np.arange(100, dtype=float) / 1000.0)
        means = quantile_forward_means(s, f, 5)
        assert len(means) == 5
        assert all(means[i] < means[i + 1] for i in range(4))

    def test_pocos_datos_devuelve_vacio(self):
        assert quantile_forward_means(pd.Series([1.0, 2.0]), pd.Series([0.1, 0.2]), 5) == []


class TestHorizonStats:
    def test_senal_perfecta(self):
        s = pd.Series(np.linspace(-1, 1, 200))
        f = s * 0.01  # retorno futuro perfectamente alineado con la señal
        regime = pd.Series([True] * 200)
        st = horizon_stats(s, f, regime, horizon=1, threshold=0.5, n_quantiles=5)
        assert st.spearman_ic == pytest.approx(1.0)
        assert st.hit_rate == pytest.approx(1.0)
        assert st.quantile_spread > 0
        assert st.n_eff == 200  # horizonte 1 ⇒ sin solape

    def test_n_eff_cae_con_horizonte(self):
        s = pd.Series(np.linspace(-1, 1, 240))
        f = s * 0.01
        regime = pd.Series([True] * 240)
        st = horizon_stats(s, f, regime, horizon=24, threshold=0.5, n_quantiles=5)
        assert st.n_eff == 10  # 240 // 24


class TestAnalyzeEdge:
    def _df(self, n=400, seed=0):
        closes = 100 + np.cumsum(np.random.default_rng(seed).normal(0, 1, n))
        return pd.DataFrame({
            "open_time": pd.date_range("2025-01-01", periods=n, freq="1h", tz="UTC"),
            "open": closes, "high": closes + 1, "low": closes - 1,
            "close": closes, "volume": np.full(n, 1000.0),
        })

    def test_un_stat_por_horizonte(self):
        cfg = load_settings()
        stats = analyze_edge(self._df(), cfg)
        assert len(stats) == len(cfg.edge.forward_horizons)
        assert [s.horizon for s in stats] == cfg.edge.forward_horizons

    def test_ic_en_rango_valido(self):
        stats = analyze_edge(self._df(), load_settings())
        for s in stats:
            assert -1.0 <= s.spearman_ic <= 1.0
            assert -1.0 <= s.pearson_ic <= 1.0
            assert 0.0 <= s.hit_rate <= 1.0
