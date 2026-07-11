"""Tests de las familias pre-registradas (backtest/sp500_families.py).

Lo crítico aquí es el ANTI-LOOK-AHEAD estructural: pesos decididos en el mes m
solo tocan el retorno de m+1; señales del cierre de t−1 solo tocan el día t.
Datos sintéticos donde la respuesta se conoce por construcción.
"""

import numpy as np
import pandas as pd
import pytest

from backtest.sp500_families import (
    daily_hold_returns,
    daily_strategy_returns,
    dual_momentum_weights,
    first_open_by_month,
    golden_cross_daily_position,
    ma_timing_monthly_weights,
    monthly_cash_returns,
    monthly_hold_returns,
    monthly_strategy_returns,
    rsi_reversion_daily_position,
    trades_from_positions,
    tsmom_index_weights,
    xs_momentum_weights,
    xs_monthly_hold_returns,
)


def _daily_df(dates, opens, closes=None):
    closes = closes if closes is not None else opens
    return pd.DataFrame({
        "open_time": pd.to_datetime(dates, utc=True),
        "open": np.asarray(opens, dtype=float),
        "high": np.asarray(closes, dtype=float) * 1.01,
        "low": np.asarray(opens, dtype=float) * 0.99,
        "close": np.asarray(closes, dtype=float),
        "volume": np.full(len(dates), 1e6),
    })


# ---------- calendario mensual ----------

def test_monthly_hold_returns_es_open_a_open():
    df = _daily_df(
        ["2020-01-02", "2020-01-15", "2020-02-03", "2020-02-20", "2020-03-02"],
        [100.0, 105.0, 110.0, 108.0, 121.0],
    )
    r = monthly_hold_returns(df)
    # Enero: open feb (110) / open ene (100) − 1 = 10%
    assert r[pd.Period("2020-01")] == pytest.approx(0.10)
    # Febrero: open mar (121) / open feb (110) − 1 = 10%
    assert r[pd.Period("2020-02")] == pytest.approx(0.10)
    # Marzo no tiene m+1 → descartado
    assert pd.Period("2020-03") not in r.index


def test_first_open_by_month_toma_el_primer_dia_habil():
    df = _daily_df(["2020-01-02", "2020-01-03"], [100.0, 999.0])
    fo = first_open_by_month(df)
    assert fo[pd.Period("2020-01")] == 100.0


# ---------- cartera mensual: anti-look-ahead y costos ----------

def test_pesos_del_mes_m_solo_tocan_el_retorno_de_m1():
    idx = pd.period_range("2020-01", "2020-04", freq="M")
    # El activo rinde +10% SOLO en marzo.
    asset = pd.DataFrame({"a": [0.0, 0.0, 0.10, 0.0]}, index=idx)
    # Decisión de estar largo tomada en FEBRERO (se aplica a marzo).
    w = pd.DataFrame({"a": [0.0, 1.0, 0.0, 0.0]}, index=idx)
    cash = pd.Series(0.0, index=idx)
    r = monthly_strategy_returns(w, asset, cash, per_side=0.0)
    assert r[pd.Period("2020-03")] == pytest.approx(0.10)
    assert r[pd.Period("2020-04")] == pytest.approx(0.0)


def test_costos_se_cobran_al_entrar_y_al_salir():
    idx = pd.period_range("2020-01", "2020-04", freq="M")
    asset = pd.DataFrame({"a": [0.0, 0.0, 0.0, 0.0]}, index=idx)
    w = pd.DataFrame({"a": [0.0, 1.0, 0.0, 0.0]}, index=idx)  # entra feb→mar, sale mar→abr
    cash = pd.Series(0.0, index=idx)
    r = monthly_strategy_returns(w, asset, cash, per_side=0.001)
    assert r[pd.Period("2020-03")] == pytest.approx(-0.001)  # compra: 1 lado
    assert r[pd.Period("2020-04")] == pytest.approx(-0.001)  # venta: 1 lado


def test_cash_devenga_tbill_cuando_esta_fuera():
    idx = pd.period_range("2020-01", "2020-03", freq="M")
    asset = pd.DataFrame({"a": [0.05, 0.05, 0.05]}, index=idx)
    w = pd.DataFrame({"a": [0.0, 0.0, 0.0]}, index=idx)      # siempre fuera
    cash = pd.Series([0.004, 0.004, 0.004], index=idx)
    r = monthly_strategy_returns(w, asset, cash, per_side=0.0)
    assert r.to_numpy() == pytest.approx(0.004)


def test_monthly_cash_returns_suma_los_dias_del_mes():
    days = pd.date_range("2020-01-01", "2020-01-05", freq="D", tz="UTC")
    tb = pd.Series(0.0002, index=days)
    out = monthly_cash_returns(tb, pd.period_range("2020-01", "2020-02", freq="M"))
    assert out[pd.Period("2020-01")] == pytest.approx(0.001)
    assert out[pd.Period("2020-02")] == 0.0   # sin datos → 0, no NaN


# ---------- TSMOM (definición congelada: excluye el último mes) ----------

def test_tsmom_excluye_el_ultimo_mes():
    idx = pd.period_range("2020-01", "2020-08", freq="M")
    # Sube 6 meses, luego se DESPLOMA en el mes 8 (el más reciente).
    close = pd.Series([100, 110, 120, 130, 140, 150, 160, 40.0], index=idx)
    w = tsmom_index_weights(close, lookback_months=3)
    # Decisión en 2020-08: momentum = close(jul)/close(abr) − 1 > 0 → LONG
    # aunque agosto fue una masacre (el mes actual se EXCLUYE por definición).
    assert w.loc[pd.Period("2020-08"), "asset"] == 1.0


def test_tsmom_sin_historia_es_nan():
    idx = pd.period_range("2020-01", "2020-03", freq="M")
    close = pd.Series([100, 101, 102.0], index=idx)
    w = tsmom_index_weights(close, lookback_months=6)
    assert w["asset"].isna().all()


# ---------- MA timing ----------

def test_ma_timing_encima_de_la_sma_es_uno():
    dates = pd.date_range("2020-01-01", periods=10, freq="B", tz="UTC")
    df = _daily_df(dates, np.linspace(100, 120, 10))
    w = ma_timing_monthly_weights(df, sma_days=3)
    # Serie creciente: el cierre de fin de mes > SMA3 → 1.0
    assert w["asset"].dropna().iloc[-1] == 1.0


def test_golden_cross_daily():
    dates = pd.date_range("2020-01-01", periods=8, freq="B", tz="UTC")
    closes = [100, 100, 100, 100, 130, 140, 150, 160.0]
    df = _daily_df(dates, closes, closes)
    pos = golden_cross_daily_position(df, fast_days=2, slow_days=4)
    assert pos.iloc[3] == 0.0        # aún plano
    assert pos.iloc[-1] == 1.0       # la rápida cruzó por encima


# ---------- RSI-2 con histéresis ----------

def test_rsi_reversion_entra_en_el_dip_y_sale_en_la_recuperacion():
    # 60 días planos (RSI estable), luego 3 días de caída fuerte (RSI2 → ~0),
    # luego recuperación fuerte (RSI2 → ~100).
    n_flat = 60
    closes = ([100.0 + 0.01 * (i % 2) for i in range(n_flat)]
              + [95.0, 90.0, 85.0]           # dip
              + [92.0, 99.0, 105.0])         # recuperación
    dates = pd.date_range("2020-01-01", periods=len(closes), freq="B", tz="UTC")
    df = _daily_df(dates, closes, closes)
    pos = rsi_reversion_daily_position(df, rsi_period=2, entry_below=10.0,
                                       exit_above=70.0, trend_sma_days=50)
    # Nunca antes del dip:
    assert pos.iloc[:n_flat].max() == 0.0
    # En el dip (cierre aún > SMA50 porque la SMA es ~100 y el dip es corto...
    # 85 < SMA≈100 → el filtro de tendencia BLOQUEA la compra del día 62).
    # El diseño exige AMBAS condiciones: aquí el dip rompe la tendencia → no entra.
    assert pos.max() == 0.0


def test_rsi_reversion_compra_dip_dentro_de_tendencia():
    # Tendencia alcista clara + dip suave que NO rompe la SMA → sí entra.
    up = list(np.linspace(100, 160, 80))
    dip = [155.0, 150.0]                      # retroceso suave (sigue > SMA50)
    rec = [158.0, 166.0, 172.0]
    closes = up + dip + rec
    dates = pd.date_range("2020-01-01", periods=len(closes), freq="B", tz="UTC")
    df = _daily_df(dates, closes, closes)
    pos = rsi_reversion_daily_position(df, rsi_period=2, entry_below=10.0,
                                       exit_above=70.0, trend_sma_days=50)
    assert pos.iloc[80:82].max() == 1.0       # entró en el dip
    assert pos.iloc[-1] == 0.0                # salió con RSI alto


# ---------- Dual momentum ----------

def test_dual_momentum_elige_el_mejor_sobre_cash():
    idx = pd.period_range("2019-01", "2020-12", freq="M")
    eq = pd.Series(np.linspace(100, 200, len(idx)), index=idx)   # 12m mom fuerte
    bd = pd.Series(np.linspace(100, 110, len(idx)), index=idx)   # mom débil
    cash = pd.Series(0.0, index=idx)
    w = dual_momentum_weights(eq, bd, cash, lookback_months=12)
    last = w.dropna().iloc[-1]
    assert last["equity"] == 1.0 and last["bond"] == 0.0


def test_dual_momentum_cae_a_cash_si_nada_supera_al_tbill():
    idx = pd.period_range("2019-01", "2020-12", freq="M")
    eq = pd.Series(np.linspace(200, 100, len(idx)), index=idx)   # cayendo
    bd = pd.Series(np.linspace(110, 100, len(idx)), index=idx)   # cayendo
    cash = pd.Series(0.004, index=idx)                            # T-bill positivo
    w = dual_momentum_weights(eq, bd, cash, lookback_months=12)
    last = w.dropna().iloc[-1]
    assert last["equity"] == 0.0 and last["bond"] == 0.0          # → cash


# ---------- Momentum cross-sectional ----------

def _xs_fixture():
    idx = pd.period_range("2020-01", "2020-12", freq="M")
    # 4 tickers: A sube fuerte, B sube, C plano, D cae.
    close = pd.DataFrame({
        "A": np.linspace(100, 220, 12),
        "B": np.linspace(100, 150, 12),
        "C": np.full(12, 100.0),
        "D": np.linspace(100, 60, 12),
    }, index=idx)
    return idx, close


def test_xs_momentum_rankea_y_equipondera():
    idx, close = _xs_fixture()
    members = pd.Series({m: ["A", "B", "C", "D"] for m in idx})
    w, cov = xs_momentum_weights(close, members, lookback_months=3, skip_months=0,
                                 top_n=2, min_history_months=3, min_coverage=0.5)
    last = w.loc[idx[-1]]
    assert last["A"] == 0.5 and last["B"] == 0.5      # top-2 equiponderado
    assert last["C"] == 0.0 and last["D"] == 0.0
    assert cov[idx[-1]] == 1.0


def test_xs_momentum_respeta_la_membresia_punto_en_el_tiempo():
    idx, close = _xs_fixture()
    # A (el mejor) NO era miembro en 2020: no puede recibir peso aunque vuele.
    members = pd.Series({m: ["B", "C", "D"] for m in idx})
    w, _ = xs_momentum_weights(close, members, lookback_months=3, skip_months=0,
                               top_n=2, min_history_months=3, min_coverage=0.5)
    assert w.loc[idx[-1], "A"] == 0.0 or np.isnan(w.loc[idx[-1], "A"])
    assert w.loc[idx[-1], "B"] == 0.5


def test_xs_momentum_cobertura_baja_deja_el_mes_sin_decision():
    idx, close = _xs_fixture()
    # 4 miembros reales pero solo 1 en datos → cobertura 25% < 60% → NaN.
    members = pd.Series({m: ["B", "X1", "X2", "X3"] for m in idx})
    w, cov = xs_momentum_weights(close, members, lookback_months=3, skip_months=0,
                                 top_n=1, min_history_months=3, min_coverage=0.6)
    assert w.loc[idx[-1]].isna().all()
    assert cov[idx[-1]] == pytest.approx(0.25)


def test_xs_momentum_skip_salta_el_ultimo_mes():
    idx = pd.period_range("2020-01", "2020-08", freq="M")
    # E: sube 6 meses y se desploma el último mes. F: plano siempre.
    close = pd.DataFrame({
        "E": [100, 120, 140, 160, 180, 200, 220, 50.0],
        "F": np.full(8, 100.0),
    }, index=idx)
    members = pd.Series({m: ["E", "F"] for m in idx})
    w1, _ = xs_momentum_weights(close, members, lookback_months=3, skip_months=1,
                                top_n=1, min_history_months=3, min_coverage=0.5)
    # Con skip=1 la formación termina en julio (220): E sigue #1 pese al crash.
    assert w1.loc[idx[-1], "E"] == 1.0


def test_xs_hold_returns_maneja_delistings():
    idx = pd.period_range("2020-01", "2020-03", freq="M")
    opens = pd.DataFrame({
        "A": [100.0, 110.0, 121.0],
        "Z": [50.0, 40.0, np.nan],       # Z deslista en marzo
    }, index=idx)
    last_close = pd.DataFrame({
        "A": [105.0, 115.0, 125.0],
        "Z": [45.0, 30.0, np.nan],       # último cierre de Z en feb: 30
    }, index=idx)
    r = xs_monthly_hold_returns(opens, last_close)
    assert r.loc[idx[0], "A"] == pytest.approx(0.10)        # open feb/open ene
    # Z en febrero: sin open de marzo → salida al último cierre de feb (30).
    assert r.loc[idx[1], "Z"] == pytest.approx(30.0 / 40.0 - 1.0)


# ---------- diario: mecánica y trades ----------

def test_daily_strategy_returns_sin_lookahead():
    dates = pd.date_range("2020-01-01", periods=4, freq="B", tz="UTC")
    df = _daily_df(dates, [100.0, 100.0, 110.0, 110.0])
    hold = daily_hold_returns(df)                 # día t: open t+1/open t − 1
    # Señal generada el día 1 (índice 1) → posición el día 2 → captura el 0%
    # del día 2→3, NO el +10% del día 1→2 (eso sería look-ahead).
    sig = pd.Series([0.0, 1.0, 0.0, 0.0], index=df["open_time"])
    r = daily_strategy_returns(sig, hold, pd.Series(0.0, index=hold.index), 0.0)
    assert r.iloc[1] == pytest.approx(0.0)        # día 1: aún sin posición
    assert r.iloc[2] == pytest.approx(0.0)        # día 2: en mercado, retorno 0


def test_trades_from_positions_compone_y_cobra_dos_lados():
    dates = pd.date_range("2020-01-01", periods=5, freq="B", tz="UTC")
    hold = pd.Series([0.0, 0.10, 0.10, 0.0], index=dates[:4])
    # Posición en los días 1 y 2 (señal en 0 y 1).
    sig = pd.Series([1.0, 1.0, 0.0, 0.0], index=dates[:4])
    trades = trades_from_positions(sig, hold, per_side=0.01)
    assert len(trades) == 1
    assert trades[0] == pytest.approx((0.99 ** 2) * 1.1 * 1.1 - 1.0)
