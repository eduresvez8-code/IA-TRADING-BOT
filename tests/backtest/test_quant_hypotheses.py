"""Tests de los deciders de backtest/quant_hypotheses.py (variante "dejar correr").

Cubren el cambio 2026-07-08: cortar pérdidas rápido (stop ATR explícito),
dejar correr las ganancias (tp=None, sin techo), salida solo por reversión
de la condición de entrada. Valores a mano, sin I/O.
"""

import numpy as np
import pandas as pd

from backtest.quant_hypotheses import (
    make_donchian_decider,
    make_dow_decider,
    make_hour_seasonality_decider,
    make_rsi_reversion_decider,
)


def _ts_hour(h: int) -> pd.Timestamp:
    return pd.Timestamp(f"2024-01-01 {h:02d}:00:00", tz="UTC")


# ---------- Donchian: tp opcional y salida por tiempo opcional ----------

def _donchian(closes, *, rr, max_hold, funding=None):
    closes = np.asarray(closes, dtype=float)
    atrs = np.full(len(closes), 2.0)
    funding = (np.full(len(closes), 0.0002) if funding is None
               else np.asarray(funding, dtype=float))
    return make_donchian_decider(
        closes, atrs, funding, entry_period=3, exit_ema_period=2,
        funding_min_frac=0.0001, funding_max_frac=0.0005,
        atr_mult=2.0, take_profit_rr=rr, max_hold_bars=max_hold)


def test_donchian_con_rr_pone_take_profit_fijo():
    # Comportamiento histórico intacto: rr=3 → tp = close + 3·(close − stop).
    closes = [100.0, 100.0, 100.0, 110.0]
    d = _donchian(closes, rr=3.0, max_hold=30)
    dec = d(3, None, 0.0, 0)
    assert dec is not None and dec[0] == "enter" and dec[1] == "LONG"
    stop, tp = dec[3], dec[4]
    assert stop == 110.0 - 2.0 * 2.0
    assert tp == 110.0 + 3.0 * (110.0 - stop)


def test_donchian_sin_rr_no_pone_techo():
    # Variante "dejar correr": rr=None → tp None (el motor solo vigila el stop).
    closes = [100.0, 100.0, 100.0, 110.0]
    d = _donchian(closes, rr=None, max_hold=None)
    dec = d(3, None, 0.0, 0)
    assert dec is not None and dec[4] is None
    assert dec[3] == 110.0 - 2.0 * 2.0          # el stop ATR sigue siendo el freno


def test_donchian_sin_max_hold_no_sale_por_tiempo():
    # Con max_hold=None la única salida por señal es el cruce de la EMA rápida.
    # Cierres crecientes → la EMA queda debajo → un LONG jamás sale por tiempo.
    closes = [100.0, 100.0, 100.0, 110.0, 111.0, 112.0, 113.0, 114.0]
    d = _donchian(closes, rr=None, max_hold=None)
    d(3, None, 0.0, 0)                           # decide la entrada
    for i in range(4, 8):                        # sostiene: nunca ("exit",)
        assert d(i, "LONG", 0.0, 0) is None


def test_donchian_max_hold_sigue_funcionando_si_se_pide():
    # Regresión: la salida por tiempo se conserva cuando max_hold está configurado.
    closes = [100.0, 100.0, 100.0, 110.0, 111.0, 112.0, 113.0, 114.0]
    d = _donchian(closes, rr=None, max_hold=2)
    d(3, None, 0.0, 0)
    d(4, "LONG", 0.0, 0)                         # entry_bar=4 (transición plano→pos)
    assert d(5, "LONG", 0.0, 0) is None          # 1 vela sostenida
    assert d(6, "LONG", 0.0, 0) == ("exit",)     # 2 velas → backstop de tiempo


# ---------- Estacionalidad horaria: stop explícito + tp None ----------

def test_hour_decider_entra_con_stop_atr_y_sin_techo():
    closes = np.array([100.0, 100.0])
    atrs = np.array([5.0, 5.0])
    d = make_hour_seasonality_decider(closes, atrs, entry_open_hour=22,
                                      hold_hours=2, atr_mult=2.0)
    # La señal se emite en la hora 21 (para entrar en la APERTURA de las 22).
    dec = d(0, None, 0.0, _ts_hour(21))
    assert dec == ("enter", "LONG", 1.0, 100.0 - 2.0 * 5.0, None)
    # Fuera de la hora de señal: nada.
    assert d(1, None, 0.0, _ts_hour(10)) is None


def test_hour_decider_sale_cuando_la_ventana_paso():
    closes = np.array([100.0] * 3)
    atrs = np.array([5.0] * 3)
    d = make_hour_seasonality_decider(closes, atrs, entry_open_hour=22,
                                      hold_hours=2, atr_mult=2.0)
    # signal_out = (22-1+2) % 24 = 23 → a esa hora se señala la salida.
    assert d(1, "LONG", 0.0, _ts_hour(22)) is None
    assert d(2, "LONG", 0.0, _ts_hour(23)) == ("exit",)


def test_hour_decider_sin_atr_valido_no_entra():
    closes = np.array([100.0])
    atrs = np.array([np.nan])
    d = make_hour_seasonality_decider(closes, atrs, entry_open_hour=22,
                                      hold_hours=2, atr_mult=2.0)
    assert d(0, None, 0.0, _ts_hour(21)) is None


# ---------- Día-de-la-semana: stop explícito + tp None ----------

def test_dow_decider_entra_con_stop_y_sale_al_pasar_el_dia():
    closes = np.array([100.0] * 3)
    atrs = np.array([4.0] * 3)
    d = make_dow_decider(closes, atrs, entry_weekday=0, hold_days=1, atr_mult=2.0)
    sunday = pd.Timestamp("2023-12-31", tz="UTC")    # dayofweek=6 → señal del lunes
    monday = pd.Timestamp("2024-01-01", tz="UTC")    # dayofweek=0 → señal de salida
    dec = d(0, None, 0.0, sunday)
    assert dec == ("enter", "LONG", 1.0, 100.0 - 2.0 * 4.0, None)
    assert d(1, "LONG", 0.0, monday) == ("exit",)


# ---------- Reversión RSI: stop explícito + salida por sobrecompra ----------

def test_rsi_decider_entra_con_stop_y_sale_en_sobrecompra():
    closes = np.array([100.0, 100.0, 100.0])
    rsi_vals = np.array([25.0, 50.0, 75.0])
    trend = np.array([90.0, 90.0, 90.0])             # close > SMA → tendencia de fondo
    atrs = np.array([3.0, 3.0, 3.0])
    d = make_rsi_reversion_decider(closes, rsi_vals, trend, atrs,
                                   oversold=30.0, overbought=70.0, atr_mult=2.0)
    dec = d(0, None, 0.0, 0)
    assert dec == ("enter", "LONG", 1.0, 100.0 - 2.0 * 3.0, None)
    # RSI intermedio: dejar correr (ni entra ni sale).
    assert d(1, "LONG", 0.0, 0) is None
    # RSI cruza a sobrecompra: la condición de entrada se revirtió → salir.
    assert d(2, "LONG", 0.0, 0) == ("exit",)


def test_rsi_decider_no_compra_contra_la_tendencia():
    closes = np.array([80.0])
    rsi_vals = np.array([25.0])
    trend = np.array([90.0])                          # close < SMA → cuchillo cayendo
    atrs = np.array([3.0])
    d = make_rsi_reversion_decider(closes, rsi_vals, trend, atrs,
                                   oversold=30.0, overbought=70.0, atr_mult=2.0)
    assert d(0, None, 0.0, 0) is None
