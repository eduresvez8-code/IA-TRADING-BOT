"""Tests de los deciders genéricos de backtest/deciders.py.

Verifican COMPORTAMIENTO (invariantes de entrada/salida/supresión), no que el
código "no truene". Valores a mano, sin I/O.
"""

import numpy as np

from backtest.deciders import (
    make_dow_decider,
    make_macross_decider,
    make_rsi_reversion_decider,
    make_tsmom_decider,
    moving_average,
)


# ---------- moving_average ----------

def test_sma_valores_a_mano():
    closes = np.array([1.0, 2.0, 3.0, 4.0])
    out = moving_average(closes, 2, "sma")
    assert np.isnan(out[0])
    assert out[1] == 1.5 and out[2] == 2.5 and out[3] == 3.5


def test_ema_reacciona_mas_rapido_que_sma():
    # Tras un salto de precio, la EMA (pondera lo reciente) queda más cerca del
    # nuevo nivel que la SMA del mismo periodo.
    closes = np.array([100.0] * 10 + [110.0] * 3)
    ema = moving_average(closes, 5, "ema")
    sma = moving_average(closes, 5, "sma")
    assert ema[-1] > sma[-1]


# ---------- TSMOM ----------

def _tsmom(closes, *, lookback=2, allow_short=True):
    closes = np.asarray(closes, dtype=float)
    atrs = np.full(len(closes), 2.0)
    return make_tsmom_decider(closes, atrs, lookback, 2.0, allow_short=allow_short)


def test_tsmom_entra_long_con_momentum_positivo_y_stop_atr():
    d = _tsmom([100.0, 100.0, 110.0])
    dec = d(2, None, 0.0, 0)
    assert dec == ("enter", "LONG", 1.0, 110.0 - 2.0 * 2.0, None)


def test_tsmom_sale_cuando_el_signo_se_invierte():
    d = _tsmom([100.0, 100.0, 110.0, 90.0])
    d(2, None, 0.0, 0)
    assert d(3, "LONG", 0.0, 0) == ("exit",)


def test_tsmom_short_bloqueado_sin_allow_short():
    # Momentum negativo → señal SHORT, pero en cash-account queda plano.
    d = _tsmom([100.0, 100.0, 90.0], allow_short=False)
    assert d(2, None, 0.0, 0) is None


def test_tsmom_no_reentra_tras_stop_hasta_reiniciar_el_signo():
    # Vela 2: entra LONG. Vela 3: la posición desapareció SIN exit nuestro → fue
    # el stop. Con momentum aún positivo NO reentra (supresión); cuando el
    # momentum cruza a ≤0 y vuelve a positivo, la supresión se levanta.
    closes = [100.0, 100.0, 110.0, 112.0, 111.0, 90.0, 80.0, 120.0, 130.0]
    d = _tsmom(closes, lookback=2)
    assert d(2, None, 0.0, 0)[0] == "enter"      # entra LONG
    d(3, "LONG", 0.0, 0)                          # sostiene
    assert d(4, None, 0.0, 0) is None             # stop saltó → suprimido (mom>0)
    d(5, None, 0.0, 0)                            # mom 90/110-1 <0 → levanta supresión
    dec = d(7, None, 0.0, 0)                      # mom 120/90-1 >0 → puede reentrar
    assert dec is not None and dec[1] == "LONG"


# ---------- MA-cross ----------

def test_macross_entra_y_flipa_en_el_cruce():
    closes = np.array([100.0] * 4)
    atrs = np.full(4, 1.0)
    fast = np.array([np.nan, 11.0, 11.0, 9.0])
    slow = np.array([np.nan, 10.0, 10.0, 10.0])
    d = make_macross_decider(closes, atrs, fast, slow, atr_mult=2.0, allow_short=True)
    dec = d(1, None, 0.0, 0)
    assert dec == ("enter", "LONG", 1.0, 100.0 - 2.0 * 1.0, None)
    assert d(2, "LONG", 0.0, 0) is None           # sigue por encima → sostiene
    assert d(3, "LONG", 0.0, 0) == ("exit",)      # cruce en contra → sale


def test_macross_short_bloqueado_sin_allow_short():
    closes = np.array([100.0, 100.0])
    atrs = np.full(2, 1.0)
    fast = np.array([9.0, 9.0])
    slow = np.array([10.0, 10.0])
    d = make_macross_decider(closes, atrs, fast, slow, atr_mult=2.0, allow_short=False)
    assert d(1, None, 0.0, 0) is None


# ---------- Reversión RSI ----------

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


# ---------- Día-de-semana (calendario bursátil real) ----------

def test_dow_decider_entra_cuando_la_proxima_vela_es_el_dia_objetivo():
    closes = np.array([100.0, 101.0, 102.0])
    atrs = np.full(3, 4.0)
    # next_weekday[0]=0 → la próxima vela real es lunes → señal en la vela 0.
    next_wd = np.array([0, 1, 2])
    d = make_dow_decider(closes, atrs, next_wd, entry_weekday=0, hold_days=1,
                         atr_mult=2.0)
    dec = d(0, None, 0.0, 0)
    assert dec == ("enter", "LONG", 1.0, 100.0 - 2.0 * 4.0, None)


def test_dow_decider_sale_tras_hold_days_velas_de_trading():
    closes = np.array([100.0] * 4)
    atrs = np.full(4, 4.0)
    next_wd = np.array([0, 1, 2, 3])
    d = make_dow_decider(closes, atrs, next_wd, entry_weekday=0, hold_days=2,
                         atr_mult=2.0)
    d(0, None, 0.0, 0)                    # señala la entrada (entry_bar=0)
    assert d(1, "LONG", 0.0, 0) is None   # 1 vela sostenida (< hold)
    assert d(2, "LONG", 0.0, 0) == ("exit",)  # 2 velas → sale


def test_dow_decider_festivo_desplaza_la_senal_sin_romperla():
    # Un viernes cuya próxima vela real es MARTES (lunes festivo): con
    # entry_weekday=0 (lunes) esa vela NO señala — el patrón sigue el
    # calendario real, no un offset fijo de 7 días.
    closes = np.array([100.0])
    atrs = np.array([4.0])
    next_wd = np.array([1])               # próxima vela real: martes
    d = make_dow_decider(closes, atrs, next_wd, entry_weekday=0, hold_days=1,
                         atr_mult=2.0)
    assert d(0, None, 0.0, 0) is None
