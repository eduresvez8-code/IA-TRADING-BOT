"""Deciders genéricos por-activo para el BacktestEngine (herencia de la
investigación cripto 2026-06/07, ya depurados de todo lo perp-only).

Cada estrategia se expresa como un `decider(i, position_side, score, ts)` que el
`BacktestEngine` consume (decide al cierre de t, ejecuta en la apertura de t+1,
sin look-ahead). Los deciders IGNORAN el `score` EMA del motor: traen su propia
señal precalculada en arrays cerrados por closure.

Familias retenidas (todas agnósticas del activo — funcionan sobre cualquier
DataFrame OHLCV diario, hoy acciones/ETF del S&P 500):
    - TSMOM        — momentum de series de tiempo: el signo del retorno a N
                     velas marca el lado.
    - MA-cross     — cruce de medias móviles (golden/death cross generalizado).
    - RSI-reversión— comprar el dip (RSI<oversold) DENTRO de tendencia alcista.
    - Día-de-semana— anomalía de calendario, versión calendario BURSÁTIL real
                     (huecos de fin de semana/festivos, no ciclo fijo de 7 días).

Estructura de salida común ("cortar pérdidas rápido, dejar correr ganancias"):
stop ATR explícito anclado al cierre causal de la vela de decisión
(close ∓ mult·ATR) y tp=None — sin techo de ganancia. El motor dimensiona por
la distancia al stop (riesgo fijo % equity).

Se BORRARON (2026-07-11, pivote a S&P 500): Donchian+funding, funding extremo,
lead-lag BTC→alt, estacionalidad horaria 24/7 — todas dependían de conceptos de
futuros perpetuos cripto (funding rate, mercado 24/7) o murieron en la
investigación (ver docs/research/).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def _stop_level(side: str, ref_close: float, atr_val: float, mult: float) -> float | None:
    """Nivel de stop explícito: close ∓ mult·ATR. None si el ATR no es válido."""
    if atr_val is None or math.isnan(atr_val) or atr_val <= 0:
        return None
    return ref_close - mult * atr_val if side == "LONG" else ref_close + mult * atr_val


def moving_average(closes: np.ndarray, period: int, kind: str) -> np.ndarray:
    """Media móvil de los cierres: 'sma' (simple) o 'ema' (exponencial).

    La SMA pondera igual las N velas (más lenta, menos whipsaws); la EMA pondera
    más lo reciente (reacciona antes, más señales falsas). Ambas causales.
    """
    s = pd.Series(closes)
    if kind == "ema":
        return s.ewm(span=period, adjust=False).mean().to_numpy()
    return s.rolling(period).mean().to_numpy()


def make_tsmom_decider(closes: np.ndarray, atrs: np.ndarray, lookback: int,
                       atr_mult: float, *, allow_short: bool = True):
    """TSMOM: signo del retorno a `lookback` velas marca el lado.

    Siempre busca estar en mercado en la dirección del momentum; sale cuando el
    signo se invierte. El stop ATR es un freno de desastre: tras saltar, NO se
    reentra en la misma dirección hasta que el signo del momentum se reinicie
    (cruce de cero) — así un stop no provoca churn de reentradas a costo.

    `allow_short=False` (defecto natural en acciones cash, sin margen): las
    señales SHORT se ignoran y el bot queda plano — equivale a "long o cash".
    """
    mom = np.full(len(closes), np.nan)
    if lookback < len(closes):
        mom[lookback:] = closes[lookback:] / closes[:-lookback] - 1.0

    # Estado mínimo para distinguir "salida por señal" de "salida por stop/TP".
    st = {"prev_pos": None, "exit_emitted": False, "suppress": None}

    def decider(i, position_side, score, ts):
        # ¿La posición desapareció sin que NOSOTROS emitiéramos exit? → fue stop/TP.
        if (st["prev_pos"] is not None and position_side is None
                and not st["exit_emitted"]):
            st["suppress"] = st["prev_pos"]      # no reentrar esa dirección aún
        st["exit_emitted"] = False               # el flag vive exactamente una vela
        st["prev_pos"] = position_side

        m = mom[i]
        if math.isnan(m):
            return None
        # La supresión se levanta cuando el momentum deja de apuntar a esa dirección.
        if st["suppress"] == "LONG" and m <= 0:
            st["suppress"] = None
        elif st["suppress"] == "SHORT" and m >= 0:
            st["suppress"] = None

        side = "LONG" if m > 0 else ("SHORT" if m < 0 else None)
        if position_side is None:
            if side is None or side == st["suppress"]:
                return None
            if side == "SHORT" and not allow_short:
                return None
            stop = _stop_level(side, closes[i], atrs[i], atr_mult)
            if stop is None:
                return None
            return ("enter", side, 1.0, stop, None)   # TP None → salida por señal/stop
        # Sosteniendo: salir cuando el signo se invierte.
        if (position_side == "LONG" and m < 0) or (position_side == "SHORT" and m > 0):
            st["exit_emitted"] = True
            return ("exit",)
        return None

    return decider


def make_macross_decider(closes: np.ndarray, atrs: np.ndarray, fast: np.ndarray,
                         slow: np.ndarray, *, atr_mult: float, allow_short: bool):
    """Cruce de medias: el lado lo marca el signo de (fast − slow).

    LONG cuando la media rápida está por encima de la lenta; SHORT cuando por
    debajo (si allow_short). Flip en el cruce opuesto. Igual que el TSMOM, el
    stop ATR es un freno: tras saltar, NO se reentra en la misma dirección hasta
    que las medias se vuelvan a cruzar → un stop no genera churn de reentradas.
    Lógica clásica del golden/death cross, neta de costos.
    """
    st = {"prev_pos": None, "exit_emitted": False, "suppress": None}

    def decider(i, position_side, score, ts):
        if (st["prev_pos"] is not None and position_side is None
                and not st["exit_emitted"]):
            st["suppress"] = st["prev_pos"]
        st["exit_emitted"] = False
        st["prev_pos"] = position_side

        f, s = fast[i], slow[i]
        if math.isnan(f) or math.isnan(s):
            return None
        side = "LONG" if f > s else ("SHORT" if f < s else None)
        if st["suppress"] == "LONG" and f <= s:
            st["suppress"] = None
        elif st["suppress"] == "SHORT" and f >= s:
            st["suppress"] = None

        if position_side is None:
            if side is None or side == st["suppress"]:
                return None
            if side == "SHORT" and not allow_short:
                return None
            stop = _stop_level(side, closes[i], atrs[i], atr_mult)
            if stop is None:
                return None
            return ("enter", side, 1.0, stop, None)
        # Sosteniendo: salir cuando las medias cruzan en contra.
        if (position_side == "LONG" and f < s) or (position_side == "SHORT" and f > s):
            st["exit_emitted"] = True
            return ("exit",)
        return None

    return decider


def make_rsi_reversion_decider(closes: np.ndarray, rsi_vals: np.ndarray,
                               trend_sma: np.ndarray, atrs: np.ndarray, *,
                               oversold: float, overbought: float,
                               atr_mult: float):
    """Reversión a la media: LONG cuando RSI<oversold Y close>SMA (comprar el
    dip DENTRO de una tendencia alcista de fondo), salir cuando RSI>overbought
    (la condición de entrada se revirtió: de sobreventa a sobrecompra). El
    filtro SMA evita 'cazar el cuchillo' en tendencia bajista.

    Stop ATR explícito + tp=None: la ganancia no tiene techo — si el rebote se
    convierte en tendencia, la posición sigue hasta que el RSI cruce a
    sobrecompra o salte el stop.
    """
    def decider(i, position_side, score, ts):
        r = rsi_vals[i]
        if math.isnan(r) or math.isnan(trend_sma[i]):
            return None
        if position_side is None:
            if r < oversold and closes[i] > trend_sma[i]:
                stop = _stop_level("LONG", closes[i], atrs[i], atr_mult)
                if stop is None:
                    return None
                return ("enter", "LONG", 1.0, stop, None)
            return None
        if r > overbought:
            return ("exit",)
        return None

    return decider


def make_dow_decider(closes: np.ndarray, atrs: np.ndarray, next_weekday: np.ndarray,
                     *, entry_weekday: int, hold_days: int, atr_mult: float):
    """Efecto día-de-semana en calendario BURSÁTIL real (huecos de fin de
    semana/festivos, no un ciclo fijo de 7 días como en cripto 24/7).

    `next_weekday[i]` = día de la semana de la SIGUIENTE vela de trading real
    (precalculado desde los datos: un festivo que corre el día se resuelve
    solo). Entra si la PRÓXIMA vela real cae en `entry_weekday`; sale tras
    `hold_days` VELAS de trading contadas desde la entrada — estado explícito.
    """
    st = {"entry_bar": None}

    def decider(i, position_side, score, ts):
        if position_side is None:
            st["entry_bar"] = None
            if i < len(next_weekday) and next_weekday[i] == entry_weekday:
                stop = _stop_level("LONG", closes[i], atrs[i], atr_mult)
                if stop is None:
                    return None
                st["entry_bar"] = i
                return ("enter", "LONG", 1.0, stop, None)
            return None
        if st["entry_bar"] is not None and (i - st["entry_bar"]) >= hold_days:
            return ("exit",)
        return None

    return decider
