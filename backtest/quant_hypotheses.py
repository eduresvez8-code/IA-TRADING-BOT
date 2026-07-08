"""Hipótesis quant de la investigación Perplexity (2026-06) → deciders del motor.

Cada hipótesis se expresa como un `decider(i, position_side, score, ts)` que el
`BacktestEngine` consume (decide al cierre de t, ejecuta en la apertura de t+1, sin
look-ahead). Los deciders IGNORAN el `score` EMA del motor: traen su propia señal
precalculada (TSMOM, funding extremo, Donchian) en arrays cerrados por closure.

Familias implementadas (todas perp-only, direccionales, datos gratis):
    H1  TSMOM diario        — signo del retorno a N días; siempre en mercado.
    H2  Funding extremo     — contrarian al crowding + filtro de tendencia MA.
    H3  Ruptura Donchian 4h — breakout de canal + banda de funding (SIN gate de OI).

NO incluye las hipótesis de POSICIONAMIENTO (open interest / long-short ratio):
Binance solo sirve ~30 días de ese dato gratis → no son backtesteables. Se
documentan como no-verificables en el runner.

Stops: explícitos, anclados al cierre causal de la vela de decisión (close ∓
mult·ATR). El motor dimensiona por la distancia al stop (riesgo fijo % equity).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

# Periodos de funding por día (Binance: cada 8h → 3/día) y días/año para anualizar.
_FUNDING_PERIODS_PER_DAY = 3
_DAYS_PER_YEAR = 365


def annualize_funding_pct(funding_frac: np.ndarray) -> np.ndarray:
    """Funding 8h (fracción, p.ej. 1e-4) → tasa ANUAL en % (1e-4 → +10.95%).

    FR_ann% = FR_8h · 3 periodos/día · 365 días · 100. Es la convención que usó
    Perplexity para fijar los umbrales (-20%, +40%).
    """
    return funding_frac * _FUNDING_PERIODS_PER_DAY * _DAYS_PER_YEAR * 100.0


def align_funding_to_bars(bars: pd.DataFrame, funding: pd.DataFrame) -> np.ndarray:
    """Funding 8h → un valor por vela (asof backward: el último funding vigente).

    Para cada vela en `bars.open_time` toma el `funding_rate` cuyo `funding_time`
    es el más reciente ≤ open_time. Causal por construcción (nunca usa un funding
    futuro). Devuelve un array de fracciones alineado posicionalmente a `bars`.
    """
    left = bars[["open_time"]].reset_index(drop=True)
    right = funding[["funding_time", "funding_rate"]].sort_values("funding_time")
    merged = pd.merge_asof(
        left, right, left_on="open_time", right_on="funding_time",
        direction="backward",
    )
    return merged["funding_rate"].to_numpy(dtype=float)


def daily_ma_on_bars(bars: pd.DataFrame, ma_days: int) -> np.ndarray:
    """MA de cierres DIARIOS proyectada (causal) sobre la rejilla de `bars`.

    Resamplea a 1d, calcula la SMA de `ma_days`, la DESPLAZA un día (cada vela usa
    la MA del día ANTERIOR ya cerrado → sin look-ahead) y la propaga hacia adelante
    a cada vela. Devuelve un array alineado a `bars` (NaN durante el warmup).
    """
    s = bars.set_index("open_time")["close"]
    daily_close = s.resample("1D").last()
    daily_ma = daily_close.rolling(ma_days).mean().shift(1)  # día anterior cerrado
    # Reindexa a las velas: asof backward = la MA diaria vigente en cada vela.
    ma_on_bars = daily_ma.reindex(
        daily_ma.index.union(s.index)).ffill().reindex(s.index)
    return ma_on_bars.to_numpy(dtype=float)


def _stop_level(side: str, ref_close: float, atr_val: float, mult: float) -> float | None:
    """Nivel de stop explícito: close ∓ mult·ATR. None si el ATR no es válido."""
    if atr_val is None or math.isnan(atr_val) or atr_val <= 0:
        return None
    return ref_close - mult * atr_val if side == "LONG" else ref_close + mult * atr_val


def make_tsmom_decider(closes: np.ndarray, atrs: np.ndarray, lookback: int,
                       atr_mult: float):
    """H1 — TSMOM diario: signo del retorno a `lookback` días marca el lado.

    Siempre busca estar en mercado en la dirección del momentum; sale cuando el
    signo se invierte. El stop ATR es un freno de desastre: tras saltar, NO se
    reentra en la misma dirección hasta que el signo del momentum se reinicie
    (cruce de cero) — así un stop no provoca churn de reentradas a costo.
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


def moving_average(closes: np.ndarray, period: int, kind: str) -> np.ndarray:
    """Media móvil de los cierres: 'sma' (simple) o 'ema' (exponencial).

    La SMA pondera igual las N velas (más lenta, menos whipsaws); la EMA pondera
    más lo reciente (reacciona antes, más señales falsas). Ambas causales.
    """
    s = pd.Series(closes)
    if kind == "ema":
        return s.ewm(span=period, adjust=False).mean().to_numpy()
    return s.rolling(period).mean().to_numpy()


def make_macross_decider(closes: np.ndarray, atrs: np.ndarray, fast: np.ndarray,
                         slow: np.ndarray, *, atr_mult: float, allow_short: bool):
    """H4 — Cruce de medias: el lado lo marca el signo de (fast − slow).

    LONG cuando la media rápida está por encima de la lenta; SHORT cuando por debajo
    (si allow_short). Flip en el cruce opuesto. Igual que el TSMOM, el stop ATR es un
    freno: tras saltar, NO se reentra en la misma dirección hasta que el cruce se
    reinicie (las medias se vuelven a cruzar) → un stop no genera churn de reentradas.
    Esta es la lógica clásica del golden/death cross, simétrica y neta de costos.
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


def make_funding_decider(closes: np.ndarray, atrs: np.ndarray,
                         funding_ann_pct: np.ndarray, trend_ma: np.ndarray,
                         *, neg_thr: float, pos_thr: float,
                         normal_low: float, normal_high: float, atr_mult: float):
    """H2 — Funding extremo direccional (contrarian + filtro de tendencia).

    LONG  si funding anual < neg_thr (crowd corto pagando) Y close > MA (tendencia
          alcista de fondo): se apuesta al rebote, no a cazar el cuchillo.
    SHORT si funding anual > pos_thr (euforia larga) Y close < MA.
    Entra solo cuando la condición se ACTIVA (fresca) → no reentra mientras siga
    vigente, evitando churn tras un stop. Sale cuando el funding vuelve a la banda
    normal O el precio cruza la MA en contra.
    """
    def cond_long(i):
        return funding_ann_pct[i] < neg_thr and closes[i] > trend_ma[i]

    def cond_short(i):
        return funding_ann_pct[i] > pos_thr and closes[i] < trend_ma[i]

    def decider(i, position_side, score, ts):
        if i == 0 or math.isnan(funding_ann_pct[i]) or math.isnan(trend_ma[i]):
            return None
        if position_side is None:
            # Entrada fresca: la condición pasa de falsa (vela previa) a verdadera.
            if cond_long(i) and not cond_long(i - 1):
                stop = _stop_level("LONG", closes[i], atrs[i], atr_mult)
                return ("enter", "LONG", 1.0, stop, None) if stop else None
            if cond_short(i) and not cond_short(i - 1):
                stop = _stop_level("SHORT", closes[i], atrs[i], atr_mult)
                return ("enter", "SHORT", 1.0, stop, None) if stop else None
            return None
        # Salida: funding normalizado O precio cruzó la MA en contra.
        if position_side == "LONG":
            if funding_ann_pct[i] >= normal_low or closes[i] < trend_ma[i]:
                return ("exit",)
        else:  # SHORT
            if funding_ann_pct[i] <= normal_high or closes[i] > trend_ma[i]:
                return ("exit",)
        return None

    return decider


def make_donchian_decider(closes: np.ndarray, atrs: np.ndarray,
                          funding_frac: np.ndarray, *, entry_period: int,
                          exit_ema_period: int, funding_min_frac: float,
                          funding_max_frac: float, atr_mult: float,
                          take_profit_rr: float | None, max_hold_bars: int | None):
    """H3 — Ruptura Donchian 4h + banda de funding (SIN el gate de OI).

    LONG  si close rompe el máximo de los `entry_period` cierres previos Y el
          funding 8h está en [min, max] (demanda apalancada moderada, no eufórica).
    SHORT si close rompe el mínimo previo Y el funding ≤ -min.
    Stop ATR + take-profit a `take_profit_rr`× el riesgo (gestionados por el motor).
    Salida adicional: cruce de la EMA rápida en contra, o `max_hold_bars` velas.

    Variante "dejar correr" (2026-07-08): `take_profit_rr=None` NO coloca techo de
    ganancia (tp=None → el motor solo vigila el stop) y `max_hold_bars=None` quita
    la salida por tiempo. La pierna sale SOLO por el cruce de la EMA rápida en
    contra (trailing de tendencia) o por el stop ATR — la estructura "cortar
    pérdidas rápido, dejar correr las ganancias".
    OJO: la confirmación por open interest de la receta original se OMITE — Binance
    no sirve histórico de OI gratis. Esto prueba una versión PARCIAL de la hipótesis.
    """
    s = pd.Series(closes)
    # Canal causal: máx/mín de los N cierres ANTERIORES (shift(1) excluye el actual).
    donchian_high = s.shift(1).rolling(entry_period).max().to_numpy()
    donchian_low = s.shift(1).rolling(entry_period).min().to_numpy()
    ema_exit = s.ewm(span=exit_ema_period, adjust=False).mean().to_numpy()

    st = {"prev_pos": None, "entry_bar": None}

    def decider(i, position_side, score, ts):
        # Detecta la vela de entrada (transición plano→posición) para el max_hold.
        if st["prev_pos"] is None and position_side is not None:
            st["entry_bar"] = i
        st["prev_pos"] = position_side

        if math.isnan(donchian_high[i]) or math.isnan(funding_frac[i]):
            return None

        if position_side is None:
            fr = funding_frac[i]
            if closes[i] > donchian_high[i] and funding_min_frac <= fr <= funding_max_frac:
                stop = _stop_level("LONG", closes[i], atrs[i], atr_mult)
                if stop is None:
                    return None
                # rr=None → sin techo de ganancia: el motor solo vigila el stop.
                tp = (closes[i] + take_profit_rr * (closes[i] - stop)
                      if take_profit_rr is not None else None)
                return ("enter", "LONG", 1.0, stop, tp)
            if closes[i] < donchian_low[i] and fr <= -funding_min_frac:
                stop = _stop_level("SHORT", closes[i], atrs[i], atr_mult)
                if stop is None:
                    return None
                tp = (closes[i] - take_profit_rr * (stop - closes[i])
                      if take_profit_rr is not None else None)
                return ("enter", "SHORT", 1.0, stop, tp)
            return None

        # Salida por tiempo (backstop, solo si está configurada) o por cruce
        # de la EMA rápida en contra.
        if (max_hold_bars is not None and st["entry_bar"] is not None
                and (i - st["entry_bar"]) >= max_hold_bars):
            return ("exit",)
        if position_side == "LONG" and closes[i] < ema_exit[i]:
            return ("exit",)
        if position_side == "SHORT" and closes[i] > ema_exit[i]:
            return ("exit",)
        return None

    return decider


def make_leadlag_decider(closes: np.ndarray, atrs: np.ndarray,
                         leader_ret_lag: np.ndarray, target_sma: np.ndarray,
                         leader_close: np.ndarray, leader_sma: np.ndarray,
                         *, atr_mult: float, allow_short: bool):
    """H5 — Lead-lag cascade: el signo del retorno del LÍDER (BTC) a N horas marca
    el lado del TARGET (alt), filtrado por un régimen SMA que líder y target deben
    compartir.

    Señal (causal, ya desfasada 1 vela en `leader_ret_lag`):
        LONG  si leader_ret_lag > 0 Y target_close > target_SMA Y leader_close > leader_SMA
        SHORT si leader_ret_lag < 0 Y target_close < target_SMA Y leader_close < leader_SMA
    El doble gate de régimen (ambos activos de acuerdo) es la condición del JSON de
    Perplexity (rank 4/5). Flip cuando la señal se invierte; supresión tras stop para
    no reentrar la misma dirección hasta que la señal se reinicie (igual que TSMOM/MA).
    """
    st = {"prev_pos": None, "exit_emitted": False, "suppress": None}

    def _side(i):
        m = leader_ret_lag[i]
        if math.isnan(m) or math.isnan(target_sma[i]) or math.isnan(leader_sma[i]):
            return None
        up_regime = closes[i] > target_sma[i] and leader_close[i] > leader_sma[i]
        dn_regime = closes[i] < target_sma[i] and leader_close[i] < leader_sma[i]
        if m > 0 and up_regime:
            return "LONG"
        if m < 0 and dn_regime:
            return "SHORT"
        return None

    def decider(i, position_side, score, ts):
        if (st["prev_pos"] is not None and position_side is None
                and not st["exit_emitted"]):
            st["suppress"] = st["prev_pos"]
        st["exit_emitted"] = False
        st["prev_pos"] = position_side

        side = _side(i)
        # La supresión se levanta cuando la señal deja de apuntar a esa dirección.
        if st["suppress"] == "LONG" and side != "LONG":
            st["suppress"] = None
        elif st["suppress"] == "SHORT" and side != "SHORT":
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
        # Sosteniendo: salir cuando la señal se invierte o se apaga contra la posición.
        if position_side == "LONG" and side != "LONG":
            st["exit_emitted"] = True
            return ("exit",)
        if position_side == "SHORT" and side != "SHORT":
            st["exit_emitted"] = True
            return ("exit",)
        return None

    return decider


def _utc_dt(ts):
    """Convierte el open_time del motor (epoch ms int, o datetime) a datetime UTC."""
    if isinstance(ts, (int, float)):
        return pd.Timestamp(ts, unit="ms", tz="UTC")
    return pd.Timestamp(ts).tz_localize("UTC") if pd.Timestamp(ts).tzinfo is None else pd.Timestamp(ts)


def make_hour_seasonality_decider(closes: np.ndarray, atrs: np.ndarray, *,
                                  entry_open_hour: int, hold_hours: int,
                                  atr_mult: float):
    """H6 — Estacionalidad horaria: LONG en la APERTURA de `entry_open_hour` UTC,
    sostener `hold_hours` velas 1h, salir cuando la ventana horaria pasó. Sin
    predecir precio: apuesta a un patrón de flujo por hora del día (SSRN 4081000).

    Estructura de salida (2026-07-08, "dejar correr"): stop ATR EXPLÍCITO
    (close ∓ mult·ATR de la vela de decisión) como único freno de pérdida y
    tp=None — sin techo de ganancia. Antes estos deciders no pasaban stop/tp y
    el motor les aplicaba su default con take-profit FIJO (bt.take_profit_rr);
    eso capaba las ganancias, justo lo contrario de la filosofía a probar.

    Como el motor decide al cierre de i y ejecuta en la apertura de i+1: para entrar
    en la apertura de la hora H hay que señalar cuando la vela i es la hora (H-1);
    para salir en la apertura de H+hold hay que señalar en la hora (H+hold-1).
    """
    signal_in = (entry_open_hour - 1) % 24
    signal_out = (entry_open_hour - 1 + hold_hours) % 24

    def decider(i, position_side, score, ts):
        h = _utc_dt(ts).hour
        if position_side is None:
            if h == signal_in:
                stop = _stop_level("LONG", closes[i], atrs[i], atr_mult)
                if stop is None:
                    return None
                return ("enter", "LONG", 1.0, stop, None)  # tp None → sin techo
            return None
        if h == signal_out:
            return ("exit",)
        return None

    return decider


def make_dow_decider(closes: np.ndarray, atrs: np.ndarray, *,
                     entry_weekday: int, hold_days: int, atr_mult: float):
    """H7 — Efecto día-de-la-semana en velas DIARIAS: LONG el día `entry_weekday`
    (0=lunes), sostener `hold_days` velas; la salida es "el día pasó" (reversión
    de la condición de entrada). El motor entra en la apertura del día siguiente
    al de la señal, así que señalamos el día previo al de entrada deseado.

    Igual que H6: stop ATR explícito + tp=None (sin techo de ganancia).
    """
    signal_in = (entry_weekday - 1) % 7

    def decider(i, position_side, score, ts):
        wd = _utc_dt(ts).dayofweek
        if position_side is None:
            if wd == signal_in:
                stop = _stop_level("LONG", closes[i], atrs[i], atr_mult)
                if stop is None:
                    return None
                return ("enter", "LONG", 1.0, stop, None)
            return None
        # Salir tras hold_days: señalamos hold_days-1 días después de la entrada real.
        if wd == (entry_weekday - 1 + hold_days) % 7:
            return ("exit",)
        return None

    return decider


def make_rsi_reversion_decider(closes: np.ndarray, rsi_vals: np.ndarray,
                               trend_sma: np.ndarray, atrs: np.ndarray, *,
                               oversold: float, overbought: float,
                               atr_mult: float):
    """H8 — Reversión a la media: LONG cuando RSI<oversold Y close>SMA (comprar el
    dip DENTRO de una tendencia alcista de fondo), salir cuando RSI>overbought
    (la condición de entrada se revirtió: de sobreventa a sobrecompra). El
    filtro SMA evita 'cazar el cuchillo' en tendencia bajista.

    Igual que H6/H7: stop ATR explícito + tp=None. La ganancia no tiene techo:
    si el rebote se convierte en tendencia, la posición sigue hasta que el RSI
    cruce a sobrecompra o salte el stop.
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
