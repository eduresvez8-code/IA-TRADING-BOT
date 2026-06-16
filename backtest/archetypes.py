"""Tres arquetipos de estrategia para el escáner de investigación quant.

Cada arquetipo es una filosofía con raíz matemática OPUESTA:

  - "trend"           Arq. 1 — Seguimiento de tendencia: EMA-cross + RSI (la señal
                      actual del bot). Gana en expansiones macro, sufre en lateral.
  - "mean_reversion"  Arq. 2 — Reversión a la media: entra en las Bandas de
                      Bollinger con RSI en extremos y sale en la media central.
                      Monetiza rangos; un stop a ATR la protege de tendencias.
  - "breakout"        Arq. 3 — Ruptura de volatilidad: entra al romper el canal de
                      Donchian con volumen, stop en el lado OPUESTO del canal y TP
                      por RR. Captura el inicio de impulsos violentos.

Cada arquetipo expone un *decider* compatible con `BacktestEngine.run(decider=…)`:
una función `decide(i, position_side, score, ts)` que mira indicadores
precalculados (cerrados en el closure, causales) y devuelve la acción de la vela
— `("enter", side, size_factor[, stop_px, tp_px])`, `("exit",)` o `None`.

Cero hardcoding: todos los umbrales vienen de `Settings`. El sizing (riesgo fijo)
y los costos los aplica el motor, idénticos para los tres → comparación limpia.
"""

from __future__ import annotations

import pandas as pd

from src.core.config import Settings
from src.quant.indicators import atr, bollinger_bands, donchian_channel, rsi, sma

ARCHETYPES = ("trend", "mean_reversion", "breakout")

# Etiquetas legibles para los reportes.
ARCHETYPE_LABELS = {
    "trend": "Tendencia (EMA-cross + RSI)",
    "mean_reversion": "Reversión a la media (Bollinger + RSI)",
    "breakout": "Ruptura de volatilidad (Donchian + volumen)",
}


def make_mean_reversion_decider(df: pd.DataFrame, settings: Settings, *, allow_short: bool):
    """Arq. 2. Entra en las bandas con RSI extremo; sale en la media central.

    LONG  : cierre ≤ banda inferior  y  RSI < oversold  → stop a ATR, sin TP.
    SHORT : cierre ≥ banda superior  y  RSI > overbought → stop a ATR, sin TP.
    Salida: el cierre cruza la media central (SMA). Sin TP fijo: la tesis es
            "vuelve a la media", no "alcanza un objetivo RR". El stop ATR es el
            seguro contra una tendencia que no revierte.
    """
    mr = settings.mean_reversion
    close = df["close"]
    middle, upper, lower = bollinger_bands(close, mr.bb_period, mr.bb_num_std)
    rsi_s = rsi(close, settings.quant.rsi_period)
    atr_s = atr(df, settings.risk.atr_period)
    atr_mult = settings.risk.atr_stop_multiplier

    c = close.to_numpy()
    mid, up, lo = middle.to_numpy(), upper.to_numpy(), lower.to_numpy()
    r, a = rsi_s.to_numpy(), atr_s.to_numpy()

    def decide(i, position_side, score, ts):
        if pd.isna(mid[i]) or pd.isna(r[i]) or pd.isna(a[i]):
            return None
        if position_side is None:
            if c[i] <= lo[i] and r[i] < mr.rsi_oversold:
                return ("enter", "LONG", 1.0, c[i] - atr_mult * a[i], None)
            if allow_short and c[i] >= up[i] and r[i] > mr.rsi_overbought:
                return ("enter", "SHORT", 1.0, c[i] + atr_mult * a[i], None)
            return None
        if position_side == "LONG" and c[i] >= mid[i]:
            return ("exit",)
        if position_side == "SHORT" and c[i] <= mid[i]:
            return ("exit",)
        return None

    return decide


def make_breakout_decider(df: pd.DataFrame, settings: Settings, *, allow_short: bool):
    """Arq. 3 (refinado, estilo Turtle). Ruptura con volumen y régimen volátil;
    stop duro al lado opuesto del canal de entrada, salida TRAILING por canal corto.

    Entrada LONG : cierre > máximo del canal de entrada previo (Donchian N)
                   Y volumen > mult × media(volumen)            (impulso real)
                   Y ATR > expansión × media(ATR)               (volatilidad expansiva).
                   Stop duro = mínimo del canal de entrada (lado opuesto). SIN TP.
    Salida LONG  : el cierre rompe a la baja el canal de SALIDA previo (Donchian
                   M<N) → trailing al estilo Turtle: deja correr al ganador y lo
                   suelta cuando la estructura de corto plazo se rompe.
    SHORT: espejo.

    Anti look-ahead: canales, media de volumen y media de ATR se desplazan una
    vela (`.shift(1)`); la vela t solo usa niveles conocidos ANTES de t.
    Por qué los cambios vs. la v1 (TP por RR): un TP fijo sobre un stop de canal
    ancho cortaba a los ganadores — lo contrario de lo que busca un breakout. El
    trailing y el filtro de volatilidad atacan ese modo de fallo directamente.
    """
    bo = settings.breakout
    high, low, close, volume = df["high"], df["low"], df["close"], df["volume"]
    don_up, don_lo = donchian_channel(high, low, bo.donchian_period)
    don_up_prev = don_up.shift(1).to_numpy()
    don_lo_prev = don_lo.shift(1).to_numpy()
    exit_up, exit_lo = donchian_channel(high, low, bo.exit_donchian_period)
    exit_up_prev = exit_up.shift(1).to_numpy()
    exit_lo_prev = exit_lo.shift(1).to_numpy()
    vol_ma_prev = sma(volume, bo.volume_ma_period).shift(1).to_numpy()
    atr_s = atr(df, settings.risk.atr_period)
    atr_avg = sma(atr_s, bo.atr_filter_period)
    a, a_avg = atr_s.to_numpy(), atr_avg.to_numpy()

    c = close.to_numpy()
    v = volume.to_numpy()
    mult = bo.volume_multiplier
    expand = bo.atr_expansion_mult

    def decide(i, position_side, score, ts):
        # Salida TRAILING: ruptura del canal de salida (corto) en contra.
        if position_side == "LONG":
            if not pd.isna(exit_lo_prev[i]) and c[i] < exit_lo_prev[i]:
                return ("exit",)
            return None
        if position_side == "SHORT":
            if not pd.isna(exit_up_prev[i]) and c[i] > exit_up_prev[i]:
                return ("exit",)
            return None
        # Entrada: ruptura + volumen + régimen de volatilidad expansiva.
        if (pd.isna(don_up_prev[i]) or pd.isna(vol_ma_prev[i]) or pd.isna(a_avg[i])):
            return None
        if v[i] <= mult * vol_ma_prev[i]:
            return None  # ruptura sin volumen: no confirmada
        if a[i] <= expand * a_avg[i]:
            return None  # volatilidad en compresión: ruptura poco fiable
        if c[i] > don_up_prev[i]:
            stop = don_lo_prev[i]
            if stop < c[i]:
                return ("enter", "LONG", 1.0, stop, None)   # sin TP: deja correr
        if allow_short and c[i] < don_lo_prev[i]:
            stop = don_up_prev[i]
            if stop > c[i]:
                return ("enter", "SHORT", 1.0, stop, None)
        return None

    return decide


def make_decider(archetype: str, df: pd.DataFrame, settings: Settings, *, allow_short: bool):
    """Devuelve el decider del arquetipo, o None para 'trend' (decider del motor).

    Para 'trend' el motor usa su decider por defecto (EMA-cross + RSI por umbral,
    Sprint 3) — es exactamente el Arquetipo 1, sin duplicar la lógica.
    """
    if archetype == "trend":
        return None
    if archetype == "mean_reversion":
        return make_mean_reversion_decider(df, settings, allow_short=allow_short)
    if archetype == "breakout":
        return make_breakout_decider(df, settings, allow_short=allow_short)
    raise ValueError(f"arquetipo desconocido: {archetype!r}")
