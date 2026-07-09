"""Barrido en ACCIONES (2026-07-08): la misma batería de estrategias probadas
en cripto, aplicada a 10 acciones grandes de sectores distintos, con datos
diarios 2010-2026 (yfinance, gratis). Reutiliza los MISMOS deciders y grids ya
verificados (Cero Hardcoding intacto: nada nuevo se hardcodea aquí).

    uv run python -m backtest.run_stocks

Familias EXCLUIDAS por no aplicar a acciones: Donchian (exige funding rate,
concepto de perp), estacionalidad horaria (las acciones no operan 24/7).

Protocolo idéntico a toda la sesión: split FIJO por mitades ANTES de mirar
resultados, config elegida SOLO por train, test medido una vez, grid completo.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.core.config import load_settings
from src.quant.indicators import atr, rsi, sma
from backtest.engine import BacktestEngine
from backtest.quant_hypotheses import (
    make_macross_decider,
    make_rsi_reversion_decider,
    make_tsmom_decider,
    moving_average,
)

TICKERS = ["XOM", "MSFT", "WMT", "GE", "JNJ", "PG", "T", "BAC", "CVX", "IBM"]
_DATA_DIR = Path("data/stocks")


def load_stock(ticker: str) -> pd.DataFrame:
    return pd.read_parquet(_DATA_DIR / f"{ticker}.parquet")


def run_ma_cross(cfg, engine):
    qh = cfg.quant_hypotheses
    print("\n### MA-CROSS en acciones (1d) — por ticker, top 10 por train ###")
    pares = []
    for kind in qh.ma_cross_types:
        for fast_p, slow_p in qh.ma_cross_pairs:
            for t in TICKERS:
                df = load_stock(t).reset_index(drop=True)
                closes = df["close"].to_numpy(dtype=float)
                atrs = atr(df, cfg.risk.atr_period).to_numpy()
                fast = moving_average(closes, fast_p, kind)
                slow = moving_average(closes, slow_p, kind)
                decider = make_macross_decider(
                    closes, atrs, fast, slow,
                    atr_mult=qh.atr_stop_mult, allow_short=qh.ma_cross_allow_short)
                mid = len(df) // 2
                s1 = engine.run(df.iloc[:mid], t, "1d", decider=decider).metrics
                s2 = engine.run(df.iloc[mid:], t, "1d", decider=decider).metrics
                pares.append((f"{kind.upper()}{fast_p}/{slow_p}", t, s1.sharpe, s2.sharpe,
                             s1.n_trades, s2.n_trades))
    pares.sort(key=lambda r: -r[2])
    _print_top(pares, top=10)
    return pares


def run_tsmom(cfg, engine):
    qh = cfg.quant_hypotheses
    print("\n### TSMOM en acciones (1d) — por ticker, top 10 por train ###")
    pares = []
    for lookback in qh.tsmom_lookback_days_grid:
        for t in TICKERS:
            df = load_stock(t).reset_index(drop=True)
            closes = df["close"].to_numpy(dtype=float)
            atrs = atr(df, cfg.risk.atr_period).to_numpy()
            decider = make_tsmom_decider(closes, atrs, lookback, qh.atr_stop_mult)
            mid = len(df) // 2
            s1 = engine.run(df.iloc[:mid], t, "1d", decider=decider).metrics
            s2 = engine.run(df.iloc[mid:], t, "1d", decider=decider).metrics
            pares.append((f"TSMOM-{lookback}d", t, s1.sharpe, s2.sharpe,
                         s1.n_trades, s2.n_trades))
    pares.sort(key=lambda r: -r[2])
    _print_top(pares, top=10)
    return pares


def run_rsi_reversion(cfg, engine):
    qh = cfg.quant_hypotheses
    print("\n### RSI-reversión en acciones (1d) — por ticker ###")
    pares = []
    for t in TICKERS:
        df = load_stock(t).reset_index(drop=True)
        closes = df["close"].to_numpy(dtype=float)
        rsi_vals = rsi(df["close"], qh.rsi_reversion_period).to_numpy()
        trend = sma(df["close"], qh.rsi_reversion_trend_sma).to_numpy()
        atrs = atr(df, cfg.risk.atr_period).to_numpy()
        decider = make_rsi_reversion_decider(
            closes, rsi_vals, trend, atrs, oversold=qh.rsi_reversion_oversold,
            overbought=qh.rsi_reversion_overbought, atr_mult=qh.atr_stop_mult)
        mid = len(df) // 2
        s1 = engine.run(df.iloc[:mid], t, "1d", decider=decider).metrics
        s2 = engine.run(df.iloc[mid:], t, "1d", decider=decider).metrics
        pares.append(("RSI-rev", t, s1.sharpe, s2.sharpe, s1.n_trades, s2.n_trades))
    pares.sort(key=lambda r: -r[2])
    _print_top(pares, top=10)
    return pares


def _make_stock_dow_decider(closes: np.ndarray, atrs: np.ndarray, next_wd: np.ndarray,
                            *, entry_weekday: int, hold_days: int, atr_mult: float):
    """Efecto día-de-semana adaptado a calendario BURSÁTIL (huecos reales de fin
    de semana/festivos, no un ciclo fijo de 7 días como en cripto 24/7).

    `next_wd[i]` = día de la semana de la SIGUIENTE vela de trading real (ya
    precalculado desde los datos, así que un festivo que corre el "viernes
    siguiente" a jueves se resuelve solo). Entra si la PRÓXIMA vela real cae en
    `entry_weekday`; sale tras `hold_days` VELAS de trading (no días calendario)
    contadas desde la entrada — estado explícito, igual que TSMOM/MA-cross.
    """
    from backtest.quant_hypotheses import _stop_level

    st = {"entry_bar": None}

    def decider(i, position_side, score, ts):
        if position_side is None:
            st["entry_bar"] = None
            if i < len(next_wd) and next_wd[i] == entry_weekday:
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


def run_dow(cfg, engine):
    qh = cfg.quant_hypotheses
    print("\n### Efecto día-de-la-semana en acciones (1d, la anomalía clásica 'lunes') ###")
    print("(adaptado al calendario bursátil real: entra cuando la PRÓXIMA vela de")
    print(" trading, no un offset fijo de 7 días, cae en el día objetivo)")
    pares = []
    for t in TICKERS:
        df = load_stock(t).reset_index(drop=True)
        closes = df["close"].to_numpy(dtype=float)
        atrs = atr(df, cfg.risk.atr_period).to_numpy()
        wd = df["open_time"].dt.dayofweek.to_numpy()
        next_wd = np.roll(wd, -1)  # next_wd[i] = día de semana de la vela i+1
        decider = _make_stock_dow_decider(
            closes, atrs, next_wd, entry_weekday=qh.dow_entry_weekday,
            hold_days=qh.dow_hold_days, atr_mult=qh.atr_stop_mult)
        mid = len(df) // 2
        s1 = engine.run(df.iloc[:mid], t, "1d", decider=decider).metrics
        s2 = engine.run(df.iloc[mid:], t, "1d", decider=decider).metrics
        pares.append(("DoW-Mon", t, s1.sharpe, s2.sharpe, s1.n_trades, s2.n_trades))
    pares.sort(key=lambda r: -r[2])
    _print_top(pares, top=10)
    return pares


def _print_top(pares, *, top: int) -> None:
    header = ["config", "ticker", "Sh train", "Sh test", "n1", "n2"]
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for cfg_name, t, s1, s2, n1, n2 in pares[:top]:
        print(f"| {cfg_name} | {t} | {s1:+.2f} | {s2:+.2f} | {n1} | {n2} |")
    pos = sum(1 for r in pares if r[3] > 0)
    print(f"Test positivo: {pos}/{len(pares)} pares (referencia de ruido: 50% esperado por azar)")


def main() -> int:
    cfg = load_settings()
    engine = BacktestEngine(cfg)
    print("=" * 84)
    print("BARRIDO EN ACCIONES — 10 tickers, 2010-2026, misma batería que cripto")
    print("=" * 84)
    run_ma_cross(cfg, engine)
    run_tsmom(cfg, engine)
    run_rsi_reversion(cfg, engine)
    run_dow(cfg, engine)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
