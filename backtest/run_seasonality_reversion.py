"""H6/H7/H8 — Barrido FINAL (Perplexity batch 3): estacionalidad horaria,
efecto día-de-la-semana y reversión RSI+SMA. NETO de costos + split OOS.

    uv run python -m backtest.run_seasonality_reversion

Las tres son las únicas familias NUEVAS y BACKTESTEABLES del último JSON. El resto
del batch es no-testeable con datos gratis (basis trade → necesita spot; OI/L-S ratio
→ solo 30 días gratis) o son overlays de sizing (vol targeting, HMM) que no originan.
"""

from __future__ import annotations

import pandas as pd

from src.core.config import load_settings
from src.quant.indicators import atr, rsi, sma
from backtest.engine import BacktestEngine
from backtest.quant_hypotheses import (
    make_dow_decider,
    make_hour_seasonality_decider,
    make_rsi_reversion_decider,
)
from backtest.run_backtest import load_parquet
from backtest.run_quant_hypotheses import ASSETS, Row, _table, resample


def _split_sharpe(engine, df, asset, tf, make_decider):
    """Sharpe full | 1ª mitad | 2ª mitad, con el decider ya construido por closure."""
    def run(slice_df):
        s = slice_df.reset_index(drop=True)
        return engine.run(s, asset, tf, decider=make_decider(s)).metrics
    mid = len(df) // 2
    full = engine.run(df.reset_index(drop=True), asset, tf,
                      decider=make_decider(df.reset_index(drop=True)))
    m1 = run(df.iloc[:mid])
    m2 = run(df.iloc[mid:])
    return full, m1, m2


def main() -> int:
    cfg = load_settings()
    qh = cfg.quant_hypotheses
    engine = BacktestEngine(cfg)
    rt = 2 * (cfg.backtest.commission_pct + cfg.backtest.slippage_pct)

    print("=" * 82)
    print("BARRIDO FINAL — Estacionalidad + Reversión RSI (neto de costos, histórico real)")
    print(f"Costo ida-vuelta base ≈ {rt:.2f}% + slippage ATR dinámico")
    print("=" * 82)

    # Deciders-fábrica por familia (reciben el df del tramo para indicadores causales).
    # Desde 2026-07-08 los tres deciders llevan stop ATR explícito + tp=None
    # ("dejar correr"): necesitan closes/atrs del tramo.
    def mk_hour(df):
        closes = df["close"].to_numpy(dtype=float)
        atrs = atr(df, cfg.risk.atr_period).to_numpy()
        return make_hour_seasonality_decider(
            closes, atrs, entry_open_hour=qh.seasonality_entry_open_hour_utc,
            hold_hours=qh.seasonality_hold_hours, atr_mult=qh.atr_stop_mult)

    def mk_dow(df):
        closes = df["close"].to_numpy(dtype=float)
        atrs = atr(df, cfg.risk.atr_period).to_numpy()
        return make_dow_decider(
            closes, atrs, entry_weekday=qh.dow_entry_weekday,
            hold_days=qh.dow_hold_days, atr_mult=qh.atr_stop_mult)

    def mk_rsi(df):
        closes = df["close"].to_numpy(dtype=float)
        rsi_vals = rsi(df["close"], qh.rsi_reversion_period).to_numpy()
        trend = sma(df["close"], qh.rsi_reversion_trend_sma).to_numpy()
        atrs = atr(df, cfg.risk.atr_period).to_numpy()
        return make_rsi_reversion_decider(
            closes, rsi_vals, trend, atrs,
            oversold=qh.rsi_reversion_oversold,
            overbought=qh.rsi_reversion_overbought, atr_mult=qh.atr_stop_mult)

    families = [
        (f"Hour-{qh.seasonality_entry_open_hour_utc}h", "1h", lambda a: load_parquet(a, "1h"), mk_hour),
        ("DoW-Mon", "1d", lambda a: resample(load_parquet(a, "1h"), "1D"), mk_dow),
        ("RSI-rev", "1h", lambda a: load_parquet(a, "1h"), mk_rsi),
    ]

    for label, tf, loader, mk in families:
        full_rows: list[Row] = []
        split_lines = []
        for asset in ASSETS:
            df = loader(asset)
            full, m1, m2 = _split_sharpe(engine, df, asset, tf, mk)
            full_rows.append(Row(label, asset, tf, full))
            stable = ("sí" if (m1.sharpe > 0 and m2.sharpe > 0)
                      else ("no" if (m1.sharpe < 0 and m2.sharpe < 0) else "mixto"))
            split_lines.append(
                f"| {asset} | {m1.sharpe:+.2f} | {m1.n_trades} | "
                f"{m2.sharpe:+.2f} | {m2.n_trades} | {stable} |")
        print(_table(f"{label} — full-sample (neto de costos)", full_rows))
        print(f"\n#### {label} — split OOS (Sharpe 1ª | 2ª)")
        print("\n| Activo | Sharpe 1ª | n1 | Sharpe 2ª | n2 | ¿estable? |")
        print("|" + "---|" * 6)
        print("\n".join(split_lines))
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
