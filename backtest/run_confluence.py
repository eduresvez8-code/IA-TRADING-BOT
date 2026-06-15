"""A/B del sentimiento sobre el histórico real: quant solo vs quant + sentimiento.

    uv run python -m backtest.run_confluence          # timeframe base (5m)
    uv run python -m backtest.run_confluence --htf    # timeframe superior (1h)

Carga las velas de data/candles/ y los scores de sentimiento de SQLite (los que
hayas acumulado con src.sentiment.build_history). Corre dos veces por símbolo —
sin y con sentimiento, por la MISMA ruta de confluencia — y un walk-forward de
robustez. Si aún no hay corpus de sentimiento, ambos brazos coinciden (baseline).
"""

from __future__ import annotations

import argparse
import asyncio

from src.core.config import load_settings
from src.data.storage import Storage
from backtest.confluence import events_from_rows, run_confluence, walk_forward
from backtest.report import format_result_markdown
from backtest.run_backtest import load_parquet


async def main(*, htf: bool) -> int:
    cfg = load_settings()
    tf = cfg.market.htf_timeframe if htf else cfg.market.timeframe
    storage = await Storage(cfg.storage.db_path, cfg.storage.candles_dir).init()
    try:
        for symbol in cfg.market.symbols:
            try:
                df = load_parquet(symbol, tf)
            except FileNotFoundError as e:
                print(f"[WARN] {e}")
                continue

            start_ms = int(df["open_time"].iloc[0].timestamp() * 1000)
            end_ms = int(df["open_time"].iloc[-1].timestamp() * 1000)
            rows = await storage.get_sentiment_scores(since_ms=start_ms, until_ms=end_ms)
            events = events_from_rows(rows)

            print(f"\n{'=' * 70}\n{symbol} {tf}: {len(df):,} velas · "
                  f"{len(events)} scores de sentimiento en el rango\n{'=' * 70}")

            base = run_confluence(df, symbol, tf, None, cfg)
            treat = run_confluence(df, symbol, tf, events or None, cfg)
            print("── Baseline (quant solo, vía confluencia) ──")
            print(format_result_markdown(base))
            print("── Con sentimiento ──")
            print(format_result_markdown(treat))

            folds = walk_forward(df, symbol, tf, n_folds=4,
                                 sentiment_events=events or None, settings=cfg)
            print("Walk-forward (equity final por tramo):")
            for lo, hi, res in folds:
                print(f"  velas [{lo:>6},{hi:>6}) → {res.final_equity:,.2f} USDT "
                      f"({len(res.trades)} trades)")
            if not events:
                print("\n  ⚠ Sin scores de sentimiento: ejecuta src.sentiment.build_history "
                      "para poblar el corpus y ver el brazo 'con sentimiento' de verdad.")
    finally:
        await storage.close()
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A/B de sentimiento sobre histórico")
    parser.add_argument("--htf", action="store_true", help="usar el timeframe superior")
    args = parser.parse_args()
    raise SystemExit(asyncio.run(main(htf=args.htf)))
