"""Corre el backtester sobre el histórico real descargado (Parquet) y reporta.

Uso:
    uv run python -m backtest.run_backtest            # todos los símbolos, tf base
    uv run python -m backtest.run_backtest --htf      # usa el timeframe superior (1h)

Carga las velas de data/candles/, simula la estrategia ema_cross_rsi con costos
y escribe un reporte en backtest/reports/.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.core.config import load_settings
from backtest.engine import BacktestEngine
from backtest.report import format_result_markdown, write_report

CANDLES_DIR = Path("data/candles")
REPORTS_DIR = Path("backtest/reports")


def load_parquet(symbol: str, timeframe: str) -> pd.DataFrame:
    path = CANDLES_DIR / f"{symbol}_{timeframe}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"No existe {path} — ejecuta primero src.data.download_history"
        )
    df = pd.read_parquet(path)
    df.sort_values("open_time", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest sobre histórico Parquet")
    parser.add_argument("--htf", action="store_true",
                        help="usar el timeframe superior (htf) en vez del base")
    args = parser.parse_args()

    cfg = load_settings()
    timeframe = cfg.market.htf_timeframe if args.htf else cfg.market.timeframe
    engine = BacktestEngine(cfg)

    results = []
    for symbol in cfg.market.symbols:
        try:
            df = load_parquet(symbol, timeframe)
        except FileNotFoundError as e:
            print(f"[WARN] {e}")
            continue
        print(f"[..] Backtest {symbol} {timeframe}: {len(df):,} velas")
        result = engine.run(df, symbol, timeframe)
        results.append(result)
        print(format_result_markdown(result))

    if not results:
        print("[!] Sin datos para backtestear.")
        return

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    md_path = write_report(results, REPORTS_DIR, stamp)
    print(f"\n[OK] Reporte escrito en {md_path}")


if __name__ == "__main__":
    main()
