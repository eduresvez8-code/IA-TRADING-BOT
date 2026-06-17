"""Descarga el histórico diario del universo de perps a un único Parquet ($0).

    uv run python -m src.data.download_universe

Lista los perpetuos USDT (TRADING) y baja sus velas diarias de los últimos
`cross_sectional.history_days`. Guarda un panel largo en
`storage.universe_dir/daily.parquet`: [symbol, open_time, close, quote_volume].
Descarta activos con menos de `cross_sectional.min_history_days` velas.
"""

import asyncio
from pathlib import Path

import httpx
import pandas as pd

from src.core.config import load_settings
from src.data.universe_client import fetch_daily_klines, fetch_perp_symbols


async def main() -> int:
    cfg = load_settings()
    xs = cfg.cross_sectional
    out_dir = Path(cfg.storage.universe_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    async with httpx.AsyncClient(timeout=30.0) as client:
        symbols = await fetch_perp_symbols(client)
        print(f"universo: {len(symbols)} perps USDT TRADING. Descargando diarios "
              f"({xs.history_days} días)...", flush=True)
        frames = []
        kept = skipped = 0
        for i, sym in enumerate(symbols):
            try:
                df = await fetch_daily_klines(sym, xs.history_days, client=client)
            except Exception as e:
                print(f"  ERR {sym}: {type(e).__name__}", flush=True)
                skipped += 1
                continue
            if len(df) < xs.min_history_days:
                skipped += 1
            else:
                df.insert(0, "symbol", sym)
                frames.append(df)
                kept += 1
            if (i + 1) % 100 == 0:
                print(f"  ...{i + 1}/{len(symbols)}", flush=True)
            await asyncio.sleep(0.03)

    panel = pd.concat(frames, ignore_index=True)
    path = out_dir / "daily.parquet"
    panel.to_parquet(path, index=False)
    print(f"✓ {kept} activos guardados ({skipped} descartados por histórico corto) "
          f"→ {path}")
    print(f"  rango: {panel.open_time.min():%Y-%m-%d} → {panel.open_time.max():%Y-%m-%d} "
          f"· {len(panel):,} filas")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
