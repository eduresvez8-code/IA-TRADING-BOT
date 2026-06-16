"""Descarga funding rate y basis (premium) históricos a Parquet ($0, sin API key).

    uv run python -m src.data.download_funding

Usa `scan.symbols`, `scan.history_days` y `funding_edge.premium_interval` de la
config. Guarda en `storage.funding_dir`:
    {symbol}_funding.parquet          → [funding_time, funding_rate]   (8h)
    {symbol}_premium_{interval}.parquet → [open_time, premium_*]       (basis)
"""

import asyncio
from pathlib import Path

from src.core.config import load_settings
from src.data.funding_client import (
    fetch_funding_rate_history,
    fetch_premium_index_klines,
)


async def main() -> int:
    cfg = load_settings()
    fdir = Path(cfg.storage.funding_dir)
    fdir.mkdir(parents=True, exist_ok=True)
    days = cfg.scan.history_days
    interval = cfg.funding_edge.premium_interval

    for sym in cfg.scan.symbols:
        try:
            fr = await fetch_funding_rate_history(sym, days)
            fr.to_parquet(fdir / f"{sym}_funding.parquet", index=False)
            pr = await fetch_premium_index_klines(sym, interval, days)
            pr.to_parquet(fdir / f"{sym}_premium_{interval}.parquet", index=False)
            fr_rng = (f"{fr.funding_time.min():%Y-%m-%d}→{fr.funding_time.max():%Y-%m-%d}"
                      if len(fr) else "vacío")
            pr_rng = (f"{pr.open_time.min():%Y-%m-%d}→{pr.open_time.max():%Y-%m-%d}"
                      if len(pr) else "vacío")
            print(f"OK {sym}: funding {len(fr):,} ({fr_rng}) · premium {len(pr):,} ({pr_rng})",
                  flush=True)
        except Exception as e:
            print(f"ERR {sym}: {type(e).__name__}: {e}", flush=True)
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
