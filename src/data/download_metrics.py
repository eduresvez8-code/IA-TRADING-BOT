"""Descarga métricas históricas de posicionamiento desde Binance Vision.

Qué resuelve: la API REST de Binance solo sirve ~30 días de open interest y
long/short ratio (futures/data/*), lo que hacía estas hipótesis no-backtesteables.
Binance Vision (data.binance.vision, el repositorio público de datos bulk)
publica el histórico COMPLETO a 5 minutos desde ~2021-12, un zip por día:

    data/futures/um/daily/metrics/{SYM}/{SYM}-metrics-{YYYY-MM-DD}.zip

Cada CSV trae: create_time, symbol, sum_open_interest, sum_open_interest_value,
count_toptrader_long_short_ratio, sum_toptrader_long_short_ratio,
count_long_short_ratio, sum_taker_long_short_vol_ratio.

Uso:
    uv run python -m src.data.download_metrics --start 2023-06-16 --end 2026-06-14
    uv run python -m src.data.download_metrics --symbol BTCUSDT --start 2024-01-01 --end 2024-12-31

Sin --symbol usa los símbolos de config/settings.yaml. Salida: un parquet 5m por
símbolo en storage.metrics_dir. Fuente pública, sin API key, $0.
"""

from __future__ import annotations

import argparse
import asyncio
import io
import logging
import zipfile
from datetime import date
from pathlib import Path

import httpx
import pandas as pd

from src.core.config import load_settings

log = logging.getLogger(__name__)

BASE_URL = "https://data.binance.vision/data/futures/um/daily/metrics"
# Concurrencia moderada: Vision es un CDN estático, pero no hay que martillarlo.
_CONCURRENCY = 16
_RETRIES = 3


def metrics_url(symbol: str, day: date) -> str:
    """URL del zip diario de métricas de un símbolo. Función pura → testeable."""
    return f"{BASE_URL}/{symbol}/{symbol}-metrics-{day:%Y-%m-%d}.zip"


def parse_metrics_zip(payload: bytes) -> pd.DataFrame:
    """Zip diario → DataFrame 5m tipado, con create_time en UTC.

    El CSV trae create_time como texto local del archivo ('YYYY-MM-DD HH:MM:SS');
    lo fijamos a UTC explícito para que el merge con las velas (UTC) no corra el
    dato 5 horas en silencio. format='mixed' tolera los días en que Binance
    cambió el formato del timestamp (pasó en 2024).
    """
    zf = zipfile.ZipFile(io.BytesIO(payload))
    with zf.open(zf.namelist()[0]) as f:
        df = pd.read_csv(f)
    df["create_time"] = pd.to_datetime(df["create_time"], utc=True, format="mixed")
    return df


def consolidate_days(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatena los días, deduplica por timestamp y ordena. Función pura.

    Vision a veces repite el registro de medianoche en el zip del día anterior
    y el del siguiente; sin drop_duplicates el resample 1h contaría doble.
    """
    df = pd.concat(frames, ignore_index=True)
    return (df.drop_duplicates("create_time")
              .sort_values("create_time")
              .reset_index(drop=True))


async def _fetch_day(client: httpx.AsyncClient, sem: asyncio.Semaphore,
                     symbol: str, day: date) -> pd.DataFrame | None:
    async with sem:
        for attempt in range(_RETRIES):
            try:
                r = await client.get(metrics_url(symbol, day))
                if r.status_code == 404:  # día sin datos (símbolo aún no listado)
                    return None
                r.raise_for_status()
                return parse_metrics_zip(r.content)
            except (httpx.HTTPError, zipfile.BadZipFile) as e:
                if attempt == _RETRIES - 1:
                    log.warning("%s %s: %s (agotados los reintentos)", symbol, day, e)
                    return None
                await asyncio.sleep(2.0 * (attempt + 1))
    return None


async def download_symbol_metrics(client: httpx.AsyncClient, symbol: str,
                                  start: date, end: date, out_dir: Path) -> Path | None:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{symbol}_metrics_5m.parquet"
    days = pd.date_range(start, end, freq="D").date
    sem = asyncio.Semaphore(_CONCURRENCY)
    results = await asyncio.gather(
        *(_fetch_day(client, sem, symbol, d) for d in days))
    frames = [f for f in results if f is not None]
    if not frames:
        log.warning("%s: ningún día disponible en [%s, %s]", symbol, start, end)
        return None
    df = consolidate_days(frames)
    df.to_parquet(out_path)
    missing = len(days) - len(frames)
    print(f"✓ {symbol}: {len(df):,} filas 5m "
          f"({df.create_time.min():%Y-%m-%d} → {df.create_time.max():%Y-%m-%d}), "
          f"{missing} días faltantes → {out_path}")
    return out_path


async def run(args: argparse.Namespace) -> None:
    settings = load_settings()
    symbols = [args.symbol] if args.symbol else settings.market.symbols
    out_dir = Path(settings.storage.metrics_dir)
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    async with httpx.AsyncClient(timeout=30) as client:
        for symbol in symbols:
            await download_symbol_metrics(client, symbol, start, end, out_dir)


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", help="ej. BTCUSDT (default: settings.yaml)")
    parser.add_argument("--start", required=True, help="YYYY-MM-DD (UTC)")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD (UTC, inclusive)")
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
