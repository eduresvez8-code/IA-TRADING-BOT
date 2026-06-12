"""Descarga histórico de velas (API pública de mainnet) a archivos Parquet.

Uso:
    uv run python -m src.data.download_history --days 365
    uv run python -m src.data.download_history --symbol BTCUSDT --timeframe 5m --days 30

Sin argumentos usa los símbolos y timeframes de config/settings.yaml.
"""

import argparse
import asyncio
import logging

from binance import AsyncClient

from src.core.config import load_settings
from src.data.binance_client import download_history
from src.data.storage import Storage


async def run(args: argparse.Namespace) -> None:
    settings = load_settings()
    symbols = [args.symbol] if args.symbol else settings.market.symbols
    timeframes = ([args.timeframe] if args.timeframe
                  else [settings.market.timeframe, settings.market.htf_timeframe])
    storage = Storage(settings.storage.db_path, settings.storage.candles_dir)

    # Sin API key: los datos de mercado de mainnet son públicos.
    client = await AsyncClient.create()
    try:
        for symbol in symbols:
            for tf in timeframes:
                df = await download_history(client, symbol, tf, args.days)
                path = storage.save_history_parquet(df, symbol, tf)
                print(f"✓ {symbol} {tf}: {len(df):,} velas "
                      f"({df.open_time.min():%Y-%m-%d} → {df.open_time.max():%Y-%m-%d}) "
                      f"→ {path}")
    finally:
        await client.close_connection()


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", help="ej. BTCUSDT (default: settings.yaml)")
    parser.add_argument("--timeframe", help="ej. 5m, 1h (default: settings.yaml)")
    parser.add_argument("--days", type=int, default=365)
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
