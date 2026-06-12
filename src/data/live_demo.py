"""Demo del Sprint 1: velas cerradas en vivo → consola + SQLite.

Uso:
    uv run python -m src.data.live_demo --timeframe 1m --count 3

Termina solo tras recibir `count` velas cerradas (con 1m, ~1 min por vela).
"""

import argparse
import asyncio
import logging

from binance import AsyncClient

from src.core.config import load_settings
from src.core.models import Candle
from src.data.binance_client import stream_candles
from src.data.storage import Storage


async def run(args: argparse.Namespace) -> None:
    settings = load_settings()
    storage = await Storage(settings.storage.db_path,
                            settings.storage.candles_dir).init()
    client = await AsyncClient.create()
    received = 0
    done = asyncio.Event()

    async def on_candle(c: Candle) -> None:
        nonlocal received
        await storage.save_candle(c)
        print(f"✓ vela cerrada {c.symbol} {c.timeframe} "
              f"O={c.open:,.2f} H={c.high:,.2f} L={c.low:,.2f} C={c.close:,.2f} "
              f"vol={c.volume:,.3f} @ {c.open_time:%H:%M} UTC → guardada en SQLite")
        received += 1
        if received >= args.count:
            done.set()

    print(f"Escuchando {args.symbol} {args.timeframe} "
          f"(esperando {args.count} velas cerradas)...")
    task = asyncio.create_task(
        stream_candles(client, args.symbol, args.timeframe, on_candle))
    try:
        await done.wait()
    finally:
        task.cancel()
        stored = await storage.get_candles(args.symbol, args.timeframe)
        print(f"Velas de {args.symbol} {args.timeframe} en la BD: {len(stored)}")
        await storage.close()
        await client.close_connection()


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--timeframe", default="1m",
                        help="1m para feedback rápido en la demo")
    parser.add_argument("--count", type=int, default=3)
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
