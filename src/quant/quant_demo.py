"""Demo del Quant Engine sobre datos históricos reales (Sprint 1).

Carga los Parquet descargados, calcula los tres indicadores y emite
la señal actual para cada símbolo y timeframe.

Uso:
    uv run python -m src.quant.quant_demo
"""

from pathlib import Path

import pandas as pd

from src.core.config import load_settings
from src.quant.indicators import atr, ema, rsi
from src.quant.strategy import compute_signal

CANDLES_DIR = Path("data/candles")
TAIL = 5  # filas a mostrar en la tabla de indicadores


def load_parquet(symbol: str, timeframe: str) -> pd.DataFrame:
    path = CANDLES_DIR / f"{symbol}_{timeframe}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"No existe {path} — ejecuta primero download_history.py")
    df = pd.read_parquet(path)
    df.sort_values("open_time", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def show_indicators(df: pd.DataFrame, symbol: str, timeframe: str) -> None:
    cfg = load_settings()
    q = cfg.quant

    close = df["close"]
    ema_fast = ema(close, q.ema_fast_period)
    ema_slow = ema(close, q.ema_slow_period)
    rsi_vals = rsi(close, q.rsi_period)
    atr_vals = atr(df, cfg.risk.atr_period)

    summary = pd.DataFrame(
        {
            f"EMA({q.ema_fast_period})": ema_fast,
            f"EMA({q.ema_slow_period})": ema_slow,
            f"RSI({q.rsi_period})": rsi_vals,
            f"ATR({cfg.risk.atr_period})": atr_vals,
            "close": close,
        }
    )

    print(f"\n{'='*60}")
    print(f"  {symbol} / {timeframe}  — últimas {TAIL} velas")
    print(f"{'='*60}")
    print(summary.tail(TAIL).to_string(float_format="{:.4f}".format))

    sig = compute_signal(df, symbol)
    if sig is None:
        print("  [!] Datos insuficientes para generar señal.")
    else:
        print(f"\n  Señal: score={sig.score:+.4f}  strategy={sig.strategy}")
        for k, v in sig.features.items():
            print(f"    {k}: {v:.4f}")


def main() -> None:
    cfg = load_settings()
    for symbol in cfg.market.symbols:
        for tf in (cfg.market.timeframe, cfg.market.htf_timeframe):
            try:
                df = load_parquet(symbol, tf)
                show_indicators(df, symbol, tf)
            except FileNotFoundError as e:
                print(f"[WARN] {e}")


if __name__ == "__main__":
    main()
