"""Runner del estudio de posicionamiento: flujo taker + OI + long/short ratio.

    uv run python -m backtest.run_positioning [--klines-dir DIR] [--metrics-dir DIR]

Requiere:
  - Velas 1h CON taker_buy_base/quote_volume (las de data/candles NO las traen;
    ver informe 2026-07-06 — se descargan del fapi público).
  - Métricas 5m de Binance Vision: `uv run python -m src.data.download_metrics
    --start 2023-06-16 --end 2026-06-14`.

Protocolo anti-selección (lección del artefacto run_ma_split, 2026-07-06):
  1. El grid COMPLETO se reporta con su Sharpe de TRAIN y de TEST.
  2. La "elegida" es la mejor por TRAIN. El TEST se imprime para todas, pero la
     decisión ya está tomada: JAMÁS reordenar por la columna de test.
Resultado de la sesión 2026-07-06: ninguna config pasó el listón (ver informe).
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd

from src.core.config import load_settings
from backtest.positioning import (
    annualized_sharpe,
    net_strategy_returns,
    rolling_zscore,
    split_by_date,
    taker_imbalance,
    threshold_positions,
)

_BARS_PER_YEAR_1H = 24 * 365

_METRIC_COLS = ["sum_open_interest", "count_long_short_ratio",
                "sum_toptrader_long_short_ratio", "sum_taker_long_short_vol_ratio"]


def load_klines_1h(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path).set_index("open_time").sort_index()
    df["ret"] = df["close"].pct_change()
    return df


def load_metrics_1h(path: Path, index: pd.DatetimeIndex, ffill_limit: int) -> pd.DataFrame:
    """5m → 1h causal: el valor de la barra T es el último registro < T+1h."""
    m = pd.read_parquet(path).set_index("create_time").sort_index()
    out = m[_METRIC_COLS].resample("1h", label="left", closed="left").last()
    return out.reindex(index).ffill(limit=ffill_limit)


def main() -> int:
    cfg = load_settings()
    pr = cfg.positioning_research
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--klines-dir", default=None,
                        help="dir con {SYM}.parquet 1h incl. taker_buy_base")
    parser.add_argument("--metrics-dir", default=cfg.storage.metrics_dir)
    args = parser.parse_args()

    if args.klines_dir is None:
        raise SystemExit("--klines-dir es obligatorio: las velas de data/candles "
                         "no traen taker_buy_base (ver docstring)")

    split_ts = pd.Timestamp(pr.train_test_split_date)
    cost = pr.cost_per_side_pct / 100.0
    symbols = cfg.market.symbols

    data: dict[str, pd.DataFrame] = {}
    feats: dict[str, dict[str, pd.Series]] = {}
    for s in symbols:
        px = load_klines_1h(Path(args.klines_dir) / f"{s}.parquet")
        m = load_metrics_1h(Path(args.metrics_dir) / f"{s}_metrics_5m.parquet",
                            px.index, pr.metrics_ffill_limit_bars)
        imb = taker_imbalance(px["volume"], px["taker_buy_base"])
        f: dict[str, pd.Series] = {}
        for k in pr.imbalance_ma_bars_grid:
            f[f"imb_ma{k}"] = rolling_zscore(imb.rolling(k).mean(), pr.zscore_window_bars)
        f["doi24_z"] = rolling_zscore(m["sum_open_interest"].pct_change(24),
                                      pr.zscore_window_bars)
        f["glsr_z"] = rolling_zscore(m["count_long_short_ratio"], pr.zscore_window_bars)
        f["tlsr_z"] = rolling_zscore(m["sum_toptrader_long_short_ratio"],
                                     pr.zscore_window_bars)
        f["smart_dumb"] = f["tlsr_z"] - f["glsr_z"]
        data[s], feats[s] = px, f

    fnames = list(feats[symbols[0]].keys())
    print(f"{'config':26s} {'ShTRAIN':>8} {'ShTEST':>8}")
    results: list[tuple[str, float, float]] = []
    for fname in fnames:
        for direction, dlab in ((+1, "mom"), (-1, "con")):
            for th in pr.entry_threshold_grid:
                tr_parts, te_parts = [], []
                for s in symbols:
                    pos = threshold_positions(feats[s][fname], th, direction)
                    pnl = net_strategy_returns(pos, data[s]["ret"], cost)
                    tr, te = split_by_date(pnl, split_ts)
                    tr_parts.append(tr)
                    te_parts.append(te)
                tr = pd.concat(tr_parts, axis=1).mean(axis=1).dropna()
                te = pd.concat(te_parts, axis=1).mean(axis=1).dropna()
                results.append((f"{fname}|{dlab}|z{th}",
                                annualized_sharpe(tr, _BARS_PER_YEAR_1H),
                                annualized_sharpe(te, _BARS_PER_YEAR_1H)))
    # Orden por TRAIN (la única columna legal para elegir).
    results.sort(key=lambda r: -(r[1] if not math.isnan(r[1]) else -99.0))
    for cfg_name, shtr, shte in results:
        print(f"{cfg_name:26s} {shtr:>8.2f} {shte:>8.2f}")
    best = results[0]
    print(f"\n>> Elegida EN TRAIN: {best[0]} → Sharpe train {best[1]:.2f}, "
          f"test {best[2]:.2f} (listón: >0.5 en ambos)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
