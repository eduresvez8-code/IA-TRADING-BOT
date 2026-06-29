"""Test de estabilidad fuera de muestra del TSMOM diario (split-sample).

    uv run python -m backtest.run_tsmom_split

Para cada activo y cada lookback del grid, corre el TSMOM en la PRIMERA mitad del
histórico y en la SEGUNDA por separado, y compara el Sharpe. Un edge real debe
sobrevivir en ambas mitades; si solo aparece en una, es un artefacto de régimen
(p.ej. un único tramo alcista) y NO se puede confiar en él hacia adelante.

NO es un walk-forward formal (solo 2 tramos), pero es el filtro mínimo de honestidad
antes de tomarse en serio cualquier Sharpe de una sola pasada.
"""

from __future__ import annotations

import pandas as pd

from src.core.config import load_settings
from src.quant.indicators import atr
from backtest.engine import BacktestEngine
from backtest.quant_hypotheses import make_tsmom_decider
from backtest.run_backtest import load_parquet
from backtest.run_quant_hypotheses import ASSETS, resample


def _sharpe_on(engine, cfg, df: pd.DataFrame, asset: str, lookback: int) -> tuple[float, int]:
    df = df.reset_index(drop=True)
    closes = df["close"].to_numpy(dtype=float)
    atrs = atr(df, cfg.risk.atr_period).to_numpy()
    decider = make_tsmom_decider(closes, atrs, lookback, cfg.quant_hypotheses.atr_stop_mult)
    res = engine.run(df, asset, "1d", decider=decider)
    return res.metrics.sharpe, res.metrics.n_trades


def main() -> int:
    cfg = load_settings()
    engine = BacktestEngine(cfg)
    grid = cfg.quant_hypotheses.tsmom_lookback_days_grid

    print("=" * 78)
    print("TSMOM diario — estabilidad fuera de muestra (Sharpe 1ª mitad | 2ª mitad)")
    print("Un edge real sobrevive en AMBAS mitades; si solo en una → régimen, no edge.")
    print("=" * 78)

    header = ["Lookback", "Activo", "Sharpe 1ª", "n1", "Sharpe 2ª", "n2", "¿estable?"]
    print("\n| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))

    for lookback in grid:
        for asset in ASSETS:
            daily = resample(load_parquet(asset, "1h"), "1D")
            mid = len(daily) // 2
            first, second = daily.iloc[:mid], daily.iloc[mid:]
            s1, n1 = _sharpe_on(engine, cfg, first, asset, lookback)
            s2, n2 = _sharpe_on(engine, cfg, second, asset, lookback)
            stable = "sí" if (s1 > 0 and s2 > 0) else ("no" if (s1 < 0 and s2 < 0) else "mixto")
            print(f"| {lookback}d | {asset} | {s1:+.2f} | {n1} | {s2:+.2f} | {n2} | {stable} |")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
