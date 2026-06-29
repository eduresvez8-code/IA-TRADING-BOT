"""Estabilidad fuera de muestra del sweep de medias móviles (split-sample).

    uv run python -m backtest.run_ma_split

Para CADA configuración del sweep (par × tipo × timeframe), corre el cruce en la
PRIMERA mitad del histórico y en la SEGUNDA por separado, promedia el Sharpe entre los
5 activos en cada mitad, y ordena por el Sharpe de la 2ª mitad (la "fuera de muestra").

Un edge real sobrevive en AMBAS mitades. Si una config solo brilla en la 1ª (el tramo
alcista 2023-24) y se cae en la 2ª, es trend-following halagado por el régimen, NO edge.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.core.config import load_settings
from src.quant.indicators import atr
from backtest.engine import BacktestEngine
from backtest.quant_hypotheses import make_macross_decider, moving_average
from backtest.run_ma_sweep import _frame
from backtest.run_quant_hypotheses import ASSETS


def _sharpe(engine, cfg, df: pd.DataFrame, asset: str, tf: str,
            fast_p: int, slow_p: int, kind: str) -> float:
    df = df.reset_index(drop=True)
    closes = df["close"].to_numpy(dtype=float)
    atrs = atr(df, cfg.risk.atr_period).to_numpy()
    fast = moving_average(closes, fast_p, kind)
    slow = moving_average(closes, slow_p, kind)
    decider = make_macross_decider(
        closes, atrs, fast, slow,
        atr_mult=cfg.quant_hypotheses.atr_stop_mult,
        allow_short=cfg.quant_hypotheses.ma_cross_allow_short)
    return engine.run(df, asset, tf, decider=decider).metrics.sharpe


def main() -> int:
    cfg = load_settings()
    qh = cfg.quant_hypotheses
    engine = BacktestEngine(cfg)

    print("=" * 84)
    print("CRUCES DE MEDIAS — estabilidad fuera de muestra (Sharpe medio 1ª | 2ª mitad)")
    print("Ordenado por la 2ª mitad (OOS). Edge real = positivo en AMBAS, no solo en la 1ª.")
    print("=" * 84)

    results = []
    for tf in qh.ma_cross_timeframes:
        for kind in qh.ma_cross_types:
            for fast_p, slow_p in qh.ma_cross_pairs:
                s1s, s2s = [], []
                for asset in ASSETS:
                    df = _frame(asset, tf)
                    mid = len(df) // 2
                    s1s.append(_sharpe(engine, cfg, df.iloc[:mid], asset, tf, fast_p, slow_p, kind))
                    s2s.append(_sharpe(engine, cfg, df.iloc[mid:], asset, tf, fast_p, slow_p, kind))
                results.append({
                    "label": f"{kind.upper()}{fast_p}/{slow_p}@{tf}",
                    "s1": float(np.mean(s1s)), "s2": float(np.mean(s2s)),
                    "pos2": sum(1 for s in s2s if s > 0),
                })

    results.sort(key=lambda d: d["s2"], reverse=True)
    head = ["Config", "Sharpe 1ª", "Sharpe 2ª (OOS)", "Pos 2ª/5", "¿degrada?"]
    print("\n| " + " | ".join(head) + " |")
    print("|" + "---|" * len(head))
    for d in results:
        degrada = "no" if d["s2"] >= d["s1"] - 0.1 else f"-{d['s1'] - d['s2']:.2f}"
        print(f"| {d['label']} | {d['s1']:+.2f} | {d['s2']:+.2f} | {d['pos2']}/5 | {degrada} |")

    n_oos_pos = sum(1 for d in results if d["s2"] > 0.2)
    print(f"\nConfigs con Sharpe OOS > 0.2 (umbral mínimo de interés): {n_oos_pos}/{len(results)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
