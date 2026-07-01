"""H5 — Lead-lag cascade (Perplexity rank 4/5, Guo 2024): ¿el retorno de BTC a N
horas predice la dirección del altcoin, NETO de costos y fuera de muestra?

    uv run python -m backtest.run_leadlag

Dos tablas:
  (1) Full-sample — métricas netas de costos por (target, lag).
  (2) Split OOS  — Sharpe 1ª mitad | 2ª mitad. Un edge real sobrevive en AMBAS;
      si solo en una → régimen (un tramo alcista), no edge (misma lección que TSMOM).

Causal: leader_ret_lag[i] usa closes hasta la vela i; el motor ejecuta en la apertura
de i+1 (sin look-ahead), equivalente al .shift(1) del JSON. Cero Hardcoding: leader,
targets, lags, SMA de régimen y dirección viven en config.quant_hypotheses.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.core.config import load_settings
from src.quant.indicators import atr
from backtest.engine import BacktestEngine
from backtest.quant_hypotheses import make_leadlag_decider, moving_average
from backtest.run_backtest import load_parquet
from backtest.run_quant_hypotheses import Row, _table


def _merge(leader_df: pd.DataFrame, target_df: pd.DataFrame) -> pd.DataFrame:
    """Alinea líder y target por open_time (inner join) → un frame con OHLCV del
    target más `leader_close`. El inner join descarta velas sin pareja (robusto)."""
    lead = leader_df[["open_time", "close"]].rename(columns={"close": "leader_close"})
    return target_df.merge(lead, on="open_time", how="inner").reset_index(drop=True)


def _run(engine, cfg, df: pd.DataFrame, asset: str, lag: int):
    df = df.reset_index(drop=True)
    closes = df["close"].to_numpy(dtype=float)
    leader_close = df["leader_close"].to_numpy(dtype=float)
    atrs = atr(df, cfg.risk.atr_period).to_numpy()
    qh = cfg.quant_hypotheses

    leader_ret_lag = np.full(len(closes), np.nan)
    if lag < len(closes):
        leader_ret_lag[lag:] = leader_close[lag:] / leader_close[:-lag] - 1.0
    target_sma = moving_average(closes, qh.leadlag_regime_sma, "sma")
    leader_sma = moving_average(leader_close, qh.leadlag_regime_sma, "sma")

    decider = make_leadlag_decider(
        closes, atrs, leader_ret_lag, target_sma, leader_close, leader_sma,
        atr_mult=qh.atr_stop_mult, allow_short=qh.leadlag_allow_short)
    return engine.run(df, asset, "1h", decider=decider)


def main() -> int:
    cfg = load_settings()
    engine = BacktestEngine(cfg)
    qh = cfg.quant_hypotheses
    leader_df = load_parquet(qh.leadlag_leader, "1h")

    rt_cost = 2 * (cfg.backtest.commission_pct + cfg.backtest.slippage_pct)
    print("=" * 82)
    print("H5 — LEAD-LAG CASCADE (BTC lidera) — neto de costos, histórico real 1h")
    print(f"Líder: {qh.leadlag_leader} | targets: {qh.leadlag_target_assets} | "
          f"lags: {qh.leadlag_lag_hours_grid}h | régimen SMA{qh.leadlag_regime_sma} "
          f"(ambos de acuerdo) | costo ida-vuelta base ≈ {rt_cost:.2f}%")
    print("=" * 82)

    full_rows: list[Row] = []
    for asset in qh.leadlag_target_assets:
        merged = _merge(leader_df, load_parquet(asset, "1h"))
        for lag in qh.leadlag_lag_hours_grid:
            res = _run(engine, cfg, merged, asset, lag)
            full_rows.append(Row(f"LeadLag-{lag}h", asset, "1h", res))
    print(_table("Full-sample (neto de costos)", full_rows))

    print("\n\n### Split OOS — Sharpe 1ª mitad | 2ª mitad (edge real sobrevive en AMBAS)")
    header = ["Lag", "Target", "Sharpe 1ª", "n1", "Sharpe 2ª", "n2", "¿estable?"]
    print("\n| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for asset in qh.leadlag_target_assets:
        merged = _merge(leader_df, load_parquet(asset, "1h"))
        mid = len(merged) // 2
        for lag in qh.leadlag_lag_hours_grid:
            r1 = _run(engine, cfg, merged.iloc[:mid], asset, lag).metrics
            r2 = _run(engine, cfg, merged.iloc[mid:], asset, lag).metrics
            s1, s2 = r1.sharpe, r2.sharpe
            stable = ("sí" if (s1 > 0 and s2 > 0)
                      else ("no" if (s1 < 0 and s2 < 0) else "mixto"))
            print(f"| {lag}h | {asset} | {s1:+.2f} | {r1.n_trades} | "
                  f"{s2:+.2f} | {r2.n_trades} | {stable} |")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
