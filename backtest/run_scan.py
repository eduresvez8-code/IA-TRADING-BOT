"""Escáner multi-activo de arquetipos: matriz [Activo × Arquetipo] sobre 3 años.

    uv run python -m backtest.run_scan          # timeframe base (5m)
    uv run python -m backtest.run_scan --htf    # timeframe superior (1h)
    uv run python -m backtest.run_scan --both    # ambos timeframes

Corre los tres arquetipos (tendencia / reversión / ruptura) sobre cada símbolo de
`scan.symbols`, con riesgo (`risk_per_trade_pct`) y costos (comisión + slippage)
FIJOS de config — comparación limpia. Por cada combo imprime retorno, PF, win
rate, max drawdown, expectancy y nº de trades, y un walk-forward de `scan.
walk_forward_folds` tramos: el juez de si el edge es consistente o un tramo
afortunado. Marca como candidato a "Ancla Cuántica" (★) el combo con
PF>`edge_profit_factor_min`, expectancy>0 y TODOS los tramos rentables.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

from src.core.config import Settings, load_settings
from backtest.archetypes import ARCHETYPE_LABELS, ARCHETYPES, make_decider
from backtest.engine import BacktestEngine, BacktestResult
from backtest.run_backtest import load_parquet


def run_archetype(df, symbol, tf, archetype, cfg: Settings) -> BacktestResult:
    decider = make_decider(archetype, df, cfg, allow_short=cfg.backtest.allow_short)
    return BacktestEngine(cfg).run(df, symbol, tf, decider=decider)


def walk_forward_archetype(df, symbol, tf, archetype, cfg: Settings) -> list[BacktestResult]:
    """Backtestea el arquetipo en `n_folds` tramos contiguos (recalcula indicadores
    dentro de cada tramo → sin fuga de información entre periodos)."""
    n_folds = cfg.scan.walk_forward_folds
    df = df.reset_index(drop=True)
    n = len(df)
    size = n // n_folds
    out = []
    for k in range(n_folds):
        lo = k * size
        hi = (k + 1) * size if k < n_folds - 1 else n
        fold = df.iloc[lo:hi].reset_index(drop=True)
        out.append(run_archetype(fold, symbol, tf, archetype, cfg))
    return out


@dataclass
class ScanRow:
    symbol: str
    archetype: str
    total_return: float
    profit_factor: float
    win_rate: float
    max_drawdown: float
    expectancy: float
    n_trades: int
    wf_folds_positive: int
    wf_n_folds: int
    wf_min_pf: float
    is_edge: bool


def evaluate(symbol, tf, archetype, cfg: Settings) -> ScanRow:
    df = load_parquet(symbol, tf)
    res = run_archetype(df, symbol, tf, archetype, cfg)
    m = res.metrics
    folds = walk_forward_archetype(df, symbol, tf, archetype, cfg)
    positive = sum(1 for f in folds if f.final_equity > f.initial_capital)
    min_pf = min(f.metrics.profit_factor for f in folds)
    is_edge = (
        m.profit_factor > cfg.scan.edge_profit_factor_min
        and m.expectancy > 0
        and positive == len(folds)
    )
    return ScanRow(
        symbol=symbol, archetype=archetype,
        total_return=m.total_return, profit_factor=m.profit_factor,
        win_rate=m.win_rate, max_drawdown=m.max_drawdown,
        expectancy=m.expectancy, n_trades=m.n_trades,
        wf_folds_positive=positive, wf_n_folds=len(folds), wf_min_pf=min_pf,
        is_edge=is_edge,
    )


def _pf(x: float) -> str:
    return "∞" if math.isinf(x) else f"{x:.2f}"


_ARCH_SHORT = {"trend": "Tendencia", "mean_reversion": "Reversión", "breakout": "Ruptura"}


def format_matrix(rows: list[ScanRow], tf: str) -> str:
    lines = [
        f"\n### Matriz de arquetipos · timeframe {tf}\n",
        "| Símbolo | Arquetipo | Retorno % | PF | Win % | Max DD % | Expectancy | Trades | WF tramos+ | Edge |",
        "|---|---|---:|---:|---:|---:|---:|---:|:--:|:--:|",
    ]
    for r in rows:
        lines.append(
            f"| {r.symbol} | {_ARCH_SHORT[r.archetype]} | {r.total_return * 100:+.2f} "
            f"| {_pf(r.profit_factor)} | {r.win_rate * 100:.1f} "
            f"| {r.max_drawdown * 100:.2f} | {r.expectancy:+.2f} | {r.n_trades} "
            f"| {r.wf_folds_positive}/{r.wf_n_folds} | {'★' if r.is_edge else '·'} |"
        )
    return "\n".join(lines)


def scan_timeframe(cfg: Settings, tf: str) -> list[ScanRow]:
    rows: list[ScanRow] = []
    for symbol in cfg.scan.symbols:
        try:
            load_parquet(symbol, tf)
        except FileNotFoundError as e:
            print(f"[WARN] {e}")
            continue
        for archetype in ARCHETYPES:
            rows.append(evaluate(symbol, tf, archetype, cfg))
    return rows


def main(*, timeframes: list[str]) -> int:
    cfg = load_settings()
    print("Escáner multi-activo de arquetipos — laboratorio de estrategia.")
    print(f"Universo: {', '.join(cfg.scan.symbols)}")
    print(f"Riesgo fijo: {cfg.risk.risk_per_trade_pct}% · comisión {cfg.backtest.commission_pct}%/lado "
          f"· slippage {cfg.backtest.slippage_pct}%/lado")
    print("Arquetipos: " + " | ".join(f"{k}={v}" for k, v in ARCHETYPE_LABELS.items()))
    print(f"Edge ★ = PF>{cfg.scan.edge_profit_factor_min}, expectancy>0 y "
          f"{cfg.scan.walk_forward_folds}/{cfg.scan.walk_forward_folds} tramos rentables.")

    all_edges: list[tuple[str, ScanRow]] = []
    for tf in timeframes:
        rows = scan_timeframe(cfg, tf)
        print(format_matrix(rows, tf))
        all_edges += [(tf, r) for r in rows if r.is_edge]

    print("\n## Candidatos a Ancla Cuántica (edge consistente)")
    if not all_edges:
        print("Ninguna combinación [Activo × Arquetipo] supera el umbral de edge "
              "con consistencia walk-forward. La eficiencia de los majores se mantiene.")
    else:
        for tf, r in sorted(all_edges, key=lambda x: x[1].profit_factor, reverse=True):
            print(f"  ★ {r.symbol} · {_ARCH_SHORT[r.archetype]} · {tf} → "
                  f"PF {_pf(r.profit_factor)}, ret {r.total_return * 100:+.2f}%, "
                  f"expectancy {r.expectancy:+.2f}, WF {r.wf_folds_positive}/{r.wf_n_folds}, "
                  f"min PF tramo {_pf(r.wf_min_pf)}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Escáner multi-activo de arquetipos")
    parser.add_argument("--htf", action="store_true", help="usar el timeframe superior")
    parser.add_argument("--both", action="store_true", help="ambos timeframes")
    args = parser.parse_args()
    cfg0 = load_settings()
    if args.both:
        tfs = [cfg0.market.htf_timeframe, cfg0.market.timeframe]
    elif args.htf:
        tfs = [cfg0.market.htf_timeframe]
    else:
        tfs = [cfg0.market.timeframe]
    raise SystemExit(main(timeframes=tfs))
