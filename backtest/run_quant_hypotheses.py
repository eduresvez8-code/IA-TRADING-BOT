"""Backtest de las hipótesis quant de Perplexity sobre el histórico real.

    uv run python -m backtest.run_quant_hypotheses

Corre, NETO de costos (comisión + slippage del motor), las tres familias
backtesteables sobre los 5 perps (BTC/ETH/SOL/XRP/BNB), ~3 años (2023-06→2026-06):

    H1  TSMOM diario        (1d)  · barrido de lookback del grid de config
    H2  Funding extremo     (4h)
    H3  Ruptura Donchian    (4h)  · SIN gate de OI (no backtesteable)

Rellena con números REALES las métricas que Perplexity devolvió en null: Sharpe,
profit factor, win rate, nº trades, trades/mes, drag de comisiones y CAGR neto.
Las hipótesis de POSICIONAMIENTO (open interest / long-short ratio) NO se corren:
Binance solo sirve ~30 días de ese dato gratis → no son verificables (se anota).
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.core.config import load_settings
from src.quant.indicators import atr
from backtest.engine import BacktestEngine
from backtest.metrics import bars_per_year
from backtest.quant_hypotheses import (
    align_funding_to_bars,
    annualize_funding_pct,
    daily_ma_on_bars,
    make_donchian_decider,
    make_funding_decider,
    make_tsmom_decider,
)
from backtest.run_backtest import load_parquet

# Universo del research = el universo que opera el bot (market.symbols) y rutas de
# datos desde storage. Antes eran literales duplicados aquí (violación de Cero
# Hardcoding: "nunca hardcodear símbolos"); ahora una sola fuente de verdad en
# settings.yaml. Los demás runners (run_ma_sweep/split, run_tsmom_split,
# run_seasonality_reversion) importan ASSETS de aquí.
_SETTINGS = load_settings()
ASSETS = list(_SETTINGS.market.symbols)
FUNDING_DIR = Path(_SETTINGS.storage.funding_dir)
REPORTS_DIR = Path("backtest/reports")

_AGG = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}


def resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Reagrupa velas 1h a un timeframe superior ('4h', '1D'). open_time = inicio."""
    out = (df.set_index("open_time").resample(rule).agg(_AGG).dropna().reset_index())
    return out


def load_funding(symbol: str) -> pd.DataFrame:
    return pd.read_parquet(FUNDING_DIR / f"{symbol}_funding.parquet")


class Row:
    """Una fila de resultados lista para tabular."""

    def __init__(self, label: str, asset: str, tf: str, res):
        m = res.metrics
        years = max(len(res.equity_curve) / bars_per_year(tf), 1e-9)
        commissions = sum(t.commission for t in res.trades)
        self.label = label
        self.asset = asset
        self.tf = tf
        self.n_trades = m.n_trades
        self.trades_per_month = m.n_trades / (years * 12)
        self.win_rate = m.win_rate * 100
        self.profit_factor = m.profit_factor
        self.sharpe = m.sharpe
        self.sortino = m.sortino
        self.cagr = m.cagr * 100
        self.max_dd = m.max_drawdown * 100
        self.exposure = m.exposure * 100
        self.total_return = m.total_return * 100
        self.cost_drag = commissions / res.initial_capital / years * 100

    def cells(self) -> list[str]:
        pf = "∞" if self.profit_factor == float("inf") else f"{self.profit_factor:.2f}"
        return [
            self.label, self.asset, f"{self.n_trades:d}", f"{self.trades_per_month:.1f}",
            f"{self.win_rate:.0f}%", pf, f"{self.sharpe:+.2f}",
            f"{self.cagr:+.1f}%", f"{self.max_dd:.0f}%", f"{self.cost_drag:.1f}%",
        ]


_HEADERS = ["Estrategia", "Activo", "Trades", "T/mes", "Win", "PF", "Sharpe",
            "CAGR", "MaxDD", "Drag"]


def _table(title: str, rows: list[Row]) -> str:
    lines = [f"\n### {title}", "", "| " + " | ".join(_HEADERS) + " |",
             "|" + "---|" * len(_HEADERS)]
    for r in rows:
        lines.append("| " + " | ".join(r.cells()) + " |")
    return "\n".join(lines)


def run_tsmom(cfg, engine) -> list[Row]:
    qh = cfg.quant_hypotheses
    rows: list[Row] = []
    for lookback in qh.tsmom_lookback_days_grid:
        for asset in ASSETS:
            df = resample(load_parquet(asset, "1h"), "1D")
            closes = df["close"].to_numpy(dtype=float)
            atrs = atr(df, cfg.risk.atr_period).to_numpy()
            decider = make_tsmom_decider(closes, atrs, lookback, qh.atr_stop_mult)
            res = engine.run(df, asset, "1d", decider=decider)
            rows.append(Row(f"TSMOM-{lookback}d", asset, "1d", res))
    return rows


def run_funding(cfg, engine) -> list[Row]:
    qh = cfg.quant_hypotheses
    rows: list[Row] = []
    for asset in ASSETS:
        df = resample(load_parquet(asset, "1h"), "4h")
        closes = df["close"].to_numpy(dtype=float)
        atrs = atr(df, cfg.risk.atr_period).to_numpy()
        funding_ann = annualize_funding_pct(align_funding_to_bars(df, load_funding(asset)))
        trend_ma = daily_ma_on_bars(df, qh.funding_trend_ma_days)
        decider = make_funding_decider(
            closes, atrs, funding_ann, trend_ma,
            neg_thr=qh.funding_extreme_neg_ann_pct, pos_thr=qh.funding_extreme_pos_ann_pct,
            normal_low=qh.funding_normal_low_ann_pct, normal_high=qh.funding_normal_high_ann_pct,
            atr_mult=qh.atr_stop_mult)
        res = engine.run(df, asset, "4h", decider=decider)
        rows.append(Row("Funding-extremo", asset, "4h", res))
    return rows


def run_donchian(cfg, engine) -> list[Row]:
    qh = cfg.quant_hypotheses
    rows: list[Row] = []
    for asset in ASSETS:
        df = resample(load_parquet(asset, "1h"), "4h")
        closes = df["close"].to_numpy(dtype=float)
        atrs = atr(df, cfg.risk.atr_period).to_numpy()
        funding_frac = align_funding_to_bars(df, load_funding(asset))
        decider = make_donchian_decider(
            closes, atrs, funding_frac,
            entry_period=qh.donchian_entry_period, exit_ema_period=qh.donchian_exit_ema,
            funding_min_frac=qh.donchian_funding_min_8h_pct / 100.0,
            funding_max_frac=qh.donchian_funding_max_8h_pct / 100.0,
            atr_mult=qh.atr_stop_mult, take_profit_rr=qh.donchian_take_profit_rr,
            max_hold_bars=qh.donchian_max_hold_bars)
        res = engine.run(df, asset, "4h", decider=decider)
        rows.append(Row("Donchian-4h", asset, "4h", res))
    return rows


def main() -> int:
    cfg = load_settings()
    engine = BacktestEngine(cfg)
    rt_cost = 2 * (cfg.backtest.commission_pct + cfg.backtest.slippage_pct)

    print("=" * 78)
    print("BACKTEST DE HIPÓTESIS QUANT (Perplexity) — neto de costos, histórico real")
    print(f"Costo ida-vuelta base ≈ {rt_cost:.2f}% (comisión {cfg.backtest.commission_pct}%"
          f" + slippage {cfg.backtest.slippage_pct}% por lado, + componente ATR dinámico)")
    print("Sharpe/Sortino anualizados · CAGR y Drag = comisiones netas anualizadas en %")
    print("=" * 78)

    tsmom = run_tsmom(cfg, engine)
    funding = run_funding(cfg, engine)
    donchian = run_donchian(cfg, engine)

    report = [
        "# Backtest de hipótesis quant (Perplexity)",
        f"\nGenerado: {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC} · "
        f"costo ida-vuelta base ≈ {rt_cost:.2f}% + slippage ATR dinámico.",
        "\nMétricas NETAS de costos. 'Drag' = comisiones anualizadas (% del capital).",
        _table("H1 — TSMOM diario (1d) · barrido de lookback", tsmom),
        _table("H2 — Funding extremo direccional (4h)", funding),
        _table("H3 — Ruptura Donchian 4h (SIN gate de OI — versión parcial)", donchian),
        "\n### No backtesteable (datos no disponibles)",
        "\nLas hipótesis de POSICIONAMIENTO (filtro de régimen por open interest y "
        "long-short ratio) NO se pueden validar: Binance solo sirve ~30 días de ese "
        "histórico gratis. Quedan como forward-test en vivo, no como backtest.",
    ]
    print(_table("H1 — TSMOM diario (1d) · barrido de lookback", tsmom))
    print(_table("H2 — Funding extremo direccional (4h)", funding))
    print(_table("H3 — Ruptura Donchian 4h (SIN gate de OI)", donchian))

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = REPORTS_DIR / f"quant_hypotheses_{stamp}.md"
    out.write_text("\n".join(report), encoding="utf-8")
    print(f"\n[OK] Reporte escrito en {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
