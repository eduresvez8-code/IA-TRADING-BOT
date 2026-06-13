"""Renderizado de resultados de backtest: Markdown legible + CSV de trades.

Separamos el cálculo (engine/metrics) de la presentación (aquí): las métricas
son datos puros; este módulo solo les da formato humano. Así el reporte se
puede cambiar sin tocar la lógica de simulación.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from backtest.engine import BacktestResult


def _pct(x: float) -> str:
    return f"{x * 100:+.2f}%"


def _ratio(x: float) -> str:
    return "∞" if math.isinf(x) else f"{x:.2f}"


def format_result_markdown(result: BacktestResult) -> str:
    """Una sección Markdown con la tabla de métricas de un símbolo."""
    m = result.metrics
    pnl_abs = result.final_equity - result.initial_capital
    lines = [
        f"### {result.symbol} · {result.timeframe}",
        "",
        f"- **Capital inicial → final**: {result.initial_capital:,.2f} → "
        f"{result.final_equity:,.2f} USDT ({pnl_abs:+,.2f})",
        "",
        "| Métrica | Valor |",
        "|---|---|",
        f"| Retorno total | {_pct(m.total_return)} |",
        f"| CAGR | {_pct(m.cagr)} |",
        f"| Sharpe (anualizado) | {_ratio(m.sharpe)} |",
        f"| Sortino (anualizado) | {_ratio(m.sortino)} |",
        f"| Max drawdown | {_pct(m.max_drawdown)} |",
        f"| Win rate | {_pct(m.win_rate)} |",
        f"| Profit factor | {_ratio(m.profit_factor)} |",
        f"| Nº de trades | {m.n_trades} |",
        f"| Exposure (tiempo en mercado) | {_pct(m.exposure)} |",
        f"| PnL medio ganador / perdedor | {m.avg_win:+,.2f} / {m.avg_loss:+,.2f} |",
        f"| Expectancy (PnL medio/trade) | {m.expectancy:+,.2f} |",
        f"| Duración media (velas) | {m.avg_bars_held:.1f} |",
        "",
    ]
    return "\n".join(lines)


def trades_to_dataframe(result: BacktestResult) -> pd.DataFrame:
    """Tabla de trades para exportar a CSV / inspeccionar."""
    return pd.DataFrame([t.__dict__ for t in result.trades])


def write_report(results: list[BacktestResult], out_dir: str | Path,
                 stamp: str) -> Path:
    """Escribe el reporte Markdown y un CSV de trades por símbolo.

    Returns:
        Ruta del .md generado.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    md = [f"# Reporte de Backtest — {stamp}", "",
          "Estrategia: `ema_cross_rsi` · costos: comisión + slippage por lado.", ""]
    for r in results:
        md.append(format_result_markdown(r))
        trades_df = trades_to_dataframe(r)
        if not trades_df.empty:
            csv_path = out_dir / f"trades_{r.symbol}_{r.timeframe}_{stamp}.csv"
            trades_df.to_csv(csv_path, index=False)
            md.append(f"_Trades exportados a_ `{csv_path.name}`\n")

    md_path = out_dir / f"backtest_{stamp}.md"
    md_path.write_text("\n".join(md), encoding="utf-8")
    return md_path
