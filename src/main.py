"""Punto de entrada del bot cuantitativo S&P 500.

Hoy el proyecto está en fase de INVESTIGACIÓN/BACKTEST: no hay trading en vivo
ni broker conectado. Este módulo ofrece el smoke test de integridad
(`--check`): carga la config, valida el protocolo pre-registrado e importa los
módulos críticos. Si algo está roto, muere aquí en el segundo 0.

    uv run python -m src.main --check
"""

from __future__ import annotations

import argparse
import sys


def run_check() -> int:
    """Smoke test: config + imports. Devuelve 0 si todo carga."""
    from src.core.config import load_settings

    cfg = load_settings()

    # Imports críticos: si un módulo del pipeline no importa, fallar aquí.
    from backtest.engine import BacktestEngine          # noqa: F401
    from backtest.metrics import compute_metrics        # noqa: F401
    from src.quant.strategy import compute_signal       # noqa: F401
    from src.risk.manager import RiskManager            # noqa: F401

    print("✅ Config OK")
    print(f"   benchmark: {cfg.market.benchmark_symbol} | índice: {cfg.market.index_symbol}")
    print(f"   split pre-registrado (test desde): {cfg.research.test_start_date}")
    print(f"   criterio de éxito: Sharpe test > {cfg.research.success_sharpe_min}, "
          f"CI bootstrap {cfg.research.bootstrap_ci:.0%} excluye 0, "
          f"concentración < {cfg.research.concentration_max:.0%}")
    print("✅ Imports OK (engine, metrics, strategy, risk manager)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Bot cuantitativo S&P 500")
    parser.add_argument("--check", action="store_true",
                        help="smoke test: config + imports")
    args = parser.parse_args(argv)

    if args.check:
        return run_check()
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
