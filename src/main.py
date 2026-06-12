"""Orquestador del bot.

Por ahora solo implementa `--check`: smoke test de configuración e imports.
Los módulos del pipeline se conectan aquí a medida que avanzan los sprints.
"""

import argparse
import sys


def check() -> int:
    from src.core.config import load_secrets, load_settings

    settings = load_settings()
    secrets = load_secrets()

    print(f"✓ settings.yaml válido — símbolos: {settings.market.symbols}, "
          f"timeframe: {settings.market.timeframe}")
    print(f"✓ riesgo por trade: {settings.risk.risk_per_trade_pct}% | "
          f"pérdida diaria máx: {settings.risk.max_daily_loss_pct}% | "
          f"drawdown máx: {settings.risk.max_drawdown_pct}%")

    missing = [name for name, value in [
        ("BINANCE_API_KEY", secrets.binance_api_key),
        ("ANTHROPIC_API_KEY", secrets.anthropic_api_key),
    ] if not value]
    if missing:
        print(f"⚠ .env incompleto (faltan: {', '.join(missing)}) — "
              f"necesario a partir del Sprint 1")
    else:
        print(f"✓ .env cargado (testnet={secrets.binance_testnet})")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Bot de trading híbrido")
    parser.add_argument("--check", action="store_true",
                        help="validar configuración e imports, sin operar")
    args = parser.parse_args()

    if args.check:
        return check()

    print("El pipeline en vivo se implementa en los Sprints 1-6. "
          "Usa --check para validar la configuración.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
