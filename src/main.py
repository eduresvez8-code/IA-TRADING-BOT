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

    # Sprint 5: confluencia + risk manager listos para integrarse al pipeline.
    from src.decision.confluence import decide  # noqa: F401
    from src.risk.manager import RiskManager  # noqa: F401

    c = settings.confluence
    print(f"✓ confluencia — quant fuerte ≥{c.quant_strong_threshold} | "
          f"sentimiento confirma ≥{c.sentiment_confirm_threshold} | "
          f"tamaño reducido ×{c.reduced_size_factor} | "
          f"cortos {'ON' if c.allow_short else 'OFF (Spot long-only)'}")
    print(f"✓ risk manager — máx {settings.risk.max_open_positions} posiciones | "
          f"TP {settings.risk.take_profit_rr}×SL | "
          f"feed obsoleto >{settings.risk.stale_feed_seconds:.0f}s | "
          f"exposición máx {settings.risk.max_portfolio_exposure_pct:.0f}%")

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
