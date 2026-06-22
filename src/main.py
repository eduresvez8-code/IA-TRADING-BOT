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

    # Sprints 5-6: confluencia + risk manager + execution listos para el pipeline.
    from src.decision.confluence import decide  # noqa: F401
    from src.risk.manager import RiskManager  # noqa: F401
    from src.execution.executor import Executor  # noqa: F401

    c = settings.confluence
    o = settings.orchestrator
    print(f"✓ confluencia (Opción 2: noticia origina, quant=régimen) — "
          f"noticia origina ≥{c.sentiment_confirm_threshold} | "
          f"régimen fuerte ≥{c.quant_strong_threshold} ({o.regime_htf_bars} velas "
          f"{settings.market.htf_timeframe}) | tamaño reducido ×{c.reduced_size_factor} | "
          f"cortos {'ON (simétrico)' if c.allow_short else 'OFF'}")
    print(f"✓ risk manager (Futuros USD-M) — máx {settings.risk.max_open_positions} "
          f"posiciones | TP {settings.risk.take_profit_rr}×SL | "
          f"feed obsoleto >{settings.risk.stale_feed_seconds:.0f}s | "
          f"leverage máx {settings.risk.max_leverage}x | "
          f"margen máx {settings.risk.max_portfolio_margin_pct:.0f}%")
    from src.orchestrator.engine import Orchestrator  # noqa: F401
    print(f"✓ execution — hedge mode al arrancar | stops sobre "
          f"{settings.execution.stop_working_type} | "
          f"reconciliación ±{settings.execution.reconcile_position_tolerance:.1%}")
    print(f"✓ orchestrator — warmup {settings.orchestrator.warmup_candles} velas | "
          f"política: una pierna por símbolo (flip en señal opuesta)")

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


async def preflight() -> int:
    """Validación de testnet SIN operar: claves, cuenta, hedge mode, filtros.

    El paso seguro antes de `--live`: confirma que las credenciales conectan, que
    hay saldo (testnet), fija el modo cobertura y lee los metadatos de cada
    símbolo. No envía ninguna orden. Si esto pasa, `--live` arrancará en limpio.
    """
    from binance.exceptions import BinanceAPIException

    from src.core.config import load_secrets, load_settings
    from src.execution.binance_futures import BinanceFuturesExchange

    settings, secrets = load_settings(), load_secrets()
    if not (secrets.binance_api_key and secrets.binance_api_secret):
        print("⚠ faltan BINANCE_API_KEY/SECRET en .env — crea claves en la testnet "
              "de Futuros (https://testnet.binancefuture.com) y añádelas a .env.")
        return 1
    if not secrets.binance_testnet:
        print("⚠ BINANCE_TESTNET=false en .env — el paper trading debe ir contra "
              "testnet. Pon BINANCE_TESTNET=true antes de continuar.")
        return 1

    exchange = await BinanceFuturesExchange.connect(
        secrets.binance_api_key, secrets.binance_api_secret,
        testnet=secrets.binance_testnet)
    try:
        acct = await exchange.get_account()
        print(f"✓ conectado a Futuros (testnet={secrets.binance_testnet})")
        print(f"✓ saldo wallet: {acct.wallet_balance:,.2f} USDT | "
              f"disponible: {acct.available_balance:,.2f} USDT")
        if acct.wallet_balance <= 0:
            print("  ⚠ saldo 0 — pide fondos ficticios en el faucet de la testnet.")
        if acct.positions:
            for p in acct.positions:
                print(f"  • posición abierta: {p.symbol} {p.position_side.value} "
                      f"qty={p.qty} entry={p.entry_price}")
        else:
            print("✓ sin posiciones abiertas (cuenta limpia)")

        dual = await exchange.get_position_mode()
        if not dual:
            await exchange.set_position_mode(True)  # intenta; el testnet puede vetarlo
            dual = await exchange.get_position_mode()
        if dual:
            print("✓ hedge mode (dual side) activo")
        else:
            print("✓ modo ONE-WAY (el testnet no permite hedge) — el adaptador "
                  "traduce a positionSide=BOTH; el bot opera 1 pierna por símbolo")

        for symbol in settings.market.symbols:
            f = await exchange.get_symbol_filters(symbol)
            print(f"✓ {symbol}: tick={f.tick_size} step={f.step_size} "
                  f"minNotional={f.min_notional}")
        print("\n▶ Preflight OK. Ya puedes lanzar:  uv run python -m src.main --live")
        return 0
    except BinanceAPIException as e:
        print(f"⚠ Binance rechazó la llamada (code {e.code}): {e.message}")
        if e.code == -2015:
            print("  → clave inválida, sin permiso de Futuros, o IP no autorizada.\n"
                  "    Verifica que la clave es de testnet.binancefuture.com (NO la de\n"
                  "    mainnet) y que BINANCE_TESTNET=true en .env.")
        elif e.code == -2014:
            print("  → formato de API-key inválido (revisa que no haya espacios/comillas).")
        return 1
    finally:
        await exchange.close()


async def status() -> int:
    """Foto del estado: saldo, posiciones abiertas y últimas órdenes enviadas.

    El comando para responder "¿qué ha hecho el bot?" sin mirar la terminal del
    lazo ni la UI de Binance. Solo lee; no opera.
    """
    from binance.exceptions import BinanceAPIException

    from src.core.config import load_secrets, load_settings
    from src.data.storage import Storage
    from src.execution.binance_futures import BinanceFuturesExchange

    settings, secrets = load_settings(), load_secrets()
    storage = await Storage(settings.storage.db_path, settings.storage.candles_dir).init()
    try:
        orders = await storage.get_orders(limit=10)
        print(f"Órdenes enviadas (total recientes): {len(orders)}")
        for o in orders:
            print(f"  {o['symbol']} {o['side']}/{o['position_side']} {o['type']} "
                  f"status={o['status']} · {o['decision_reason']}")
        if not orders:
            print("  (ninguna todavía — el bot aún no ha cruzado el umbral de señal)")
    finally:
        await storage.close()

    if not (secrets.binance_api_key and secrets.binance_api_secret):
        return 0
    exchange = await BinanceFuturesExchange.connect(
        secrets.binance_api_key, secrets.binance_api_secret, testnet=secrets.binance_testnet)
    try:
        acct = await exchange.get_account()
        print(f"\nSaldo testnet: {acct.wallet_balance:,.2f} USDT | "
              f"posiciones abiertas: {len(acct.positions)}")
        for p in acct.positions:
            print(f"  {p.symbol} {p.position_side.value} qty={p.qty} "
                  f"entry={p.entry_price} uPnL={p.unrealized_pnl:+.2f}")
    except BinanceAPIException as e:
        print(f"\n⚠ no se pudo leer la cuenta (code {e.code}): {e.message}")
    finally:
        await exchange.close()
    return 0


async def live() -> int:
    """Lazo en vivo: datos de SPOT mainnet (públicos) + órdenes a FUTUROS testnet.

    Capa operativa (red): se valida en testnet. La lógica de decisión está
    cubierta por tests. Requiere BINANCE_API_KEY/SECRET de la testnet de Futuros.
    """
    from binance import AsyncClient

    from src.core.config import load_secrets, load_settings
    from src.data.storage import Storage
    from src.execution.binance_futures import BinanceFuturesExchange
    from src.execution.executor import Executor
    from src.orchestrator.engine import Orchestrator
    from src.sentiment.events import build_event_fetch
    from src.sentiment.slow_path import build_sentiment_fetch

    settings, secrets = load_settings(), load_secrets()
    if not (secrets.binance_api_key and secrets.binance_api_secret):
        print("⚠ faltan claves de Binance en .env — el modo en vivo necesita "
              "credenciales de la testnet de Futuros.")
        return 1

    # Datos: spot mainnet, públicos (sin claves). Órdenes: futuros testnet.
    data_client = await AsyncClient.create()
    exchange = await BinanceFuturesExchange.connect(
        secrets.binance_api_key, secrets.binance_api_secret,
        testnet=secrets.binance_testnet)
    storage = await Storage(settings.storage.db_path, settings.storage.candles_dir).init()
    executor = Executor(exchange, settings, storage)
    orch = Orchestrator(executor, settings)

    # --- Fast Path (Plan V2 §2.5(ii)): productor de eventos real (RSS→shock→Claude).
    # Se construye SIEMPRE, pero queda INERTE hasta que se valide en testnet: run()
    # solo arranca el _event_loop/streams markPrice si settings.event.enabled (gate
    # maestro, hoy false). Construir el closure no toca la red ni gasta tokens; la
    # primera llamada a Claude ocurre dentro de _event_loop, que no se lanza con el
    # gate cerrado. Así el cableado queda listo para el día que el gate flipee.
    event_fetch = build_event_fetch(settings, secrets)
    if settings.event.enabled and not secrets.anthropic_api_key:
        # Salvaguarda: con el gate abierto, _event_loop llamaría a Claude y, sin
        # clave, fallaría en bucle bajo _supervise (restart infinito). Fallar-rápido.
        print("⚠ event.enabled=true pero falta ANTHROPIC_API_KEY en .env — el Fast "
              "Path no puede analizar noticias. Añádela o vuelve a poner enabled=false.")
        return 1

    # --- Slow Path (Plan V2): overlay de sentimiento de noticias. Se construye
    # SIEMPRE, pero el GATE DE SEGURIDAD settings.sentiment.enabled (default false)
    # decide si run() arranca el _sentiment_loop. Con el flag apagado el callable
    # nunca se invoca → cero tokens de Claude y la señal quant queda PURA (no se
    # altera la línea base de paper trading). Activarlo es decisión explícita.
    async def _on_scored(item, score) -> None:
        ts_ms = int(score.analyzed_at.timestamp() * 1000)
        await storage.save_news(item)
        await storage.save_sentiment_score(score, ts_ms=ts_ms)

    sentiment_fetch = build_sentiment_fetch(settings, secrets, on_scored=_on_scored)
    if settings.sentiment.enabled and not secrets.anthropic_api_key:
        # Misma salvaguarda que el Fast Path: gate abierto + sin clave = restart loop.
        print("⚠ sentiment.enabled=true pero falta ANTHROPIC_API_KEY en .env — el "
              "overlay de sentimiento no puede analizar noticias. Añádela o vuelve a "
              "poner enabled=false.")
        return 1

    print("▶ Iniciando lazo en vivo (Futuros testnet). Ctrl-C para detener.")
    try:
        await orch.run(
            data_client,
            sentiment_fetch=sentiment_fetch,
            event_fetch=event_fetch,
        )
    finally:
        await data_client.close_connection()
        await exchange.close()
        await storage.close()
    return 0


def dashboard() -> int:
    """Dashboard de observabilidad en tiempo real (READ-ONLY, proceso aparte).

    Sirve en http://host:port (loopback por defecto) una página única que repolla
    la SQLite del bot y muestra equity, posiciones, decisiones, órdenes y noticias.
    Nunca envía órdenes ni toca el exchange: abre la base en modo `ro`. Se puede
    correr en paralelo a `--live` (otra terminal) o sobre una BD ya existente.
    """
    from src.dashboard.server import serve

    serve()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Bot de trading híbrido")
    parser.add_argument("--check", action="store_true",
                        help="validar configuración e imports, sin operar")
    parser.add_argument("--preflight", action="store_true",
                        help="validar conexión/cuenta/hedge mode en testnet, sin operar")
    parser.add_argument("--status", action="store_true",
                        help="ver saldo, posiciones y últimas órdenes (qué ha hecho el bot)")
    parser.add_argument("--live", action="store_true",
                        help="lazo en vivo contra la testnet de Futuros (requiere claves)")
    parser.add_argument("--dashboard", action="store_true",
                        help="dashboard READ-ONLY en tiempo real (http local, no opera)")
    args = parser.parse_args()

    if args.check:
        return check()
    if args.preflight:
        import asyncio
        return asyncio.run(preflight())
    if args.status:
        import asyncio
        return asyncio.run(status())
    if args.live:
        import asyncio
        return asyncio.run(live())
    if args.dashboard:
        return dashboard()

    print("Usa --check (config), --preflight (conexión), --status (qué hizo), "
          "--live (operar en testnet) o --dashboard (visor en tiempo real).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
