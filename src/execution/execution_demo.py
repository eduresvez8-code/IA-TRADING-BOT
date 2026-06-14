"""Demo del Execution Engine de extremo a extremo, contra el fake en memoria.

Recorre el lazo completo en Futuros USD-M sin red:
    arranque (impone hedge mode) → señal → confluencia → Risk Manager →
    apertura (entrada + SL + TP) → reconciliación → snapshot del PortfolioState
    → flip de dirección (cierra LONG, abre SHORT) → log auditado.

    uv run python -m src.execution.execution_demo
"""

import asyncio
from datetime import datetime, timezone

from src.core.config import load_settings
from src.core.models import PositionSide, SentimentScore, Signal, SymbolFilters
from src.data.storage import Storage
from src.decision.confluence import decide
from src.execution.executor import Executor
from src.execution.fake_exchange import FakeFuturesExchange
from src.risk.manager import RiskManager

NOW = datetime.now(timezone.utc)
SYMBOL = "BTCUSDT"
PRICE = 1000.0
ATR = 50.0


def _sig(score: float) -> Signal:
    return Signal(symbol=SYMBOL, score=score, strategy="ema_cross_rsi", timestamp=NOW)


def _sent(score: float) -> SentimentScore:
    return SentimentScore(news_id="demo", symbol_scope=[SYMBOL], score=score,
                          confidence=0.8, high_impact=False, analyzed_at=NOW)


async def _open(execu, rm, cfg, filt, sig, sent, label):
    decision = decide(sig, sent, cfg)
    state = await execu.snapshot_portfolio()
    a = rm.assess(decision, price=PRICE, atr=ATR, state=state, filters=filt,
                  confidence=sent.confidence)
    print(f"  {label}: confluencia {decision.action.value} ({decision.reason})")
    if not a.approved:
        print(f"    risk VETÓ ({a.reason})")
        return None
    report = await execu.open_position(a.order)
    o = report.order
    print(f"    abierta {o.side.value}/{o.position_side.value} {o.leverage}x "
          f"qty={o.quantity:.3f} SL={o.stop_loss:.1f} TP={o.take_profit:.1f} "
          f"→ entry={report.entry.status}, protectoras={len(report.protective)}")
    return a.order


async def main() -> int:
    cfg = load_settings()
    filt = SymbolFilters(symbol=SYMBOL, tick_size="0.1", step_size="0.001",
                         min_qty="0.001", min_notional="5")
    # startup() fija leverage y filtros de TODOS los símbolos de la config.
    filters = {filt.symbol: filt}
    for sym in cfg.market.symbols:
        filters.setdefault(sym, SymbolFilters(symbol=sym, tick_size="0.01",
                           step_size="0.001", min_qty="0.001", min_notional="5"))
    ex = FakeFuturesExchange(wallet_balance=10_000.0, filters=filters,
                             prices={SYMBOL: PRICE}, dual_mode=False)
    storage = await Storage(":memory:", "data/candles").init()
    execu = Executor(ex, cfg, storage)
    rm = RiskManager(cfg)

    print("▶ Arranque")
    await execu.startup()
    print(f"  modo de posición: {'HEDGE' if await ex.get_position_mode() else 'ONE-WAY'} "
          f"· leverage BTCUSDT={ex.leverage[SYMBOL]}x")

    print("\n▶ Señal alcista confirmada → abrir LONG")
    long_order = await _open(execu, rm, cfg, filt, _sig(0.8), _sent(0.6), "LONG")

    print("\n▶ Reconciliación tras la apertura")
    recon = await execu.reconcile([(SYMBOL, PositionSide.LONG, long_order.quantity)])
    print(f"  consistente={recon.consistent} (discrepancias={len(recon.discrepancies)})")

    print("\n▶ Snapshot del PortfolioState (alimenta al Risk Manager)")
    state = await execu.snapshot_portfolio()
    print(f"  wallet={state.wallet_balance:.2f} available={state.available_balance:.2f} "
          f"committed_margin={state.committed_margin:.2f} piernas={state.open_positions}")

    print("\n▶ Flip: la señal gira a bajista → cerrar LONG y abrir SHORT")
    closed = await execu.close_position(SYMBOL, PositionSide.LONG)
    print(f"  cierre LONG: {closed.status} qty={closed.executed_qty:.3f}")
    await _open(execu, rm, cfg, filt, _sig(-0.8), _sent(-0.7), "SHORT")
    state = await execu.snapshot_portfolio()
    print(f"  estado: available={state.available_balance:.2f} piernas={state.open_positions}")

    print("\n▶ Reconciliación con discrepancia simulada (el exchange no tiene la pierna)")
    recon = await execu.reconcile([(SYMBOL, PositionSide.LONG, 5.0)])  # esperamos algo que no existe
    print(f"  consistente={recon.consistent} → halt si False (circuit breaker c)")

    print("\n▶ Log auditado (SQLite)")
    for o in reversed(await storage.get_orders()):
        print(f"  [{o['type']:>18}] {o['side']}/{o['position_side']} "
              f"qty={o['quantity']} price={o['price']} {o['status']} ({o['decision_reason']})")
    await storage.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
