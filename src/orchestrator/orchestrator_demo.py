"""Demo del orquestador, dirigido a mano contra los fakes (sin red).

Recorre el lazo de decisión vela a vela:
    warmup → abrir LONG → flip a SHORT → resync (SL/TP cerró) → halt (reconciliación)

    uv run python -m src.orchestrator.orchestrator_demo
"""

import asyncio
from datetime import datetime, timedelta, timezone

from src.core.config import load_settings
from src.core.models import PositionSide, SentimentScore, Signal, SymbolFilters
from src.data.storage import Storage
from src.execution.exchange import ExchangePosition
from src.execution.executor import Executor
from src.execution.fake_exchange import FakeFuturesExchange
from src.orchestrator.alerts import RecordingAlertSink
from src.orchestrator.engine import Orchestrator

SYMBOL = "BTCUSDT"
PRICE = 1000.0
T0 = datetime(2026, 6, 14, 0, 0, tzinfo=timezone.utc)


class StubSignal:
    """signal_fn controlable: devuelve la señal que le fijemos por fase."""

    def __init__(self):
        self.score = 0.0
        self.atr = 50.0

    def __call__(self, df, sym):
        return Signal(symbol=sym, score=self.score, strategy="demo",
                      timestamp=T0, features={"atr": self.atr})


def _candle(i: int):
    from src.core.models import Candle
    return Candle(symbol=SYMBOL, timeframe="5m", open_time=T0 + timedelta(minutes=5 * i),
                  open=PRICE, high=PRICE + 5, low=PRICE - 5, close=PRICE, volume=10.0)


def _sent(score: float) -> SentimentScore:
    return SentimentScore(news_id="d", symbol_scope=[SYMBOL], score=score,
                          confidence=0.8, high_impact=False, analyzed_at=T0)


async def main() -> int:
    cfg = load_settings().model_copy(deep=True)
    cfg.orchestrator.warmup_candles = 3  # demo corto

    filters = {s: SymbolFilters(symbol=s, tick_size="0.1", step_size="0.001",
                                min_qty="0.001", min_notional="5")
               for s in cfg.market.symbols}
    ex = FakeFuturesExchange(wallet_balance=10_000.0, filters=filters,
                             prices={SYMBOL: PRICE}, dual_mode=False)
    storage = await Storage(":memory:", "data/candles").init()
    execu = Executor(ex, cfg, storage)
    rec = RecordingAlertSink()
    sig = StubSignal()
    store: dict = {}
    orch = Orchestrator(execu, cfg, alerts=rec, sentiment_store=store, signal_fn=sig)

    await orch.startup()
    print(f"▶ Arranque · modo {'HEDGE' if await ex.get_position_mode() else 'ONE-WAY'} "
          f"· warmup={cfg.orchestrator.warmup_candles}\n")

    i = 0

    async def feed(n=1):
        nonlocal i
        for _ in range(n):
            await orch.on_closed_candle(_candle(i), now=T0 + timedelta(minutes=5 * i))
            i += 1

    print("▶ Warmup (3 velas) + señal alcista confirmada → abrir LONG")
    sig.score = 0.8
    store[SYMBOL] = _sent(0.6)
    await feed(4)
    print(f"  piernas: {[ (k[0], k[1].value) for k in orch.expected ]}")

    print("\n▶ Señal gira a bajista confirmada → FLIP a SHORT")
    sig.score = -0.8
    store[SYMBOL] = _sent(-0.7)
    await feed(1)
    print(f"  piernas: {[ (k[0], k[1].value) for k in orch.expected ]}")

    print("\n▶ Un SL/TP cierra la pierna en el exchange → RESYNC (benigno)")
    ex.positions.pop((SYMBOL, PositionSide.SHORT))  # simula el fill del stop
    sig.score = 0.1  # señal débil: no abrimos, solo reconciliamos
    store[SYMBOL] = _sent(0.0)
    await feed(1)
    print(f"  piernas tras resync: {[ (k[0], k[1].value) for k in orch.expected ]}")

    print("\n▶ Aparece una pierna que NO abrimos → HALT (circuit breaker c)")
    ex.positions[(SYMBOL, PositionSide.LONG)] = ExchangePosition(
        symbol=SYMBOL, position_side=PositionSide.LONG, qty=1.0,
        entry_price=PRICE, initial_margin=333.0)
    await feed(1)
    print(f"  halted={orch.halted}")

    print("\n▶ Alertas emitidas")
    for level, event, detail in rec.alerts:
        print(f"  [{level.value:>8}] {event}: {detail}")

    await storage.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
