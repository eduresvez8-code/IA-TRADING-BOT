"""Demo del orquestador, dirigido a mano contra los fakes (sin red).

Recorre el lazo con el blindaje del Sprint 7.2:
    warmup → abrir LONG → confirmar → FLIP (cerrar / abrir en ciclos distintos)
    → lag de visibilidad (in-flight, sin resync falso) → resync por SL/TP
    → HALT tras la ventana de gracia.

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


def _legs(orch):
    return [(k[0], k[1].value) for k in orch.expected]


async def main() -> int:
    cfg = load_settings().model_copy(deep=True)
    cfg.orchestrator.warmup_candles = 3
    grace = cfg.orchestrator.reconcile_grace_cycles

    filters = {s: SymbolFilters(symbol=s, tick_size="0.1", step_size="0.001",
                                min_qty="0.001", min_notional="5")
               for s in cfg.market.symbols}
    ex = FakeFuturesExchange(wallet_balance=10_000.0, filters=filters,
                             prices={SYMBOL: PRICE}, dual_mode=False)
    storage = await Storage(":memory:", "data/candles").init()
    execu = Executor(ex, cfg, storage)
    rec = RecordingAlertSink()
    sig = StubSignal()
    orch = Orchestrator(execu, cfg, alerts=rec, sentiment_store={}, signal_fn=sig)
    await orch.startup()
    print(f"▶ Arranque · modo {'HEDGE' if await ex.get_position_mode() else 'ONE-WAY'} "
          f"· warmup={cfg.orchestrator.warmup_candles} · gracia={grace}\n")

    i = 0

    async def feed(n=1):
        nonlocal i
        for _ in range(n):
            await orch.on_closed_candle(_candle(i), now=T0 + timedelta(minutes=5 * i))
            i += 1

    print("▶ Warmup + señal alcista confirmada → abrir LONG (queda EN VUELO)")
    sig.score = 0.8
    orch.sentiment_store[SYMBOL] = _sent(0.6)
    await feed(3)   # warmup (3 velas) + abre LONG en la última
    print(f"  piernas={_legs(orch)} · in_flight={[k[0] for k in orch._in_flight]}")

    print("\n▶ El exchange tarda en reportar la pierna recién abierta (latencia de fill)")
    ex.hidden.add((SYMBOL, PositionSide.LONG))  # aún no visible en get_account
    await feed(1)
    print(f"  en vuelo, SIN resync falso: piernas={_legs(orch)} "
          f"in_flight={[k[0] for k in orch._in_flight]}")
    ex.hidden.discard((SYMBOL, PositionSide.LONG))
    await feed(1)   # ya visible → se confirma (sale de in_flight)
    print(f"  confirmada: piernas={_legs(orch)} in_flight={[k[0] for k in orch._in_flight]}")

    print("\n▶ Señal gira a bajista → FLIP DESACOPLADO (cerrar ahora, abrir luego)")
    sig.score = -0.8
    orch.sentiment_store[SYMBOL] = _sent(-0.7)
    await feed(1)   # solo cierra el LONG
    print(f"  tras el cierre: piernas={_legs(orch)}")
    await feed(1)   # ciclo siguiente: abre SHORT con snapshot fresco
    await feed(1)   # confirma el SHORT
    print(f"  tras la apertura inversa: piernas={_legs(orch)}")

    print("\n▶ Un SL/TP cierra la pierna en el exchange → RESYNC (benigno)")
    ex.positions.pop((SYMBOL, PositionSide.SHORT), None)
    sig.score = 0.1
    orch.sentiment_store[SYMBOL] = _sent(0.0)
    await feed(1)
    print(f"  piernas tras resync: {_legs(orch)}")

    print(f"\n▶ Pierna desconocida sostenida → HALT tras {grace} ciclos de gracia")
    ex.positions[(SYMBOL, PositionSide.LONG)] = ExchangePosition(
        symbol=SYMBOL, position_side=PositionSide.LONG, qty=1.0,
        entry_price=PRICE, initial_margin=333.0)
    for _ in range(grace):
        await feed(1)
        print(f"  halted={orch.halted}")

    print("\n▶ Alertas emitidas")
    for level, event, detail in rec.alerts:
        print(f"  [{level.value:>8}] {event}: {detail}")

    await storage.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
