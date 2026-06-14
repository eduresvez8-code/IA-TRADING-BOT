"""Tests de integración del orquestador, dirigidos a mano contra los fakes.

Cubren el manejador por vela: warmup, apertura, señal débil, flip, resync por
SL/TP, halt por reconciliación, alerta de kill switch y salud del feed. La señal
se inyecta (StubSignal) para no depender de los valores exactos del quant engine.
"""

from datetime import datetime, timedelta, timezone

from src.core.config import load_settings
from src.core.models import Candle, PositionSide, SentimentScore, Signal, SymbolFilters
from src.execution.exchange import ExchangePosition
from src.execution.executor import Executor
from src.execution.fake_exchange import FakeFuturesExchange
from src.orchestrator.alerts import RecordingAlertSink
from src.orchestrator.engine import Orchestrator

CFG = load_settings().model_copy(deep=True)
CFG.orchestrator.warmup_candles = 2  # demos y tests cortos

T0 = datetime(2026, 6, 14, 0, 0, tzinfo=timezone.utc)
SYMBOL = "BTCUSDT"
FILTERS = {s: SymbolFilters(symbol=s, tick_size="0.1", step_size="0.001",
                            min_qty="0.001", min_notional="5")
           for s in CFG.market.symbols}


class StubSignal:
    def __init__(self, score: float = 0.0, atr: float = 50.0):
        self.score = score
        self.atr = atr

    def __call__(self, df, sym):
        return Signal(symbol=sym, score=self.score, strategy="stub",
                      timestamp=T0, features={"atr": self.atr})


def _sent(score: float) -> SentimentScore:
    return SentimentScore(news_id="n", symbol_scope=[SYMBOL], score=score,
                          confidence=0.8, high_impact=False, analyzed_at=T0)


def _candle(i: int) -> Candle:
    return Candle(symbol=SYMBOL, timeframe="5m", open_time=T0 + timedelta(minutes=5 * i),
                  open=1000.0, high=1005.0, low=995.0, close=1000.0, volume=10.0)


def _t(i: int) -> datetime:
    return T0 + timedelta(minutes=5 * i)


async def build():
    ex = FakeFuturesExchange(wallet_balance=10_000.0, filters=FILTERS,
                             prices={"BTCUSDT": 1000.0, "ETHUSDT": 2000.0}, dual_mode=False)
    execu = Executor(ex, CFG)
    rec = RecordingAlertSink()
    sig = StubSignal()
    orch = Orchestrator(execu, CFG, alerts=rec, sentiment_store={}, signal_fn=sig)
    await orch.startup()
    return ex, orch, rec, sig


async def feed(orch, indices):
    for i in indices:
        await orch.on_closed_candle(_candle(i), now=_t(i))


# ------------------------------ warmup ------------------------------

async def test_no_opera_durante_warmup():
    ex, orch, rec, sig = await build()
    sig.score = 0.8
    orch.sentiment_store[SYMBOL] = _sent(0.6)
    await feed(orch, [0])  # 1 vela < warmup(2)
    assert orch.expected == {}
    assert "open" not in rec.events()


# ------------------------------ apertura ------------------------------

async def test_abre_long_con_senal_confirmada():
    ex, orch, rec, sig = await build()
    sig.score = 0.8
    orch.sentiment_store[SYMBOL] = _sent(0.6)
    await feed(orch, [0, 1])  # la 2ª vela alcanza el warmup y dispara
    assert (SYMBOL, PositionSide.LONG) in orch.expected
    assert (SYMBOL, PositionSide.LONG) in ex.positions
    assert "open" in rec.events()


async def test_senal_debil_no_abre():
    ex, orch, rec, sig = await build()
    sig.score = 0.1  # |quant| < umbral fuerte → confluencia HOLD
    await feed(orch, [0, 1])
    assert orch.expected == {}
    assert "open" not in rec.events()


# ------------------------------ flip ------------------------------

async def test_flip_long_a_short():
    ex, orch, rec, sig = await build()
    sig.score = 0.8
    orch.sentiment_store[SYMBOL] = _sent(0.6)
    await feed(orch, [0, 1])  # abre LONG
    assert (SYMBOL, PositionSide.LONG) in ex.positions

    sig.score = -0.8
    orch.sentiment_store[SYMBOL] = _sent(-0.7)
    await feed(orch, [2])  # señal opuesta confirmada → flip
    assert (SYMBOL, PositionSide.LONG) not in ex.positions
    assert (SYMBOL, PositionSide.SHORT) in ex.positions
    assert (SYMBOL, PositionSide.SHORT) in orch.expected
    assert "flip" in rec.events()


# ------------------------------ reconciliación ------------------------------

async def test_resync_cuando_sl_tp_cierra_la_pierna():
    ex, orch, rec, sig = await build()
    sig.score = 0.8
    orch.sentiment_store[SYMBOL] = _sent(0.6)
    await feed(orch, [0, 1])  # abre LONG
    ex.positions.pop((SYMBOL, PositionSide.LONG))  # simula el fill de un SL/TP

    sig.score = 0.1  # señal débil: no abre, solo reconcilia
    await feed(orch, [2])
    assert orch.halted is False
    assert orch.expected == {}            # resincronizado con el exchange
    assert "resync" in rec.events()


async def test_halt_por_pierna_desconocida():
    ex, orch, rec, sig = await build()
    # Pierna en el exchange que el bot nunca abrió → divergencia peligrosa.
    ex.positions[(SYMBOL, PositionSide.LONG)] = ExchangePosition(
        symbol=SYMBOL, position_side=PositionSide.LONG, qty=1.0,
        entry_price=1000.0, initial_margin=333.0)
    sig.score = 0.8
    await feed(orch, [0, 1])
    assert orch.halted is True
    assert "reconcile_halt" in rec.events()


async def test_halted_no_opera():
    ex, orch, rec, sig = await build()
    orch.halted = True
    sig.score = 0.8
    orch.sentiment_store[SYMBOL] = _sent(0.6)
    await feed(orch, [0, 1, 2])
    assert orch.expected == {}


# ------------------------------ kill switch + feed ------------------------------

async def test_kill_switch_dispara_alerta_y_no_abre():
    ex, orch, rec, sig = await build()
    sig.score = 0.8
    orch.sentiment_store[SYMBOL] = _sent(0.6)
    await feed(orch, [0, 1])              # abre LONG (fija el pico del wallet en 10k)
    ex.wallet_balance = 8_900.0           # drawdown del 11%
    await feed(orch, [2])
    assert "kill_switch_drawdown" in rec.events()
    # no se abrió una segunda pierna; la LONG sigue (la gestionan los SL/TP)
    assert (SYMBOL, PositionSide.SHORT) not in ex.positions


async def test_check_feed_health_detecta_feed_obsoleto():
    ex, orch, rec, sig = await build()
    await feed(orch, [0])  # registra last_candle_time
    stale = CFG.risk.stale_feed_seconds
    healthy = orch.check_feed_health(now=_t(0) + timedelta(seconds=stale + 5))
    assert healthy is False
    assert orch.halted is True
    assert "stale_feed" in rec.events()
