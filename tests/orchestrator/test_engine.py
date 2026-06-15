"""Tests de integración del orquestador, dirigidos a mano contra los fakes.

Sprint 7.2: además del lazo de decisión, cubren el blindaje de concurrencia y
ciclo de vida con latencia y visibilidad de fills SIMULADAS (ex.hidden):
FLIP desacoplado, in-flight anti-resync-falso, ventana de gracia, open no
confirmado, backfill REST, hueco→rewarm, adopción de posiciones y persistencia.
"""

import asyncio
from datetime import datetime, timedelta, timezone

from src.core.config import load_settings
from src.core.models import (
    Candle,
    OrderType,
    PositionSide,
    SentimentScore,
    Side,
    Signal,
    SymbolFilters,
)
from src.data.storage import Storage
from src.execution.exchange import ExchangePosition, OrderResult
from src.execution.executor import Executor
from src.execution.fake_exchange import FakeFuturesExchange
from src.orchestrator.alerts import RecordingAlertSink
from src.orchestrator.engine import Orchestrator

CFG = load_settings().model_copy(deep=True)
CFG.orchestrator.warmup_candles = 2  # tests cortos
GRACE = CFG.orchestrator.reconcile_grace_cycles  # 3

T0 = datetime(2026, 6, 14, 0, 0, tzinfo=timezone.utc)
SYMBOL = "BTCUSDT"
LONG, SHORT = PositionSide.LONG, PositionSide.SHORT
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


def _leg(qty: float = 1.0) -> ExchangePosition:
    return ExchangePosition(symbol=SYMBOL, position_side=LONG, qty=qty,
                            entry_price=1000.0, initial_margin=333.0)


def make_env(*, dual_mode=False, storage=None, backfill_fn=None):
    ex = FakeFuturesExchange(wallet_balance=10_000.0, filters=FILTERS,
                             prices={"BTCUSDT": 1000.0, "ETHUSDT": 2000.0}, dual_mode=dual_mode)
    execu = Executor(ex, CFG, storage=storage)
    rec = RecordingAlertSink()
    sig = StubSignal()
    orch = Orchestrator(execu, CFG, alerts=rec, sentiment_store={}, signal_fn=sig,
                        backfill_fn=backfill_fn)
    return ex, execu, orch, rec, sig


async def build(**kw):
    ex, execu, orch, rec, sig = make_env(**kw)
    await orch.startup()
    return ex, orch, rec, sig


async def feed(orch, indices):
    for i in indices:
        await orch.on_closed_candle(_candle(i), now=_t(i))


# ------------------------------ lazo básico ------------------------------

async def test_no_opera_durante_warmup():
    ex, orch, rec, sig = await build()
    sig.score = 0.8
    orch.sentiment_store[SYMBOL] = _sent(0.6)
    await feed(orch, [0])
    assert orch.expected == {} and "open" not in rec.events()


async def test_abre_long_con_senal_confirmada():
    ex, orch, rec, sig = await build()
    sig.score = 0.8
    orch.sentiment_store[SYMBOL] = _sent(0.6)
    await feed(orch, [0, 1])
    assert (SYMBOL, LONG) in orch.expected
    assert (SYMBOL, LONG) in ex.positions
    assert "open" in rec.events()


async def test_senal_debil_no_abre():
    ex, orch, rec, sig = await build()
    sig.score = 0.1
    await feed(orch, [0, 1])
    assert orch.expected == {} and "open" not in rec.events()


async def test_halted_no_opera():
    ex, orch, rec, sig = await build()
    orch.halted = True
    sig.score = 0.8
    orch.sentiment_store[SYMBOL] = _sent(0.6)
    await feed(orch, [0, 1, 2])
    assert orch.expected == {}


# ------------------------------ FLIP desacoplado ------------------------------

async def test_flip_cierra_y_abre_en_ciclos_distintos():
    ex, orch, rec, sig = await build()
    sig.score = 0.8
    orch.sentiment_store[SYMBOL] = _sent(0.6)
    await feed(orch, [0, 1])  # abre LONG
    assert (SYMBOL, LONG) in ex.positions

    sig.score = -0.8
    orch.sentiment_store[SYMBOL] = _sent(-0.7)
    await feed(orch, [2])  # SOLO cierra el LONG
    assert (SYMBOL, LONG) not in ex.positions
    assert (SYMBOL, SHORT) not in ex.positions   # aún no abre el inverso
    assert orch.expected == {}
    assert "flip_close" in rec.events()

    await feed(orch, [3])  # ciclo siguiente: abre SHORT con snapshot fresco
    assert (SYMBOL, SHORT) in ex.positions


# ------------------------------ reconciliación ------------------------------

async def test_resync_tras_confirmacion_cuando_sl_tp_cierra():
    ex, orch, rec, sig = await build()
    sig.score = 0.8
    orch.sentiment_store[SYMBOL] = _sent(0.6)
    await feed(orch, [0, 1])  # abre LONG (en vuelo)
    await feed(orch, [2])     # ciclo de confirmación: promueve a confirmada
    assert (SYMBOL, LONG) not in orch._in_flight

    ex.positions.pop((SYMBOL, LONG))  # un SL/TP cierra la pierna
    sig.score = 0.1
    await feed(orch, [3])
    assert orch.halted is False
    assert orch.expected == {}
    assert "resync" in rec.events()


async def test_in_flight_evita_resync_falso_por_lag():
    ex, orch, rec, sig = await build()
    sig.score = 0.8
    orch.sentiment_store[SYMBOL] = _sent(0.6)
    await feed(orch, [0, 1])  # abre LONG (in_flight=0)

    ex.hidden.add((SYMBOL, LONG))  # el exchange aún no reporta la pierna (lag)
    sig.score = 0.1
    await feed(orch, [2])
    # sigue en vuelo: ni se resync-ea ni se reabre
    assert (SYMBOL, LONG) in orch.expected
    assert (SYMBOL, LONG) in orch._in_flight
    assert "resync" not in rec.events()

    ex.hidden.discard((SYMBOL, LONG))  # por fin la reporta
    await feed(orch, [3])
    assert (SYMBOL, LONG) not in orch._in_flight  # confirmada
    assert (SYMBOL, LONG) in orch.expected


async def test_halt_tras_gracia_por_pierna_desconocida():
    ex, orch, rec, sig = await build()
    ex.positions[(SYMBOL, LONG)] = _leg()  # pierna que el bot nunca abrió
    sig.score = 0.8
    await feed(orch, [0, 1])  # ciclo 1
    assert orch.halted is False
    await feed(orch, [2])     # ciclo 2
    assert orch.halted is False
    await feed(orch, [3])     # ciclo 3 → alcanza la gracia → HALT
    assert orch.halted is True
    assert "reconcile_halt" in rec.events()


async def test_gracia_se_reinicia_si_la_anomalia_desaparece():
    ex, orch, rec, sig = await build()
    ex.positions[(SYMBOL, LONG)] = _leg()
    sig.score = 0.8
    await feed(orch, [0, 1])  # ciclo 1: sospecha
    assert orch.halted is False and orch._suspect_counts
    ex.positions.pop((SYMBOL, LONG))  # transitoria: desaparece
    await feed(orch, [2])
    assert orch._suspect_counts == {} and orch.halted is False


async def test_open_no_confirmado_expira():
    ex, orch, rec, sig = await build()
    sig.score = 0.8
    orch.sentiment_store[SYMBOL] = _sent(0.6)
    await feed(orch, [0, 1])  # abre LONG (in_flight=0)
    ex.hidden.add((SYMBOL, LONG))  # nunca se confirma
    sig.score = 0.1
    await feed(orch, [2, 3, 4, 5])  # edad 1,2,3,4 → >grace(3) expira
    assert (SYMBOL, LONG) not in orch._in_flight
    assert (SYMBOL, LONG) not in orch.expected
    assert "open_unconfirmed" in rec.events()


async def test_lock_serializa_ciclos_concurrentes():
    ex, orch, rec, sig = await build()
    sig.score = 0.8
    orch.sentiment_store[SYMBOL] = _sent(0.6)
    await feed(orch, [0])  # warmup
    await asyncio.gather(
        orch.on_closed_candle(_candle(1), now=_t(1)),
        orch.on_closed_candle(_candle(2), now=_t(2)),
    )
    longs = [k for k in ex.positions if k[1] == LONG]
    assert len(longs) == 1  # el lock + held evitan la doble apertura


# ------------------------------ warmup / backfill ------------------------------

async def test_warmup_rellena_buffer_via_backfill():
    n = CFG.orchestrator.warmup_candles
    seed = [_candle(i) for i in range(n)]

    async def fake_backfill(sym, k):
        return seed[:k]

    ex, orch, rec, sig = await build(backfill_fn=fake_backfill)
    await orch.warmup(SYMBOL)
    assert len(orch.buffers[SYMBOL]) == n


async def test_hueco_dispara_rewarm():
    calls = []

    async def fake_backfill(sym, k):
        calls.append(k)
        return [_candle(i) for i in range(k)]

    ex, orch, rec, sig = await build(backfill_fn=fake_backfill)
    await orch.on_closed_candle(_candle(0), now=_t(0))
    await orch.on_closed_candle(_candle(1), now=_t(1))   # contigua
    await orch.on_closed_candle(_candle(5), now=_t(5))   # hueco (salta velas)
    assert SYMBOL not in orch._needs_rewarm  # el rewarm lo limpió
    assert calls                              # backfill fue invocado


# ------------------------------ adopción y persistencia ------------------------------

async def test_adopta_posicion_con_stop():
    ex, execu, orch, rec, sig = make_env(dual_mode=True)
    ex.positions[(SYMBOL, LONG)] = _leg()
    ex.resting[SYMBOL] = [OrderResult(
        order_id="1", symbol=SYMBOL, status="NEW", side=Side.SELL, position_side=LONG,
        type=OrderType.STOP_MARKET, executed_qty=0.0, avg_price=0.0)]
    await orch.startup()
    assert orch.expected[(SYMBOL, LONG)] == 1.0
    assert orch.halted is False
    assert "adopt" in rec.events()


async def test_posicion_desnuda_tras_reinicio_halt():
    ex, execu, orch, rec, sig = make_env(dual_mode=True)
    ex.positions[(SYMBOL, LONG)] = _leg()  # sin STOP en resting
    await orch.startup()
    assert orch.halted is True
    assert "naked_position" in rec.events()


async def test_recarga_estado_de_sesion(tmp_path):
    storage = await Storage(tmp_path / "t.db", tmp_path / "c").init()
    await storage.save_session_state(peak_wallet=12_000.0, day_start_wallet=11_000.0,
                                     day="2026-06-14", kill_switch=True)
    ex, execu, orch, rec, sig = make_env(dual_mode=True, storage=storage)
    await orch.startup()
    peak, day_start, _ = execu.session_state()
    assert peak == 12_000.0 and day_start == 11_000.0
    assert orch.risk.kill_switch_active is True
    await storage.close()


async def test_persiste_estado_de_sesion_tras_operar(tmp_path):
    storage = await Storage(tmp_path / "t.db", tmp_path / "c").init()
    ex, execu, orch, rec, sig = make_env(storage=storage)
    await orch.startup()
    sig.score = 0.8
    orch.sentiment_store[SYMBOL] = _sent(0.6)
    await feed(orch, [0, 1])  # opera → snapshot fija el pico → se persiste
    st = await storage.load_session_state()
    assert st is not None and st["peak_wallet"] == 10_000.0
    await storage.close()


# ------------------------------ kill switch + feed ------------------------------

async def test_kill_switch_dispara_alerta_y_no_abre():
    ex, orch, rec, sig = await build()
    sig.score = 0.8
    orch.sentiment_store[SYMBOL] = _sent(0.6)
    await feed(orch, [0, 1])     # abre LONG, fija el pico en 10k
    ex.wallet_balance = 8_900.0  # drawdown del 11%
    await feed(orch, [2])
    assert "kill_switch_drawdown" in rec.events()
    assert (SYMBOL, SHORT) not in ex.positions


async def test_check_feed_health_detecta_feed_obsoleto():
    ex, orch, rec, sig = await build()
    await feed(orch, [0])
    stale = CFG.risk.stale_feed_seconds
    healthy = orch.check_feed_health(now=_t(0) + timedelta(seconds=stale + 5))
    assert healthy is False and orch.halted is True
    assert "stale_feed" in rec.events()
