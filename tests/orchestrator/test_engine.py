"""Tests de integración del orquestador, dirigidos a mano contra los fakes.

Sprint 7.2: además del lazo de decisión, cubren el blindaje de concurrencia y
ciclo de vida con latencia y visibilidad de fills SIMULADAS (ex.hidden):
FLIP desacoplado, in-flight anti-resync-falso, ventana de gracia, open no
confirmado, backfill REST, hueco→rewarm, adopción de posiciones y persistencia.
"""

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from src.core.config import load_settings
from src.core.models import (
    Candle,
    EventIntent,
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
# TTL holgado: los tests del lazo reusan un sentimiento fijo (analyzed_at=T0) a
# lo largo de varias velas; no deben verse afectados por la caducidad. Los tests
# dedicados al TTL usan un CFG propio con un TTL realista (ver _cfg_ttl).
CFG.confluence.sentiment_ttl_seconds = 10_000
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


def make_env(*, dual_mode=False, storage=None, backfill_fn=None, cfg=CFG):
    ex = FakeFuturesExchange(wallet_balance=10_000.0, filters=FILTERS,
                             prices={"BTCUSDT": 1000.0, "ETHUSDT": 2000.0}, dual_mode=dual_mode)
    execu = Executor(ex, cfg, storage=storage)
    rec = RecordingAlertSink()
    sig = StubSignal()
    orch = Orchestrator(execu, cfg, alerts=rec, sentiment_store={}, signal_fn=sig,
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
    # El umbral escala con el timeframe (stale_feed_intervals × intervalo de 5m).
    stale = orch._stale_threshold_seconds()
    # Dentro del umbral (≈1 intervalo) el feed está sano; pasado el umbral, HALT.
    assert orch.check_feed_health(now=_t(0) + timedelta(seconds=stale - 5)) is True
    healthy = orch.check_feed_health(now=_t(0) + timedelta(seconds=stale + 5))
    assert healthy is False and orch.halted is True
    assert "stale_feed" in rec.events()


def test_stale_threshold_escala_con_timeframe():
    ex, execu, orch, rec, sig = make_env()
    # 5m × stale_feed_intervals (2.0) = 600s, mayor que el absoluto de 30s.
    assert orch._stale_threshold_seconds() == max(
        CFG.risk.stale_feed_seconds, CFG.risk.stale_feed_intervals * 300)


# ------------------------------ TTL de sentimiento ------------------------------

def _cfg_ttl(ttl_seconds: int):
    """Copia del CFG con un TTL de sentimiento realista (los tests del lazo usan
    uno holgado; estos miden la caducidad explícitamente)."""
    c = CFG.model_copy(deep=True)
    c.confluence.sentiment_ttl_seconds = ttl_seconds
    return c


def test_fresh_sentiment_caduca_por_ttl():
    # Frontera exacta + purga del store. _sent tiene analyzed_at=T0; con TTL=300s,
    # a 299s sigue vigente (no se purga); a 301s caduca → None y se purga.
    ex, execu, orch, rec, sig = make_env(cfg=_cfg_ttl(300))
    fresh = _sent(0.6)
    orch.sentiment_store[SYMBOL] = fresh
    assert orch._fresh_sentiment(SYMBOL, T0 + timedelta(seconds=299)) is fresh
    assert SYMBOL in orch.sentiment_store
    assert orch._fresh_sentiment(SYMBOL, T0 + timedelta(seconds=301)) is None
    assert SYMBOL not in orch.sentiment_store


async def test_sentimiento_opuesto_caducado_no_bloquea_el_trade():
    # Quant fuerte LONG + sentimiento bajista pero CADUCADO (analyzed_at de hace
    # 30 min, TTL 300s) → se trata como "sin noticia": no hay conflicto y el LONG
    # se abre (con tamaño reducido). Es el bug que el TTL corrige: sin él, ese
    # score viejo seguiría vetando el trade por sentiment_conflict.
    ex, orch, rec, sig = await build(cfg=_cfg_ttl(300))
    sig.score = 0.8
    orch.sentiment_store[SYMBOL] = SentimentScore(
        news_id="old", symbol_scope=[SYMBOL], score=-0.7, confidence=0.8,
        high_impact=False, analyzed_at=T0 - timedelta(minutes=30))
    await feed(orch, [0, 1])
    assert (SYMBOL, LONG) in ex.positions
    assert SYMBOL not in orch.sentiment_store  # el score caduco fue purgado


async def test_sentimiento_opuesto_fresco_si_bloquea():
    # Contraste del anterior: el MISMO sentimiento bajista, pero FRESCO en la vela
    # i=1, sí dispara sentiment_conflict → HOLD → no se abre el LONG.
    ex, orch, rec, sig = await build(cfg=_cfg_ttl(300))
    sig.score = 0.8
    orch.sentiment_store[SYMBOL] = SentimentScore(
        news_id="new", symbol_scope=[SYMBOL], score=-0.7, confidence=0.8,
        high_impact=False, analyzed_at=_t(1))
    await feed(orch, [0, 1])
    assert (SYMBOL, LONG) not in ex.positions


# ------------------------------ Fast Path (Plan V2 §2.3) ------------------------------


def _cfg_event(**over):
    """Copia del CFG con el Fast Path ENCENDIDO (y overrides de event.*)."""
    c = CFG.model_copy(deep=True)
    c.event.enabled = True
    for k, v in over.items():
        setattr(c.event, k, v)
    return c


def _shock(score: float = 0.7, *, confidence: float = 0.8,
           analyzed_at: datetime = T0) -> SentimentScore:
    return SentimentScore(news_id="ev", symbol_scope=[SYMBOL], score=score,
                          confidence=confidence, high_impact=True,
                          event_kind="shock", analyzed_at=analyzed_at)


def _ev_candle(i: int, close: float) -> Candle:
    return Candle(symbol=SYMBOL, timeframe="5m", open_time=T0 + timedelta(minutes=5 * i),
                  open=close, high=close + 5, low=close - 5, close=close, volume=10.0)


def _reason(rec, event: str) -> str:
    """El detalle de la primera alerta con ese nombre (para leer el reason)."""
    return next(d for _, e, d in rec.alerts if e == event)


# ---- helpers puros: impulso y resolución de scope ----

def test_price_impulse_bps_mide_el_movimiento_de_la_ultima_vela():
    ex, execu, orch, rec, sig = make_env()
    orch.buffers[SYMBOL] = [_ev_candle(0, 1000.0), _ev_candle(1, 1010.0)]
    # window 60s, intervalo 300s → n=1: retorno de la última vela = +100 bps.
    assert orch._price_impulse_bps(SYMBOL, 60) == pytest.approx(100.0)


def test_price_impulse_bps_negativo_y_sin_datos():
    ex, execu, orch, rec, sig = make_env()
    orch.buffers[SYMBOL] = [_ev_candle(0, 1000.0), _ev_candle(1, 990.0)]
    assert orch._price_impulse_bps(SYMBOL, 60) == pytest.approx(-100.0)
    orch.buffers[SYMBOL] = [_ev_candle(0, 1000.0)]  # <2 velas → no medible
    assert orch._price_impulse_bps(SYMBOL, 60) == 0.0


def test_resolve_scope_wildcard_e_interseccion():
    ex, execu, orch, rec, sig = make_env()
    assert set(orch._resolve_scope(["*"])) == set(CFG.market.symbols)
    assert orch._resolve_scope(["BTCUSDT", "DOGEUSDT"]) == ["BTCUSDT"]  # filtra lo no seguido
    assert orch._resolve_scope(["DOGEUSDT"]) == []


# ---- on_event: gate maestro y circuit breakers ----

async def test_on_event_disabled_no_opera():
    # CFG por defecto trae event.enabled=false: el Fast Path no abre nada.
    ex, orch, rec, sig = await build()
    orch.buffers[SYMBOL] = [_ev_candle(0, 1000.0), _ev_candle(1, 1010.0)]
    await orch.on_event(EventIntent(symbol=SYMBOL, sentiment=_shock(0.7)), now=T0)
    assert orch.expected == {} and (SYMBOL, LONG) not in ex.positions


async def test_on_event_halted_no_opera():
    ex, orch, rec, sig = await build(cfg=_cfg_event())
    orch.halted = True
    orch.buffers[SYMBOL] = [_ev_candle(0, 1000.0), _ev_candle(1, 1010.0)]
    await orch.on_event(EventIntent(symbol=SYMBOL, sentiment=_shock(0.7)), now=T0)
    assert orch.expected == {}


async def test_on_event_sin_warmup_no_abre():
    ex, orch, rec, sig = await build(cfg=_cfg_event())
    orch.buffers[SYMBOL] = [_ev_candle(0, 1000.0)]  # 1 vela < warmup(2)
    await orch.on_event(EventIntent(symbol=SYMBOL, sentiment=_shock(0.7)), now=T0)
    assert orch.expected == {}
    assert "event_not_warm" in rec.events()


# ---- on_event: originación y sus puertas ----

async def test_on_event_origina_long_y_registra_in_flight():
    # Camino feliz: shock alcista + impulso alcista (+100 bps ≥ 8) → abre LONG.
    ex, orch, rec, sig = await build(cfg=_cfg_event())
    orch.buffers[SYMBOL] = [_ev_candle(0, 1000.0), _ev_candle(1, 1010.0)]
    await orch.on_event(EventIntent(symbol=SYMBOL, sentiment=_shock(0.7)), now=T0)
    assert (SYMBOL, LONG) in ex.positions
    assert (SYMBOL, LONG) in orch.expected
    assert (SYMBOL, LONG) in orch._in_flight        # como in-flight: el lazo de velas NO hace HALT
    assert orch._last_event_trade[SYMBOL] == T0     # cooldown armado
    assert "event_open" in rec.events()


async def test_on_event_sin_impulso_no_abre():
    # shock alcista pero precio plano (0 bps < 8): el mercado no respalda → HOLD.
    ex, orch, rec, sig = await build(cfg=_cfg_event())
    orch.buffers[SYMBOL] = [_ev_candle(0, 1000.0), _ev_candle(1, 1000.0)]
    await orch.on_event(EventIntent(symbol=SYMBOL, sentiment=_shock(0.7)), now=T0)
    assert orch.expected == {}
    assert _reason(rec, "event_hold").endswith("event_no_impulse")


async def test_on_event_no_shock_no_abre():
    # Aunque llegue a on_event, decide_event rechaza un kind != shock.
    ex, orch, rec, sig = await build(cfg=_cfg_event())
    orch.buffers[SYMBOL] = [_ev_candle(0, 1000.0), _ev_candle(1, 1010.0)]
    sent = SentimentScore(news_id="x", symbol_scope=[SYMBOL], score=0.7,
                          confidence=0.8, event_kind="none", analyzed_at=T0)
    await orch.on_event(EventIntent(symbol=SYMBOL, sentiment=sent), now=T0)
    assert orch.expected == {}
    assert _reason(rec, "event_hold").endswith("event_not_shock")


async def test_on_event_respeta_cooldown():
    ex, orch, rec, sig = await build(cfg=_cfg_event())
    orch.buffers[SYMBOL] = [_ev_candle(0, 1000.0), _ev_candle(1, 1010.0)]
    orch._last_event_trade[SYMBOL] = T0 - timedelta(seconds=100)  # cooldown=900s
    await orch.on_event(EventIntent(symbol=SYMBOL, sentiment=_shock(0.7)), now=T0)
    assert orch.expected == {}
    assert _reason(rec, "event_hold").endswith("event_cooldown")


# ---- productor / cola / consumidor ----

async def test_enqueue_event_resuelve_scope_y_encola():
    ex, orch, rec, sig = await build(cfg=_cfg_event())
    sc = SentimentScore(news_id="m", symbol_scope=["*"], score=0.7, confidence=0.8,
                        event_kind="shock", analyzed_at=T0)
    await orch._enqueue_event(sc)
    encolados = {orch._event_queue.get_nowait().symbol for _ in CFG.market.symbols}
    assert encolados == set(CFG.market.symbols)   # un intent por símbolo del wildcard


async def test_on_event_sin_baseline_suficiente_abre_sin_recorte():
    # Con solo 2 velas en el buffer, _compute_atr_baseline devuelve None
    # (vol_regime_lookback=20 requiere ≥21 velas). El assess usa vol_damp=1.0.
    # El trade debe abrirse igualmente (la falta de baseline no es un veto).
    ex, orch, rec, sig = await build(cfg=_cfg_event())
    orch.buffers[SYMBOL] = [_ev_candle(0, 1000.0), _ev_candle(1, 1010.0)]
    await orch.on_event(EventIntent(symbol=SYMBOL, sentiment=_shock(0.7)), now=T0)
    assert (SYMBOL, LONG) in ex.positions
    assert (SYMBOL, LONG) in orch.expected


def _flat_candle(i: int, half_range: float) -> Candle:
    """Vela de cierre PLANO (=1000) con rango H-L controlado = 2×half_range.

    Cierre plano ⇒ True Range = 2×half_range en TODAS las velas (sin gaps), así el
    ATR es determinista y exactamente igual al rango. Permite fijar la línea base
    de volatilidad del buffer con precisión. Cierre plano ⇒ impulso=0, por eso los
    tests que la usan desactivan el gate de impulso (confirm_impulse_bps=0).
    """
    return Candle(symbol=SYMBOL, timeframe="5m",
                  open_time=T0 + timedelta(minutes=5 * i),
                  open=1000.0, high=1000.0 + half_range, low=1000.0 - half_range,
                  close=1000.0, volume=10.0)


async def test_on_event_atr_expandido_abre_qty_menor_que_regimen_normal():
    # Lo que pide la spec 2.4-D: con ATR ACTUAL idéntico (stub=50) en ambos
    # escenarios, el de régimen EXPANDIDO abre tamaño menor que el NORMAL. Como el
    # ATR actual es el mismo, el stop es el mismo: la única diferencia de qty es el
    # amortiguador vol_damp (se aísla así de cualquier otro efecto).
    #   - normal:    velas de rango 50 → baseline=50 → vol_ratio=50/50=1   ≤ cap=2 → vol_damp=1.0
    #   - expandido: velas de rango 10 → baseline=10 → vol_ratio=50/10=5   >  cap=2 → vol_damp=0.4
    # vol_regime_lookback=2 y atr_period=2 hacen el buffer de test manejable; con
    # cierre plano el ATR(2) converge exactamente al rango. confirm_impulse_bps=0
    # desactiva el gate de impulso (las velas planas no producen impulso).
    cfg = _cfg_event(confirm_impulse_bps=0)
    cfg.risk = cfg.risk.model_copy(update={"vol_regime_lookback": 2, "atr_period": 2})

    ex_n, orch_n, rec_n, sig_n = await build(cfg=cfg)
    sig_n.atr = 50.0
    orch_n.buffers[SYMBOL] = [_flat_candle(i, 25.0) for i in range(5)]  # rango 50
    await orch_n.on_event(EventIntent(symbol=SYMBOL, sentiment=_shock(0.7)), now=T0)
    qty_normal = orch_n.expected[(SYMBOL, LONG)]

    ex_e, orch_e, rec_e, sig_e = await build(cfg=cfg)
    sig_e.atr = 50.0
    orch_e.buffers[SYMBOL] = [_flat_candle(i, 5.0) for i in range(5)]   # rango 10
    await orch_e.on_event(EventIntent(symbol=SYMBOL, sentiment=_shock(0.7)), now=T0)
    qty_expandido = orch_e.expected[(SYMBOL, LONG)]

    # qty_normal = (10000×0.005×0.5×1.0)/(2.5×50) = 0.2 ; expandido = ×0.4 = 0.08.
    assert qty_normal == pytest.approx(0.2)
    assert qty_expandido == pytest.approx(qty_normal * 0.4)  # = 0.08
    assert qty_expandido < qty_normal


async def test_event_consumer_entrega_intents_a_on_event():
    # El consumidor solo es plomería: cada intent encolado llega a on_event.
    ex, orch, rec, sig = await build(cfg=_cfg_event())
    visto = []

    async def fake_on_event(intent, *, now=None):
        visto.append(intent.symbol)

    orch.on_event = fake_on_event
    await orch._enqueue_event(_shock(0.7))   # scope [SYMBOL] → un intent
    task = asyncio.create_task(orch._event_consumer())
    await asyncio.wait_for(orch._event_queue.join(), timeout=1.0)
    task.cancel()
    assert visto == [SYMBOL]
