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

from src.core.config import QuantConfig

CFG = load_settings().model_copy(deep=True)
CFG.orchestrator.warmup_candles = 2  # tests cortos
# Estos tests validan la MECÁNICA del orquestador (ratio HTF, buckets, régimen,
# stale, time-stop), diseñada para base 5m / HTF 1h / quant EMA. El settings.yaml
# enviado ahora usa 1h/4h/SMA 50/200; fijamos aquí la config histórica para
# DESACOPLAR estos tests de la config de producción (la validan test_config.py).
CFG.market = CFG.market.model_copy(update={"timeframe": "5m", "htf_timeframe": "1h"})
CFG.quant = QuantConfig(ma_type="ema", ema_fast_period=9, ema_slow_period=21,
                        rsi_period=14, ema_weight=0.6)
# EMA solo necesita 35 velas HTF; el 230 del settings.yaml (para SMA200) haría un
# buffer_target de 2760 velas y el _downtrend_buffer generaría precios NEGATIVOS.
CFG.orchestrator.regime_htf_bars = 50
# TTL holgado: los tests del lazo reusan un sentimiento fijo (analyzed_at=T0) a
# lo largo de varias velas; no deben verse afectados por la caducidad. Los tests
# dedicados al TTL usan un CFG propio con un TTL realista (ver _cfg_ttl).
CFG.confluence.sentiment_ttl_seconds = 10_000


@pytest.fixture(autouse=True)
def _force_ema_in_signal(monkeypatch):
    """El signal_fn=compute_signal que inyectan algunos tests lee la config GLOBAL
    (load_settings), no el CFG del orquestador. Forzamos EMA ahí también para que el
    régimen caliente con los datos cortos de estos tests."""
    monkeypatch.setattr("src.quant.strategy.load_settings", lambda *a, **k: CFG)
# El repo en vivo tiene el quant apagado (news_only); estos tests del engine ejercen
# el veto/confirm de régimen, que solo existe con el quant encendido.
CFG.confluence.quant_regime_enabled = True
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


def make_env(*, dual_mode=False, storage=None, backfill_fn=None, cfg=CFG, signal_fn=None):
    ex = FakeFuturesExchange(wallet_balance=10_000.0, filters=FILTERS,
                             prices={"BTCUSDT": 1000.0, "ETHUSDT": 2000.0}, dual_mode=dual_mode)
    execu = Executor(ex, cfg, storage=storage)
    rec = RecordingAlertSink()
    sig = signal_fn if signal_fn is not None else StubSignal()
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
    orch.sentiment_store.pop(SYMBOL, None)  # sin noticia fresca → no reabre (Opción 2)
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
    orch.sentiment_store.pop(SYMBOL, None)  # sin noticia fresca → no reabre (Opción 2)
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


async def test_sentimiento_caducado_no_origina():
    # Opción 2: un sentimiento CADUCADO (analyzed_at de hace 30 min, TTL 300s) se
    # purga y se trata como "sin noticia" → no_news_origination → no se abre nada.
    # (Antes, con el quant originando, el LONG se abría igual; ahora, sin noticia
    # fresca, NO hay trade: el quant ya no tiene gatillo.)
    ex, orch, rec, sig = await build(cfg=_cfg_ttl(300))
    orch.sentiment_store[SYMBOL] = SentimentScore(
        news_id="old", symbol_scope=[SYMBOL], score=0.7, confidence=0.8,
        high_impact=False, analyzed_at=T0 - timedelta(minutes=30))
    await feed(orch, [0, 1])
    assert (SYMBOL, LONG) not in ex.positions
    assert (SYMBOL, SHORT) not in ex.positions
    assert SYMBOL not in orch.sentiment_store  # el score caduco fue purgado


async def test_noticia_fresca_bajista_origina_short():
    # Inversión Opción 2 a nivel engine: la NOTICIA pone la dirección. Un titular
    # bajista FRESCO origina un SHORT (régimen aún neutro en el test → tamaño
    # reducido), SIN que el quant tenga que estar bajista. Hedge mode para abrir
    # el corto de forma explícita.
    ex, orch, rec, sig = await build(dual_mode=True, cfg=_cfg_ttl(300))
    orch.sentiment_store[SYMBOL] = SentimentScore(
        news_id="new", symbol_scope=[SYMBOL], score=-0.7, confidence=0.8,
        high_impact=False, analyzed_at=_t(1))
    await feed(orch, [0, 1])
    assert (SYMBOL, SHORT) in ex.positions
    assert (SYMBOL, LONG) not in ex.positions


# ------------------------------ Régimen HTF (Opción 2) ------------------------------


def _downtrend_buffer(n: int) -> list[Candle]:
    """n velas de 5m en bajada monótona (closes decrecientes) → régimen 1h bajista."""
    out = []
    for i in range(n):
        price = 2000.0 * (1 - 0.0005 * i)
        out.append(Candle(symbol=SYMBOL, timeframe="5m",
                          open_time=T0 + timedelta(minutes=5 * i),
                          open=price, high=price * 1.001, low=price * 0.999,
                          close=price, volume=10.0))
    return out


def test_htf_ratio_y_buffer_target_derivados():
    # 1h/5m = 12 velas base por vela HTF; el buffer objetivo cubre el régimen
    # (regime_htf_bars * ratio), nunca menos que el warmup operativo.
    ex, execu, orch, rec, sig = make_env()
    assert orch._htf_ratio() == 12
    assert orch._buffer_target() == max(
        CFG.orchestrator.warmup_candles, CFG.orchestrator.regime_htf_bars * 12)


def test_buffer_df_htf_solo_buckets_completos():
    # 24 velas de 5m = 2 horas EXACTAS → 2 velas de 1h completas. Una vela 25 abre
    # un bucket parcial que se descarta (causal: no se usa la 1h en formación).
    ex, execu, orch, rec, sig = make_env()
    orch.buffers[SYMBOL] = _downtrend_buffer(24)
    assert len(orch._buffer_df_htf(SYMBOL)) == 2
    orch.buffers[SYMBOL] = _downtrend_buffer(25)
    assert len(orch._buffer_df_htf(SYMBOL)) == 2          # el bucket parcial se descarta
    orch.buffers[SYMBOL] = _downtrend_buffer(23)
    assert len(orch._buffer_df_htf(SYMBOL)) == 1          # el 2º bucket queda incompleto


def test_regime_signal_neutro_sin_suficientes_velas():
    # Con pocas velas no hay buckets HTF → régimen None (el engine lo trata neutro).
    ex, execu, orch, rec, sig = make_env()
    orch.buffers[SYMBOL] = _downtrend_buffer(6)
    assert orch._regime_signal(SYMBOL) is None


async def test_regime_signal_lee_tendencia_real_en_htf():
    # Con el quant REAL sobre un downtrend largo, el régimen sale fuertemente
    # bajista (|score| ≥ umbral). Esta es la señal que confirma/veta la noticia.
    from src.quant.strategy import compute_signal
    ex, orch, rec, sig = await build(signal_fn=compute_signal)
    orch.buffers[SYMBOL] = _downtrend_buffer(orch._buffer_target())
    regime = orch._regime_signal(SYMBOL)
    assert regime is not None
    assert regime.score <= -CFG.confluence.quant_confirm_threshold


async def test_regimen_fuerte_opuesto_veta_la_noticia_end_to_end():
    # Integración: régimen 1h fuertemente BAJISTA + noticia ALCISTA fresca →
    # regime_conflict → HOLD → no se abre nada (ni LONG ni SHORT). Prueba que la
    # tendencia HTF realmente VETA la originación por noticia en el lazo completo.
    from src.quant.strategy import compute_signal
    ex, orch, rec, sig = await build(dual_mode=True, cfg=_cfg_ttl(300),
                                     signal_fn=compute_signal)
    n = orch._buffer_target()
    orch.buffers[SYMBOL] = _downtrend_buffer(n)
    orch.sentiment_store[SYMBOL] = SentimentScore(
        news_id="bull", symbol_scope=[SYMBOL], score=0.7, confidence=0.8,
        high_impact=False, analyzed_at=T0 + timedelta(minutes=5 * n))
    nxt = Candle(symbol=SYMBOL, timeframe="5m",
                 open_time=T0 + timedelta(minutes=5 * n),
                 open=1400.0, high=1401.0, low=1399.0, close=1400.0, volume=10.0)
    await orch.on_closed_candle(nxt, now=T0 + timedelta(minutes=5 * n))
    assert (SYMBOL, LONG) not in ex.positions
    assert (SYMBOL, SHORT) not in ex.positions


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


def _seed_markprice(orch, sym=SYMBOL, *, last=1000.0, ref=1000.0, now=T0, span=70):
    """Siembra el micro-buffer markPrice@1s (Fase 2.5(i)) vía el push real.

    `span`+1 ticks a 1s terminando en `now`: el más viejo en `now-span` (cubre la
    ventana si span ≥ window), todos a `ref` salvo el último a `last`. Impulso
    resultante sobre la ventana = (last/ref - 1)·1e4. Usa el push de producción
    (_ingest_mark_price) para ejercitar también el desalojo por timestamp.
    """
    for k in range(span, 0, -1):
        orch._ingest_mark_price(sym, now - timedelta(seconds=k), ref)
    orch._ingest_mark_price(sym, now, last)


def _reason(rec, event: str) -> str:
    """El detalle de la primera alerta con ese nombre (para leer el reason)."""
    return next(d for _, e, d in rec.alerts if e == event)


# ---- helpers puros: impulso y resolución de scope ----

def test_price_impulse_bps_mide_sobre_ticks_markprice():
    # Fase 2.5(i): el impulso se mide sobre los ticks markPrice@1s reales en la
    # ventana [now-60, now], no sobre la vela 5m cerrada. ref=1000 → last=1010 = +100bps.
    ex, execu, orch, rec, sig = make_env()
    _seed_markprice(orch, last=1010.0, ref=1000.0, now=T0)
    assert orch._price_impulse_bps(SYMBOL, 60, T0) == pytest.approx(100.0)


def test_price_impulse_bps_negativo_y_vacio():
    ex, execu, orch, rec, sig = make_env()
    _seed_markprice(orch, last=990.0, ref=1000.0, now=T0)
    assert orch._price_impulse_bps(SYMBOL, 60, T0) == pytest.approx(-100.0)
    # Sin ticks → None (fallar-cerrado), NO 0.0: el 0.0 pasaría la ablación a ciegas.
    ex2, execu2, orch2, rec2, sig2 = make_env()
    assert orch2._price_impulse_bps(SYMBOL, 60, T0) is None


def test_price_impulse_bps_stale_devuelve_none():
    # El tick más reciente es más viejo que markprice_stale_seconds (5s) → feed
    # congelado → None (no operamos sobre un precio muerto).
    ex, execu, orch, rec, sig = make_env()
    _seed_markprice(orch, last=1010.0, ref=1000.0, now=T0 - timedelta(seconds=10))
    assert orch._price_impulse_bps(SYMBOL, 60, T0) is None


def test_price_impulse_bps_ventana_no_cubierta_devuelve_none():
    # Solo 30s de histórico para una ventana de 60s → None (mediría media ventana).
    ex, execu, orch, rec, sig = make_env()
    _seed_markprice(orch, last=1010.0, ref=1000.0, now=T0, span=30)
    assert orch._price_impulse_bps(SYMBOL, 60, T0) is None


def test_price_impulse_bps_frio_pocos_ticks_devuelve_none():
    # Buffer que cubre la ventana pero con < markprice_min_ticks (5) ticks dentro
    # de ella → None (no nos fiamos de un impulso con apenas datos).
    ex, execu, orch, rec, sig = make_env()
    orch._ingest_mark_price(SYMBOL, T0 - timedelta(seconds=70), 1000.0)  # cubre ventana
    orch._ingest_mark_price(SYMBOL, T0, 1010.0)                          # solo 1 en ventana
    assert orch._price_impulse_bps(SYMBOL, 60, T0) is None


def test_resolve_scope_wildcard_e_interseccion():
    ex, execu, orch, rec, sig = make_env()
    assert set(orch._resolve_scope(["*"])) == set(CFG.market.symbols)
    # ADAUSDT NO está en el universo (a diferencia de DOGE, que ahora sí operamos).
    assert orch._resolve_scope(["BTCUSDT", "ADAUSDT"]) == ["BTCUSDT"]  # filtra lo no seguido
    assert orch._resolve_scope(["ADAUSDT"]) == []
    # FIX DEUDA_TICKER: el ticker de activo base que devuelve Claude ahora machea el
    # par completo (BTC → BTCUSDT), antes caía a [] y solo entraba por "*".
    assert orch._resolve_scope(["BTC"]) == ["BTCUSDT"]


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
    orch.buffers[SYMBOL] = [_ev_candle(0, 1000.0), _ev_candle(1, 1010.0)]  # warmup
    _seed_markprice(orch, last=1010.0, now=T0)                              # +100 bps
    await orch.on_event(EventIntent(symbol=SYMBOL, sentiment=_shock(0.7)), now=T0)
    assert (SYMBOL, LONG) in ex.positions
    assert (SYMBOL, LONG) in orch.expected
    assert (SYMBOL, LONG) in orch._in_flight        # como in-flight: el lazo de velas NO hace HALT
    assert orch._last_event_trade[SYMBOL] == T0     # cooldown armado
    assert "event_open" in rec.events()


async def test_on_event_sin_impulso_no_abre():
    # shock alcista pero precio plano (0 bps < 8): el mercado no respalda → HOLD.
    # OJO: hay precio en vivo (deque sembrado plano), así que NO es event_no_price;
    # el impulso es válido pero insuficiente → event_no_impulse.
    ex, orch, rec, sig = await build(cfg=_cfg_event())
    orch.buffers[SYMBOL] = [_ev_candle(0, 1000.0), _ev_candle(1, 1000.0)]  # warmup
    _seed_markprice(orch, last=1000.0, now=T0)                              # 0 bps (plano)
    await orch.on_event(EventIntent(symbol=SYMBOL, sentiment=_shock(0.7)), now=T0)
    assert orch.expected == {}
    assert _reason(rec, "event_hold").endswith("event_no_impulse")


async def test_on_event_no_shock_no_abre():
    # Aunque llegue a on_event (con precio en vivo), decide_event rechaza kind != shock.
    ex, orch, rec, sig = await build(cfg=_cfg_event())
    orch.buffers[SYMBOL] = [_ev_candle(0, 1000.0), _ev_candle(1, 1010.0)]  # warmup
    _seed_markprice(orch, last=1010.0, now=T0)                              # precio en vivo
    sent = SentimentScore(news_id="x", symbol_scope=[SYMBOL], score=0.7,
                          confidence=0.8, event_kind="none", analyzed_at=T0)
    await orch.on_event(EventIntent(symbol=SYMBOL, sentiment=sent), now=T0)
    assert orch.expected == {}
    assert _reason(rec, "event_hold").endswith("event_not_shock")


async def test_on_event_respeta_cooldown():
    ex, orch, rec, sig = await build(cfg=_cfg_event())
    orch.buffers[SYMBOL] = [_ev_candle(0, 1000.0), _ev_candle(1, 1010.0)]  # warmup
    _seed_markprice(orch, last=1010.0, now=T0)                              # precio en vivo
    orch._last_event_trade[SYMBOL] = T0 - timedelta(seconds=100)  # cooldown=900s
    await orch.on_event(EventIntent(symbol=SYMBOL, sentiment=_shock(0.7)), now=T0)
    assert orch.expected == {}
    assert _reason(rec, "event_hold").endswith("event_cooldown")


async def test_on_event_sin_precio_en_vivo_no_abre_ni_con_ablacion():
    # Fallar-cerrado (Fase 2.5(i)): sin ticks markPrice NO se abre, AUNQUE el gate
    # de impulso esté ablado (confirm_impulse_bps=0). Entrar sin precio en vivo
    # viola el control de riesgo; el None se resuelve en el orquestador (event_no_price).
    ex, orch, rec, sig = await build(cfg=_cfg_event(confirm_impulse_bps=0))
    orch.buffers[SYMBOL] = [_ev_candle(0, 1000.0), _ev_candle(1, 1010.0)]  # warmup ok
    # deque markPrice vacío a propósito (no se siembra)
    await orch.on_event(EventIntent(symbol=SYMBOL, sentiment=_shock(0.7)), now=T0)
    assert orch.expected == {} and (SYMBOL, LONG) not in ex.positions
    assert "event_no_price" in rec.events()


async def test_on_event_stale_no_abre():
    # Variante de fallar-cerrado: hay ticks pero el más reciente es stale (>5s) →
    # event_no_price, no se abre.
    ex, orch, rec, sig = await build(cfg=_cfg_event())
    orch.buffers[SYMBOL] = [_ev_candle(0, 1000.0), _ev_candle(1, 1010.0)]  # warmup ok
    _seed_markprice(orch, last=1010.0, now=T0 - timedelta(seconds=30))      # último tick viejo
    await orch.on_event(EventIntent(symbol=SYMBOL, sentiment=_shock(0.7)), now=T0)
    assert orch.expected == {}
    assert "event_no_price" in rec.events()


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
    orch.buffers[SYMBOL] = [_ev_candle(0, 1000.0), _ev_candle(1, 1010.0)]  # warmup
    _seed_markprice(orch, last=1010.0, now=T0)                              # +100 bps
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
    _seed_markprice(orch_n, last=1000.0, now=T0)  # precio en vivo (gate de impulso ablado)
    await orch_n.on_event(EventIntent(symbol=SYMBOL, sentiment=_shock(0.7)), now=T0)
    qty_normal = orch_n.expected[(SYMBOL, LONG)]

    ex_e, orch_e, rec_e, sig_e = await build(cfg=cfg)
    sig_e.atr = 50.0
    orch_e.buffers[SYMBOL] = [_flat_candle(i, 5.0) for i in range(5)]   # rango 10
    _seed_markprice(orch_e, last=1000.0, now=T0)  # precio en vivo (gate de impulso ablado)
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


# ----------------- gate de seguridad del Slow Path (sentiment.enabled) -----------------

async def test_sentiment_loop_gate_false_no_llama_a_claude():
    # El gate de presupuesto: con sentiment.enabled=false, el loop retorna de
    # inmediato (early-return) → CERO llamadas a Claude y store intacto (quant puro).
    cfg = CFG.model_copy(deep=True)
    cfg.sentiment.enabled = False
    orch = make_env(cfg=cfg)[2]
    calls: list[int] = []

    async def fetch():
        calls.append(1)
        return {SYMBOL: _sent(0.5)}

    await orch._sentiment_loop(fetch)        # NO se cuelga: retorna sin entrar al while
    assert calls == []                        # nunca se invocó al productor (ni Claude)
    assert orch.sentiment_store == {}         # señal quant intacta


async def test_sentiment_loop_gate_true_actualiza_el_store():
    # Con el flag encendido, el loop sí sondea y vuelca el sentimiento al store.
    cfg = CFG.model_copy(deep=True)
    cfg.sentiment.enabled = True
    orch = make_env(cfg=cfg)[2]

    async def fetch():
        return {SYMBOL: _sent(0.5)}

    task = asyncio.create_task(orch._sentiment_loop(fetch))
    for _ in range(100):                      # cede control hasta la 1ª actualización
        if SYMBOL in orch.sentiment_store:
            break
        await asyncio.sleep(0)
    task.cancel()                             # cancelamos antes del sleep del poll
    assert orch.sentiment_store[SYMBOL].score == 0.5


# ------------------------------ time-stop (cierre por tiempo) ------------------------------

async def test_time_stop_cierra_posicion_que_supera_el_hold():
    # Una pierna que vive más de max_position_hold_candles se cierra por tiempo,
    # aunque el catalizador (sentimiento) siga vigente.
    cfg = CFG.model_copy(deep=True)
    cfg.orchestrator.max_position_hold_candles = 2   # ~2 velas (10 min) de hold máx
    ex, execu, orch, rec, sig = make_env(cfg=cfg)
    await orch.startup()
    sig.score = 0.8
    orch.sentiment_store[SYMBOL] = _sent(0.6)
    await feed(orch, [0, 1])                  # abre LONG en la vela 1
    assert (SYMBOL, LONG) in ex.positions
    await feed(orch, [2])                      # 5 min < 10: aún no vence
    assert (SYMBOL, LONG) in ex.positions
    await feed(orch, [3])                      # 10 min ≥ 2*5: time-stop cierra
    assert (SYMBOL, LONG) not in ex.positions
    assert (SYMBOL, LONG) not in orch.expected
    assert "time_stop" in rec.events()


async def test_time_stop_desactivado_no_cierra():
    # Con max_position_hold_candles=0 (off) la pierna no se cierra por tiempo.
    cfg = CFG.model_copy(deep=True)
    cfg.orchestrator.max_position_hold_candles = 0
    ex, execu, orch, rec, sig = make_env(cfg=cfg)
    await orch.startup()
    sig.score = 0.8
    orch.sentiment_store[SYMBOL] = _sent(0.6)
    await feed(orch, [0, 1, 2, 3, 4, 5])
    assert (SYMBOL, LONG) in ex.positions
    assert "time_stop" not in rec.events()


# ------------------------------ auto-recuperación del HALT ------------------------------

async def test_feed_recupera_levanta_el_halt_por_stale():
    ex, orch, rec, sig = await build()
    orch.last_candle_time[SYMBOL] = T0
    stale = orch._stale_threshold_seconds()
    far = T0 + timedelta(seconds=stale + 60)
    assert orch.check_feed_health(now=far) is False          # feed congelado → halt
    assert orch.halted is True and orch.halt_reason == "stale_feed"
    orch.last_candle_time[SYMBOL] = far                       # llega vela fresca
    assert orch.check_feed_health(now=far + timedelta(seconds=1)) is True
    assert orch.halted is False and orch.halt_reason is None  # se auto-levanta
    assert "feed_recovered" in rec.events()


async def test_halt_de_reconciliacion_no_se_auto_recupera():
    # Un halt por divergencia exige revisión humana: un feed sano NO lo levanta.
    ex, orch, rec, sig = await build()
    orch.halted = True
    orch.halt_reason = "reconcile_halt"
    orch.last_candle_time[SYMBOL] = T0
    assert orch.check_feed_health(now=T0 + timedelta(seconds=1)) is True
    assert orch.halted is True and orch.halt_reason == "reconcile_halt"


async def test_portfolio_state_estampa_feed_age_y_halted():
    # El snapshot que ve el Risk Manager lleva la edad del feed y el flag de halt
    # (antes quedaban en 0.0/False → los circuit breakers (a)/(c) del RM, muertos).
    ex, orch, rec, sig = await build()
    orch.last_candle_time[SYMBOL] = T0
    state = await orch._portfolio_state(SYMBOL, T0 + timedelta(seconds=42))
    assert state.feed_age_seconds == pytest.approx(42.0)
    assert state.halted is False
