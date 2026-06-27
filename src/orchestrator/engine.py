"""Orquestador en vivo: el lazo que une todos los motores del bot.

Por cada vela cerrada (`on_closed_candle`): reconcilia el estado con el exchange,
calcula el ATR base (5m) para los stops y el RÉGIMEN del quant sobre velas
resampleadas al HTF (Opción 2: el quant ya no origina, solo confirma/dimensiona),
lee el último sentimiento (que ORIGINA la dirección), cruza en la confluencia,
pide veredicto al Risk Manager y aplica la política de UNA pierna por símbolo.

Blindaje de concurrencia y ciclo de vida (Sprint 7.2):
    - Un `asyncio.Lock` serializa la sección crítica (reconciliar→decidir→actuar):
      ninguna tarea concurrente observa un estado de cuenta a medio aplicar.
    - Registro `_in_flight`: piernas que abrimos pero el exchange aún no confirma
      en `get_account` (latencia del fill). La reconciliación las ignora, evitando
      tanto un HALT falso (pierna "desconocida") como un RESYNC falso (creer que
      un SL/TP la cerró). Se promueven al verlas; si nunca aparecen, expiran.
    - Ventana de gracia: una pierna desconocida debe persistir N ciclos antes de
      disparar el HALT (circuit breaker c), no a la primera observación.
    - FLIP desacoplado: una señal opuesta solo CIERRA; la apertura inversa ocurre
      en el ciclo siguiente con un snapshot fresco, sin colisión de margen.
    - Backfill REST + adopción de posiciones + persistencia de estado de sesión:
      el bot sobrevive a reinicios en caliente sin huecos ni perder el kill switch.

Inyectables para tests: `signal_fn` (quant engine) y `backfill_fn` (fuente de
velas históricas), además del `sentiment_store`.
"""

from __future__ import annotations

import asyncio
import logging
import statistics
from collections import deque
from datetime import date, datetime, timedelta, timezone
from typing import Awaitable, Callable

import pandas as pd
from binance.exceptions import BinanceAPIException

from src.core.config import Settings, load_settings
from src.core.scope import resolve_scope
from src.core.models import (
    Action,
    Candle,
    EventIntent,
    OrderType,
    PositionSide,
    SentimentScore,
    Signal,
)
from src.data.binance_client import (
    CLOCK_AHEAD_CODE,
    interval_to_ms,
    rest_kline_to_candle,
    retry_with_backoff,
    stream_candles,
    stream_mark_price,
)
from src.decision.confluence import decide, decide_event
from src.execution.executor import Executor
from src.orchestrator.alerts import AlertLevel, AlertSink, LoggingAlertSink
from src.orchestrator.policy import (
    PositionAction,
    classify_reconciliation,
    decide_position_action,
)
from src.quant.indicators import atr as _atr_series
from src.quant.strategy import compute_signal
from src.risk.manager import RiskManager

logger = logging.getLogger("ia_trading.orchestrator")

_CRITICAL_VETOES = {"kill_switch_drawdown", "daily_loss_limit", "portfolio_margin"}

LegKey = tuple[str, PositionSide]
SignalFn = Callable[[pd.DataFrame, str], Signal | None]
BackfillFn = Callable[[str, int], Awaitable[list[Candle]]]


class Orchestrator:
    def __init__(
        self,
        executor: Executor,
        settings: Settings | None = None,
        *,
        risk: RiskManager | None = None,
        alerts: AlertSink | None = None,
        sentiment_store: dict[str, SentimentScore] | None = None,
        signal_fn: SignalFn = compute_signal,
        backfill_fn: BackfillFn | None = None,
    ):
        self.executor = executor
        self.cfg = settings or load_settings()
        self.risk = risk or RiskManager(self.cfg)
        self.alerts = alerts or LoggingAlertSink()
        self.sentiment_store = sentiment_store if sentiment_store is not None else {}
        self.signal_fn = signal_fn
        self._backfill_fn = backfill_fn

        self.buffers: dict[str, list[Candle]] = {}
        self.expected: dict[LegKey, float] = {}     # piernas confirmadas
        self._in_flight: dict[LegKey, int] = {}      # piernas abiertas sin confirmar (→ edad)
        self._suspect_counts: dict[LegKey, int] = {}  # anomalías y su antigüedad (gracia)
        self._needs_rewarm: set[str] = set()
        self.last_candle_time: dict[str, datetime] = {}
        self.halted = False

        self._lock = asyncio.Lock()
        self._interval = timedelta(milliseconds=interval_to_ms(self.cfg.market.timeframe))

        # --- Fast Path (Plan V2 §2.3): cola de eventos + cooldown por símbolo ---
        # El productor (_event_loop) empuja EventIntents; el consumidor
        # (_event_consumer) los pasa a on_event, que comparte el MISMO self._lock
        # que el lazo de velas (un solo punto de serialización). _last_event_trade
        # alimenta el cooldown de decide_event.
        self._event_queue: asyncio.Queue[EventIntent] = asyncio.Queue()
        self._last_event_trade: dict[str, datetime] = {}

        # --- Fast Path (Plan V2 §2.5(i)): micro-buffer rodante de markPrice@1s ---
        # Deque por símbolo de (timestamp, mark_price). Lo alimenta el productor WS
        # (_ingest_mark_price, push SÍNCRONO) y lo lee _price_impulse_bps dentro de
        # self._lock (sin await entre snapshot y cálculo → atómico). Reemplaza la
        # fuente del impulso (antes: vela 5m cerrada, ventana pre-noticia).
        self._markprice: dict[str, deque[tuple[datetime, float]]] = {}

        # Inicio de la sesión en vivo (epoch ms): frontera del PnL realizado por
        # símbolo que muestra el dashboard. Lo fija run(); aquí se inicializa
        # defensivamente por si _realized_pnl_loop se invoca fuera de run() (tests).
        self._session_start_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

    # ------------------------------ arranque ------------------------------

    async def startup(self) -> "Orchestrator":
        """Impone hedge mode, recarga el estado de sesión y adopta posiciones."""
        await self.executor.startup()
        await self._load_session()
        await self._adopt_positions()
        # Snapshot inicial: el dashboard muestra datos frescos desde el primer segundo
        # sin tener que esperar el próximo cierre de vela (hasta 5 min de espera).
        acct = await self.executor.exchange.get_account()
        await self._record_equity(acct, datetime.now(timezone.utc))
        return self

    async def _load_session(self) -> None:
        storage = self.executor.storage
        if storage is None:
            return
        st = await storage.load_session_state()
        if st is None:
            return
        self.executor.load_session(
            peak_wallet=st["peak_wallet"], day_start_wallet=st["day_start_wallet"],
            day=date.fromisoformat(st["day"]),
        )
        self.risk.kill_switch_active = st["kill_switch"]
        logger.info("estado de sesión recargado: %s", st)

    async def _adopt_positions(self) -> None:
        """Adopta las piernas que el exchange ya tiene (reinicio en caliente).

        Sin esto, un reinicio con posiciones abiertas las vería como "desconocidas"
        y dispararía un HALT. Verifica además que cada pierna tenga su STOP_MARKET:
        una posición desnuda tras un reinicio es el riesgo #1 → halt y alerta.
        """
        acct = await self.executor.exchange.get_account()
        for p in acct.positions:
            key = (p.symbol, p.position_side)
            self.expected[key] = p.qty
            open_orders = await self.executor.exchange.get_open_orders(p.symbol)
            has_stop = any(o.type == OrderType.STOP_MARKET for o in open_orders)
            if not has_stop:
                self.halted = True
                self.alerts.alert(AlertLevel.CRITICAL, "naked_position",
                                  f"{p.symbol} {p.position_side.value} sin STOP tras reinicio "
                                  f"— revisión manual requerida")
            else:
                self.alerts.alert(AlertLevel.INFO, "adopt",
                                  f"{p.symbol} {p.position_side.value} qty={p.qty}")

    # ------------------------- warmup / backfill -------------------------

    async def warmup(self, sym: str) -> bool:
        """Rellena el buffer del símbolo con velas históricas REST (contiguas)."""
        return await self._backfill_buffer(sym)

    async def _backfill_buffer(self, sym: str) -> bool:
        if self._backfill_fn is None:
            return False
        candles = await self._backfill_fn(sym, self._buffer_target())
        self.buffers[sym] = list(candles)
        if candles:
            # El feed está FRESCO justo tras el backfill (acabamos de traer datos):
            # marcamos la hora de llegada, no el open_time de la última vela (que ya
            # es ~1 intervalo viejo y dispararía un stale_feed falso al arrancar).
            self.last_candle_time[sym] = datetime.now(timezone.utc)
        self._needs_rewarm.discard(sym)
        return True

    async def _rewarm(self, sym: str) -> None:
        """Tras un hueco: re-backfillea (contiguo) o, sin fuente, reinicia el buffer."""
        if await self._backfill_buffer(sym):
            self.alerts.alert(AlertLevel.WARNING, "rewarm",
                              f"{sym}: buffer re-backfilleado tras un hueco de velas")
        else:
            buf = self.buffers.get(sym, [])
            self.buffers[sym] = buf[-1:] if buf else []  # arranca contiguo desde aquí
            self._needs_rewarm.discard(sym)
            self.alerts.alert(AlertLevel.WARNING, "rewarm_degraded",
                              f"{sym}: sin fuente de backfill — buffer reiniciado")

    # ------------------------- manejador por vela -------------------------

    async def on_closed_candle(self, candle: Candle, *, now: datetime | None = None) -> None:
        now = now or datetime.now(timezone.utc)
        sym = candle.symbol
        self._ingest_candle(sym, candle, now)

        if self.halted:
            return
        if sym in self._needs_rewarm:
            await self._rewarm(sym)
            return  # re-calentando: operamos en el ciclo siguiente
        if len(self.buffers.get(sym, [])) < self.cfg.orchestrator.warmup_candles:
            return

        async with self._lock:  # sección crítica serializada
            await self._cycle(sym, candle, now)

    def _ingest_candle(self, sym: str, candle: Candle, now: datetime) -> None:
        self.last_candle_time[sym] = now
        buf = self.buffers.setdefault(sym, [])
        if buf:
            last = buf[-1].open_time
            if candle.open_time <= last:
                return  # duplicada o atrasada: ignora
            if candle.open_time > last + self._interval:
                self._needs_rewarm.add(sym)  # hueco: faltan velas intermedias
        buf.append(candle)
        # Retén lo suficiente para formar el régimen HTF, con holgura para que el
        # EMA del HTF tenga ventana estable; nunca menos del mínimo histórico (200).
        maxlen = max(self._buffer_target() * 2, 200)
        if len(buf) > maxlen:
            del buf[: len(buf) - maxlen]

    def _fresh_sentiment(self, sym: str, now: datetime) -> SentimentScore | None:
        """El sentimiento del store SOLO si no ha caducado (TTL en vivo).

        El `_sentiment_loop` hace `store.update(...)` cada poll y solo pisa las
        claves que el fetch devuelve; un símbolo sin noticia fresca conservaría su
        último score para siempre, y `_cycle` lo reusaría en cada vela. Aquí lo
        caducamos contra `analyzed_at`: pasado `sentiment_ttl_seconds`, es como no
        tener noticia (None) y se purga la clave (el poller la reescribe si llega
        una fresca). Seguro sin lock extra: corre dentro de la sección crítica y
        no hay `await` entre el get y el pop, así que es atómico frente al poller.
        """
        sent = self.sentiment_store.get(sym)
        if sent is None:
            return None
        age = (now - sent.analyzed_at).total_seconds()
        if age > self.cfg.confluence.sentiment_ttl_seconds:
            self.sentiment_store.pop(sym, None)  # purga: no volver a evaluarla
            return None
        return sent

    def _ingest_mark_price(self, sym: str, ts: datetime, price: float) -> None:
        """Push SÍNCRONO de un tick markPrice@1s al deque del símbolo (Fase 2.5(i)).

        Sin `await`: corre atómico frente a la lectura de `_price_impulse_bps`
        (ambos en el mismo event loop). Retiene `markprice_buffer_seconds` usando el
        event time del PROPIO tick como reloj (consistente, inmune al skew con el
        wall-clock); desaloja por la izquierda lo que cae fuera de la ventana.
        """
        dq = self._markprice.get(sym)
        if dq is None:
            dq = self._markprice[sym] = deque()
        dq.append((ts, price))
        cutoff = ts - timedelta(seconds=self.cfg.event.markprice_buffer_seconds)
        while dq and dq[0][0] < cutoff:
            dq.popleft()

    def _price_impulse_bps(
        self, sym: str, window_seconds: int, now: datetime
    ) -> float | None:
        """Impulso de precio (con signo, en bps) sobre los ticks markPrice@1s reales.

        Mide el retorno POST-noticia sobre la ventana correcta `[now−window, now]`,
        no la vela 5m cerrada (que terminaba ANTES de la noticia: error de ventana,
        ver Fase 2.5 del plan). FALLA-CERRADO con `None` en tres casos —el
        orquestador traduce `None` a "no operar", aun con el gate de impulso ablado
        (`confirm_impulse_bps=0`): entrar sin precio en vivo viola el control de
        riesgo. Casos:
            1. deque vacío → None.
            2. stale: el tick más reciente es más viejo que `markprice_stale_seconds`
               (feed congelado) → None.
            3. ventana no cubierta: el tick más viejo es posterior a `now−window`
               (no hay histórico para medir la ventana completa) → None.
            4. frío: menos de `markprice_min_ticks` ticks dentro de la ventana → None.
        Si pasa todo: ref = primer tick con ts ≥ now−window; retorno (ref→último).
        """
        ev = self.cfg.event
        dq = self._markprice.get(sym)
        if not dq:
            return None
        newest_ts, newest_px = dq[-1]
        if (now - newest_ts).total_seconds() > ev.markprice_stale_seconds:
            return None
        cutoff = now - timedelta(seconds=window_seconds)
        if dq[0][0] > cutoff:                      # el buffer no cubre la ventana
            return None
        in_window = [(ts, px) for ts, px in dq if ts >= cutoff]
        if len(in_window) < ev.markprice_min_ticks:
            return None
        ref_px = in_window[0][1]
        if ref_px <= 0:
            return None
        return (newest_px / ref_px - 1.0) * 10_000.0

    def _compute_atr_baseline(self, sym: str) -> float | None:
        """Mediana del ATR sobre vol_regime_lookback velas (línea base de volatilidad).

        La mediana es robusta contra el propio spike que intentamos medir: la media
        se inflaría con el outlier que está en curso. Si el buffer es insuficiente,
        devuelve None → el Risk Manager usa vol_damp=1.0 (sin recorte), no un veto.
        """
        buf = self.buffers.get(sym, [])
        lookback = self.cfg.risk.vol_regime_lookback
        if len(buf) < lookback + 1:
            return None
        atr_vals = _atr_series(self._buffer_df(sym), self.cfg.risk.atr_period)
        valid = [v for v in atr_vals if not pd.isna(v)]
        if len(valid) < lookback:
            return None
        return statistics.median(valid[-lookback:])

    def _resolve_scope(self, scope: list[str]) -> list[str]:
        """Resuelve el symbol_scope de una noticia a los símbolos que operamos.

        Delega en `src/core/scope.resolve_scope` (única fuente de verdad, compartida
        con el Slow Path): "*" → todos; en otro caso machea por nombre completo o por
        ACTIVO BASE (BTC → BTCUSDT, vía market.quote_assets). Ignora lo no seguido.
        """
        return resolve_scope(
            scope, self.cfg.market.symbols, self.cfg.market.quote_assets
        )

    async def _cycle(self, sym: str, candle: Candle, now: datetime) -> None:
        acct = await self.executor.exchange.get_account()
        actual = {(p.symbol, p.position_side): p.qty for p in acct.positions}

        # Observabilidad (dashboard): foto de equity por ciclo, ANTES de cualquier
        # early-return de reconciliación → la curva de capital y la señal de
        # liveness siguen frescas aunque el ciclo no llegue a operar.
        await self._record_equity(acct, now)

        # --- 1a. Piernas en vuelo: promover si el exchange ya las confirma ---
        grace = self.cfg.orchestrator.reconcile_grace_cycles
        for key in list(self._in_flight):
            if key in actual:
                del self._in_flight[key]                 # confirmada por el exchange
            else:
                self._in_flight[key] += 1
                if self._in_flight[key] > grace:         # nunca apareció: no cuajó
                    del self._in_flight[key]
                    self.expected.pop(key, None)
                    self.alerts.alert(AlertLevel.WARNING, "open_unconfirmed", str(key))

        # --- 1b. Clasificar (ignora lo en vuelo) ---
        report = classify_reconciliation(
            self.expected, actual, self.cfg.execution.reconcile_position_tolerance,
            in_flight=set(self._in_flight),
        )
        for key in report.resync_keys:                   # cierres benignos por SL/TP
            self.expected.pop(key, None)
            self._suspect_counts.pop(key, None)
        if report.resync_keys:
            self.alerts.alert(AlertLevel.WARNING, "resync",
                              f"cerradas por SL/TP: {self._fmt_keys(report.resync_keys)}")

        # --- 1c. Sospechas: ventana de gracia antes del HALT ---
        seen = set(report.suspect_keys)
        for key in list(self._suspect_counts):
            if key not in seen:
                del self._suspect_counts[key]            # transitorio resuelto → reinicia
        for key in seen:
            self._suspect_counts[key] = self._suspect_counts.get(key, 0) + 1
        if any(c >= grace for c in self._suspect_counts.values()):
            self.halted = True
            self.alerts.alert(AlertLevel.CRITICAL, "reconcile_halt",
                              f"divergencia sostenida: esperado={self._fmt(self.expected)} "
                              f"real={self._fmt(actual)}")
            return
        if seen:
            self.alerts.alert(AlertLevel.WARNING, "reconcile_suspect",
                              f"anomalía en gracia {self._fmt_keys(seen)} — re-chequeando")
            return  # en gracia: no operamos esta vela

        # --- 2. Señal base (timeframe de entrada) → ATR para los stops ---
        # El ATR de los stops DEBE salir del timeframe base (5m): un trade
        # intradía con stops de 1h sería ~3.5x más ancho de lo debido. Por eso la
        # señal base se mantiene en la vela de trabajo aunque la DIRECCIÓN ya no
        # salga de aquí (Opción 2).
        base_signal = self.signal_fn(self._buffer_df(sym), sym)
        if base_signal is None:
            return
        atr = base_signal.features.get("atr")
        if atr is None:
            return

        # --- 2b. Régimen (Opción 2): el quant sobre velas resampleadas al HTF.
        # Ya NO origina la dirección; solo confirma/dimensiona la apuesta de la
        # noticia. Si aún no hay suficientes velas HTF cerradas, régimen NEUTRO
        # (score 0): la noticia opera con tamaño reducido, no se bloquea.
        regime = self._regime_signal(sym)
        if regime is None:
            regime = base_signal.model_copy(update={"score": 0.0})

        # --- 3. Confluencia: la NOTICIA origina; el régimen confirma (TTL fresco) ---
        sentiment = self._fresh_sentiment(sym, now)
        decision = decide(regime, sentiment, self.cfg, as_of=now)
        await self._record_decision(decision, "slow")  # feed del dashboard (incl. HOLD)

        # --- 4. Veredicto del Risk Manager ---
        state = await self.executor.snapshot_portfolio(now=now)
        confidence = sentiment.confidence if sentiment is not None else 1.0
        assessment = self.risk.assess(
            decision, price=candle.close, atr=atr, state=state,
            filters=self.executor.filters[sym], confidence=confidence,
        )
        if not assessment.approved and assessment.reason in _CRITICAL_VETOES:
            self.alerts.alert(AlertLevel.CRITICAL, assessment.reason, sym)

        # --- 5. Política de una pierna por símbolo ---
        want = assessment.order.position_side if assessment.approved else None
        held = next((ps for (s, ps) in (set(self.expected) | set(self._in_flight))
                     if s == sym), None)
        action = decide_position_action(held, want)
        if action == PositionAction.OPEN:
            await self._open(assessment.order, now, mark_price=candle.close)
        elif action == PositionAction.FLIP:
            # FLIP desacoplado: solo cerramos; la apertura inversa ocurre en el
            # ciclo siguiente, con snapshot fresco y sin colisión de margen.
            await self.executor.close_position(sym, held, now=now)
            self.expected.pop((sym, held), None)
            self._in_flight.pop((sym, held), None)
            self.alerts.alert(AlertLevel.INFO, "flip_close",
                              f"{sym}: cerrada {held.value}; apertura inversa en el próximo ciclo")

        await self._persist_session(now)

    async def _open(self, order, now: datetime, *, mark_price: float) -> None:
        key = (order.symbol, order.position_side)
        report = await self.executor.open_position(order, now=now, mark_price=mark_price)
        if report.ok:
            # Con IOC parcial, confirmed_qty puede ser < order.quantity. Registrar
            # la cantidad REAL para que la reconciliación no detecte una divergencia
            # espuria y dispare un HALT (el bug que esta línea corrige).
            qty = report.confirmed_qty if report.confirmed_qty > 0 else order.quantity
            self.expected[key] = qty
            self._in_flight[key] = 0   # pendiente de que el exchange la confirme
            self.alerts.alert(AlertLevel.INFO, "open",
                              f"{order.symbol} {order.position_side.value} qty={qty:.6f}")
        else:
            self.alerts.alert(AlertLevel.WARNING, "open_failed",
                              f"{order.symbol}: {report.detail}")

    # ------------------------- Fast Path: consumo de eventos -------------------------

    async def on_event(self, intent: EventIntent, *, now: datetime | None = None) -> None:
        """Consume un EventIntent del Fast Path (Plan V2 §2.3).

        Adquiere el MISMO `self._lock` que el lazo de velas: un evento que llega a
        mitad de un `_cycle` espera su turno, nunca observa un estado a medio
        aplicar. Gateado por `event.enabled` (gate maestro) y por `halted` (si un
        circuit breaker disparó, el Fast Path tampoco abre). La apertura va por el
        MISMO `_open`, así que queda registrada como in-flight y el lazo de velas
        no la confunde con una pierna desconocida (→ 0 HALTs por el Fast Path).
        """
        now = now or datetime.now(timezone.utc)
        if not self.cfg.event.enabled:
            return
        if self.halted:
            return
        async with self._lock:
            await self._handle_event(intent, now)

    async def _handle_event(self, intent: EventIntent, now: datetime) -> None:
        sym = intent.symbol
        ev = self.cfg.event

        # (1) Necesitamos buffer caliente: el stop usa ATR y el gate usa el impulso.
        if len(self.buffers.get(sym, [])) < self.cfg.orchestrator.warmup_candles:
            self.alerts.alert(AlertLevel.WARNING, "event_not_warm", sym)
            return
        signal = self.signal_fn(self._buffer_df(sym), sym)
        atr = signal.features.get("atr") if signal is not None else None
        if atr is None:
            self.alerts.alert(AlertLevel.WARNING, "event_no_atr", sym)
            return

        # (2) Línea base de régimen de volatilidad para el vol_damp (Fase 2.4).
        #     Se computa antes de decide_event (lectura síncrona del buffer, sin await).
        atr_baseline = self._compute_atr_baseline(sym)

        # (3) Plano de datos en tiempo real (Fase 2.5(i)): impulso desde el deque de
        #     markPrice. FALLA-CERRADO: None (buffer frío/stale) → no operar, SIEMPRE,
        #     aun con el gate de impulso ablado (confirm_impulse_bps=0). El None se
        #     resuelve AQUÍ (orquestador), así decide_event sigue puro (recibe float).
        impulse = self._price_impulse_bps(sym, ev.confirm_window_seconds, now)
        if impulse is None:
            self.alerts.alert(AlertLevel.WARNING, "event_no_price", sym)
            return
        decision = decide_event(
            intent.sentiment, sym, impulse, self.cfg,
            as_of=now, last_event_trade_at=self._last_event_trade.get(sym),
        )
        await self._record_decision(decision, "event")  # feed del dashboard (incl. HOLD)
        if decision.action == Action.HOLD:
            # Auditoría: por qué un evento NO operó (kill criteria §C los revisa).
            self.alerts.alert(AlertLevel.INFO, "event_hold", f"{sym}: {decision.reason}")
            return

        # (4) Veredicto del Risk Manager en modo evento (Fase 2.4): stop más ancho,
        #     presupuesto menor, vol_damp activo si hay suficiente historia.
        price = self.buffers[sym][-1].close
        state = await self.executor.snapshot_portfolio(now=now)
        assessment = self.risk.assess(
            decision, price=price, atr=atr, state=state,
            filters=self.executor.filters[sym], confidence=intent.sentiment.confidence,
            mode="event", atr_baseline=atr_baseline,
        )
        if not assessment.approved:
            if assessment.reason in _CRITICAL_VETOES:
                self.alerts.alert(AlertLevel.CRITICAL, assessment.reason, sym)
            return

        # (4) Misma política de una pierna por símbolo y el MISMO _open.
        want = assessment.order.position_side
        held = next((ps for (s, ps) in (set(self.expected) | set(self._in_flight))
                     if s == sym), None)
        action = decide_position_action(held, want)
        if action == PositionAction.OPEN:
            await self._open(assessment.order, now, mark_price=price)
            self._last_event_trade[sym] = now      # arma el cooldown del símbolo
            self.alerts.alert(AlertLevel.INFO, "event_open",
                              f"{sym} {want.value} por evento ({decision.reason})")
        elif action == PositionAction.FLIP:
            # FLIP desacoplado, igual que el Slow Path: solo cerramos aquí.
            await self.executor.close_position(sym, held, now=now)
            self.expected.pop((sym, held), None)
            self._in_flight.pop((sym, held), None)
            self.alerts.alert(AlertLevel.INFO, "event_flip_close",
                              f"{sym}: cerrada {held.value} por evento; inversa en el próximo ciclo")
        await self._persist_session(now)

    # ------------------------- Fast Path: productor + cola -------------------------

    async def _enqueue_event(self, score: SentimentScore) -> None:
        """Resuelve el scope de un score y encola un EventIntent por símbolo."""
        for sym in self._resolve_scope(score.symbol_scope):
            await self._event_queue.put(EventIntent(symbol=sym, sentiment=score))

    async def _event_loop(
        self, fetch: Callable[[], Awaitable[list[SentimentScore]]]
    ) -> None:
        """Productor: sondea shocks más rápido que el Slow Path y los encola.

        ⚠️ Capa operativa (RSS + Claude): se valida en testnet.
        """
        while True:
            for score in await fetch():
                await self._enqueue_event(score)
            await asyncio.sleep(self.cfg.event.poll_interval_seconds)

    async def _event_consumer(self) -> None:
        """Consumidor: drena la cola y entrega cada intent a on_event (serializado)."""
        while True:
            intent = await self._event_queue.get()
            try:
                await self.on_event(intent)
            finally:
                self._event_queue.task_done()

    async def _record_equity(self, acct, now: datetime) -> None:
        """Snapshot de equity del ciclo (curva de capital del dashboard).

        Read-only para el trading: solo ESCRIBE en SQLite lo que ya tenemos del
        `get_account` del ciclo; no toca el exchange. No-op si no hay storage
        (tests/demo). Una fila por ciclo → serie temporal que `session_state`
        (fila única) no podía dar.
        """
        storage = self.executor.storage
        if storage is None:
            return
        upnl = sum(p.unrealized_pnl for p in acct.positions)
        positions = [
            {"symbol": p.symbol, "side": p.position_side.value, "qty": p.qty,
             "entry_price": p.entry_price, "upnl": p.unrealized_pnl}
            for p in acct.positions
        ]
        await storage.save_equity_snapshot(
            ts_ms=int(now.timestamp() * 1000), wallet=acct.wallet_balance,
            equity=acct.wallet_balance + upnl, upnl=upnl, positions=positions,
        )

    async def _record_decision(self, decision, source: str) -> None:
        """Persiste la decisión de la confluencia (incluido HOLD) para el dashboard.

        Da trazabilidad al "¿por qué (no) operó?" que antes no dejaba rastro: los
        HOLD no generaban orden, así que eran invisibles. No-op sin storage.
        """
        storage = self.executor.storage
        if storage is None:
            return
        await storage.save_decision(
            ts_ms=int(decision.timestamp.timestamp() * 1000), symbol=decision.symbol,
            action=decision.action.value, reason=decision.reason,
            quant_score=decision.quant_score, sentiment_score=decision.sentiment_score,
            size_factor=decision.size_factor, source=source,
        )

    async def _persist_session(self, now: datetime) -> None:
        storage = self.executor.storage
        if storage is None:
            return
        peak, day_start, day = self.executor.session_state()
        day_str = day.isoformat() if isinstance(day, date) else now.date().isoformat()
        await storage.save_session_state(
            peak_wallet=peak, day_start_wallet=day_start, day=day_str,
            kill_switch=self.risk.kill_switch_active,
        )

    def _buffer_df(self, sym: str) -> pd.DataFrame:
        buf = self.buffers[sym]
        return pd.DataFrame({
            "open_time": [c.open_time for c in buf],
            "open": [c.open for c in buf], "high": [c.high for c in buf],
            "low": [c.low for c in buf], "close": [c.close for c in buf],
            "volume": [c.volume for c in buf],
        })

    # ------------------------- régimen HTF (Opción 2) -------------------------

    def _htf_ratio(self) -> int:
        """Velas base por vela HTF (ej. 1h/5m = 12). Deriva de la config, no se
        hardcodea: si cambias htf_timeframe o timeframe, el ratio se recalcula."""
        return (interval_to_ms(self.cfg.market.htf_timeframe)
                // interval_to_ms(self.cfg.market.timeframe))

    def _buffer_target(self) -> int:
        """Velas base a backfillear/retener: el MAYOR entre el warmup operativo
        (gate de 5m) y las que necesita el régimen HTF (regime_htf_bars * ratio).
        Sin esto, el buffer de 5m no tendría historia para formar las ~50 velas de
        1h que el EMA-cross necesita para leer la tendencia."""
        return max(self.cfg.orchestrator.warmup_candles,
                   self.cfg.orchestrator.regime_htf_bars * self._htf_ratio())

    def _buffer_df_htf(self, sym: str) -> pd.DataFrame:
        """Resamplea el buffer de velas base al HTF, devolviendo SOLO buckets
        completos (causal: nunca usa la vela HTF en formación). Un bucket es
        completo si contiene exactamente `_htf_ratio()` velas base; los incompletos
        (la última en curso, o huecos del feed) se descartan para no fabricar una
        vela HTF con datos parciales."""
        base = self._buffer_df(sym)
        if base.empty:
            return base
        ratio = self._htf_ratio()
        idx = pd.to_datetime(base["open_time"], utc=True)
        rule = pd.Timedelta(milliseconds=interval_to_ms(self.cfg.market.htf_timeframe))
        grouped = base.set_index(idx).resample(rule, label="left", closed="left")
        agg = grouped.agg({"open": "first", "high": "max", "low": "min",
                           "close": "last", "volume": "sum"})
        full = grouped.size() == ratio          # solo buckets HTF completos
        return agg[full].reset_index(drop=True)

    def _regime_signal(self, sym: str) -> Signal | None:
        """Señal de RÉGIMEN: el quant corrido sobre las velas resampleadas al HTF.
        Devuelve None si aún no hay velas HTF completas suficientes (el llamador lo
        trata como régimen neutro). Reusa el MISMO signal_fn que la señal base, así
        que vivo y backtest comparten la fórmula; solo cambia el timeframe de entrada."""
        htf_df = self._buffer_df_htf(sym)
        if htf_df.empty:
            return None
        return self.signal_fn(htf_df, sym)

    @staticmethod
    def _fmt(d: dict[LegKey, float]) -> dict[str, float]:
        return {f"{k[0]}:{k[1].value}": v for k, v in d.items()}

    @staticmethod
    def _fmt_keys(keys) -> list[str]:
        return [f"{k[0]}:{k[1].value}" for k in keys]

    # ------------------------- circuit breaker (a): feed -------------------------

    def _stale_threshold_seconds(self) -> float:
        """Umbral de obsolescencia del feed, escalado al timeframe.

        Con velas cerradas, entre vela y vela pasa ~1 intervalo; declarar el feed
        muerto a los stale_feed_seconds (30) en un timeframe de 5m daría un HALT
        falso en cada hueco normal. Tomamos el mayor entre el absoluto y
        stale_feed_intervals × intervalo.
        """
        r = self.cfg.risk
        return max(r.stale_feed_seconds,
                   r.stale_feed_intervals * self._interval.total_seconds())

    def check_feed_health(self, *, now: datetime | None = None) -> bool:
        """Si algún símbolo lleva sin velas más del umbral (timeframe-aware) → halt."""
        now = now or datetime.now(timezone.utc)
        stale = self._stale_threshold_seconds()
        for sym, last in self.last_candle_time.items():
            if (now - last).total_seconds() > stale:
                if not self.halted:
                    self.halted = True
                    self.alerts.alert(AlertLevel.CRITICAL, "stale_feed",
                                      f"{sym}: sin velas hace más de {stale:.0f}s")
                return False
        return True

    # ------------------------- capa operativa (red) -------------------------

    async def run(
        self,
        data_client,
        *,
        sentiment_fetch: Callable[[], Awaitable[dict[str, SentimentScore]]] | None = None,
        event_fetch: Callable[[], Awaitable[list[SentimentScore]]] | None = None,
    ) -> None:
        """Lazo en vivo: backfill + streams de velas + poller + watchdog + Fast Path.

        ⚠️ Capa operativa (websockets + RSS + Claude): se valida en testnet.
        """
        if self._backfill_fn is None:
            self._backfill_fn = self._make_rest_backfill(data_client)
        # Frontera del PnL realizado por símbolo del dashboard: ESTA corrida.
        self._session_start_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        await self.startup()
        for sym in self.cfg.market.symbols:
            await self.warmup(sym)

        tf = self.cfg.market.timeframe
        tasks = [
            self._supervise(
                lambda s=sym: stream_candles(data_client, s, tf, self.on_closed_candle),
                name=f"stream:{sym}",
            )
            for sym in self.cfg.market.symbols
        ]
        tasks.append(self._supervise(self._watchdog_loop, name="watchdog"))
        # Observabilidad: PnL realizado por símbolo para el dashboard. Tarea aparte,
        # solo si hay storage; un fallo suyo nunca toca el lazo de trading.
        if self.executor.storage is not None:
            tasks.append(self._supervise(self._realized_pnl_loop, name="realized_pnl"))
        # Slow Path: overlay de sentimiento de noticias. Gate de seguridad de
        # presupuesto (sentiment.enabled, default false): con el flag apagado NO se
        # arranca el loop → cero llamadas a Claude y la señal quant queda PURA. Mismo
        # patrón que event.enabled para el Fast Path.
        if self.cfg.sentiment.enabled and sentiment_fetch is not None:
            tasks.append(self._supervise(
                lambda: self._sentiment_loop(sentiment_fetch), name="sentiment"))
        # Fast Path: solo si está habilitado. El gate maestro evita que un despiste
        # arranque el Fast Path antes de validarlo en testnet.
        if self.cfg.event.enabled:
            # Plano de datos en tiempo real (Fase 2.5(i)): un stream markPrice@1s por
            # símbolo alimenta el micro-buffer; es prerrequisito del impulso, así que
            # arranca aunque aún no haya event_fetch cableado (Fase 2.5(ii)).
            for sym in self.cfg.market.symbols:
                tasks.append(self._supervise(
                    lambda s=sym: stream_mark_price(data_client, s, self._ingest_mark_price),
                    name=f"markprice:{sym}"))
            # Productor + consumidor de eventos: solo si hay fuente de eventos.
            if event_fetch is not None:
                tasks.append(self._supervise(
                    lambda: self._event_loop(event_fetch), name="event_producer"))
                tasks.append(self._supervise(self._event_consumer, name="event_consumer"))
        await asyncio.gather(*tasks)

    def _make_rest_backfill(self, data_client) -> BackfillFn:
        tf = self.cfg.market.timeframe

        async def backfill(sym: str, n: int) -> list[Candle]:
            rows = await retry_with_backoff(
                lambda: data_client.get_klines(symbol=sym, interval=tf, limit=n + 1))
            # descarta la última (posible vela en formación); el stream la dará al cerrar
            return [rest_kline_to_candle(r, sym, tf) for r in rows[:-1]]

        return backfill

    async def _supervise(self, factory: Callable[[], Awaitable], *, name: str,
                         backoff: float = 5.0) -> None:
        """Mantiene viva una tarea: si cae (no por cancelación), reinicia con backoff."""
        while True:
            try:
                await factory()
                return
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("tarea %s cayó; reinicio en %.0fs", name, backoff)
                self.alerts.alert(AlertLevel.WARNING, "task_restart", name)
                await asyncio.sleep(backoff)

    async def _watchdog_loop(self) -> None:
        period = max(self.cfg.risk.stale_feed_seconds / 2.0, 1.0)
        while True:
            await asyncio.sleep(period)
            self.check_feed_health()

    async def _sentiment_loop(
        self, fetch: Callable[[], Awaitable[dict[str, SentimentScore]]]
    ) -> None:
        # Defensa en profundidad del gate de presupuesto: run() ya no arranca esta
        # tarea con sentiment.enabled=false, pero si alguien la invoca fuera de run()
        # (test o futuro refactor), el early-return garantiza CERO llamadas a Claude
        # con el flag apagado. El gate primario sigue en run() (ni crea la tarea).
        if not self.cfg.sentiment.enabled:
            return
        while True:
            self.sentiment_store.update(await fetch())
            await asyncio.sleep(self.cfg.sentiment.poll_interval_seconds)

    async def _realized_pnl_loop(self) -> None:
        """Sondea el PnL REALIZADO por símbolo (income history) para el dashboard.

        Tarea APARTE del trading: si el fetch falla, NO sobrescribe la tabla (mantiene
        los últimos valores buenos) y reintenta al siguiente sondeo; un fallo fatal lo
        recoge `_supervise`. Mide desde `self._session_start_ms`, así el panel refleja
        cómo va CADA moneda en la corrida actual (excluye sesiones previas). Refresca
        TODOS los símbolos del universo (incluso 0) para no dejar valores rancios.
        """
        storage = self.executor.storage
        if storage is None:
            return
        while True:
            try:
                realized = await self.executor.exchange.get_realized_pnl(
                    self._session_start_ms)
                full = {sym: realized.get(sym, 0.0) for sym in self.cfg.market.symbols}
                await storage.save_realized_pnl(
                    realized=full,
                    ts_ms=int(datetime.now(timezone.utc).timestamp() * 1000))
            except BinanceAPIException as e:
                # -1021: reloj local adelantado (típico al despertar el Mac).
                # retry_with_backoff dentro de get_realized_pnl ya recalibró el
                # offset; aun así, aquí lo tratamos aparte para NO spinnear: si el
                # primer reajuste no bastó (macOS aún re-sincronizando por NTP),
                # esperamos un delay corto y dejamos que el siguiente ciclo reintente
                # con el reloj ya corregido, en vez de martillar la API.
                if e.code == CLOCK_AHEAD_CODE:
                    delay = self.cfg.orchestrator.clock_retry_delay_seconds
                    logger.warning("reloj desincronizado (-1021), ajustando offset; "
                                   "reintento en %.0fs", delay)
                    await asyncio.sleep(delay)
                    continue
                logger.warning("sondeo de PnL realizado falló; se reintenta",
                               exc_info=True)
            except Exception:
                logger.warning("sondeo de PnL realizado falló; se reintenta",
                               exc_info=True)
            await asyncio.sleep(self.cfg.orchestrator.realized_pnl_poll_seconds)
