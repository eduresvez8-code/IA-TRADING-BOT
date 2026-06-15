"""Orquestador en vivo: el lazo que une todos los motores del bot.

Por cada vela cerrada (`on_closed_candle`): reconcilia el estado con el exchange,
calcula la señal cuantitativa, lee el último sentimiento, cruza en la confluencia,
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
from datetime import date, datetime, timedelta, timezone
from typing import Awaitable, Callable

import pandas as pd

from src.core.config import Settings, load_settings
from src.core.models import Candle, OrderType, PositionSide, SentimentScore, Signal
from src.data.binance_client import (
    interval_to_ms,
    rest_kline_to_candle,
    retry_with_backoff,
    stream_candles,
)
from src.decision.confluence import decide
from src.execution.executor import Executor
from src.orchestrator.alerts import AlertLevel, AlertSink, LoggingAlertSink
from src.orchestrator.policy import (
    PositionAction,
    classify_reconciliation,
    decide_position_action,
)
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

    # ------------------------------ arranque ------------------------------

    async def startup(self) -> "Orchestrator":
        """Impone hedge mode, recarga el estado de sesión y adopta posiciones."""
        await self.executor.startup()
        await self._load_session()
        await self._adopt_positions()
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
        candles = await self._backfill_fn(sym, self.cfg.orchestrator.warmup_candles)
        self.buffers[sym] = list(candles)
        if candles:
            self.last_candle_time[sym] = candles[-1].open_time
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
        maxlen = max(self.cfg.orchestrator.warmup_candles * 2, 200)
        if len(buf) > maxlen:
            del buf[: len(buf) - maxlen]

    async def _cycle(self, sym: str, candle: Candle, now: datetime) -> None:
        acct = await self.executor.exchange.get_account()
        actual = {(p.symbol, p.position_side): p.qty for p in acct.positions}

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

        # --- 2. Señal cuantitativa ---
        signal = self.signal_fn(self._buffer_df(sym), sym)
        if signal is None:
            return
        atr = signal.features.get("atr")
        if atr is None:
            return

        # --- 3. Confluencia con el sentimiento más reciente ---
        sentiment = self.sentiment_store.get(sym)
        decision = decide(signal, sentiment, self.cfg)

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
            await self._open(assessment.order, now)
        elif action == PositionAction.FLIP:
            # FLIP desacoplado: solo cerramos; la apertura inversa ocurre en el
            # ciclo siguiente, con snapshot fresco y sin colisión de margen.
            await self.executor.close_position(sym, held, now=now)
            self.expected.pop((sym, held), None)
            self._in_flight.pop((sym, held), None)
            self.alerts.alert(AlertLevel.INFO, "flip_close",
                              f"{sym}: cerrada {held.value}; apertura inversa en el próximo ciclo")

        await self._persist_session(now)

    async def _open(self, order, now: datetime) -> None:
        key = (order.symbol, order.position_side)
        report = await self.executor.open_position(order, now=now)
        if report.ok:
            self.expected[key] = order.quantity
            self._in_flight[key] = 0   # pendiente de que el exchange la confirme
            self.alerts.alert(AlertLevel.INFO, "open",
                              f"{order.symbol} {order.position_side.value} qty={order.quantity}")
        else:
            self.alerts.alert(AlertLevel.WARNING, "open_failed",
                              f"{order.symbol}: {report.detail}")

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

    @staticmethod
    def _fmt(d: dict[LegKey, float]) -> dict[str, float]:
        return {f"{k[0]}:{k[1].value}": v for k, v in d.items()}

    @staticmethod
    def _fmt_keys(keys) -> list[str]:
        return [f"{k[0]}:{k[1].value}" for k in keys]

    # ------------------------- circuit breaker (a): feed -------------------------

    def check_feed_health(self, *, now: datetime | None = None) -> bool:
        """Si algún símbolo lleva sin velas más de stale_feed_seconds → halt."""
        now = now or datetime.now(timezone.utc)
        stale = self.cfg.risk.stale_feed_seconds
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
    ) -> None:
        """Lazo en vivo: backfill + streams de velas + poller + watchdog.

        ⚠️ Capa operativa (websockets + RSS + Claude): se valida en testnet.
        """
        if self._backfill_fn is None:
            self._backfill_fn = self._make_rest_backfill(data_client)
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
        if sentiment_fetch is not None:
            tasks.append(self._supervise(
                lambda: self._sentiment_loop(sentiment_fetch), name="sentiment"))
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
        while True:
            self.sentiment_store.update(await fetch())
            await asyncio.sleep(self.cfg.sentiment.poll_interval_seconds)
