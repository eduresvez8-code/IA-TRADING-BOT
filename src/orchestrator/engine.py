"""Orquestador en vivo: el lazo que une todos los motores del bot.

Por cada vela cerrada (`on_closed_candle`): reconcilia el estado con el exchange,
calcula la señal cuantitativa, lee el último sentimiento, cruza en la confluencia,
pide veredicto al Risk Manager y aplica la política de UNA pierna por símbolo
(abrir / flip / nada). Las salidas las gestionan los SL/TP ya colocados en el
exchange; un SL/TP que dispara se absorbe en la reconciliación del ciclo siguiente.

`on_closed_candle`, `check_feed_health` y las políticas son síncronas de lógica y
se prueban dirigiéndolas a mano. `run()` (websockets + poller de sentimiento +
supervisión) es la capa operativa que se valida en testnet con red.

Inyectables para tests: `signal_fn` (por defecto el quant engine) y
`sentiment_store` (dict symbol→SentimentScore que el poller mantiene en vivo).
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Awaitable, Callable

import pandas as pd

from src.core.config import Settings, load_settings
from src.core.models import Candle, PositionSide, SentimentScore, Signal
from src.data.binance_client import stream_candles
from src.decision.confluence import decide
from src.execution.executor import Executor
from src.orchestrator.alerts import AlertLevel, AlertSink, LoggingAlertSink
from src.orchestrator.policy import (
    PositionAction,
    ReconVerdict,
    classify_reconciliation,
    decide_position_action,
)
from src.quant.strategy import compute_signal
from src.risk.manager import RiskManager

logger = logging.getLogger("ia_trading.orchestrator")

# Motivos de veto que merecen alerta crítica (no solo "no operamos esta vela").
_CRITICAL_VETOES = {"kill_switch_drawdown", "daily_loss_limit", "portfolio_margin"}

SignalFn = Callable[[pd.DataFrame, str], Signal | None]


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
    ):
        self.executor = executor
        self.cfg = settings or load_settings()
        self.risk = risk or RiskManager(self.cfg)
        self.alerts = alerts or LoggingAlertSink()
        self.sentiment_store = sentiment_store if sentiment_store is not None else {}
        self.signal_fn = signal_fn

        self.buffers: dict[str, list[Candle]] = {}
        # modelo interno de piernas abiertas: (symbol, positionSide) -> qty
        self.expected: dict[tuple[str, PositionSide], float] = {}
        self.last_candle_time: dict[str, datetime] = {}
        self.halted = False

    # ------------------------------ arranque ------------------------------

    async def startup(self) -> "Orchestrator":
        """Delega en el Executor: impone hedge mode, leverage y filtros."""
        await self.executor.startup()
        return self

    # ------------------------- manejador por vela -------------------------

    async def on_closed_candle(self, candle: Candle, *, now: datetime | None = None) -> None:
        """Procesa una vela cerrada: reconcilia, decide y actúa."""
        now = now or datetime.now(timezone.utc)
        sym = candle.symbol
        self.last_candle_time[sym] = now

        buf = self.buffers.setdefault(sym, [])
        buf.append(candle)
        maxlen = max(self.cfg.orchestrator.warmup_candles * 2, 200)
        if len(buf) > maxlen:  # acota el buffer para no crecer sin fin
            del buf[: len(buf) - maxlen]

        if self.halted:
            return
        if len(buf) < self.cfg.orchestrator.warmup_candles:
            return

        # --- 1. Reconciliación: el exchange es la verdad ---
        acct = await self.executor.exchange.get_account()
        actual = {(p.symbol, p.position_side): p.qty for p in acct.positions}
        verdict = classify_reconciliation(
            self.expected, actual, self.cfg.execution.reconcile_position_tolerance
        )
        if verdict == ReconVerdict.HALT:
            self.halted = True
            self.alerts.alert(AlertLevel.CRITICAL, "reconcile_halt",
                              f"estado divergente: esperado={self.expected} real={actual}")
            return
        if verdict == ReconVerdict.RESYNC:
            self.alerts.alert(AlertLevel.WARNING, "resync",
                              f"{sym}: pierna cerrada por SL/TP — resincronizando con el exchange")
            self.expected = dict(actual)

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
        held = next((ps for (s, ps) in self.expected if s == sym), None)
        action = decide_position_action(held, want)

        if action == PositionAction.OPEN:
            await self._open(assessment.order, now)
        elif action == PositionAction.FLIP:
            await self.executor.close_position(sym, held, now=now)
            self.expected.pop((sym, held), None)
            self.alerts.alert(AlertLevel.INFO, "flip",
                              f"{sym}: {held.value} → {want.value}")
            await self._open(assessment.order, now)

    async def _open(self, order, now: datetime) -> None:
        report = await self.executor.open_position(order, now=now)
        if report.ok:
            self.expected[(order.symbol, order.position_side)] = order.quantity
            self.alerts.alert(AlertLevel.INFO, "open",
                              f"{order.symbol} {order.position_side.value} qty={order.quantity}")
        else:
            self.alerts.alert(AlertLevel.WARNING, "open_failed",
                              f"{order.symbol}: {report.detail}")

    def _buffer_df(self, sym: str) -> pd.DataFrame:
        buf = self.buffers[sym]
        return pd.DataFrame({
            "open_time": [c.open_time for c in buf],
            "open": [c.open for c in buf], "high": [c.high for c in buf],
            "low": [c.low for c in buf], "close": [c.close for c in buf],
            "volume": [c.volume for c in buf],
        })

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
        """Lazo en vivo: streams de velas + poller de sentimiento + watchdog.

        ⚠️ Capa operativa (websockets + RSS + Claude): se valida en testnet. La
        lógica de decisión (`on_closed_candle`) está cubierta por tests.
        """
        await self.startup()
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
