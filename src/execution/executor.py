"""Execution Engine: traduce Orders aprobadas en órdenes del exchange y vigila
que el estado local y el del exchange no se desincronicen (Futuros USD-M).

Responsabilidades:
    1. startup(): IMPONER hedge mode (con precondición de cuenta limpia), fijar
       el apalancamiento por símbolo y cachear los filtros de microestructura.
    2. open_position(order): entrada MARKET + SL + TP (vía translate), con
       reintentos idempotentes, y registro en el log auditado.
    3. close_position(symbol, side): cierre a mercado de una pierna (salida/flip).
    4. reconcile(expected): comparar lo que creemos tener con lo que el exchange
       reporta → circuit breaker (c) si difieren.
    5. snapshot_portfolio(): construir el PortfolioState que alimenta al Risk
       Manager (cierra el lazo del Sprint 5), llevando el pico y el inicio de día
       del wallet para el kill switch y la pérdida diaria.

El Executor habla con un `FuturesExchange` (Protocol): en tests/demo, el fake en
memoria; en producción, el adaptador real de python-binance.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable
from uuid import uuid4

from src.core.config import Settings, load_settings
from src.core.models import Order, PositionSide
from src.data.binance_client import retry_with_backoff
from src.data.storage import Storage
from src.execution.exchange import FuturesExchange, OrderRequest, OrderResult
from src.execution.translate import build_close_request, build_open_requests
from src.risk.manager import PortfolioState


class ExecutionStartupError(RuntimeError):
    """El arranque no puede dejar la cuenta en un estado seguro (hedge mode)."""


@dataclass
class ExecutionReport:
    """Resultado de abrir una posición: la entrada y sus protectoras."""

    order: Order
    entry: OrderResult | None
    protective: list[OrderResult] = field(default_factory=list)
    ok: bool = False
    detail: str = ""


@dataclass(frozen=True)
class ReconItem:
    symbol: str
    position_side: PositionSide
    expected_qty: float
    actual_qty: float


@dataclass(frozen=True)
class ReconResult:
    consistent: bool
    discrepancies: list[ReconItem] = field(default_factory=list)


def _uuid_id(tag: str) -> str:
    return f"bot-{tag}-{uuid4().hex[:18]}"


class Executor:
    def __init__(
        self,
        exchange: FuturesExchange,
        settings: Settings | None = None,
        storage: Storage | None = None,
        id_factory: Callable[[str], str] = _uuid_id,
    ):
        self.exchange = exchange
        self.cfg = settings or load_settings()
        self.storage = storage
        self._id_factory = id_factory
        self.filters: dict = {}
        # Estadísticos de sesión para el Risk Manager (kill switch / pérdida diaria).
        self._peak_wallet = 0.0
        self._day_start_wallet = 0.0
        self._day: object = None

    # ------------------- estado de sesión (persistencia) -------------------

    def load_session(self, *, peak_wallet: float, day_start_wallet: float,
                     day: object) -> None:
        """Restaura el pico y el inicio de día tras un reinicio en caliente."""
        self._peak_wallet = peak_wallet
        self._day_start_wallet = day_start_wallet
        self._day = day

    def session_state(self) -> tuple[float, float, object]:
        """(peak_wallet, day_start_wallet, day) para persistir."""
        return self._peak_wallet, self._day_start_wallet, self._day

    # ---------------------------- arranque ----------------------------

    async def startup(self) -> "Executor":
        """Impone hedge mode, fija leverage por símbolo y cachea filtros."""
        dual = await self.exchange.get_position_mode()
        if not dual:
            # Cambiar de modo exige cuenta sin posiciones (Binance -4068). Y NO
            # cerramos para forzar el cambio: sería liquidar a ciegas. Abortamos.
            acct = await self.exchange.get_account()
            if acct.positions:
                raise ExecutionStartupError(
                    "cuenta en one-way con posiciones abiertas: no se puede imponer "
                    "hedge mode sin cerrar a ciegas — revisión manual requerida"
                )
            await self.exchange.set_position_mode(True)

        for sym in self.cfg.market.symbols:
            await self.exchange.set_leverage(sym, self.cfg.risk.max_leverage)
            self.filters[sym] = await self.exchange.get_symbol_filters(sym)
        return self

    # ---------------------------- apertura ----------------------------

    async def open_position(self, order: Order, *, now: datetime | None = None) -> ExecutionReport:
        """Abre una pierna: entrada MARKET y, si llena, sus SL/TP protectores."""
        reqs = build_open_requests(
            order, working_type=self.cfg.execution.stop_working_type,
            id_factory=self._id_factory,
        )
        entry_req, protective_reqs = reqs[0], reqs[1:]

        entry_res = await self._send(entry_req, order.decision_reason, now)
        # Una entrada MARKET puede responder NEW/PARTIALLY_FILLED y llenarse un
        # instante después (Binance real/testnet). No nos fiamos del status de la
        # respuesta: confirmamos contra la POSICIÓN real antes de proteger.
        if entry_res.status not in ("FILLED", "PARTIALLY_FILLED"):
            if not await self._confirm_fill(order.symbol, order.position_side):
                # Sin pierna confirmada: no colocamos protectoras (un SL/TP
                # huérfano podría cerrar otra cosa más tarde).
                return ExecutionReport(order, entry_res, [], ok=False,
                                       detail=f"entrada no llenó (status={entry_res.status})")

        protective = [await self._send(r, order.decision_reason, now) for r in protective_reqs]
        return ExecutionReport(order, entry_res, protective, ok=True, detail="abierta")

    async def _confirm_fill(self, symbol: str, position_side: PositionSide) -> bool:
        """Confirma que la pierna existe en el exchange (entrada MARKET ya llenada).

        Reintenta `fill_confirm_retries` veces con `fill_confirm_delay_seconds`:
        absorbe el desfase entre el ACK de la orden y la aparición de la posición.
        """
        ex = self.cfg.execution
        for attempt in range(ex.fill_confirm_retries):
            acct = await self.exchange.get_account()
            leg = next((p for p in acct.positions
                        if p.symbol == symbol and p.position_side == position_side
                        and p.qty > 0), None)
            if leg is not None:
                return True
            if attempt < ex.fill_confirm_retries - 1:
                await asyncio.sleep(ex.fill_confirm_delay_seconds)
        return False

    # ---------------------------- cierre ----------------------------

    async def close_position(self, symbol: str, position_side: PositionSide, *,
                             now: datetime | None = None) -> OrderResult | None:
        """Cierra una pierna a mercado (salida por señal o flip de dirección).

        Devuelve None si no hay pierna que cerrar. Lee la cantidad real de la
        posición en el exchange — no la asume — para enviar el cierre exacto.
        """
        # Primero cancelamos las protectoras: ya no hay pierna que vigilen.
        await self.exchange.cancel_all(symbol)
        acct = await self.exchange.get_account()
        leg = next((p for p in acct.positions
                    if p.symbol == symbol and p.position_side == position_side), None)
        if leg is None:
            return None
        req = build_close_request(symbol, position_side, leg.qty,
                                  id_factory=self._id_factory)
        return await self._send(req, "close", now)

    # ---------------------------- envío + auditoría ----------------------------

    async def _send(self, req: OrderRequest, decision_reason: str,
                    now: datetime | None) -> OrderResult:
        # retry_with_backoff reintenta solo ante 429/418; el mismo client_order_id
        # del req hace idempotente cualquier reenvío.
        res = await retry_with_backoff(lambda: self.exchange.place_order(req))
        if self.storage is not None:
            await self._persist(req, res, decision_reason, now)
        return res

    async def _persist(self, req: OrderRequest, res: OrderResult,
                       decision_reason: str, now: datetime | None) -> None:
        ts = now or datetime.now(timezone.utc)
        price = res.avg_price if res.avg_price else req.stop_price
        await self.storage.save_order(
            client_order_id=req.client_order_id, ts_ms=int(ts.timestamp() * 1000),
            symbol=req.symbol, side=req.side.value,
            position_side=req.position_side.value, type=req.type.value,
            quantity=req.quantity, price=price, status=res.status,
            exchange_order_id=res.order_id, decision_reason=decision_reason,
        )

    # ---------------------------- reconciliación ----------------------------

    async def reconcile(
        self, expected: list[tuple[str, PositionSide, float]]
    ) -> ReconResult:
        """Compara las piernas esperadas con las que reporta el exchange.

        `expected` es una lista de (symbol, position_side, qty). Una diferencia
        relativa por encima de la tolerancia (o una pierna presente en un lado y
        no en el otro) marca desincronización → el orquestador debe hacer halt.
        """
        acct = await self.exchange.get_account()
        actual = {(p.symbol, p.position_side): p.qty for p in acct.positions}
        exp = {(s, ps): q for (s, ps, q) in expected}
        tol = self.cfg.execution.reconcile_position_tolerance

        discrepancies: list[ReconItem] = []
        for key in set(actual) | set(exp):
            a = actual.get(key, 0.0)
            e = exp.get(key, 0.0)
            denom = max(abs(e), abs(a), 1e-12)
            if abs(a - e) / denom > tol:
                discrepancies.append(ReconItem(key[0], key[1], e, a))
        return ReconResult(consistent=not discrepancies, discrepancies=discrepancies)

    # ---------------------------- snapshot para el Risk Manager ----------------------------

    def _update_running(self, wallet: float, now: datetime) -> None:
        day = now.date()
        if self._day != day:  # nuevo día UTC → reinicia la base de pérdida diaria
            self._day = day
            self._day_start_wallet = wallet
        self._peak_wallet = wallet if self._peak_wallet == 0.0 else max(self._peak_wallet, wallet)

    async def snapshot_portfolio(self, *, now: datetime | None = None) -> PortfolioState:
        """Construye el PortfolioState (Futuros) que consume el Risk Manager."""
        now = now or datetime.now(timezone.utc)
        acct = await self.exchange.get_account()
        self._update_running(acct.wallet_balance, now)
        committed = sum(p.initial_margin for p in acct.positions)
        return PortfolioState(
            wallet_balance=acct.wallet_balance,
            available_balance=acct.available_balance,
            committed_margin=committed,
            peak_wallet_balance=self._peak_wallet,
            day_start_wallet_balance=self._day_start_wallet,
            open_positions=len(acct.positions),
        )
