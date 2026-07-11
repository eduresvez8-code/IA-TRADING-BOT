"""Contratos de datos del sistema.

Este módulo es la "lengua franca" del bot: todos los módulos se comunican
intercambiando estos objetos, nunca diccionarios sueltos. Si un módulo emite
algo malformado, Pydantic lo rechaza aquí — en la frontera — y no dentro de un
backtest de horas o (algún día) con una posición abierta.

Regla del repo: este archivo no se modifica sin actualizar tests/test_models.py
en el mismo cambio.

2026-07-11 (pivote a S&P 500): se eliminaron los contratos del híbrido cripto
(NewsItem, SentimentScore, EventIntent, SymbolFilters de Binance, PositionSide/
OrderType de futuros). Quedan los contratos genéricos de un bot de acciones.
"""

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field, field_validator


class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class Action(str, Enum):
    """Qué hacer con un símbolo según la estrategia."""

    LONG = "LONG"
    SHORT = "SHORT"
    HOLD = "HOLD"  # no operar: sin señal


class Candle(BaseModel):
    """Vela OHLCV diaria (yfinance u otra fuente)."""

    symbol: str
    timeframe: str
    open_time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    closed: bool = True  # una vela del día en curso aún puede cambiar

    @field_validator("open_time")
    @classmethod
    def must_be_utc(cls, v: datetime) -> datetime:
        # Todo timestamp del sistema es UTC consciente (aware). Un datetime
        # naive mezclado con aware revienta comparaciones en el backtester.
        if v.tzinfo is None:
            raise ValueError("open_time debe incluir timezone (UTC)")
        return v.astimezone(timezone.utc)


class Signal(BaseModel):
    """Salida del Quant Engine, normalizada a [-1, +1].

    -1 = máxima convicción bajista, +1 = máxima convicción alcista, 0 = neutro.
    Normalizar permite comparar señales de estrategias distintas sin conocer
    sus detalles internos.
    """

    symbol: str
    score: float = Field(ge=-1.0, le=1.0)
    strategy: str  # nombre de la estrategia que la generó (auditoría)
    timestamp: datetime
    features: dict[str, float] = Field(default_factory=dict)  # ej. {"rsi": 28.5}


class Decision(BaseModel):
    """Decisión de la estrategia sobre un símbolo: qué hacer y con qué convicción."""

    symbol: str
    action: Action
    score: float = Field(ge=-1.0, le=1.0)
    size_factor: float = Field(ge=0.0, le=1.0)  # 1.0 = tamaño pleno, 0.5 = reducido
    reason: str  # regla que disparó la decisión (auditoría)
    timestamp: datetime


class Order(BaseModel):
    """Orden ya validada por el Risk Manager, lista para (algún día) un broker.

    stop_loss es obligatorio por diseño: una orden sin SL no puede existir
    en este sistema (el Risk Manager la rechaza antes de construirla).
    En acciones cash la cantidad son ACCIONES (enteras salvo que el broker
    soporte fraccionales — eso lo decide el Risk Manager, no este modelo).
    """

    symbol: str
    side: Side
    quantity: float = Field(gt=0)
    entry_price: float = Field(gt=0)
    stop_loss: float = Field(gt=0)
    take_profit: float | None = None
    decision_reason: str  # trazabilidad: qué decisión originó esta orden
    created_at: datetime

    @field_validator("stop_loss")
    @classmethod
    def stop_must_protect(cls, v: float, info) -> float:
        # El SL debe estar del lado que limita la pérdida, no la ganancia.
        side = info.data.get("side")
        entry = info.data.get("entry_price")
        if side is None or entry is None:
            return v
        if side == Side.BUY and v >= entry:
            raise ValueError("stop_loss de una compra debe ser menor que entry_price")
        if side == Side.SELL and v <= entry:
            raise ValueError("stop_loss de una venta debe ser mayor que entry_price")
        return v
