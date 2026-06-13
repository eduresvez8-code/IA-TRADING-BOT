"""Contratos de datos del sistema.

Este módulo es la "lengua franca" del bot: todos los módulos se comunican
intercambiando estos objetos, nunca diccionarios sueltos. Si un módulo emite
algo malformado, Pydantic lo rechaza aquí — en la frontera — y no a las 3am
dentro del executor con una posición abierta.

Regla del repo: este archivo no se modifica sin actualizar tests/test_models.py
en el mismo cambio.
"""

from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum

from pydantic import BaseModel, Field, field_validator


class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class Action(str, Enum):
    """Resultado de la matriz de confluencia."""

    LONG = "LONG"
    SHORT = "SHORT"
    HOLD = "HOLD"  # no operar: desacuerdo entre motores o sin señal


class Candle(BaseModel):
    """Vela OHLCV tal como llega de Binance (websocket o REST)."""

    symbol: str
    timeframe: str
    open_time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    closed: bool = True  # el websocket emite velas aún en formación

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
    Normalizar permite que la matriz de confluencia compare señales de
    estrategias distintas sin conocer sus detalles internos.
    """

    symbol: str
    score: float = Field(ge=-1.0, le=1.0)
    strategy: str  # nombre de la estrategia que la generó (auditoría)
    timestamp: datetime
    features: dict[str, float] = Field(default_factory=dict)  # ej. {"rsi": 28.5}


class NewsItem(BaseModel):
    """Noticia cruda tras la ingesta RSS, antes de analizar sentimiento."""

    id: str  # hash del link: clave de deduplicación entre feeds
    title: str
    source: str
    url: str
    published_at: datetime
    summary: str = ""


class SentimentScore(BaseModel):
    """Salida del Sentiment Engine para una noticia ya analizada."""

    news_id: str
    symbol_scope: list[str]  # símbolos afectados; ["*"] = todo el mercado
    score: float = Field(ge=-1.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)  # baja confianza → sizing reducido
    high_impact: bool = False  # FOMC/CPI/hack: puede bloquear entradas
    rationale: str = ""  # explicación de Claude, para auditoría
    analyzed_at: datetime


class Decision(BaseModel):
    """Salida de la matriz de confluencia: qué hacer y con qué convicción."""

    symbol: str
    action: Action
    quant_score: float = Field(ge=-1.0, le=1.0)
    sentiment_score: float = Field(ge=-1.0, le=1.0)
    size_factor: float = Field(ge=0.0, le=1.0)  # 1.0 = tamaño pleno, 0.5 = reducido
    reason: str  # regla de la matriz que disparó la decisión (auditoría)
    timestamp: datetime


class SymbolFilters(BaseModel):
    """Restricciones de microestructura de un par, leídas de Binance exchangeInfo.

    NO son parámetros tuneables nuestros (no van a settings.yaml): son hechos del
    exchange que el binance_client lee de `GET /api/v3/exchangeInfo` y cachea. El
    Risk Manager los recibe como input y es el último filtro antes del executor.

    Se usan Decimal (no float) porque el ajuste a stepSize/tickSize debe ser
    exacto: con float, truncar 0.3 a paso 0.1 da 0.2 por el error binario.
    """

    symbol: str
    tick_size: Decimal = Field(gt=0)     # PRICE_FILTER: paso mínimo de precio
    step_size: Decimal = Field(gt=0)     # LOT_SIZE: paso mínimo de cantidad
    min_qty: Decimal = Field(ge=0)       # LOT_SIZE: cantidad mínima por orden
    min_notional: Decimal = Field(ge=0)  # MIN_NOTIONAL: valor mínimo (qty×precio)


class Order(BaseModel):
    """Orden ya validada por el Risk Manager, lista para el executor.

    stop_loss es obligatorio por diseño: una orden sin SL no puede existir
    en este sistema (el Risk Manager la rechaza antes de construirla).
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
