"""Carga y validación de configuración.

Dos fuentes, dos responsabilidades:
- `.env` → secretos (API keys). Leído por pydantic-settings, nunca versionado.
- `config/settings.yaml` → parámetros de trading. Versionado: cambiar el
  riesgo o los símbolos queda registrado en el historial de git.

Todo se valida al arrancar: si falta una clave o un umbral está fuera de
rango, el bot muere en el segundo 0, no tras abrir una posición.
"""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Secrets(BaseSettings):
    """Variables de .env. pydantic-settings las mapea por nombre."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    binance_api_key: str = ""
    binance_api_secret: str = ""
    binance_testnet: bool = True
    anthropic_api_key: str = ""
    cryptopanic_token: str = ""  # token del free tier para noticias históricas


class MarketConfig(BaseModel):
    symbols: list[str]
    timeframe: str
    htf_timeframe: str


class RiskConfig(BaseModel):
    # Los rangos (ge/le) son la primera línea de defensa: un typo como
    # risk_per_trade_pct: 10 en el YAML no pasa de aquí.
    risk_per_trade_pct: float = Field(gt=0, le=2.0)
    max_open_positions: int = Field(ge=1, le=10)
    max_daily_loss_pct: float = Field(gt=0, le=5.0)
    max_drawdown_pct: float = Field(gt=0, le=20.0)
    atr_stop_multiplier: float = Field(gt=0)
    atr_period: int = Field(ge=2)
    # Sprint 5: parámetros del Risk Manager en vivo.
    take_profit_rr: float = Field(gt=0)
    # low_confidence_threshold en (0,1): por encima de 1 nunca reduciría, en 0
    # nunca dispararía. low_confidence_size_factor en (0,1]: 1.0 = no reduce.
    low_confidence_threshold: float = Field(gt=0.0, lt=1.0)
    low_confidence_size_factor: float = Field(gt=0.0, le=1.0)
    stale_feed_seconds: float = Field(gt=0)
    # Futuros USD-M. max_leverage: entero ≥1; le=10 ataja un apalancamiento de
    # casino (un 20x sería un typo en este bot). max_portfolio_margin_pct: % del
    # wallet comprometible como margen inicial; >100 no tiene sentido → le=100.
    max_leverage: int = Field(ge=1, le=10)
    max_portfolio_margin_pct: float = Field(gt=0, le=100.0)


class ConfluenceConfig(BaseModel):
    quant_strong_threshold: float = Field(gt=0, lt=1)
    sentiment_confirm_threshold: float = Field(gt=0, lt=1)
    reduced_size_factor: float = Field(gt=0, le=1)
    # Spot no permite ABRIR cortos; en vivo va en false. El backtest usa su
    # propia ruta y puede reactivarlos como investigación.
    allow_short: bool


class SentimentConfig(BaseModel):
    rss_feeds: list[str]
    poll_interval_seconds: int = Field(ge=30)
    claude_model: str
    heuristic_weight: float = Field(ge=0.0, le=1.0)
    escalate_score_threshold: float = Field(gt=0.0, lt=1.0)
    max_news_age_hours: int = Field(ge=1)


class QuantConfig(BaseModel):
    ema_fast_period: int = Field(ge=2, le=50)
    ema_slow_period: int = Field(ge=2, le=200)
    rsi_period: int = Field(ge=2, le=50)
    ema_weight: float = Field(ge=0.0, le=1.0)

    @field_validator("ema_slow_period")
    @classmethod
    def slow_must_exceed_fast(cls, v: int, info) -> int:
        fast = info.data.get("ema_fast_period")
        if fast is not None and v <= fast:
            raise ValueError(
                f"ema_slow_period ({v}) debe ser mayor que ema_fast_period ({fast})"
            )
        return v


class BacktestConfig(BaseModel):
    # Costos en % de notional por lado. El le=1.0 ataja un typo tipo
    # commission_pct: 40 (interpretado como 40%, no 0.04%).
    initial_capital: float = Field(gt=0)
    commission_pct: float = Field(ge=0, le=1.0)
    slippage_pct: float = Field(ge=0, le=1.0)
    # Multiplicador del slippage dinámico por volatilidad: slip = fijo + k·ATR/precio.
    # k=0 ⇒ comportamiento idéntico al slippage fijo original (regresión protegida).
    # le=5.0 ataja un typo absurdo (un k enorme inflaría el slippage sin sentido).
    slippage_atr_multiplier: float = Field(ge=0, le=5.0)
    entry_threshold: float = Field(gt=0, lt=1)
    exit_threshold: float = Field(ge=0, lt=1)
    take_profit_rr: float = Field(gt=0)
    allow_short: bool = True

    @field_validator("exit_threshold")
    @classmethod
    def exit_below_entry(cls, v: float, info) -> float:
        # Si el umbral de salida ≥ el de entrada, abriríamos y cerraríamos en la
        # misma vela (la condición de salida ya se cumple al entrar). Sin sentido.
        entry = info.data.get("entry_threshold")
        if entry is not None and v >= entry:
            raise ValueError(
                f"exit_threshold ({v}) debe ser menor que entry_threshold ({entry})"
            )
        return v


class EdgeConfig(BaseModel):
    # Diagnóstico (no afecta al trading). forward_horizons: velas hacia adelante a
    # las que medir el IC; cada una ≥1 y la lista no vacía. n_quantiles: cubos de
    # la tabla de monotonicidad; ge=2 (un solo cubo no discrimina nada), le=20
    # ataja un valor que dejaría cada cubo sin observaciones suficientes.
    forward_horizons: list[int]
    n_quantiles: int = Field(ge=2, le=20)

    @field_validator("forward_horizons")
    @classmethod
    def horizons_validos(cls, v: list[int]) -> list[int]:
        if not v:
            raise ValueError("forward_horizons no puede estar vacío")
        if any(h < 1 for h in v):
            raise ValueError("cada horizonte de forward_horizons debe ser ≥ 1 vela")
        return v


class ScanConfig(BaseModel):
    # Universo y parámetros del escáner de arquetipos (laboratorio de estrategia).
    # symbols no vacío; history_days ≥1; folds ge=2 (un solo tramo no testea
    # consistencia) le=20; edge_profit_factor_min gt=1 (PF≤1 ya es no-edge).
    symbols: list[str]
    history_days: int = Field(ge=1)
    walk_forward_folds: int = Field(ge=2, le=20)
    edge_profit_factor_min: float = Field(gt=1.0)

    @field_validator("symbols")
    @classmethod
    def symbols_no_vacio(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("scan.symbols no puede estar vacío")
        return v


class MeanReversionConfig(BaseModel):
    # Arquetipo 2. bb_num_std gt=0 (bandas no degeneradas); RSI en [0,100] con
    # oversold < overbought (si no, la zona de compra y venta se cruzarían).
    bb_period: int = Field(ge=2, le=500)
    bb_num_std: float = Field(gt=0)
    rsi_oversold: float = Field(ge=0.0, le=100.0)
    rsi_overbought: float = Field(ge=0.0, le=100.0)

    @field_validator("rsi_overbought")
    @classmethod
    def overbought_sobre_oversold(cls, v: float, info) -> float:
        os = info.data.get("rsi_oversold")
        if os is not None and v <= os:
            raise ValueError(
                f"rsi_overbought ({v}) debe ser mayor que rsi_oversold ({os})"
            )
        return v


class BreakoutConfig(BaseModel):
    # Arquetipo 3. Periodos ge=2; multiplicadores ge=0 (0 = sin filtro: cualquier
    # ruptura/volatilidad vale). exit_donchian_period es el canal de salida
    # trailing (Turtle); debe ser ≤ donchian_period (salir con un canal MÁS ancho
    # que el de entrada no tendría sentido).
    donchian_period: int = Field(ge=2, le=500)
    exit_donchian_period: int = Field(ge=2, le=500)
    volume_ma_period: int = Field(ge=2, le=500)
    volume_multiplier: float = Field(ge=0.0)
    atr_filter_period: int = Field(ge=2, le=500)
    atr_expansion_mult: float = Field(ge=0.0)

    @field_validator("exit_donchian_period")
    @classmethod
    def salida_no_mas_ancha_que_entrada(cls, v: int, info) -> int:
        entry = info.data.get("donchian_period")
        if entry is not None and v > entry:
            raise ValueError(
                f"exit_donchian_period ({v}) no debe superar donchian_period ({entry})"
            )
        return v


class ExecutionConfig(BaseModel):
    # Tolerancia RELATIVA de reconciliación en (0,1): 0.001 = 0.1%. Un valor de 0
    # marcaría como discrepancia cualquier diferencia de redondeo; ≥1 nunca
    # detectaría una desincronización. workingType solo admite los dos modos de
    # disparo de Binance Futuros.
    reconcile_position_tolerance: float = Field(gt=0.0, lt=1.0)
    stop_working_type: Literal["MARK_PRICE", "CONTRACT_PRICE"]


class OrchestratorConfig(BaseModel):
    # warmup_candles: velas mínimas antes de operar. ge=20 evita un buffer tan
    # corto que los indicadores nunca tengan datos; le=1000 ataja un typo.
    warmup_candles: int = Field(ge=20, le=1000)
    # Ciclos de gracia antes de declarar una pierna desconocida → HALT. ge=1
    # (al menos una confirmación); le=20 ataja un valor que volvería inútil el
    # circuit breaker.
    reconcile_grace_cycles: int = Field(ge=1, le=20)


class StorageConfig(BaseModel):
    db_path: str
    candles_dir: str


class Settings(BaseModel):
    market: MarketConfig
    risk: RiskConfig
    confluence: ConfluenceConfig
    sentiment: SentimentConfig
    quant: QuantConfig
    backtest: BacktestConfig
    edge: EdgeConfig
    scan: ScanConfig
    mean_reversion: MeanReversionConfig
    breakout: BreakoutConfig
    execution: ExecutionConfig
    orchestrator: OrchestratorConfig
    storage: StorageConfig


def load_settings(path: Path | None = None) -> Settings:
    path = path or PROJECT_ROOT / "config" / "settings.yaml"
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Settings.model_validate(raw)


def load_secrets() -> Secrets:
    return Secrets()
