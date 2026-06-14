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


class ExecutionConfig(BaseModel):
    # Tolerancia RELATIVA de reconciliación en (0,1): 0.001 = 0.1%. Un valor de 0
    # marcaría como discrepancia cualquier diferencia de redondeo; ≥1 nunca
    # detectaría una desincronización. workingType solo admite los dos modos de
    # disparo de Binance Futuros.
    reconcile_position_tolerance: float = Field(gt=0.0, lt=1.0)
    stop_working_type: Literal["MARK_PRICE", "CONTRACT_PRICE"]


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
    execution: ExecutionConfig
    storage: StorageConfig


def load_settings(path: Path | None = None) -> Settings:
    path = path or PROJECT_ROOT / "config" / "settings.yaml"
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Settings.model_validate(raw)


def load_secrets() -> Secrets:
    return Secrets()
