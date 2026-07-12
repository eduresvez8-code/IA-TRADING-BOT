"""Carga y validación de configuración.

Dos fuentes, dos responsabilidades:
- `.env` → secretos (claves de broker, si algún día las hay). Leído por
  pydantic-settings, nunca versionado.
- `config/settings.yaml` → parámetros de trading e investigación. Versionado:
  cambiar el riesgo o un grid queda registrado en el historial de git.

Todo se valida al arrancar: si falta una clave o un umbral está fuera de
rango, el programa muere en el segundo 0, no a mitad de un backtest de horas.

La sección `research` es el PRE-REGISTRO del protocolo anti-sobreajuste
(split temporal, grids por familia, criterios de éxito). Su validación aquí es
parte del protocolo: un split sin timezone o un grid vacío no cargan.
"""

from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _validate_aware_iso_date(v: str, field_name: str) -> str:
    """Fecha ISO con timezone explícita. Un corte naive se compara mal contra
    índices UTC y partiría el dataset en otro punto del que se cree."""
    from datetime import datetime

    try:
        ts = datetime.fromisoformat(v.replace("Z", "+00:00"))
    except (ValueError, TypeError) as e:
        raise ValueError(f"{field_name} no parseable: {v!r}") from e
    if ts.tzinfo is None:
        raise ValueError(
            f"{field_name} debe llevar zona horaria explícita "
            f"(ej. '2015-01-01T00:00:00Z'), no {v!r}"
        )
    return v


class Secrets(BaseSettings):
    """Variables de .env. pydantic-settings las mapea por nombre.

    Hoy no hay broker conectado (todo es investigación/backtest); los campos
    quedan tipados para el día en que haya paper trading — las claves irán SOLO
    aquí, jamás en YAML ni en logs.
    """

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    broker_api_key: str = ""
    broker_api_secret: str = ""
    # Financial Modeling Prep (fundamentales/earnings, 2026-07-25): verificación
    # gratuita de point-in-time ANTES de considerar cualquier proveedor de pago.
    fmp_api_key: str = ""


class MarketConfig(BaseModel):
    # Símbolos del universo base (formato yfinance). Cero Hardcoding: ningún
    # módulo asume "SPY" — todos leen de aquí.
    benchmark_symbol: str = Field(min_length=1)   # ETF total-return del S&P 500
    index_symbol: str = Field(min_length=1)       # índice de precio (historia larga)
    tbill_symbol: str = Field(min_length=1)       # T-bill 13 semanas (retorno del cash)
    vix_symbol: str = Field(min_length=1)         # volatilidad implícita (Familia 7)
    timeframe: str

    @field_validator("timeframe")
    @classmethod
    def solo_diario(cls, v: str) -> str:
        # Decisión de diseño del pivote (lección cripto): nada intradía. El
        # ruido + costos mataron todo el intradía en 4 meses de investigación.
        if v != "1d":
            raise ValueError(f"timeframe debe ser '1d' (bot diario), no {v!r}")
        return v


class DataConfig(BaseModel):
    """Capa de datos: descarga yfinance + constituyentes históricos del índice."""

    dir: str = Field(min_length=1)
    start_date: str
    # Repo GitHub (owner/nombre) con el CSV de membresía histórica del S&P 500.
    # El downloader resuelve el archivo más reciente vía la API de GitHub, así
    # el nombre exacto (que incluye una fecha) no se hardcodea.
    constituents_repo: str = Field(pattern=r"^[\w.-]+/[\w.-]+$")
    # Tickers por lote de descarga. ge=1; le=200 ataja un lote que yfinance
    # trocearía igual pero con peor manejo de errores.
    batch_size: int = Field(ge=1, le=200)
    # Pausa entre lotes (cortesía con el rate limit de Yahoo). ge=0.
    pause_seconds: float = Field(ge=0.0, le=60.0)
    # Símbolos extra fuera del índice (p.ej. el fondo de bonos de la rotación).
    extra_symbols: list[str] = Field(default_factory=list)

    @field_validator("start_date")
    @classmethod
    def fecha_valida(cls, v: str) -> str:
        from datetime import date

        try:
            date.fromisoformat(v)
        except ValueError as e:
            raise ValueError(f"data.start_date no es YYYY-MM-DD: {v!r}") from e
        return v


class RiskConfig(BaseModel):
    # Los rangos (ge/le) son la primera línea de defensa: un typo como
    # risk_per_trade_pct: 10 en el YAML no pasa de aquí.
    # Contexto acciones cash: sin apalancamiento, sin margen, sin funding.
    risk_per_trade_pct: float = Field(gt=0, le=2.0)
    max_open_positions: int = Field(ge=1, le=100)
    max_daily_loss_pct: float = Field(gt=0, le=5.0)
    max_drawdown_pct: float = Field(gt=0, le=30.0)
    atr_stop_multiplier: float = Field(gt=0, le=10.0)
    atr_period: int = Field(ge=2, le=100)
    # Techo fijo de ganancia (solo aplica con let_winners_run=false). Se valida
    # siempre para poder reactivarlo sin tocar código.
    take_profit_rr: float = Field(gt=0, le=20.0)
    # "Cortar pérdidas rápido, dejar correr las ganancias": con true NO hay
    # take-profit — la posición sale por señal contraria o por el stop.
    let_winners_run: bool
    # Cuenta cash: sin cortos. true exigiría cuenta de margen + costo de
    # préstamo de acciones, que NO es modelable con datos gratis (declarado).
    allow_short: bool
    # Brokers clásicos operan acciones ENTERAS: la cantidad se trunca (floor).
    # true solo si el broker soporta fraccionales.
    allow_fractional_shares: bool


class QuantConfig(BaseModel):
    ema_fast_period: int = Field(ge=2, le=100)
    ema_slow_period: int = Field(ge=2, le=400)
    rsi_period: int = Field(ge=2, le=50)
    ema_weight: float = Field(ge=0.0, le=1.0)
    # Tipo de media para el cruce: "ema" o "sma".
    ma_type: str = Field(default="ema")
    # Factor del squash tanh: score = tanh(factor · spread_pct). Determina la
    # ESCALA del score. gt=0; le=1000 ataja un typo que saturaría el tanh.
    score_squash_factor: float = Field(default=50.0, gt=0, le=1000.0)

    @field_validator("ma_type")
    @classmethod
    def ma_type_valido(cls, v: str) -> str:
        if v not in ("ema", "sma"):
            raise ValueError(f"ma_type debe ser 'ema' o 'sma', no '{v}'")
        return v

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
    # k=0 ⇒ solo el fijo (universo líquido diario). le=5.0 ataja un typo absurdo.
    slippage_atr_multiplier: float = Field(ge=0, le=5.0)
    entry_threshold: float = Field(gt=0, lt=1)
    exit_threshold: float = Field(ge=0, lt=1)
    take_profit_rr: float = Field(gt=0)
    allow_short: bool

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


# ---------------------------------------------------------------------------
# PRE-REGISTRO del protocolo de investigación (anti-sobreajuste).
# Estos modelos validan el protocolo declarado en settings.yaml; el documento
# hermano con la justificación vive en docs/research/2026-07-11_protocolo_sp500.md.
# ---------------------------------------------------------------------------

class XsMomentumConfig(BaseModel):
    """Familia 1 — momentum cross-sectional (Jegadeesh-Titman 1993)."""

    # Meses de formación del ranking. Cada L en [1, 24]: J-T clásico usa 3-12.
    lookback_months_grid: list[int] = Field(min_length=1)
    # Meses a saltar entre formación y ejecución (evita la reversión de corto
    # plazo de 1 mes documentada por Jegadeesh 1990). 0 o 1.
    skip_months_grid: list[int] = Field(min_length=1)
    # Tamaño de la cartera larga (top-N del ranking). ge=5 (menos no diversifica
    # nada); le=100 (más ya es el índice entero, no un factor).
    top_n_grid: list[int] = Field(min_length=1)
    # Fracción mínima de constituyentes punto-en-el-tiempo CON precio para que
    # una fecha de rebalanceo cuente. Por debajo, el sesgo de supervivencia de
    # los datos gratis domina y la fecha se descarta (se reporta la cobertura).
    min_coverage: float = Field(gt=0.0, le=1.0)
    # Historia mínima (días de trading) de un ticker para entrar al ranking.
    min_history_days: int = Field(ge=30, le=1000)

    @field_validator("lookback_months_grid")
    @classmethod
    def lookbacks_en_rango(cls, v: list[int]) -> list[int]:
        for m in v:
            if not (1 <= m <= 24):
                raise ValueError(f"xs_momentum lookback {m} fuera de [1, 24] meses")
        return v

    @field_validator("skip_months_grid")
    @classmethod
    def skips_en_rango(cls, v: list[int]) -> list[int]:
        for m in v:
            if not (0 <= m <= 3):
                raise ValueError(f"xs_momentum skip {m} fuera de [0, 3] meses")
        return v

    @field_validator("top_n_grid")
    @classmethod
    def top_n_en_rango(cls, v: list[int]) -> list[int]:
        for n in v:
            if not (5 <= n <= 100):
                raise ValueError(f"xs_momentum top_n {n} fuera de [5, 100]")
        return v


class TsmomIndexConfig(BaseModel):
    """Familia 2 — TSMOM sobre el índice (Moskowitz-Ooi-Pedersen 2012).

    Definición congelada del Frente C (2026-07-07) para comparabilidad:
    retorno de los últimos L meses EXCLUYENDO el último mes; long si >0, si no
    cash. Long-only en acciones (cuenta cash).
    """

    lookback_months_grid: list[int] = Field(min_length=1)

    @field_validator("lookback_months_grid")
    @classmethod
    def lookbacks_en_rango(cls, v: list[int]) -> list[int]:
        for m in v:
            if not (1 <= m <= 24):
                raise ValueError(f"tsmom_index lookback {m} fuera de [1, 24] meses")
        return v


class MaTimingConfig(BaseModel):
    """Familia 3 — timing por media móvil (Faber 2007)."""

    # Precio vs SMA de N días, evaluado a FIN DE MES (baja frecuencia, pocos trades).
    sma_days_grid: list[int] = Field(min_length=1)
    # Pares [rápida, lenta] del cruce clásico, evaluado a diario.
    cross_pairs: list[list[int]] = Field(min_length=1)

    @field_validator("sma_days_grid")
    @classmethod
    def smas_en_rango(cls, v: list[int]) -> list[int]:
        for d in v:
            if not (20 <= d <= 400):
                raise ValueError(f"ma_timing sma {d} fuera de [20, 400] días")
        return v

    @field_validator("cross_pairs")
    @classmethod
    def pares_validos(cls, v: list[list[int]]) -> list[list[int]]:
        for pair in v:
            if len(pair) != 2:
                raise ValueError(f"ma_timing cross_pairs: {pair} debe tener 2 valores")
            fast, slow = pair
            if not (2 <= fast < slow <= 400):
                raise ValueError(
                    f"ma_timing cross_pairs: {pair} debe cumplir 2 ≤ fast < slow ≤ 400")
        return v


class RsiReversionConfig(BaseModel):
    """Familia 4 — reversión de corto plazo RSI-2 (Connors 2008)."""

    rsi_period: int = Field(ge=2, le=30)
    entry_grid: list[float] = Field(min_length=1)   # RSI < esto → LONG
    exit_grid: list[float] = Field(min_length=1)    # RSI > esto → salir
    trend_sma_days: int = Field(ge=20, le=400)      # filtro: solo sobre la SMA

    @field_validator("entry_grid")
    @classmethod
    def entradas_en_rango(cls, v: list[float]) -> list[float]:
        for x in v:
            if not (0.0 < x < 50.0):
                raise ValueError(f"rsi_reversion entry {x} fuera de (0, 50)")
        return v

    @field_validator("exit_grid")
    @classmethod
    def salidas_en_rango(cls, v: list[float]) -> list[float]:
        for x in v:
            if not (30.0 < x < 100.0):
                raise ValueError(f"rsi_reversion exit {x} fuera de (30, 100)")
        return v


class DualMomentumConfig(BaseModel):
    """Familia 5 — rotación dual-momentum mensual (Antonacci 2014, GEM).

    Sin grid a propósito: la definición canónica del libro es 12 meses. Añadir
    un grid aquí sería invitar al sobreajuste en la familia más simple.
    """

    lookback_months: int = Field(ge=3, le=24)
    bond_symbol: str = Field(min_length=1)


class BreadthConfig(BaseModel):
    """Familia 6 — amplitud de mercado (2026-07-25, última ronda de búsqueda).

    % de miembros punto-en-el-tiempo del índice sobre su propia SMA de N
    días. La cobertura mínima reutiliza xs_momentum.min_coverage (mismo
    concepto: fracción de miembros con precio disponible), no un umbral
    nuevo por familia.
    """

    sma_days_grid: list[int] = Field(min_length=1)
    threshold_grid: list[float] = Field(min_length=1)

    @field_validator("sma_days_grid")
    @classmethod
    def smas_en_rango(cls, v: list[int]) -> list[int]:
        for d in v:
            if not (20 <= d <= 400):
                raise ValueError(f"breadth sma {d} fuera de [20, 400] días")
        return v

    @field_validator("threshold_grid")
    @classmethod
    def umbrales_en_rango(cls, v: list[float]) -> list[float]:
        for x in v:
            if not (0.0 < x < 1.0):
                raise ValueError(f"breadth threshold {x} fuera de (0, 1)")
        return v


class VixRegimeConfig(BaseModel):
    """Familia 7 — régimen de VIX vs su propia media móvil (2026-07-25)."""

    sma_days_grid: list[int] = Field(min_length=1)
    directions: list[str] = Field(min_length=1)

    @field_validator("sma_days_grid")
    @classmethod
    def smas_en_rango(cls, v: list[int]) -> list[int]:
        for d in v:
            if not (10 <= d <= 400):
                raise ValueError(f"vix_regime sma {d} fuera de [10, 400] días")
        return v

    @field_validator("directions")
    @classmethod
    def direcciones_validas(cls, v: list[str]) -> list[str]:
        for d in v:
            if d not in ("below", "above"):
                raise ValueError(f"vix_regime direction debe ser 'below' o 'above', no {d!r}")
        return v


class ResearchConfig(BaseModel):
    """Protocolo de investigación pre-registrado (2026-07-11).

    REGLA: estos valores NO se cambian después de haber mirado un solo
    resultado. Si un experimento nuevo exige otro protocolo, se declara en un
    documento nuevo en docs/research/ ANTES de correrlo.
    """

    # Split temporal único: TRAIN < test_start_date ≤ TEST (medido UNA vez).
    test_start_date: str
    # Criterios de éxito — TODOS a la vez, sobre el TEST:
    success_sharpe_min: float = Field(gt=0.0, le=5.0)
    bootstrap_iterations: int = Field(ge=1000, le=100_000)
    bootstrap_ci: float = Field(ge=0.5, lt=1.0)
    concentration_max: float = Field(gt=0.0, le=1.0)
    # Sensibilidad de costos (slippage estresado, % por lado).
    slippage_stress_pct: float = Field(ge=0.0, le=1.0)
    # El cash fuera de mercado devenga la T-bill (^IRX). Si false, devenga 0
    # (subestima las estrategias de timing — solo para diagnóstico).
    cash_earns_tbill: bool
    # Familias (grids congelados):
    xs_momentum: XsMomentumConfig
    tsmom_index: TsmomIndexConfig
    ma_timing: MaTimingConfig
    rsi_reversion: RsiReversionConfig
    dual_momentum: DualMomentumConfig
    breadth: BreadthConfig
    vix_regime: VixRegimeConfig

    @field_validator("test_start_date")
    @classmethod
    def fecha_con_timezone(cls, v: str) -> str:
        return _validate_aware_iso_date(v, "research.test_start_date")


class PaperTradingRsi2Config(BaseModel):
    """RSI-2 en forward/paper trading real (2026-07-25) — config YA
    seleccionada por train el 2026-07-11 (docs/research/2026-07-11_sp500_resultados.txt),
    operacionalizada aquí. NO es un grid nuevo ni un re-tuneo: son los mismos
    4 números ya congelados y publicados, tipados para que el runner de
    `src/paper_trading/rsi2.py` no los hardcodee como literales sueltos.
    """

    entry_below: float = Field(gt=0.0, lt=50.0)
    exit_above: float = Field(gt=30.0, lt=100.0)
    trend_sma_days: int = Field(ge=20, le=400)
    rsi_period: int = Field(ge=2, le=30)
    # Carpeta VERSIONADA en git (no en data.dir, que es regenerable/fuera de
    # git) — el log de decisiones es la memoria del experimento, no un dato
    # descargable de nuevo.
    log_dir: str = Field(min_length=1)


class PaperTradingConfig(BaseModel):
    rsi2: PaperTradingRsi2Config


class Settings(BaseModel):
    market: MarketConfig
    data: DataConfig
    risk: RiskConfig
    quant: QuantConfig
    backtest: BacktestConfig
    research: ResearchConfig
    paper_trading: PaperTradingConfig

    @model_validator(mode="after")
    def backtest_coherente_con_risk(self) -> "Settings":
        # El simulador no debe poder abrir cortos que la política de riesgo
        # prohíbe: un backtest con shorts "gratis" (sin borrow) daría una
        # expectativa imposible de replicar en la cuenta cash real.
        if self.backtest.allow_short and not self.risk.allow_short:
            raise ValueError(
                "backtest.allow_short=true con risk.allow_short=false: el "
                "simulador permitiría cortos que la política de riesgo prohíbe"
            )
        return self


def load_settings(path: Path | None = None) -> Settings:
    path = path or PROJECT_ROOT / "config" / "settings.yaml"
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Settings.model_validate(raw)


def load_secrets() -> Secrets:
    return Secrets()
