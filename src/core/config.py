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
from pydantic import BaseModel, Field, field_validator, model_validator
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
    # Quotes válidos para derivar el ACTIVO BASE de un símbolo (BTCUSDT → BTC) al
    # resolver el scope de una noticia (src/core/scope.py). Cero Hardcoding: el
    # código no asume "USDT". Solo operamos perps USD-M, así que por defecto en el
    # YAML es ["USDT"]; añadir USDC/otros aquí si algún día se opera ese quote.
    quote_assets: list[str] = Field(min_length=1)


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
    # El feed se considera obsoleto si no llega vela por más de este múltiplo del
    # intervalo de la vela (o de stale_feed_seconds, lo que sea mayor). Con velas
    # cerradas de 5m, 30s declararía el feed muerto entre vela y vela: la
    # obsolescencia debe escalar con el timeframe. 2.0 = dos velas sin llegar.
    stale_feed_intervals: float = Field(gt=0)
    # Futuros USD-M. max_leverage: entero ≥1; le=10 ataja un apalancamiento de
    # casino (un 20x sería un typo en este bot). max_portfolio_margin_pct: % del
    # wallet comprometible como margen inicial; >100 no tiene sentido → le=100.
    max_leverage: int = Field(ge=1, le=10)
    max_portfolio_margin_pct: float = Field(gt=0, le=100.0)
    # --- Sizing de evento (Fase 2.4): parámetros del modo "event" del Risk Manager ---
    # Presupuesto base para trades de evento (más pequeño que el Slow Path por mayor
    # incertidumbre). Validador cruzado: debe ser ≤ risk_per_trade_pct.
    event_risk_per_trade_pct: float = Field(gt=0, le=2.0)
    # Múltiplo de ATR para el stop de evento (más amplio que el Slow Path: el ruido
    # post-noticia es mayor). Validador cruzado: debe ser ≥ atr_stop_multiplier.
    # El stop más ancho NO aumenta el riesgo: qty = risk/stop → la qty baja.
    event_atr_stop_multiplier: float = Field(gt=0, le=10)
    # Ventana para la línea base de régimen de volatilidad (mediana del ATR sobre las
    # últimas N velas). ge=2 (mínimo para tener mediana con sentido); le=500 ataja un
    # typo que vaciaría el techo de margen mirando cientos de velas atrás.
    vol_regime_lookback: int = Field(ge=2, le=500)
    # Techo del ratio ATR_now/ATR_baseline antes de aplicar el amortiguador.
    # vol_damp = min(1.0, cap / ratio). gt=1.0: cap=1.0 recortaría ante cualquier
    # expansión mínima, demasiado conservador; le=10 ataja un typo absurdo.
    vol_expansion_cap: float = Field(gt=1.0, le=10)

    @model_validator(mode="after")
    def event_sizing_coherente(self) -> "RiskConfig":
        if self.event_risk_per_trade_pct > self.risk_per_trade_pct:
            raise ValueError(
                f"event_risk_per_trade_pct ({self.event_risk_per_trade_pct}) "
                f"debe ser ≤ risk_per_trade_pct ({self.risk_per_trade_pct})"
            )
        if self.event_atr_stop_multiplier < self.atr_stop_multiplier:
            raise ValueError(
                f"event_atr_stop_multiplier ({self.event_atr_stop_multiplier}) "
                f"debe ser ≥ atr_stop_multiplier ({self.atr_stop_multiplier})"
            )
        return self


class ConfluenceConfig(BaseModel):
    quant_strong_threshold: float = Field(gt=0, lt=1)
    sentiment_confirm_threshold: float = Field(gt=0, lt=1)
    reduced_size_factor: float = Field(gt=0, le=1)
    # Spot no permite ABRIR cortos; en vivo va en false. El backtest usa su
    # propia ruta y puede reactivarlos como investigación.
    allow_short: bool
    # TTL del sentimiento EN VIVO (segundos). El store del orquestador retiene la
    # última lectura de cada símbolo hasta que el poller la pisa; sin TTL, una
    # noticia de hace 30 min seguiría confirmando trades. Caduca contra
    # `analyzed_at`. ge=1 (un 0 caducaría el sentimiento al instante, nunca se
    # usaría); le=86400 ataja un typo (más de un día no es "noticia fresca"). NO
    # afecta al backtest, que caduca a escala de horas vía max_news_age_hours.
    sentiment_ttl_seconds: int = Field(ge=1, le=86400)


class SentimentConfig(BaseModel):
    # Gate de seguridad del overlay de sentimiento del Slow Path (Plan V2). Arranca
    # en false: con el flag apagado, `run()` NO arranca el `_sentiment_loop` → cero
    # llamadas a Claude y la señal quant queda PURA (la línea base de paper trading
    # no se altera). Activarlo gasta Claude Haiku y cambia el comportamiento en vivo,
    # así que es decisión explícita de Eduardo (mismo razonamiento que event.enabled).
    enabled: bool
    rss_feeds: list[str]
    poll_interval_seconds: int = Field(ge=30)
    fetch_timeout_seconds: int = Field(default=10, ge=5, le=60)
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
    # Una entrada MARKET puede responder NEW y llenarse microsegundos después
    # (Binance real/testnet). Antes de colocar SL/TP confirmamos el fill mirando
    # la posición real, reintentando hasta este nº de veces con esta espera.
    fill_confirm_retries: int = Field(ge=1, le=20)
    fill_confirm_delay_seconds: float = Field(gt=0.0, le=5.0)
    # Tope de slippage para entradas LIMIT-IOC marketable (Fase 1.3). 1 bps = 0.01%.
    # limit_price = mark ± cap_bps/10000. gt=0 (0 sería MARKET sin límite); le=100
    # ataja un typo (100 bps = 1%, que ya es más que la comisión taker de Binance).
    slippage_cap_bps: float = Field(gt=0, le=100)
    # Modo de la entrada de apertura: "IOC" = Immediate-Or-Cancel (llena dentro del
    # tope o cancela); "GTC" = queda resting en el libro (solo para laboratorio).
    aggressive_entry_tif: Literal["IOC", "GTC"]


class OrchestratorConfig(BaseModel):
    # warmup_candles: velas mínimas antes de operar. ge=20 evita un buffer tan
    # corto que los indicadores nunca tengan datos; le=1000 ataja un typo.
    warmup_candles: int = Field(ge=20, le=1000)
    # Ciclos de gracia antes de declarar una pierna desconocida → HALT. ge=1
    # (al menos una confirmación); le=20 ataja un valor que volvería inútil el
    # circuit breaker.
    reconcile_grace_cycles: int = Field(ge=1, le=20)
    # Velas HTF (htf_timeframe) a retener para la lectura de RÉGIMEN del quant
    # (Opción 2: quant demotado a confirmar/dimensionar). El engine deriva cuántas
    # velas base necesita backfillear/retener = regime_htf_bars * (htf/base). El
    # quant exige ema_slow_period + rsi_period velas (35 con 21+14), así que ge=35
    # garantiza que el régimen tenga datos; le=500 ataja un buffer desmedido.
    regime_htf_bars: int = Field(ge=35, le=500)


class EventConfig(BaseModel):
    """Fast Path: originación de trades por evento (Plan V2 Fase 2).

    El Slow Path decide en cada vela cerrada (5m) con el quant como contexto. El
    Fast Path se dispara por la LLEGADA de un shock (hack/ETF/depeg…) y puede
    ORIGINAR un trade sub-vela. Estos parámetros gobiernan CUÁNDO un shock cuaja
    en una orden; la lógica (decide_event) llega en Fase 2.2.

    `enabled` arranca en false: el consumidor de eventos (Fase 2.3) aún no existe;
    nada debe intentar correr el Fast Path hasta que esté cableado y validado en
    testnet (kill criteria §B/§C del plan).
    """

    enabled: bool
    # Cadencia del poller de eventos. Debe ser MÁS rápida que el poll del Slow
    # Path (sentiment.poll_interval_seconds = 120). ge=5: por debajo martillaría
    # el RSS gratis y arriesga un baneo de IP; le=60: por encima ya no compite con
    # la cadencia de la vela de 5m y deja de ser "fast".
    poll_interval_seconds: int = Field(ge=5, le=60)
    # |score| mínimo del shock para originar. gt=0 (un 0 originaría con cualquier
    # ruido); lt=1 (exactamente 1 exigiría el score máximo perfecto: nunca dispara).
    min_impact_score: float = Field(gt=0.0, lt=1.0)
    # Confianza mínima de Claude para fiarse del titular antes de originar.
    min_confidence: float = Field(gt=0.0, lt=1.0)
    # TTL del intent de evento (segundos). Mismo razonamiento que
    # confluence.sentiment_ttl_seconds: un evento viejo no se opera. ge=1, le=86400.
    ttl_seconds: int = Field(ge=1, le=86400)
    # Enfriamiento por símbolo tras un trade de evento (segundos): evita reentrar en
    # cadena con titulares correlacionados del mismo suceso. ge=0 (0 = sin
    # cooldown); le=86400 ataja un typo (un día es el máximo razonable).
    cooldown_seconds: int = Field(ge=0, le=86400)
    # Confirmación de impulso (núcleo legítimo del circuit breaker (b) del v1): el
    # precio debe haberse movido >= esto en la dirección del score dentro de
    # confirm_window_seconds. ge=0 → 0 DESACTIVA el gate (necesario para la
    # ablación A/B de los kill criteria §B); le=1000 (1000 bps = 10% es un typo).
    confirm_impulse_bps: float = Field(ge=0.0, le=1000.0)
    # Ventana en la que se mide el impulso de precio. ge=1, le=3600 (1h máximo).
    confirm_window_seconds: int = Field(ge=1, le=3600)
    # Multiplicador de tamaño de los trades de evento (más arriesgados → más
    # pequeños). gt=0 (un 0 no abriría nada); le=1 (>1 AMPLIFICARÍA, al revés).
    size_factor: float = Field(gt=0.0, le=1.0)
    # Ventana de bloqueo alrededor de un macro PROGRAMADO (refina el `scheduled` de
    # Fase 1.2: bloquear solo CERCA del dato, no para siempre). Minutos antes y
    # después. ge=0 (0 = sin bloqueo por ese lado); le=1440 (un día) ataja un typo.
    macro_block_minutes_before: int = Field(ge=0, le=1440)
    macro_block_minutes_after: int = Field(ge=0, le=1440)
    # --- Plan V2 Fase 2.5(i): plano de datos en tiempo real (micro-buffer markPrice@1s) ---
    # Retención del deque de markPrice por símbolo (segundos). El impulso se mide
    # sobre confirm_window_seconds, así que el buffer debe RETENER al menos esa
    # ventana (un validador cruzado lo obliga: buffer ≥ confirm_window). ge=1;
    # le=3600 (1h) ataja un typo que inflaría memoria sin sentido.
    markprice_buffer_seconds: int = Field(ge=1, le=3600)
    # Fallar-cerrado por feed congelado: si el tick más reciente es más viejo que
    # esto, _price_impulse_bps devuelve None (no operamos sobre un precio muerto).
    # gt=0; le=60 (a 1 tick/s, más de 60s sin tick es un feed claramente caído).
    markprice_stale_seconds: float = Field(gt=0.0, le=60.0)
    # Fallar-cerrado por buffer frío: mínimo de ticks dentro de la ventana para
    # fiarnos del impulso. ge=2 (con <2 no hay retorno medible); le=10000 ataja un
    # typo (a 1 tick/s serían casi 3h de exigencia).
    markprice_min_ticks: int = Field(ge=2, le=10000)
    # --- Plan V2 Fase 2.5(ii): guardia de frescura del event_fetch ---
    # Antigüedad máxima (por published_at) de un titular para originar. §0(A): el
    # edge es el drift POST-evento, no perseguir noticias rancias. Más laxo que
    # ttl_seconds (que mide la edad del INTENT vs analyzed_at) porque el lag de RSS
    # gratis es de 1-5 min. ge=1; le=86400 ataja un typo (un día no es "shock").
    max_headline_age_seconds: int = Field(ge=1, le=86400)

    @model_validator(mode="after")
    def markprice_buffer_cubre_la_ventana(self) -> "EventConfig":
        # Si el buffer no retiene al menos la ventana de impulso, la comprobación
        # "ventana cubierta" de _price_impulse_bps nunca pasaría → nunca operaría.
        if self.markprice_buffer_seconds < self.confirm_window_seconds:
            raise ValueError(
                f"markprice_buffer_seconds ({self.markprice_buffer_seconds}) debe "
                f"ser ≥ confirm_window_seconds ({self.confirm_window_seconds})"
            )
        return self


class FundingEdgeConfig(BaseModel):
    # Edge test de señales no-precio. premium_interval: granularidad del basis.
    # forward_horizons_hours: horizontes de retorno futuro en HORAS (la señal de
    # funding es de 8h, así que los horizontes son múltiplos naturales, no velas).
    premium_interval: str
    forward_horizons_hours: list[int]
    n_quantiles: int = Field(ge=2, le=20)

    @field_validator("forward_horizons_hours")
    @classmethod
    def horizons_validos(cls, v: list[int]) -> list[int]:
        if not v:
            raise ValueError("forward_horizons_hours no puede estar vacío")
        if any(h < 1 for h in v):
            raise ValueError("cada horizonte (horas) debe ser ≥ 1")
        return v


class CrossSectionalConfig(BaseModel):
    # Edge test del factor de momentum relativo. forward_days/rebalance_days en
    # días; momentum_skip_days≥0 (puede ser 0). vol_lookback_days≥2 para tener
    # varianza. min_assets≥2 (una cross-section de 1 no rankea nada).
    history_days: int = Field(ge=1)
    min_history_days: int = Field(ge=1)
    momentum_lookback_days: int = Field(ge=1)
    momentum_skip_days: int = Field(ge=0)
    vol_adjust: bool
    vol_lookback_days: int = Field(ge=2)
    forward_days: int = Field(ge=1)
    rebalance_days: int = Field(ge=1)
    n_quantiles: int = Field(ge=2, le=20)
    min_assets: int = Field(ge=2)
    # Portafolio long-short: liquidity_drop_pct en [0,1) (0 = sin filtro);
    # winsorize_quantile en [0,0.5) (0 = sin recorte); max_weight en (0,1].
    liquidity_drop_pct: float = Field(ge=0.0, lt=1.0)
    winsorize_quantile: float = Field(ge=0.0, lt=0.5)
    max_weight: float = Field(gt=0.0, le=1.0)


class SentimentRegimeConfig(BaseModel):
    # Umbrales del F&G (0-100) ordenados: ext_fear < fear < greed < ext_greed.
    ext_fear_below: int = Field(ge=0, le=100)
    fear_below: int = Field(ge=0, le=100)
    greed_above: int = Field(ge=0, le=100)
    ext_greed_above: int = Field(ge=0, le=100)
    forward_days: int = Field(ge=1)
    mr_lookback_days: int = Field(ge=1)
    extreme_abs_threshold: float = Field(gt=0.0, lt=50.0)
    vol_scale_min: float = Field(gt=0.0, le=1.0)

    @field_validator("fear_below", "greed_above", "ext_greed_above")
    @classmethod
    def umbrales_ordenados(cls, v: int, info) -> int:
        order = ["ext_fear_below", "fear_below", "greed_above", "ext_greed_above"]
        idx = order.index(info.field_name)
        prev = info.data.get(order[idx - 1])
        if prev is not None and v <= prev:
            raise ValueError(f"{info.field_name} ({v}) debe ser > {order[idx - 1]} ({prev})")
        return v


class StorageConfig(BaseModel):
    db_path: str
    candles_dir: str
    funding_dir: str
    universe_dir: str


class DashboardConfig(BaseModel):
    """Dashboard de observabilidad en tiempo real (proceso READ-ONLY aparte).

    Lee la misma SQLite que escribe el engine (en modo `ro`), nunca envía órdenes
    ni toca el exchange. Por eso `host` por defecto es loopback: el dashboard no se
    expone a la red. Cero Hardcoding: intervalos y tamaños de página viven aquí.
    """

    host: str
    # Puerto del servidor local. ge=1024 evita los puertos privilegiados; le=65535
    # es el máximo válido.
    port: int = Field(ge=1024, le=65535)
    # Cada cuánto repolla el navegador /api/snapshot. gt=0; le=60 (más lento ya no
    # es "tiempo real"). No martillea: es una sola lectura SQLite por refresco.
    refresh_seconds: float = Field(gt=0.0, le=60.0)
    # Puntos de la curva de capital a servir. ge=10 (una curva con sentido); le=5000
    # ataja un payload desmedido.
    equity_points: int = Field(ge=10, le=5000)
    # Filas del feed de decisiones / tabla de órdenes / panel de noticias.
    decisions_rows: int = Field(ge=1, le=1000)
    orders_rows: int = Field(ge=1, le=1000)
    news_rows: int = Field(ge=1, le=1000)
    # Múltiplo del intervalo de vela tras el cual el último snapshot se considera
    # OBSOLETO (señal de liveness: el bot probablemente está caído/halted). ge=1
    # (al menos una vela de gracia); le=20 ataja un typo que nunca avisaría.
    stale_after_intervals: float = Field(ge=1.0, le=20.0)


class QuantMatrixConfig(BaseModel):
    """Matriz de research del Slow Path (backtest/run_quant_matrix.py).

    Costos AISLADOS de `BacktestConfig` a propósito: la matriz usa un perfil de
    fricción más conservador (taker 0.05% VIP0) sin alterar los P&L de los
    backtests legacy. Cero Hardcoding: todo umbral de la Regla de Oro y del
    simulador de carry vive aquí.
    """
    # Comisión taker por lado (% del notional). 0.05 = VIP0 conservador de Binance
    # Futuros. le=1.0 ataja el typo 0.05 → 5%.
    taker_commission_pct: float = Field(ge=0, le=1.0)
    # Familia E (carry): capital desplegado = notional × esto. El cash-and-carry
    # delta-neutral inmoviliza AMBAS piernas; 2.0 = spot completo + margen perp 1x
    # (conservador, sin riesgo de liquidación). ge=1.0: no puedes desplegar menos
    # que el notional del spot que compras.
    carry_capital_multiplier: float = Field(ge=1.0)
    # Familia E: haircut de mantenimiento por periodo de 8h (bps sobre notional):
    # borrow del spot, gestión de margen. Default 0 = upper bound del carry (la
    # física spot-perp por cantidad NO exige rebalanceo de delta). ge=0.
    carry_maintenance_bps_per_period: float = Field(ge=0)
    # Regla de Oro (gate del embudo): significancia mínima |t| y Profit Factor.
    # gt=0 / gt=1: un PF≤1 no gana, un t≤0 no es señal.
    golden_min_tstat: float = Field(gt=0)
    golden_min_profit_factor: float = Field(gt=1.0)
    # Familia B — cointegración de pares.
    # pairs_lookback_hours: ventana rolling para OLS (β) y z-score (μ/σ). ge=24
    # (mínimo 1 día de datos para estimar β y z-score); le=8760 (1 año = límite
    # superior razonable; más ventana que datos no sirve de nada).
    pairs_lookback_hours: int = Field(ge=24, le=8760)
    # z_entry: umbral de entrada. ge=0.5 (por debajo entraríamos en casi todos los
    # bares); le=5.0 (más de 5σ es tan raro que nunca habría trades).
    pairs_z_entry: float = Field(ge=0.5, le=5.0)
    # z_exit: umbral de salida (cierre). ge=0 (0 = cierra cuando el spread cruza la
    # media). El validador exige z_exit < z_entry: si fueran iguales entraríamos y
    # saldríamos en el mismo bar.
    pairs_z_exit: float = Field(ge=0.0, le=5.0)

    @field_validator("pairs_z_exit")
    @classmethod
    def exit_por_debajo_del_entry(cls, v: float, info) -> float:
        entry = info.data.get("pairs_z_entry")
        if entry is not None and v >= entry:
            raise ValueError(
                f"pairs_z_exit ({v}) debe ser estrictamente menor que "
                f"pairs_z_entry ({entry})"
            )
        return v

    # --- Costos de slippage para familias DIRECCIONALES (C, D; el carry no lo usa) ---
    # Aislados de BacktestConfig por la misma razón que taker_commission_pct: la
    # matriz lleva su propio perfil de fricción. slippage fijo por lado (% del
    # notional). ge=0; le=1.0 ataja el typo 0.02 → 2%.
    slippage_pct: float = Field(ge=0, le=1.0)
    # Slippage dinámico: slip = fijo + k·ATR/precio. k=0 ⇒ solo el fijo. le=5.0
    # ataja un k absurdo que infló el slippage sin sentido.
    slippage_atr_mult: float = Field(ge=0, le=5.0)
    # Ventana del ATR usado por el slippage dinámico. ge=2 (con <2 no hay rango
    # verdadero); le=500 ataja un typo.
    atr_period: int = Field(ge=2, le=500)

    # --- Familia C — reversión a VWAP intradía (5m) ---
    # Ventana rolling del z-score de la desviación. ge=12 (1 hora de 5m: mínimo
    # para una media/desv estables); le=8640 (30 días de 5m) ataja un typo.
    vwap_z_window: int = Field(ge=12, le=8640)
    # Umbral de entrada (|z|). ge=0.5 (por debajo entraría en casi todas las
    # barras); le=5.0 (más de 5σ casi nunca dispara).
    vwap_z_entry: float = Field(ge=0.5, le=5.0)
    # Umbral de cierre. ge=0 (0 = cierra al cruzar la media). El validador exige
    # z_exit < z_entry (si no, abriría y cerraría en la misma barra).
    vwap_z_exit: float = Field(ge=0.0, le=5.0)
    # Horizonte (barras de 5m) del IC de la Etapa 1. La reversión a VWAP es
    # MULTI-barra: gatearla en h=1 la subestima. ge=1; le=288 (1 día de 5m).
    vwap_forward_horizon: int = Field(ge=1, le=288)

    @field_validator("vwap_z_exit")
    @classmethod
    def vwap_exit_por_debajo_del_entry(cls, v: float, info) -> float:
        entry = info.data.get("vwap_z_entry")
        if entry is not None and v >= entry:
            raise ValueError(
                f"vwap_z_exit ({v}) debe ser estrictamente menor que "
                f"vwap_z_entry ({entry})"
            )
        return v

    # --- Familia D — squeeze de volatilidad → ruptura (1h) ---
    # Squeeze = Bollinger DENTRO de Keltner (bb_std·σ < keltner_k·ATR): la
    # dispersión del cierre se contrae por debajo del rango verdadero medio.
    # Ventana común de BB (SMA + σ) y del ATR de Keltner. ge=10 (con <10 barras la
    # σ y el ATR son demasiado ruidosos para hablar de "régimen" de volatilidad);
    # le=500 ataja un typo. 20 es el estándar TTM.
    squeeze_bb_period: int = Field(ge=10, le=500)
    # Nº de desviaciones típicas de las Bollinger. ge=1.0 (por debajo la banda es
    # tan estrecha que casi todo cierre la rompe → ruido); le=4.0 (más de 4σ casi
    # nunca se rompe). 2.0 es el estándar.
    squeeze_bb_std: float = Field(ge=1.0, le=4.0)
    # Multiplicador del ATR de las Keltner. ge=1.0 / le=4.0 por simetría con bb_std:
    # define la referencia de "rango verdadero" contra la que se compara la banda.
    # 1.5 es el estándar TTM. El squeeze es más exigente cuanto MENOR es este k
    # relativo a bb_std (banda debe comprimirse más para caer dentro de Keltner).
    squeeze_keltner_atr_mult: float = Field(ge=1.0, le=4.0)
    # Horizonte (barras de 1h) del IC de la Etapa 1 Y del holding time-based de la
    # Etapa 2. La ruptura/continuación es multi-barra: gatearla en h=1 mide solo el
    # impulso instantáneo (contaminado por el bid-ask del propio breakout). ge=1;
    # le=168 (1 semana de 1h) ataja un holding absurdamente largo.
    squeeze_forward_horizon: int = Field(ge=1, le=168)
    # Umbral de ruptura en unidades de ancho de banda: se considera "rompió" cuando
    # |close − mid| > threshold · (bb_std·σ). 1.0 = exactamente en la banda
    # Bollinger. ge=0.5 (por debajo de media banda no es ruptura); le=3.0 (más de
    # 3 anchos de banda casi nunca dispara). Es la perilla de sensibilidad
    # "cuántas σ de ruptura" — sin sobreoptimizar (default conservador 1.0).
    squeeze_breakout_threshold: float = Field(ge=0.5, le=3.0)


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
    funding_edge: FundingEdgeConfig
    cross_sectional: CrossSectionalConfig
    sentiment_regime: SentimentRegimeConfig
    execution: ExecutionConfig
    orchestrator: OrchestratorConfig
    event: EventConfig
    storage: StorageConfig
    dashboard: DashboardConfig
    quant_matrix: QuantMatrixConfig


def load_settings(path: Path | None = None) -> Settings:
    path = path or PROJECT_ROOT / "config" / "settings.yaml"
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Settings.model_validate(raw)


def load_secrets() -> Secrets:
    return Secrets()
