# Plan Maestro: Bot de Trading Híbrido (Algorítmico + Sentimiento NLP)

> Documento vivo. Se actualiza al cierre de cada sprint.
> Decisiones base: **criptomonedas (Binance)** · **presupuesto $0/mes** ·
> **Python 3.13 + asyncio** · **testnet hasta validar métricas**.

---

## 0. Análisis previo: los cuatro desafíos que dan forma al diseño

1. **Latencia**: con herramientas gratuitas no competimos en alta frecuencia.
   El websocket de Binance entrega precios en <100 ms gratis, pero las noticias
   por RSS llegan con 1–5 minutos de retraso. Por eso el bot opera
   *swing/intraday* (velas de 5m–1h): en ese horizonte, 3 minutos de retraso en
   una noticia no invalida la señal. Diseñamos *para* el retraso, no contra él.
2. **Costo de la API de Claude**: llamar a un LLM por cada titular quema dinero.
   Pipeline de dos etapas: un filtro local gratuito descarta el ~90% del ruido
   y solo lo relevante llega a Claude Haiku (~$0.001/noticia → centavos al día).
3. **Riesgo**: el peligro #1 de un bot casero no es una mala estrategia, es un
   bug operando sin control. El Risk Manager es un módulo independiente con
   poder de veto sobre TODA orden, y los Sprints 0–5 son 100% paper trading.
4. **Trabajo con Claude Code**: módulos pequeños con una sola responsabilidad y
   contratos de datos centralizados (`src/core/models.py`). Cada sesión de
   desarrollo carga solo el módulo en curso — sin conflictos de contexto.

---

## 1. Arquitectura del Sistema Híbrido

Cinco módulos desacoplados comunicados por un bus de eventos asíncrono
(`asyncio.Queue`). Ningún módulo conoce los detalles internos de otro: solo
intercambian los modelos de `src/core/models.py`.

```
┌─────────────────┐   ┌──────────────────┐
│ DATA INGESTION  │   │  NEWS INGESTION  │
│ Binance WS/REST │   │  RSS (3 feeds)   │
└───────┬─────────┘   └────────┬─────────┘
        │ Candle               │ NewsItem
        ▼                      ▼
┌─────────────────┐   ┌──────────────────┐
│ QUANT ENGINE    │   │ SENTIMENT ENGINE │
│ TA → Signal     │   │ filtro → Claude  │
└───────┬─────────┘   └────────┬─────────┘
        │ Signal [-1,+1]       │ SentimentScore [-1,+1] + confianza
        └──────────┬───────────┘
                   ▼
        ┌──────────────────────┐
        │ MATRIZ DE CONFLUENCIA│ → Decision
        └──────────┬───────────┘
                   ▼
        ┌──────────────────────┐
        │    RISK MANAGER      │ ← veto absoluto, sizing, circuit breakers
        └──────────┬───────────┘
                   ▼ Order (siempre con stop-loss)
        ┌──────────────────────┐
        │  EXECUTION ENGINE    │ testnet → real, reconciliación
        └──────────────────────┘
```

### Matriz de Confluencia

Ambos motores emiten un score normalizado en **[-1, +1]**. La matriz cruza lo
cuantitativo con lo cualitativo:

| Quant | Sentimiento | Decisión | Tamaño |
|---|---|---|---|
| > +0.5 (fuerte) | > +0.3 (confirma) | LONG | 100% |
| > +0.5 (fuerte) | neutro [-0.3, +0.3] | LONG | 50% |
| fuerte | signo opuesto y fuerte | **HOLD** | — (la noticia puede invalidar el patrón técnico) |
| cualquiera | evento de alto impacto pendiente (FOMC, CPI, hack) | **bloqueo de entradas** | — |

(Simétrico para SHORT. Umbrales en `config/settings.yaml`, no en el código.)

---

## 2. Stack Tecnológico ($0/mes)

| Capa | Elección | Por qué |
|---|---|---|
| Lenguaje | Python 3.13 + asyncio | ecosistema quant, websockets nativos |
| Datos de mercado | `python-binance` (websocket + REST) | gratis, incluye testnet |
| Análisis técnico | `pandas` + indicadores propios en `quant/indicators.py` | escribirlos nosotros es parte del aprendizaje; `pandas-ta` de PyPI es incompatible con numpy ≥ 2 (se reevalúa su fork en el Sprint 2) |
| Noticias | `feedparser` (RSS: CoinDesk, CoinTelegraph, Decrypt) + `httpx` | gratis, sin API key |
| Sentimiento | filtro local (diccionario heurístico cripto + VADER de apoyo) → **Claude Haiku** | centavos/día |
| Base de datos | **SQLite vía `aiosqlite`** (trades, noticias, logs) + **Parquet** (velas históricas) | cero infraestructura; migrable a TimescaleDB si crece |
| Backtesting | motor propio (comisiones + slippage) | entender > caja negra |
| Config | `pydantic-settings` + `.env` + `settings.yaml` | secretos fuera del código, parámetros versionados |
| Dashboard | `streamlit` (Sprint 7) | monitoreo local rápido |

### Directrices técnicas obligatorias

- **Concurrencia en SQLite**: todo acceso a BD usa `aiosqlite` para no bloquear
  el event loop. En la inicialización de `storage.py` se ejecuta
  `PRAGMA journal_mode=WAL` **obligatoriamente**: el modo Write-Ahead Logging
  permite que precios, noticias y logs lean/escriban simultáneamente sin
  errores `database is locked`.
- **Filtro de sentimiento calibrado para cripto**: `sentiment/filter.py` NO
  depende de VADER genérico aislado (entrenado en lenguaje general, produce
  falsos negativos con jerga cripto). El núcleo es un diccionario heurístico
  con pesos de impacto: *pump, dump, rugpull, hack, exploit, depeg,
  liquidation, halving, ETF, SEC, Fed, rate hike, APR, delisting, whale*…
  VADER solo complementa. Toda noticia que matchee un término de alto impacto
  va a Claude Haiku aunque VADER la marque neutra.
- **Rate limits de Binance**: la descarga de históricos pagina respetando el
  *request weight* de la API y aplica **exponential backoff** ante respuestas
  HTTP 429/418, para evitar el baneo temporal de la IP.

---

## 3. Estructura del repositorio

```
IA TRADING/
├── CLAUDE.md                  # convenciones + protocolo didáctico (lo lee Claude Code)
├── PLAN_MAESTRO.md            # este documento
├── pyproject.toml             # dependencias (gestionadas con uv)
├── .env.example               # plantilla de secretos (.env real en .gitignore)
├── config/settings.yaml       # símbolos, timeframes, parámetros de riesgo
├── src/
│   ├── core/                  # models.py (contratos), events.py (bus), config.py
│   ├── data/                  # binance_client.py, storage.py (aiosqlite WAL + Parquet)
│   ├── quant/                 # indicators.py, strategy.py → Signal
│   ├── sentiment/             # feeds.py, filter.py (heurístico cripto), analyzer.py (Claude)
│   ├── decision/              # confluence.py → Decision
│   ├── risk/                  # manager.py: sizing, stops, circuit breakers, kill switch
│   ├── execution/             # executor.py: órdenes testnet/real, reconciliación
│   └── main.py                # orquestador (--check ya funciona)
├── backtest/                  # engine.py + reports/
├── tests/                     # pytest, espejo de src/
├── notebooks/                 # exploración de datos
└── docs/GLOSARIO.md           # glosario vivo de términos
```

**Método de trabajo con Claude Code**: una sesión = un módulo. Los contratos
(`core/models.py`) se escribieron primero y no cambian sin actualizar sus
tests. Cada módulo se valida aislado (pytest + script de demo) antes de
integrarse en `main.py`.

---

## 4. Gestión de Riesgos (no negociable)

- **Stop-loss obligatorio**: toda orden lleva SL basado en volatilidad
  (1.5 × ATR14). El modelo `Order` ni siquiera permite construir una orden sin
  SL — está validado en el contrato de datos, no en la buena voluntad.
- **Position sizing dinámico**:
  `tamaño = (capital × 1%) / distancia_al_stop`. Se reduce ×0.5 si la
  confianza del sentimiento es baja o el ATR está en percentil alto.
- **Límites duros**: máx. 3 posiciones simultáneas · pérdida diaria máx. 3%
  (bot se detiene hasta el día siguiente) · drawdown total 10% → kill switch
  con reinicio manual.
- **Circuit breakers**:
  (a) websocket caído >30 s → proteger posiciones, no abrir nuevas;
  (b) sentimiento extremo sin confirmación de precio → ignorar (posible
  noticia falsa o mal parseada);
  (c) discrepancia entre estado local y exchange al reconciliar → detener y
  alertar.
- **Secretos**: API keys solo en `.env` (gitignored). Claves reales de Binance
  con permiso de trade pero **sin permiso de retiro**. Testnet hasta que las
  métricas de paper trading justifiquen otra cosa.

---

## 5. Roadmap por Sprints

| Sprint | Entregable | Validación |
|---|---|---|
| **0** ✅ | Scaffolding: estructura, git, deps, contratos de datos, config validada | `pytest` verde, `python -m src.main --check` |
| **1** ✅ | Data Ingestion: websocket Binance, velas → aiosqlite/Parquet, descarga de histórico **con paginación por request weight y exponential backoff ante 429/418** | velas en vivo + 1 año de histórico guardado (105.119 velas 5m + 8.759 velas 1h por símbolo, 2025-06 → 2026-06) |
| **2** ✅ | Quant Engine: EMA, RSI, ATR propios + estrategia EMA-cross/RSI → Signal [-1,+1] | 47 tests verdes · demo sobre 105k velas BTC/ETH históricas |
| **3** ✅ | Backtester: motor barra-a-barra sin look-ahead (decisión en cierre t, ejecución en apertura t+1), **ejecución en GAP** (relleno al open si la vela abre cruzada respecto al stop/TP), comisiones + **slippage fijo + dinámico por ATR** por lado, sizing/stops por ATR, métricas (retorno, CAGR, Sharpe, Sortino, max drawdown, win rate, profit factor, exposure) | 94 tests verdes · backtest sobre 105k velas 5m + 8.7k velas 1h por símbolo → reportes Markdown/CSV. Hallazgo: la estrategia naïve EMA-cross/RSI pierde en todas las configs bajo ejecución realista. El aparente edge de ETHUSDT 1h (PF 1.09) era un artefacto de infravalorar el slippage: con slippage dinámico por ATR (k=0.1) cae a PF 0.83 / −15.38%. La ejecución en GAP no afecta cripto 24/7 (sin huecos) pero queda lista para activos con gaps |
| **4** ✅ | Sentiment Engine: RSS → filtro heurístico cripto (diccionario + VADER) → Claude Haiku → SentimentScore [-1,+1] | 138 tests verdes · corpus de 20 titulares etiquetados a mano · filtro descarta noticias no-cripto y calcula score local; high-impact o \|score\| ≥ 0.3 escalan a Claude; `calendar.timegm` para UTC correcto en timestamps RSS |
| **5** ✅ | Confluencia + Risk Manager completos | 169 tests verdes (+31). `decision/confluence.py`: matriz pura (Signal × SentimentScore → Decision) con un test por fila + simetría LONG/SHORT. `risk/manager.py`: evaluador sobre snapshot (`PortfolioState`) con poder de veto, sizing por volatilidad (riesgo en dinero constante + reductores por `size_factor` y por baja confianza + techo sin apalancamiento), SL obligatorio y TP por RR. Vetos cubiertos: máx. posiciones, pérdida diaria, **kill switch por drawdown con latch + reset() manual**, feed obsoleto, halt. Circuit breaker (b) (sentimiento sin confirmación de precio) emerge de "quant débil → HOLD". Demo `src.risk.risk_demo` recorre la cadena Signal→Confluencia→Risk. **Diferido**: reducción de sizing por percentil alto de ATR (necesita distribución rodante de ATR) |
| **5.2** ✅ | Pivote a **Binance Futuros USD-M** (decisión de Eduardo) | 192 tests verdes. Motivo: estructura de costes (taker 0.04% vs 0.1% Spot) + capitalizar catalizadores negativos del Sentiment Engine operando en **corto**. Cambios: (1) **rama SHORT reactivada** y simétrica (confluencia `allow_short=true`; el Risk Manager arma SELL con SL arriba/TP abajo). (2) `PortfolioState` en terminología de Futuros: `wallet_balance` (base de riesgo/DD/pérdida diaria), `available_balance` (techo físico de margen), `committed_margin` (base del margen agregado). (3) **Cero Hardcoding** para `max_leverage` (3, auto-límite anti-casino, le=10) y `max_portfolio_margin_pct` (85%, deja 15% de colchón PnL/liquidación). (4) **Sizing con margen**: qty por riesgo (1% wallet/stop ATR) intacto; el techo valida `margen_inicial = nocional/L ≤ available_balance` y `committed_margin + nuevo ≤ 85%·wallet`. Pipeline de 10 pasos + `Decimal` + recálculo de stop tras tick + filtros, intactos. `Order` lleva ahora `leverage`. Nuevos vetos: `portfolio_margin`, `insufficient_margin`. <br> *(Histórico: 5.1 endureció el modelo Spot long-only antes de este pivote.)* |
| **5.1** 🗑️ | Hardening Spot tras auditoría de microestructura (superado por 5.2) | 188 tests verdes (+19). Venue fijado: **Binance Spot, long-only**. (1) **Dinero fantasma**: el techo físico se calcula sobre `free_balance`, no sobre la equity; `PortfolioState` añade `free_balance` y `committed_notional`. (2) **Exposición agregada**: `comprometido + nueva ≤ max_portfolio_exposure_pct (95%)`, con 5% de colchón para fees/slippage → impide el 300% nocional en Spot. (3) **Microestructura**: nuevo contrato `SymbolFilters` (tick/step/min de exchangeInfo) + helpers `floor_to_step`/`round_to_tick` con `decimal.Decimal`; pipeline de 10 pasos (round SL/TP a tick → recálculo de la distancia real → sizing → floor de qty a step → veto si < minQty/minNotional, jamás inflar). (4) **Cortos OFF en vivo** (`confluence.allow_short=false`): quant bajista → HOLD `short_disabled_spot`; red de seguridad en risk (`short_not_allowed_spot`). Nuevos vetos: `portfolio_exposure`, `insufficient_free_balance`, `below_min_notional`, `below_min_qty`, `stop_rounds_to_entry`. **Pendiente**: la comisión del backtest (`commission_pct: 0.04`) aún refleja un supuesto de futures; revisar a taker Spot (~0.1%) en un sprint de retuneo |
| **6** ✅ | Execution Engine (Futuros USD-M testnet): órdenes, SL/TP, reconciliación | 223 tests verdes (+31). Arquitectura puerto-adaptador: `FuturesExchange` (Protocol) con fake en memoria (tests/demo) y adaptador real de python-binance (`binance_futures.py`, marcado para validar en testnet). `Executor`: (1) `startup()` impone hedge mode con precondición de cuenta limpia (rehúsa si one-way con posiciones — nunca cierra a ciegas), fija leverage y cachea filtros; (2) `open_position()` traduce la Order a entrada MARKET + SL `STOP_MARKET` + TP `TAKE_PROFIT_MARKET` (lado opuesto, mismo positionSide, `closePosition`, disparo MARK_PRICE), idempotente por `clientOrderId`; (3) `close_position()` cierra la pierna con su cantidad real; (4) `reconcile()` → circuit breaker (c); (5) `snapshot_portfolio()` construye el `PortfolioState` (cierra el lazo del S5, lleva pico y arranque de día). Contratos nuevos: enum `PositionSide`/`OrderType`, `position_side` en `Order` con validador de apertura. Storage: tabla `orders` (log auditado idempotente). Config: `execution.{reconcile_position_tolerance, stop_working_type}`. **Pendiente operativo**: la validación de "una semana de paper trading" la corre Eduardo con claves de testnet (red); falta el orquestador en vivo (lazo data→quant→sentiment→confluencia→risk→executor) y la política de flip "una pierna por símbolo", candidatos al Sprint 7 |
| **7** 🚧 | Orquestador en vivo + hardening (alertas, reinicio automático). *Dashboard Streamlit: diferido* | 247 tests verdes (+24). Núcleo del sprint: `orchestrator/engine.py` une todos los motores en `on_closed_candle` (reconciliación→quant→sentimiento→confluencia→risk→política→acción). Política **una pierna por símbolo** (`policy.py`, pura: abrir/flip/nada). Reconciliación con criterio (`classify_reconciliation`): resync benigno por SL/TP vs **halt** peligroso (circuit breaker c). Hardening: `AlertSink` (log/grabador, enchufable a webhook) en kill switch/halt/feed/fallos; supervisión de tareas con auto-restart+backoff; watchdog del feed (circuit breaker a). Punto de entrada `main.py --live` (datos spot mainnet + órdenes futuros testnet). **Operativo (red, lo corre Eduardo)**: la semana de paper trading en testnet y el wiring del poller de sentimiento en vivo. **Diferido a 7.1**: dashboard Streamlit de monitoreo (capa de solo-lectura sobre el storage). Revisión de métricas antes de discutir capital real. |

Cierre de cada sprint: tests verdes + demo funcional + explicación didáctica
completa + actualización de este documento y del glosario.

---

## 6. Protocolo de Explicación Continua

1. **Antes de codificar**: concepto y matemática (con fórmulas) de lo que se va
   a implementar.
2. **Después de codificar**: bloque "📖 Explicación" recorriendo el código y el
   porqué de cada decisión de diseño frente a sus alternativas.
3. **Glosario vivo**: todo término nuevo entra en `docs/GLOSARIO.md`.
4. **Checkpoint por sprint**: resumen de conceptos cubiertos y preguntas
   abiertas.

Este protocolo está fijado en `CLAUDE.md`, así que aplica automáticamente a
todas las sesiones futuras de Claude Code en este repo.
