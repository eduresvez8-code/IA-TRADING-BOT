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
| **1** | Data Ingestion: websocket Binance, velas → aiosqlite/Parquet, descarga de histórico **con paginación por request weight y exponential backoff ante 429/418** | velas en vivo + 1 año de histórico guardado |
| **2** | Quant Engine: indicadores propios (EMA, RSI, ATR) + primera estrategia | tests de indicadores contra valores de referencia |
| **3** | Backtester: comisiones, slippage, métricas (Sharpe, max drawdown, win rate) | backtest sobre el histórico + reporte |
| **4** | Sentiment Engine: RSS → filtro heurístico cripto → Claude Haiku → score | corpus de 20 titulares etiquetados a mano vs. salida del modelo |
| **5** | Confluencia + Risk Manager completos | tests de escenario por cada regla de riesgo |
| **6** | Execution en **testnet**: órdenes, SL/TP, reconciliación | una semana de paper trading con log auditado |
| **7** | Dashboard Streamlit + hardening (alertas, reinicio automático) | revisión de métricas antes de discutir capital real |

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
