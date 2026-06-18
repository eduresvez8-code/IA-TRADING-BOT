# Plan Maestro v2 — Bot Event-Driven Híbrido (Arquitectura Dual-Core)

> **Estado:** documento vivo. Es la nueva biblia arquitectónica del proyecto.
> `PLAN_MAESTRO.md` (v1) se conserva como **legado documental** (sprints 0–7.2,
> el pivote a Futuros, los 6 edge tests fallidos). Este v2 **no lo reemplaza**:
> lo continúa, corrigiendo la causalidad invertida del sistema híbrido.
>
> Decisiones base heredadas del v1: **cripto (Binance Futuros USD-M)** ·
> **$0/mes en datos** · **Python 3.13 + asyncio** · **testnet hasta validar
> métricas** · **Cero Hardcoding** · **protocolo didáctico**.

---

## 0. La verdad incómoda primero (sin esto, lo demás es decoración)

Dos hechos físicos que **dictan** el diseño y que el v1 ignoraba:

**(A) Nunca vamos a ganar la carrera de latencia.** Con RSS gratuito + poll de
120s + lag de 1–5 min del propio feed, para cuando el bot "ve" un hack o un ETF,
los market makers co-localizados ya movieron el precio. Por tanto **el edge de
noticias NO puede ser "llegar primero"**. El único edge accesible a $0/mes es el
**drift post-evento**: que el movimiento *tenga continuación* durante
minutos-horas después de que la noticia ya es pública. Si ese drift no existe a
nuestra latencia real de entrada (≈ t+3min), el Fast Path está muerto. **Toda la
validación del Fast Path mide retorno desde t+entrada_real, no desde t+0.**

**(B) No tenemos corpus histórico de noticias.** El Sprint C.3 lo documentó:
CryptoPanic free murió, el archivo histórico es de pago, solo queda Fear&Greed
(sentimiento de mercado, no eventos). **Consecuencia brutal: el edge de eventos
NO se puede backtestear a $0 hoy.** Solo se puede *forward-testear* acumulando
eventos en testnet. Esto reordena la validación: el Fast Path se valida con un
**event study forward pre-registrado** (30–60 días), no con un backtest.

---

## 1. Crítica Sistémica — qué está fundamentalmente roto en el v1

No es un bug en la línea 70. Es la **causalidad invertida** de todo el sistema.

1. **Se gastó todo el presupuesto de edge en el subsistema equivocado.** Seis
   familias quant fallaron la Golden Rule (`|t|≥2` ∧ `PF>1.15`); EMA/RSI
   **anti-predice** (IC<0). Y aun así, [confluence.py:70](src/decision/confluence.py:70)
   hace que ese score roto sea el **originador obligatorio de toda operación**.
   El único componente que procesa información *exógena y movedora de precio*
   (noticias vía Claude) tiene prohibido por diseño originar nada. Estamos
   confirmando ruido con señal.

2. **El manejo de `high_impact` está literalmente al revés.**
   [confluence.py:65-66](src/decision/confluence.py:65): un hack, un ETF, un CPI
   → `HOLD`, bloqueo total. Los eventos **más asimétricos y operables del
   cripto** son justo los que el sistema tira a la basura. Hay un núcleo legítimo
   enterrado (no entrar *dentro* de un dato macro programado con señal stale),
   pero se aplicó como veto general. Es destrucción de edge disfrazada de riesgo.

3. **La cadencia es esclava de la vela.** Todo corre en `on_closed_candle` (5m).
   Una noticia a t+10s espera hasta 5 min para *siquiera* ser evaluada, y
   entonces la gatea un score que anti-predice. El sistema es
   **arquitectónicamente incapaz** de operar eventos.

4. **No hay TTL.** `SentimentScore.analyzed_at` existe pero **nadie lo compara
   con `now`**. El `_sentiment_loop` ([engine.py:417](src/orchestrator/engine.py:417))
   solo pisa las claves que vuelven; un símbolo sin noticia fresca **conserva su
   último score para siempre**.

5. **Ejecución sin protección de precio.** El adaptador real solo sabe `MARKET`
   para entradas ([binance_futures.py:154](src/execution/binance_futures.py:154)).
   No existe ruta LIMIT/IOC. Un `MARKET` contra una vela de noticia se come el
   spread + todo el top-of-book fino → slippage destructivo.

6. **El sizing depende de un ATR retrasado.** El Risk Manager dimensiona con
   `atr` ([manager.py:155](src/risk/manager.py:155)). ATR(14) en 5m es
   *backward-looking*: en el cambio de régimen que ES la noticia, subestima la
   vol → stop instantáneo o riesgo mal medido.

7. **El scope símbolo↔noticia es lossy.** `sentiment_store` keyed por símbolo,
   pero `SentimentScore.symbol_scope` puede ser `["*"]`. El mapeo es implícito.

**Resumen:** el v1 es un motor técnico roto con un airbag de sentimiento que solo
sabe pisar el freno. El v2 invierte la jerarquía.

---

## 2. Diseño Dual-Core — la lógica del split

Dos pipelines con **cadencias, decisores, sizing y ejecución distintos**, que
**convergen en un único punto de veto y un único estado**. La regla de CLAUDE.md
("toda orden pasa por `risk/manager.py`; ningún módulo llama al executor
directamente") es la columna vertebral del diseño.

```
   SLOW PATH (Estratégico)              FAST PATH (Táctico / Evento)
   cadencia: vela cerrada (5m)          cadencia: llegada de evento (push, ~15s)
   ─────────────────────────            ──────────────────────────────
   Velas → Quant (CONTEXTO/régimen)     RSS → filtro → Claude (ORIGINADOR)
   + Cross-sectional reversal overlay   + confirmación de impulso de precio
   + Fear&Greed (régimen)               + TTL + cooldown por símbolo
            │  decide_strategic()                 │  decide_event()
            │  Decision                           │  Decision (origen=event)
            └──────────────┬──────────────────────┘
                           ▼   asyncio.Queue de "intents"
            ┌───────────────────────────────────┐
            │  ORCHESTRATOR — sección crítica     │  ← un solo asyncio.Lock
            │  (reconcile → risk → policy → open) │     (ya existe en engine.py)
            └───────────────────────────────────┘
                           ▼
            ┌───────────────────────────────────┐
            │  RISK MANAGER  (único veto)         │  modo strategic | modo event
            └───────────────────────────────────┘
                           ▼
            ┌───────────────────────────────────┐
            │  EXECUTION  — MARKET (slow)         │
            │             LIMIT-IOC capado (fast) │
            └───────────────────────────────────┘
```

**Decisiones de diseño no negociables:**

- **El Fast Path NO abre posiciones por su cuenta.** Si abriera fuera de banda,
  rompería el bookkeeping de `expected`/`_in_flight` → HALT falso. En su lugar
  **encola un `EventIntent`** que el Orchestrator consume **dentro del mismo
  `self._lock`** ([engine.py:90](src/orchestrator/engine.py:90)). Reusa `_open`,
  `decide_position_action`, `_in_flight`. Cero ejecución duplicada.
- **Un solo Risk Manager, dos modos.** No se clona. Recibe un `mode`
  (strategic/event) que selecciona el set de parámetros de sizing. Kill-switch,
  daily-loss, portfolio-margin y reconcile-halt son **idénticos y compartidos**:
  el Fast Path no puede saltarse ningún circuit breaker.
- **El quant deja de originar, PERO no se borra.** EMA/RSI pasa a **filtro de
  Contexto/Régimen** del Slow Path y **modula la convicción (sizing)**: si el
  Fast Path origina un trade por noticia que va *contra* la tendencia EMA/RSI, el
  trade **se hace igual**, pero el quant puede **reducir el tamaño** de esa
  posición (decisión de Eduardo). El edge real del Slow Path lo aporta el
  **cross-sectional reversal** (único lead con IC significativo), no EMA/RSI.

---

## 3. Implementación por Fases

> Toda variable de comportamiento nueva → `settings.yaml` + `config.py` +
> `test_config.py` (Vía B, Cero Hardcoding). Los diccionarios lingüísticos de
> `filter.py` no son números mágicos (hechos de idioma) y se mantienen.

### FASE 1 — Desacoplar y des-arriesgar el núcleo roto (sin alpha nuevo)

Objetivo: arreglar los **bugs estructurales** sin apostar todavía al edge de
noticias. Todo validable en backtest existente + testnet.

**1.1 TTL de sentimiento (resuelve crítica #4). [HECHO]**
- Nuevo `ConfluenceConfig.sentiment_ttl_seconds: int = Field(ge=1, le=86400)`.
- `decide()` gana `as_of: datetime | None` → **determinismo** (fija
  `Decision.timestamp`). **El TTL NO va dentro de la matriz pura**: el backtest
  ya caduca con `max_news_age_hours` a escala de horas (Fear&Greed diario); un
  TTL en segundos dentro de `decide()` mataría su brazo de sentimiento.
- **El TTL vive en el engine** (`_fresh_sentiment`): caduca por `analyzed_at` y
  purga la clave del store. Aplica por igual a `decide()` y a la `confidence`
  que recibe el Risk Manager.

**1.2 Split de `high_impact` (resuelve crítica #2). [HECHO]** El veto-total
inverso se reemplaza por dos clases de evento (en [filter.py](src/sentiment/filter.py)).
*Entregado:* `event_kind` en `FilterResult` y `SentimentScore` (Literal,
default "none"); `SCHEDULED_MACRO_TERMS`/`IDIOSYNCRATIC_SHOCK_TERMS` (su unión =
el `HIGH_IMPACT_TERMS` del v1 → escalación a Claude intacta); `scoring` propaga
la etiqueta (determinista, también sobre la rama de Claude); confluencia bloquea
solo `scheduled` (`scheduled_macro_block`), `shock` cae a la matriz. *Diferido a
Fase 2:* ventana temporal del bloqueo macro + ampliar scheduled a fed/rate-hike
(decisión de config con EventConfig). Detalle de diseño original abajo:
- `SCHEDULED_MACRO_TERMS` {fomc, cpi, fed, rate hike/cut…} → programado, incierto.
- `IDIOSYNCRATIC_SHOCK_TERMS` {hack, exploit, depeg, etf approval, bankruptcy…}
  → direccional, operable.
- Añadir `event_kind: Literal["none","scheduled","shock"]` a `FilterResult` y a
  `SentimentScore` (**toca `models.py` → actualizar `tests/test_models.py` en el
  mismo cambio**).
- `scheduled` → bloqueo solo dentro de una ventana `macro_block_*`, no "para
  siempre". `shock` → deja de ser HOLD; candidato a originación (Fase 2).

**1.3 Ejecución con tope de slippage (resuelve crítica #5).**
- Extender `OrderRequest` con `price`, `time_in_force`, y `OrderType.LIMIT`.
- Entrada **marketable-limit IOC**: precio = `mark ± slippage_cap`,
  `timeInForce=IOC`. Llena dentro de la banda, cancela el resto; si el libro está
  fuera de banda, **no rellena** (mejor perder el trade que comprar 2% arriba).
- Nuevos `ExecutionConfig.slippage_cap_bps`, `aggressive_entry_tif`.
- **Fix:** tras IOC parcial, registrar `expected[key] = qty_llenada_real`, no
  `order.quantity` ([engine.py:291](src/orchestrator/engine.py:291)).

### FASE 2 — Fast Path: originación por evento (corazón del rediseño)

**2.1 Nuevo `EventConfig`** (Cero Hardcoding): `enabled`, `poll_interval_seconds`
(≈15, ≠ slow 120s), `min_impact_score` (≈0.6), `min_confidence` (≈0.7),
`ttl_seconds` (≈180), `cooldown_seconds` (≈900), `confirm_impulse_bps` (≈8),
`confirm_window_seconds` (≈60), `size_factor` (≈0.5), `macro_block_minutes_before`
(≈30) / `_after` (≈5). Todo tipado en `config.py` + filas en `test_config.py`.

**2.2 `decide_event()` — nuevo decisor puro en `confluence.py`:** origina
LONG/SHORT solo si `event_kind=="shock"` ∧ `|score|≥min_impact_score` ∧
`confidence≥min_confidence` ∧ fresco (TTL) ∧ no en cooldown ∧ **confirmación de
impulso** (el precio se movió `≥confirm_impulse_bps` en la dirección dentro de
`confirm_window_seconds`). La confirmación de impulso es el **núcleo legítimo del
circuit breaker (b)** del v1 ("no operes un titular posiblemente mal parseado sin
que el precio lo respalde"), aplicado a la escala de tiempo correcta.
`decide_strategic()` (el `decide` actual) puede **vetar o reducir** un
event-intent por contexto técnico, pero el quant ya no es condición necesaria.

**2.3 `engine.py` — productor + consumidor:** `asyncio.Queue[EventIntent]`; tarea
`_event_loop` que **empuja** (no espera al poll de 120s); `on_event(intent)` que
adquiere **el mismo `self._lock`**, reconcilia, `decide_event`, sizing en **modo
event**, y enruta por el **mismo** `decide_position_action` + `_open`. ATR del
buffer rodante; si no está warm → rechaza. Resolver scope→símbolo.

**2.4 Sizing de evento en `RiskManager.assess(mode=...)`** (resuelve crítica #6):
`event_risk_per_trade_pct` (<1%), `event_atr_stop_multiplier` (>1.5),
`vol_regime_lookback` + `vol_expansion_cap` (reduce riesgo si el ATR está
expandido X× sobre su media — implementa por fin el sizing diferido del S5). El
ATR es **volatilidad** (legítima), no la señal EMA/RSI (rota).

### FASE 3 — Overlay estratégico (el único alpha real) + capital

- **Cablear el cross-sectional reversal** (primer lead real: IC negativo
  significativo en 518 perps) como **overlay de portafolio del Slow Path**, no
  scalping. Falta el paso de "edge test" a "señal viva robusta a la cola/skew".
- **EMA/RSI demotado a filtro de régimen** que modula sizing (ver §2).
- Gating de capital real: solo tras cumplir TODOS los kill criteria de §4 sobre
  ≥30–60 días de testnet. Decisión explícita de Eduardo.

---

## 4. Kill Criteria — condiciones de fallo, matemáticas y estrictas

Notación: retorno neto por operación `r_i = dir·(p_salida/p_entrada − 1) − c`,
con `c = 2·(0.0004 + slip_bps/1e4)`. Profit Factor `PF = Σr⁺ / |Σr⁻|`. t-stat de
la media `t = r̄ / (s/√N)`.

**A. Backtest (Slow path / cross-sectional — el event NO se puede backtestear):**
- **Golden Rule, sin excepción:** vivo solo si `|t|≥2` ∧ `PF>1.15` ∧ `r̄>0`, con
  slippage dinámico por ATR activado (`k≥0.1`). Mismo criterio que mató 6
  familias; no se relaja para el cross-sectional.
- Walk-forward: signo del edge **consistente en los 4 folds**. Un fold ganador
  rodeado de perdedores = overfit → kill.

**B. Forward event study (Fast Path — pre-registrado ANTES de ver datos):**
- **Drift a latencia real:** retorno medio en la dirección del score, medido
  **desde t+entrada_real (≈t+3min)**, con `|t|≥2` y `N≥30` eventos. Si el drift
  solo existe en [t+0, t+3min] → **Fast Path muerto, no va a vivo.**
- **Net edge:** `PF>1.15` **después** del `slippage_cap` realizado.
- **La confirmación de impulso debe ayudar:** ablación A/B; si no discrimina, se
  quita (complejidad muerta).

**C. Testnet operacional (gate duro, cualquier fallo = no-go):**
- **Slippage:** mediana del slippage de entrada realizado ≤ `slippage_cap_bps`, y
  tasa de no-fill del IOC < 30% (si no, el cap es tan estrecho que nunca operas).
- **Integridad estructural — tolerancia CERO:** 0 HALTs causados por aperturas
  del Fast Path; 0 posiciones desnudas; 0 trades con `age > ttl` (auditado en
  `orders.decision_reason`). Cualquiera ≠ 0 es bug estructural.
- **Riesgo:** durante ≥30 días, daily-loss y kill-switch no se disparan por mala
  calibración del sizing de evento; MAE por trade dentro del presupuesto.

**D. Kill global del proyecto (pre-mortem):** si tras Fase 2 el event study da
`|t|<2` a latencia real **y** el cross-sectional no pasa la Golden Rule en
walk-forward, **no hay edge accesible a $0/mes**: la conclusión correcta es no
arriesgar capital, no seguir añadiendo épées.

---

## 5. Secuencia de ejecución (una sesión = un módulo)

1. **F1.1** TTL (`sentiment_ttl_seconds`) + `as_of` en confluence + `_fresh_sentiment` en engine. ✅ *(409 tests)*
2. **F1.2** split `high_impact` → `event_kind` (scheduled/shock/none); confluencia bloquea solo scheduled. ✅ *(409 tests)*
3. **F1.3** LIMIT-IOC + `slippage_cap_bps` + fix de `expected` en fills parciales. *(siguiente)*
4. **F2.1–2.2** `EventConfig` + `decide_event` puro.
5. **F2.3** cola + `on_event` en el engine (dentro del lock).
6. **F2.4** sizing de evento + vol-regime en Risk Manager.
7. **F3** cross-sectional overlay + gating de capital.

Cada paso: pytest verde + demo aislada + bloque "📖 Explicación" + glosario,
antes de integrar en `main.py`. Capital real: revisión de métricas con Eduardo.
