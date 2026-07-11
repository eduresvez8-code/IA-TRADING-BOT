# Pre-registro: Familias 6-7 (amplitud de mercado, régimen VIX) + combos con RSI-2

Fijado ANTES de correr un solo backtest de estas familias. Última ronda de
búsqueda del proyecto sobre la ventana 2015-2026: pase o no pase, después de
esto se cierra la búsqueda de estrategias sobre este periodo (ver nota de
multiplicidad al final).

## Por qué estas dos y no otras ("lo mejor entre lo mejor de internet")

Eduardo pidió explorar modelos quant adicionales conocidos por funcionar bien
en el S&P 500, descartando los que no sean viables, y priorizando señales
POCO correlacionadas con lo que ya probamos (TSMOM, SMA200, dual-momentum son
esencialmente la misma pregunta — "¿tendencia alcista?" — medida tres veces;
por eso el combo RSI-2+TSMOM del 2026-07-25 no aportó nada nuevo).

**Descartadas antes de escribir código (no cumplen el presupuesto $0/mes o el
protocolo anti-sesgo de este proyecto):**

- **Factores fundamentales (value/quality: P/E, ROE, book value).** Requieren
  datos fundamentales punto-en-el-tiempo, que no están disponibles gratis con
  calidad histórica confiable — y los reportados "hoy" a menudo reflejan
  reexpresiones contables posteriores (restatements), lo que ensucia
  silenciosamente el point-in-time. Riesgo de fuga de datos sutil, sin forma
  barata de auditarlo.
- **Venta de volatilidad / estrategias con opciones (covered calls, iron
  condors, prima de riesgo de volatilidad).** Requieren históricos de cadenas
  de opciones, que no son gratis ni confiables vía yfinance. Fuera de alcance
  por presupuesto.
- **Deriva post-anuncio de resultados (PEAD).** Requiere fechas y sorpresas de
  EPS históricas con calidad point-in-time; no hay fuente gratuita confiable.
- **Pares/cointegración sectorial.** Ya se probó en cripto (Familia B,
  2026-07 anterior) y falló: los spreads son I(1) (no estacionarios), momentum
  en vez de reversión. Mismo riesgo estructural aplica a pares de acciones —
  se descarta por baja probabilidad de éxito, no por falta de datos.
- **Estacionalidad de calendario** (fin de mes, Santa Claus rally, etc.). Ya
  explorada exhaustivamente en 2026-07-08 (`finding-letwinners-sweep`,
  `finding-stocks-weekend-effect`): el único hallazgo que sobrevivió fue el
  "efecto fin de semana" en ACCIONES INDIVIDUALES, no en timing del índice
  agregado — no encaja en el marco largo/cash de este protocolo sin inventar
  un mecanismo nuevo. Se deja fuera para no reabrir una veta ya agotada.

**Elegidas (datos gratis ya verificados, información genuinamente distinta a
lo ya probado):**

1. **Amplitud de mercado (market breadth).** En vez de mirar solo el precio
   de SPY, mira CUÁNTAS acciones del índice están, cada una, por encima de su
   propia tendencia. Es información transversal (todo el universo), no una
   sola serie de precio — la fuente de información es distinta de TSMOM/SMA
   200/dual-momentum aunque el "sabor" (tendencia) sea similar. No requiere
   descargas nuevas: usa los 774 tickers del universo punto-en-el-tiempo ya
   descargados para la Familia 1 (XS momentum).
2. **Régimen de VIX.** El VIX mide el miedo IMPLÍCITO en el precio de las
   opciones (volatilidad esperada), no el momentum del precio spot. Es una
   fuente de información genuinamente distinta: dos días pueden tener el
   mismo precio de SPY y un VIX muy diferente. Dato gratis vía yfinance
   (`^VIX`), verificado: valores ~13-14 en feb-2024, consistentes con el
   VIX real de esa fecha (sin problema de escala, a diferencia de ^TNX que
   se descartó por ambigüedad de unidades no verificada).

## Definiciones y matemática

**Amplitud (Familia 6).** Para cada día t, con M_t = miembros punto-en-el-
tiempo del índice en el mes de t (misma membresía ya usada en XS momentum):

    amplitud_t = (1 / |M_t|) · Σ_{i ∈ M_t} 1{close_i,t > SMA_N(close_i)_t}

Solo cuenta si el ticker tiene precio Y suficiente historia para su propia
SMA_N (si no, queda fuera del numerador Y del denominador de ese día — no se
rellena con 0 ni con 1). Si la cobertura (elegibles/miembros) cae bajo el
piso ya declarado en `research.xs_momentum.min_coverage` (0.60, reusado, no
un umbral nuevo), el día queda SIN decisión (NaN).

Señal: long SPY si amplitud_t > umbral, si no cash. Grid (`research.breadth`):
N ∈ sma_days_grid = {100, 200} (mismas ventanas que Familia 3) × umbral ∈
threshold_grid = {0.40, 0.50, 0.60} → 6 configuraciones.

**Régimen VIX (Familia 7).** Para cada día t:

    calma_t   = 1{VIX_t < SMA_N(VIX)_t}      (dirección "below")
    miedo_t   = 1{VIX_t > SMA_N(VIX)_t}       (dirección "above")

Ambas direcciones son hipótesis legítimas en la literatura (mercados suelen
rendir mejor con volatilidad baja/decreciente — pero también hay reversión
tras picos de miedo) — se deja que el TRAIN decida cuál, nunca a mano. Señal:
long SPY si el régimen elegido es 1, si no cash. Grid (`research.vix_regime`):
N ∈ sma_days_grid = {50, 100, 200} × dirección ∈ {below, above} → 6 configs.

**Combos con RSI-2 (regime gating, mismo mecanismo que TSMOM del 2026-07-25).**
RSI-2 propio (entry<10, exit>70) NO se re-tunea — se reproduce la selección
ya publicada. El filtro de régimen (amplitud o VIX) exige su propia condición
== 1 para ABRIR una posición nueva; nunca fuerza salida anticipada. La config
del filtro (N y umbral/dirección) se selecciona por Sharpe de TRAIN de la
COMBINACIÓN (no del filtro standalone) entre las 6 configs de su grid — mismo
patrón ya usado con TSMOM.

## Split, costos, criterios (idénticos al protocolo madre)

TRAIN < 2015-01-01 ≤ TEST · 2 pb/lado (estrés 5 pb) · cash devenga T-bill.
Selección SOLO por Sharpe de TRAIN, test medido UNA vez, grid COMPLETO
reportado. Los 5 criterios de siempre: Sharpe test > 0.5, bootstrap CI 90%
excluye 0, concentración < 60%, ambas mitades > 0, supera Sharpe B&H SPY.

**Nota sobre "acertar en la mayoría de las ocasiones":** se reportará la tasa
de acierto (win-rate) por trade como dato DESCRIPTIVO adicional, pero NO como
criterio de éxito. Un win-rate alto no es lo mismo que tener ventaja: una
estrategia puede acertar el 90% de las veces y perder dinero en conjunto si
el 10% de pérdidas es mucho más grande que las ganancias típicas (es
exactamente el perfil de riesgo de vender opciones/seguros — "recoger
monedas frente a una apisonadora"). El listón sigue siendo el mismo de
siempre.

## Nota de multiplicidad (cumpliendo la disciplina de las dos rondas previas)

Esta ronda evalúa 4 configuraciones ganadoras contra el TEST (Familia 6
standalone, Familia 7 standalone, RSI-2+Familia 6, RSI-2+Familia 7) — más
carga que las dos extensiones anteriores (que evaluaban solo 1 cada una).
Es comparable en tamaño a la ronda original de 5 familias del 2026-07-11,
donde se aceptó evaluar 5 ganadores simultáneamente porque el criterio de
éxito exige los 5 gates a la vez (no "el mejor de los 5"), y se reporta el
grid completo sin cherry-picking.

**Esta es la ÚLTIMA ronda de búsqueda sobre la ventana 2015-2026.** Pase o no
pase alguna configuración, después de esto el proyecto NO genera más
variantes sobre este periodo — ni reencuadres de métrica, ni nuevos filtros,
ni nuevas familias. Si nada pasa, el siguiente paso es forward/paper trading
real, no una cuarta ronda de backtest.
