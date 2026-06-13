# Glosario vivo

Términos en orden de aparición en el proyecto. Se amplía en cada sprint.

## Sprint 0

- **OHLCV**: Open, High, Low, Close, Volume — los cinco datos que resumen el
  precio en un intervalo de tiempo (una "vela"). Es la unidad básica de todo
  análisis técnico.
- **Vela (candle) / timeframe**: agrupación del precio en intervalos fijos
  (5m, 1h…). Una vela "cerrada" ya no cambia; una en formación sí — por eso
  `Candle.closed` existe: operar sobre velas sin cerrar produce señales falsas.
- **Stop-Loss (SL)**: orden que cierra la posición automáticamente si el precio
  va en contra hasta cierto nivel. Convierte una pérdida potencialmente
  ilimitada en una pérdida conocida y acotada.
- **Take-Profit (TP)**: el espejo del SL — cierra la posición al alcanzar un
  objetivo de ganancia.
- **Position sizing**: cuánto comprar/vender. Aquí: arriesgar un % fijo del
  capital por trade → `tamaño = (capital × riesgo%) / distancia_al_stop`. La
  distancia al stop depende de la volatilidad, así el riesgo en dinero es
  constante aunque el mercado cambie.
- **ATR (Average True Range)**: medida de volatilidad — cuánto se mueve el
  precio "típicamente" por vela. Usamos 1.5×ATR para colocar stops: en mercado
  volátil el stop se aleja (evita salidas por ruido), en mercado quieto se
  acerca.
- **Drawdown**: caída desde el máximo histórico del capital. El "max drawdown"
  mide el peor momento de la estrategia; nuestro kill switch salta al 10%.
- **Paper trading / testnet**: operar con dinero ficticio contra un mercado
  (simulado o real). Binance ofrece una testnet gratuita idéntica a la API
  real.
- **Circuit breaker**: mecanismo que detiene el sistema ante una condición
  anómala (conexión caída, datos inconsistentes) antes de que cause daño.
- **WAL (Write-Ahead Logging)**: modo de SQLite donde las escrituras van
  primero a un log separado, permitiendo que lecturas y escrituras convivan
  sin bloquearse (`database is locked`). Imprescindible cuando varios módulos
  async comparten una BD.
- **Event loop / asyncio**: modelo de concurrencia de Python donde un solo
  hilo alterna entre tareas mientras esperan I/O (red, disco). Una operación
  bloqueante (ej. SQLite síncrono) congela TODO el bot — por eso `aiosqlite`.
- **Pydantic / contrato de datos**: validación automática de estructuras. Si
  un módulo produce un dato inválido (score fuera de [-1,1], orden sin SL),
  explota inmediatamente en la frontera del modelo, no horas después en otro
  módulo.
- **Exponential backoff**: ante un error de API (ej. rate limit), reintentar
  esperando tiempos crecientes (1s, 2s, 4s, 8s…) en vez de martillar el
  servidor y ganarse un baneo de IP.
- **Request weight (Binance)**: cada endpoint de la API "pesa" distinto contra
  tu cuota por minuto. Descargar histórico sin contar el peso acumulado
  termina en HTTP 429 (rate limit) o 418 (baneo temporal).

## Sprint 1

- **REST vs Websocket**: REST es "pregunta-respuesta" (pides 1000 velas, te
  las dan, fin) — ideal para histórico. Websocket es una conexión permanente
  donde el servidor te *empuja* datos cuando ocurren — ideal para tiempo real.
  Usamos REST para el pasado y websocket para el presente.
- **Kline**: nombre que da Binance a una vela OHLCV. El mensaje websocket
  trae el campo `x` (is closed): `false` mientras la vela se forma, `true` al
  cerrar. Solo las cerradas alimentan al quant engine.
- **Paginación por cursor**: la API limita cada respuesta a 1000 velas; para
  1 año de velas de 5m (~105.000) se piden páginas sucesivas moviendo un
  cursor (`startTime`) a la vela siguiente a la última recibida. Una página
  vacía o parcial = llegamos al presente.
- **Retry-After**: header HTTP con el que el servidor te dice cuántos
  segundos esperar tras un rate limit. Ignorarlo y reintentar antes alarga el
  castigo — por eso nuestro backoff toma el máximo entre su cálculo
  exponencial y este header.
- **Idempotencia**: propiedad de una operación que da el mismo resultado si
  se ejecuta una o N veces. `INSERT OR REPLACE` + clave primaria
  (symbol, timeframe, open_time) hace que re-guardar una vela tras una
  reconexión no duplique filas.
- **Parquet / formato columnar**: archivo que guarda los datos por columnas
  en lugar de por filas. Leer "solo los cierres de 2025" no toca las demás
  columnas, y comprime ~10× mejor que CSV. Es el estándar para datasets de
  backtesting.
- **Epoch ms**: milisegundos desde el 1/1/1970 UTC. Es el formato nativo de
  tiempo de Binance y el que usamos como clave en SQLite: un entero ordena
  más rápido y sin ambigüedades de zona horaria que un string de fecha.
- **Mainnet vs testnet (datos vs órdenes)**: los *datos* de mercado se toman
  siempre de mainnet (precios reales, públicos, sin API key); las *órdenes*
  van a testnet. La testnet se resetea y su liquidez es ficticia: backtestear
  con sus precios sería estudiar un mercado que no existe.

## Sprint 2

- **EMA (Exponential Moving Average)**: media móvil que pondera más los precios
  recientes. Fórmula: `EMA_t = precio_t × α + EMA_{t-1} × (1-α)`, con
  `α = 2/(n+1)`. Con `n=9`, el precio de hace 9 velas pesa solo ~13%.
  Contraste con SMA: la SMA da el mismo peso a todas las velas de la ventana,
  reacciona más lento a cambios de tendencia.
- **EMA cross (cruce de medias)**: señal clásica. Cuando la EMA rápida (n=9)
  cruza *por encima* de la lenta (n=21), indica inicio de tendencia alcista;
  cruzar *por debajo* indica bajista. Usamos el spread porcentual continuo
  (no solo el evento de cruce) para obtener una señal gradual en [-1,+1].
- **Suavizado de Wilder**: variante del suavizado exponencial con
  `α = 1/n` en lugar del estándar `2/(n+1)`. Más conservador (más inercia).
  Wilder lo usó para RSI y ATR porque reduce el ruido de las ganancias/pérdidas
  diarias. En pandas: `ewm(com = n-1)`.
- **RSI (Relative Strength Index)**: oscilador de momentum creado por Welles
  Wilder (1978). `RSI = 100 − 100/(1+RS)`, donde `RS = avg_gain / avg_loss`
  con suavizado de Wilder. Rango [0,100]. Por encima de 70 = sobrecompra
  (probable corrección); por debajo de 30 = sobreventa (posible rebote).
  Un RSI=50 es neutral.
- **ATR (Average True Range)**: medida de volatilidad que captura gaps entre
  velas. `TR = max(H-L, |H-C_prev|, |L-C_prev|)`. El ATR es el suavizado de
  Wilder del TR sobre `n` períodos. Una vela típica de BTCUSDT en 5m tiene
  ATR ≈ 60 USDT; en 1h ≈ 400 USDT (observado en datos de 2025-2026).
- **tanh (tangente hiperbólica)**: función `tanh(x) = (eˣ - e⁻ˣ)/(eˣ + e⁻ˣ)`.
  Mapea cualquier número real a (-1, 1) de forma suave y simétrica. Útil como
  "compresor" de señales: con un factor de escala 50, un spread de EMA del 1%
  produce score ≈ 0.46; del 3% → ≈ 0.91. Evita cortes bruscos que distorsionen
  el comportamiento cerca de los límites.
- **Función pura (pure function)**: función sin efectos secundarios y sin
  estado propio — misma entrada produce siempre la misma salida. `indicators.py`
  está diseñado así para poder testearlo con valores de referencia y reutilizarlo
  en backtesting sin preocuparse por orden de llamada ni estado interno.
- **Score normalizado [-1,+1]**: convención del bot para todas las señales.
  -1 = máxima convicción bajista, +1 = alcista, 0 = neutro. Normalizar permite
  que la matriz de confluencia combine señales de estrategias distintas (técnica
  + sentimiento) con una escala común, sin conocer sus detalles internos.

## Sprint 3

- **Backtesting**: simular una estrategia sobre datos históricos, barra a
  barra, como si se viviera en tiempo real, para estimar cómo se habría
  comportado. Su valor no es prometer ganancias futuras, sino descartar
  estrategias malas *antes* de arriesgar dinero.
- **Sesgo de anticipación (look-ahead bias)**: usar, para decidir en la vela
  t, información que en la realidad no existía hasta t+1. Es el error #1 del
  backtesting amateur. Lo evitamos decidiendo con el cierre de t y ejecutando
  en la apertura de t+1, y vigilando stops solo con velas posteriores a la
  entrada.
- **Comisión maker/taker**: el exchange cobra un % del notional por operar.
  *Maker* (pones una orden límite que aporta liquidez) es más barato; *taker*
  (orden a mercado que retira liquidez) es más caro. Un bot que entra a mercado
  paga taker. Se cobra en cada lado (entrada y salida).
- **Slippage (deslizamiento)**: diferencia entre el precio esperado y el
  realmente obtenido, porque el libro de órdenes se movió o tu tamaño barrió
  varios niveles. Se modela como un % adverso: compras un poco más caro,
  vendes un poco más barato.
- **Notional**: valor monetario de una posición = cantidad × precio. La base
  sobre la que se calcula la comisión y el límite de "sin apalancamiento"
  (notional ≤ capital).
- **Marcado a mercado (mark-to-market)**: valorar una posición abierta al
  precio actual, incluyendo el PnL *no realizado*. La curva de equity del
  backtester se marca a mercado en cada vela para que el drawdown refleje el
  sufrimiento intra-trade, no solo el resultado al cerrar.
- **Curva de equity (equity curve)**: serie temporal del valor del capital.
  Es la entrada de casi todas las métricas de riesgo (Sharpe, drawdown).
- **Ratio de Sharpe**: retorno medio por unidad de riesgo TOTAL.
  `Sharpe = mean(r)/std(r) × √(barras_por_año)`. Anualizado para comparar
  estrategias de distinto timeframe. >1 decente, >2 bueno. Castiga TODA la
  volatilidad, también la de las subidas.
- **Ratio de Sortino**: como Sharpe pero usando solo la desviación a la baja
  `σ_down = √(mean(min(r,0)²))`. Más justo: no penaliza la volatilidad al alza,
  que a nadie le molesta.
- **Anualización**: escalar una métrica de su frecuencia nativa (por vela) a
  base anual. La varianza crece lineal con el tiempo y la desviación con su
  raíz, de ahí el factor `√(barras_por_año)`: 5m → 105.120, 1h → 8.760.
- **Profit factor**: ganancia bruta / pérdida bruta (en valor absoluto). >1 =
  rentable; 2 = ganas el doble de lo que pierdes. ∞ si no hubo pérdidas.
- **Win rate**: fracción de trades ganadores. Por sí solo engaña: un 30% de
  aciertos puede ser muy rentable si las ganancias son grandes y las pérdidas
  pequeñas (hay que leerlo junto al profit factor y la expectancy).
- **Expectancy**: PnL medio esperado por trade. Positivo = la estrategia gana
  en promedio; negativo = pierde aunque el win rate parezca alto.
- **Exposure (tiempo en mercado)**: fracción de velas con una posición abierta.
  Baja exposure con buen retorno = capital ocioso la mayor parte del tiempo
  (menos riesgo de mercado, pero también menos oportunidades).
- **CAGR (Compound Annual Growth Rate)**: retorno anualizado compuesto,
  `(E_final/E_inicial)^(1/años) − 1`. Permite comparar backtests de distinta
  duración en una tasa anual común.
- **Riesgo:Beneficio (RR)**: relación entre lo que arriesgas (distancia al
  stop) y lo que apuntas a ganar (distancia al take-profit). RR=2 → el objetivo
  está al doble de distancia que el stop; con RR=2 basta acertar >33% para
  empatar.

## Sprint 3.1 (refinamientos de ejecución)

- **Gap (hueco de precio)**: salto entre el cierre de una vela y la apertura de
  la siguiente sin cotización intermedia (flash-crash, noticia, baja liquidez).
  Si el `open` ya está más allá de tu stop/TP, el mercado nunca cotizó ese nivel.
- **Ejecución en gap**: rellenar al `open` (no al nivel) cuando la vela abre ya
  cruzada. En un stop el gap es EN CONTRA (open peor que el stop → más pérdida);
  en un take-profit el gap es A FAVOR (open mejor que el TP → más ganancia).
  Asumir siempre el fill exacto al nivel es un sesgo optimista que la ejecución
  en gap corrige. Mantiene la asimetría honesta del backtester.
- **Slippage dinámico (por ATR)**: deslizamiento que crece con la volatilidad,
  `slip = slip_fijo + k·ATR/precio`, en vez de un % constante. En velas
  agitadas el libro de órdenes se mueve más y el fill empeora. `k=0` lo apaga y
  reproduce exactamente el slippage fijo (condición de regresión).
- **Número mágico / umbral oculto**: constante incrustada en la lógica que
  cambia el comportamiento del bot sin estar en `settings.yaml`. Prohibidos por
  la Política de Cero Hardcoding (ver `CLAUDE.md`): todo parámetro vive en la
  config y se tipa en `config.py`.

## Sprint 4

- **RSS (Really Simple Syndication)**: formato de feed web que publica
  titulares y resúmenes de noticias en XML. Los medios cripto (CoinDesk,
  CoinTelegraph, Decrypt) lo ofrecen gratis. `feedparser` lo parsea en Python;
  `httpx` lo descarga de forma asíncrona sin bloquear el event loop.
- **VADER (Valence Aware Dictionary and sEntiment Reasoner)**: modelo de
  sentimiento basado en reglas, entrenado en texto de redes sociales. Devuelve
  un `compound` en [-1, 1]: positivo = texto positivo. Sus fortalezas son frases
  con signos de puntuación, mayúsculas y emojis; su debilidad es el argot
  cripto ("depeg", "halving", "rugpull" no están en su vocabulario → falsos
  neutros). Por eso solo complementa al diccionario heurístico propio.
- **Diccionario heurístico cripto**: tabla `{término → peso}` donde cada
  entrada es una anotación lingüística calibrada para cripto ("hack" → -0.9,
  "ETF approval" → +0.9). No son números mágicos: son juicios de dominio sobre
  el impacto histórico de cada evento. El score heurístico es la media de los
  pesos de los términos que matchean en el texto.
- **Score local vs. score Claude**: el filtro heurístico produce un `local_score`
  barato (sin API). Si supera `escalate_score_threshold` o el ítem es
  `high_impact`, el bot llama a Claude Haiku y sustituye el score local por el
  score más matizado del LLM. El pipeline de dos etapas mantiene el costo en
  céntimos al día.
- **High-impact flag**: señal de que un evento es potencialmente de alto
  impacto sistémico (hack, FOMC, halving, depeg, ETF) aunque su score local sea
  ambiguo. Cualquier ítem con este flag va a Claude obligatoriamente.
- **Escalado (escalation)**: decisión de delegar el análisis de una noticia de
  la etapa barata (filtro local) a la etapa cara (Claude). Condición:
  `is_high_impact OR |local_score| >= escalate_score_threshold`.
- **compound (VADER)**: el score resumen de VADER. Se calcula como suma
  normalizada de los pesos de cada token del texto; oscila en [-1, 1]. Por
  encima de +0.05 VADER lo clasifica como positivo; por debajo de -0.05, como
  negativo.
- **schema JSON estricto (Claude)**: pedirle al LLM que devuelva SOLO un JSON
  con campos predefinidos (score, confidence, high_impact, symbol_scope,
  rationale). Evita respuestas libres que requieran parsing frágil y fuerza al
  modelo a tomar una decisión explícita en lugar de "podría ser positivo o
  negativo dependiendo de…". Pydantic valida los rangos en la frontera.
- **Deduplicación por hash de URL**: generar un ID con los primeros 16 hex de
  SHA-256(url). La misma noticia desde dos feeds produce el mismo hash → se
  inserta una sola vez. SHA-256 es determinista y resistente a colisiones para
  este volumen de URLs.
- **calendar.timegm vs. time.mktime**: `time.mktime` interpreta un
  `struct_time` como hora local — si el servidor está en UTC-5, convierte
  incorrectamente. `calendar.timegm` lo interpreta siempre como UTC, que es
  el estándar de feedparser. Usar el incorrecto produce timestamps desplazados
  que desordena la deduplicación por antigüedad.

## Sprint 5

- **Confluencia**: exigir que varias fuentes de evidencia independientes apunten
  en la misma dirección antes de arriesgar. Aquí cruzamos el eje técnico (quant)
  con el cualitativo (sentimiento). Principio de diseño: el quant manda la
  DIRECCIÓN, el sentimiento solo CONFIRMA (tamaño pleno), calla (tamaño reducido)
  o VETA (la noticia contradice el patrón). Nunca se abre por sentimiento solo.
- **Matriz de confluencia**: tabla de decisión que mapea (señal quant × señal
  sentimiento) → (acción, tamaño). Vive en `decision/confluence.py` como función
  pura; cada celda es un test de escenario. Umbrales en `settings.yaml`
  (`quant_strong_threshold`, `sentiment_confirm_threshold`, `reduced_size_factor`),
  jamás en el código.
- **size_factor (factor de tamaño)**: multiplicador en [0,1] que la confluencia
  adjunta a la `Decision`. 1.0 = convicción plena (técnica + noticia de acuerdo),
  0.5 = solo técnica, 0.0 = no operar. El Risk Manager lo multiplica dentro de
  la fórmula de sizing: separa "qué tan convencidos estamos" de "cuánto dinero
  arriesgamos por unidad de convicción".
- **Poder de veto (Risk Manager)**: toda orden pasa por `risk/manager.py` antes
  del executor; ningún módulo llama al executor directamente. El Risk Manager
  puede rechazar cualquier `Decision` (devuelve `approved=False` con el motivo).
  Es la defensa contra el peligro #1 de un bot casero: un bug operando sin freno.
- **Evaluador sobre snapshot**: el Risk Manager no es dueño del estado de la
  cartera; recibe una foto (`PortfolioState`: equity, pico, equity de inicio de
  día, posiciones abiertas, salud del feed) y emite un veredicto. La persistencia
  de ese estado la lleva el orquestador. Diseño puro ⇒ trivialmente testeable.
- **Riesgo en dinero constante**: con `qty = (equity × riesgo% × size_factor) /
  distancia_al_stop` y `distancia_al_stop = k·ATR`, la pérdida si salta el stop
  es siempre ≈ el mismo % del capital, sin importar la volatilidad. En mercado
  agitado (ATR alto) el stop se aleja y compras MENOS cantidad; en mercado quieto,
  más. Es la idea central del position sizing por volatilidad.
- **Apalancamiento / sin apalancamiento**: apalancar es operar un notional mayor
  que tu capital (multiplica ganancias y pérdidas). Aquí lo prohibimos: tras el
  sizing, `qty` se topa en `equity/precio` para que `notional ≤ equity`. Un stop
  muy ajustado podría pedir una cantidad enorme; este techo la contiene.
- **Kill switch (con latch)**: corte de emergencia que salta al superar el
  drawdown máximo (10%). *Latcha*: una vez activo, bloquea TODA orden hasta un
  `reset()` manual, aunque la equity rebote. Un corte que se rearma solo no
  protege de nada — obliga a una revisión humana antes de volver a operar.
- **Límite de pérdida diaria**: si la equity cae ≥3% respecto a la del inicio del
  día UTC, no se abren nuevas entradas hasta el día siguiente (el orquestador
  resetea `day_start_equity` al cambiar de día). A diferencia del kill switch,
  se rearma solo con el cambio de día, no requiere intervención.
- **Confianza del sentimiento (sizing)**: el `SentimentScore` trae un `confidence`
  en [0,1]. Si está por debajo de `low_confidence_threshold`, el Risk Manager
  reduce el tamaño con `low_confidence_size_factor`. Un análisis de Claude poco
  seguro pesa menos en cuánto capital se arriesga.
- **Circuit breaker (a) feed obsoleto**: si el último precio tiene más de
  `stale_feed_seconds` (30 s) de antigüedad, se bloquean entradas nuevas: operar
  sobre un precio viejo es operar a ciegas. (La detección real de la antigüedad
  es del feed de datos; el Risk Manager solo respeta el dato.)
- **Circuit breaker (b) sentimiento sin confirmación**: no se actúa sobre un
  titular extremo si el precio no lo confirma (quant débil). En la matriz, "quant
  débil → HOLD" lo encarna: una noticia falsa o mal parseada no abre una posición.
- **Circuit breaker (c) discrepancia de reconciliación**: si el estado local y el
  del exchange no cuadran, se levanta `halted` y el Risk Manager veta todo. La
  *detección* de la discrepancia llega en el Sprint 6 (execution); aquí solo se
  respeta el flag.

## Sprint 5.1 (hardening Spot — auditoría de microestructura)

- **Spot vs. Futures/Margin**: en *Spot* compras/vendes el activo al contado con
  tu propio capital — sin apalancamiento y **sin poder abrir cortos** (un `SELL`
  solo cierra algo que ya tienes). En *Futures/Margin* operas con margen
  prestado: hay leverage y los cortos son nativos. El bot es Spot long-only, así
  que la rama SHORT se desactiva en vivo (`confluence.allow_short=false`).
- **Equity vs. free_balance (saldo libre)**: la *equity* es el valor TOTAL de la
  cuenta (USDT libre + valor de mercado de lo abierto); el *free_balance* es el
  USDT que puedes gastar AHORA. El riesgo (1%) se calcula sobre la equity (para
  que sea constante), pero el techo FÍSICO de la orden se calcula sobre el saldo
  libre: gastar más cash del libre da `INSUFFICIENT_BALANCE` en el exchange.
- **Dinero fantasma**: el bug de dimensionar contra la equity total ignorando lo
  ya comprometido. Con $10k de equity pero $2k libres, una orden de $3.5k "cabe
  en la equity" pero el exchange la rechaza. Se corrige topando por free_balance.
- **Exposición agregada**: suma de los nocionales (qty×precio, *mark-to-market*)
  de todas las posiciones abiertas. En Spot, `comprometido + nueva orden` nunca
  puede superar un % del capital (`max_portfolio_exposure_pct`, 95%): sin este
  tope, 3 señales simultáneas intentarían comprometer hasta el 300%, imposible.
- **Colchón de exposición (buffer)**: el `100 − 95 = 5%` que dejamos sin
  desplegar. Cubre la comisión taker y el slippage de la orden; sin él, gastar el
  100% del cash dejaría sin fondos para pagar el propio fee de la compra.
- **Filtros de microestructura (Binance exchangeInfo)**: reglas del exchange por
  par. `LOT_SIZE`/**stepSize** (la cantidad debe ser múltiplo de este paso),
  `PRICE_FILTER`/**tickSize** (el precio debe ser múltiplo de este paso) y
  `MIN_NOTIONAL`/**minNotional** (qty×precio mínimo, ~$5–10 en pares USDT). Una
  orden que los viola muere en el exchange; el Risk Manager es el último filtro
  que las cumple antes del executor. No son parámetros nuestros: son hechos del
  exchange (no van a `settings.yaml`).
- **Truncar (floor) vs. redondear**: la cantidad se ajusta al stepSize SIEMPRE
  hacia abajo (floor), nunca hacia arriba: subir aumentaría el riesgo y el cash
  comprometido. El SL/TP se redondean al tickSize más cercano, y luego se
  recalcula la distancia real al stop para que el sizing corresponda al stop que
  de verdad se coloca.
- **Decimal vs. float**: la aritmética de pasos exige `decimal.Decimal`. En float
  binario, `0.3 // 0.1 == 2.0` (porque `0.3/0.1 == 2.9999…`), lo que truncaría
  mal una cantidad. `Decimal` da el `3` correcto. Un floor mal hecho violaría el
  riesgo por unos satoshis — un bug real de microestructura.
- **Rechazar, nunca inflar (minNotional)**: si la confluencia reduce el tamaño
  (baja confianza) y el nocional cae bajo el mínimo de Binance, la orden se
  RECHAZA. Subir la cantidad hasta el mínimo rompería el presupuesto de riesgo:
  una operación que no cabe sin violar el riesgo, simplemente no se hace.
