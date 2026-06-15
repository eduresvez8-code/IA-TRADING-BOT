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

## Sprint 5.2 (pivote a Futuros USD-M)

- **Futuros USD-M**: contratos perpetuos liquidados en USDT. A diferencia de
  Spot, hay **apalancamiento** y los **cortos son nativos** (no necesitas poseer
  el activo para venderlo). Es lo que permite capitalizar catalizadores
  negativos del Sentiment Engine (hacks, exploits, FUD) operando en corto.
- **Apalancamiento (leverage)**: multiplicador que permite controlar un nocional
  mayor que el margen aportado. Con leverage L, abrir un nocional N solo
  inmoviliza `N/L` de margen. NO cambia la cantidad (esa la fija el riesgo); solo
  decide cuánto margen consume. El bot se auto-limita con `max_leverage` (3x):
  nada de apalancamiento de casino.
- **Margen inicial**: colateral que el exchange bloquea para abrir una posición,
  `margen_inicial = nocional / leverage`. Es lo que de verdad "cuesta" abrir en
  futuros, y contra lo que se valida el techo de entrada (no contra el nocional).
- **wallet_balance**: colateral total de la cuenta de futuros SIN contar el PnL
  no realizado. Es la base del riesgo (1%), del drawdown (kill switch) y de la
  pérdida diaria — magnitudes que no deben inflarse con ganancias aún sin cerrar.
- **available_balance**: margen libre que el exchange reporta AHORA para abrir
  nuevas posiciones (descontados el margen comprometido y las pérdidas no
  realizadas). Es el **techo físico** real de una apertura: `margen_inicial ≤
  available_balance`, o el exchange devuelve margen insuficiente.
- **committed_margin (margen agregado)**: margen inicial ya inmovilizado por las
  posiciones abiertas. El bot exige `committed_margin + nuevo ≤
  max_portfolio_margin_pct (85%) del wallet`, dejando un 15% de colchón para
  fluctuaciones de PnL no realizado y comisiones de liquidación.
- **Stop vs. liquidación**: el precio de liquidación está a ~`1/L` de movimiento
  adverso (33% con 3x); nuestro stop (≈1.5·ATR) está mucho más cerca, así que el
  SL salta MUCHO antes de que el exchange liquide. Por eso un leverage bajo con
  stops ajustados es seguro: nunca llegamos a la zona de liquidación.
- **PnL no realizado**: ganancia/pérdida de una posición abierta valorada a
  precio de mercado, aún sin cerrar. El `available_balance` lo descuenta (una
  posición perdiendo reduce el margen libre); el `wallet_balance` no lo incluye.

## Sprint 6 (Execution Engine)

- **Position mode (One-Way vs Hedge)**: configuración de la cuenta de Futuros.
  En *one-way* hay una posición NETA por símbolo (un SELL netea un LONG previo);
  en *hedge* (`dualSidePosition=true`) conviven piernas LONG y SHORT
  independientes. El bot impone hedge al arrancar para que su suposición de
  posiciones independientes sea cierta y un SHORT no netee accidentalmente un LONG.
- **side vs positionSide**: dos ejes ortogonales de una orden en hedge mode.
  `positionSide` (LONG/SHORT) dice EN QUÉ cubo opero (lo fija la dirección);
  `side` (BUY/SELL) dice si ABRO o CIERRO. Matriz: abrir LONG=BUY+LONG, cerrar
  LONG=SELL+LONG, abrir SHORT=SELL+SHORT, cerrar SHORT=BUY+SHORT. El `side` solo
  es ambiguo (un SELL puede abrir-short o cerrar-long), por eso el `positionSide`
  es explícito, nunca inferido.
- **STOP_MARKET / TAKE_PROFIT_MARKET**: órdenes condicionales que disparan a
  mercado al tocar `stopPrice`. Son las protectoras de la posición: lado opuesto
  al de entrada, mismo `positionSide`, con `closePosition=true` (cierran la
  pierna entera, inmunes al drift de cantidad por fills parciales).
- **closePosition vs reduceOnly**: dos formas de marcar "esto cierra, no abre".
  `reduceOnly` se RECHAZA en hedge mode (el par side+positionSide ya lo
  determina); `closePosition=true` sí se usa, pero solo en órdenes condicionales.
  Un cierre a mercado, en cambio, lleva la `quantity` de la pierna.
- **workingType (MARK_PRICE vs CONTRACT_PRICE)**: precio sobre el que dispara un
  stop. `MARK_PRICE` (precio de marca, una media robusta) evita que un wick de
  manipulación en el último precio active el stop; `CONTRACT_PRICE` usa el último
  negociado. Usamos MARK_PRICE.
- **clientOrderId (idempotencia)**: identificador que asignamos a cada orden
  ANTES de enviarla. Si un envío hace timeout pero el exchange sí la ejecutó,
  reintentar con el mismo id hace que Binance rechace el duplicado en vez de
  abrir dos. La PK del log auditado es este id, por la misma razón.
- **Reconciliación**: comparar periódicamente lo que el bot CREE tener (estado
  local) con lo que el exchange REPORTA. Una diferencia por encima de la
  tolerancia activa el circuit breaker (c): el bot se detiene en vez de operar
  sobre una imagen falsa de la cartera.
- **Protocol / adaptador (puerto-adaptador)**: `FuturesExchange` es un Protocol
  que define lo que el Executor necesita del exchange. Un fake en memoria y el
  adaptador real de python-binance lo implementan; el Executor no sabe cuál usa,
  así que su lógica se prueba entera sin red. Es el patrón puerto-adaptador.
- **exchangeInfo**: endpoint de Binance con los metadatos de cada par (filtros de
  microestructura, precisión). El executor lo lee al arrancar para poblar los
  `SymbolFilters` reales; en futuros el mínimo es `MIN_NOTIONAL.notional` (en spot
  era `minNotional`).

## Sprint 7 (orquestador en vivo + hardening)

- **Orquestador**: el lazo que une todos los motores. Por cada vela cerrada
  ejecuta data→quant→sentimiento→confluencia→risk→executor. Es el "director de
  orquesta" que hasta ahora faltaba: convierte módulos aislados en un bot que opera.
- **Una pierna por símbolo (política de flip)**: regla de gestión de posición. Si
  estamos planos y hay señal aprobada → abrir; si ya tenemos esa dirección → no
  duplicar; si llega la dirección OPUESTA → *flip* (cerrar la actual y abrir la
  nueva). Evita acumular piernas LONG y SHORT a la vez (doble margen y funding).
- **Fuente de verdad = el exchange**: el orquestador deriva qué tiene abierto del
  snapshot del exchange cada ciclo, no de un contador interno. Así un SL/TP que
  cerró una pierna se absorbe solo, sin quedar desincronizado.
- **Resync vs. halt (reconciliación)**: al comparar el modelo interno con el
  exchange, una pierna esperada que ya no está = cierre por SL/TP → *resync*
  (benigno, resincronizamos); una pierna desconocida o con cantidad divergente =
  peligro → *halt* (circuit breaker c, el bot se detiene). Distinguirlos evita
  tanto falsos paros como ignorar un riesgo real.
- **AlertSink**: abstracción de "esto debe verlo un humano". Desacopla el evento
  (kill switch, halt, feed caído, fallo de orden) del canal (log hoy; webhook
  Telegram/Discord mañana) sin tocar el orquestador.
- **Supervisión de tareas (auto-restart + backoff)**: cada tarea async (stream de
  velas, poller de sentimiento, watchdog) se vigila; si cae por un error no
  esperado, se reinicia tras una espera. La caída de una pieza no tumba el bot.
- **Watchdog del feed**: tarea periódica que comprueba si alguna vela lleva sin
  llegar más de `stale_feed_seconds`; si es así, activa el circuit breaker (a) y
  detiene nuevas entradas (operar sobre un precio viejo es operar a ciegas).
- **Warmup (calentamiento)**: las velas mínimas que el buffer debe acumular antes
  de operar, para que EMA/RSI/ATR tengan suficiente historia y no emitan señales
  sobre datos insuficientes.

## Sprint 7.2 (hardening de concurrencia y ciclo de vida)

- **Serialización con `asyncio.Lock`**: un candado que garantiza que la sección
  crítica (reconciliar→decidir→actuar→actualizar estado) se ejecute de a una por
  vez. Sin él, la vela de un símbolo podría leer la cuenta mientras otro símbolo
  está a mitad de una apertura → estado intermedio observado → decisiones falsas.
- **Orden en vuelo (in-flight)**: pierna que YA abrimos (fill confirmado por el
  REST) pero que el endpoint de cuenta del exchange aún no reporta, por latencia
  de propagación. El registro `_in_flight` las marca para que la reconciliación
  las ignore: ni las trata como "desconocidas" (evita HALT falso) ni como
  "cerradas por SL/TP" (evita RESYNC falso y una doble apertura). Se *promueven*
  a confirmadas cuando el exchange por fin las reporta; si nunca aparecen tras la
  gracia, se declaran no confirmadas y se descartan.
- **Ventana de gracia (reconciliación)**: una pierna "desconocida" (presente en
  el exchange, ausente de nuestro modelo) no dispara el HALT a la primera vista,
  sino tras N observaciones consecutivas. Absorbe blips transitorios de latencia;
  solo una divergencia *sostenida* es un peligro real (circuit breaker c).
- **Eventual consistencia (exchange)**: el motor de matching y el endpoint de
  cuenta de Binance no se actualizan en el mismo instante. Una orden puede estar
  FILLED en el ACK del REST y el `availableBalance`/posición tardar decenas de ms
  en reflejarlo. Todo el blindaje de in-flight/gracia existe por esto.
- **FLIP desacoplado**: ante una señal opuesta, el bot solo CIERRA la pierna
  actual en esta vela; la apertura inversa ocurre en la vela siguiente, con un
  snapshot fresco. Evita la carrera de margen (abrir antes de que el exchange
  libere el margen del cierre → INSUFFICIENT_BALANCE). Encaja con el horizonte
  swing: una vela de latencia es irrelevante.
- **Backfill REST**: al (re)arrancar, rellenar el buffer de velas pidiendo las
  últimas N *cerradas* por REST (contiguas y autoritativas del exchange) en lugar
  de esperar horas a reconstruirlo del stream en vivo. La detección de huecos
  fuerza un re-backfill: nunca se calculan indicadores sobre una serie con saltos.
- **Adopción de posiciones**: tras un reinicio en caliente, el bot lee las
  piernas que el exchange ya tiene y las incorpora a su modelo (`expected`) en
  vez de verlas como "desconocidas" y detenerse. Verifica que cada una conserve
  su STOP protector.
- **Posición desnuda (naked)**: una posición abierta SIN stop-loss en el exchange
  (p. ej. si un reinicio dejó la entrada pero perdió sus protectoras). Es el
  riesgo #1; al detectarla en la adopción, el bot hace HALT y alerta para
  revisión manual en vez de operar a ciegas.
- **Persistencia de estado de sesión**: guardar en SQLite el pico de wallet, el
  wallet de inicio de día y el latch del kill switch, para que sobrevivan a un
  reinicio. Si no, tras una caída el drawdown se mediría desde cero y el kill
  switch podría no saltar ante una pérdida ocurrida a través del corte.

## Sprint C (fundación de datos de sentimiento histórico)

- **El muro de datos del sentimiento**: para backtestear si el sentimiento da
  edge hace falta una serie HISTÓRICA de sentimiento, pero el RSS solo entrega
  los últimos titulares y no persistíamos nada. Sin corpus histórico, la
  hipótesis no se puede medir — de ahí esta fundación de datos.
- **CryptoPanic**: agregador de noticias cripto con API paginada (free tier) que
  sí permite recuperar histórico, a diferencia del RSS. Se mapea al mismo
  `NewsItem` con el MISMO hash de URL, así una noticia que llega por RSS y por
  CryptoPanic se deduplica igual.
- **Corpus acumulativo**: el free tier limita la profundidad de histórico, así
  que la estrategia es ejecutar la ingesta periódicamente y ACUMULAR en SQLite
  (idempotente por hash de URL), construyendo el dataset con el tiempo.
- **Alineación por `published_at` (anti look-ahead)**: cada SentimentScore se
  indexa por el instante en que la noticia se PUBLICÓ (cuándo la información
  estuvo disponible), no por cuándo la analizamos. Usar `analyzed_at` metería en
  el backtest información del futuro — el error #1 del backtesting.
- **Confianza local = |score local|**: a un ítem que no merece el análisis de
  Claude se le asigna una confianza igual a la magnitud de su score local (baja
  por construcción, < umbral de escalación). El Risk Manager le da así menos peso
  al sizing — coherente con que es una señal barata y menos fiable.
- **Paginación por cursor `next`**: CryptoPanic devuelve en cada página una URL
  `next` a la siguiente; se sigue ese cursor hasta agotarlo o el tope de páginas.
  Ante un 429 (rate limit del free tier) se reintenta con backoff exponencial.
  *(Nota: CryptoPanic eliminó su free tier el 1-abr-2026; el cliente queda para
  uso de pago. La fuente $0 elegida pasó a ser el índice Fear & Greed — abajo.)*
- **Índice Fear & Greed (alternative.me)**: gauge diario [0,100] de sentimiento
  de mercado cripto, calculado de volatilidad, momentum, dominancia, tendencias
  y **redes sociales (Twitter/X ~15%)**. Gratis, sin API key, con histórico
  completo desde 2018 — tras investigar todo el mercado, resultó ser la ÚNICA
  fuente de sentimiento histórico a $0 (las noticias cripto históricas gratis
  desaparecieron: todos los proveedores pusieron el archivo tras un muro de pago
  en 2025-26). Se mapea a nuestro score con `(valor−50)/50`.
- **Interpretación momentum vs. contraria (Fear & Greed)**: mapear codicia→alcista
  asume *seguimiento de tendencia* (la confluencia trata el sentimiento como
  confirmación). La lectura *contraria* clásica (codicia extrema = techo → bajista)
  es la hipótesis opuesta; se prueba negando el score. Cuál funciona es una
  pregunta empírica que el backtest debe responder, no un supuesto a hardcodear.

## Sprint C.2 (backtest de confluencia + walk-forward)

- **A/B honesto (sentimiento ON/OFF)**: el backtester corre la MISMA
  `confluence.decide` que el bot en vivo; la única diferencia entre los dos
  brazos es si se le pasa la serie de sentimiento o no. Así la diferencia de
  métricas aísla la contribución del sentimiento, sin cambiar la ruta de decisión.
- **Decider inyectable**: el motor de backtest acepta una función que decide la
  acción de cada vela. Por defecto es la estrategia por umbrales del Sprint 3
  (comportamiento intacto); la ruta de confluencia inyecta su propio decider con
  el sentimiento. Un único núcleo de ejecución (sizing, stops, GAP, costos) para
  ambos → backtest y vivo no divergen.
- **Alineación anti-look-ahead**: cada vela t solo ve el sentimiento con
  `published_at <= t` y dentro de la ventana de antigüedad. Es un merge de dos
  series ordenadas, O(n+m). Usar el score de una noticia antes de su publicación
  sería ver el futuro — el sesgo de anticipación clásico.
- **Walk-forward (robustez)**: dividir el histórico en tramos contiguos y
  backtestear cada uno por separado. Sin optimización de parámetros (aún no hay
  perillas que ajustar) no es el walk-forward "de optimización", sino un test de
  consistencia: ¿el resultado se repite entre periodos o fue un tramo afortunado?
  Si la estrategia pierde en los 4 tramos, no hay edge — y eso es un hallazgo,
  no un fracaso.
