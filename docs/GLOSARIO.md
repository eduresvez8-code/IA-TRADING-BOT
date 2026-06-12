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
