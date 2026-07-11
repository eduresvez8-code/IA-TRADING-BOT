# Prompt para transformar el proyecto en un bot cuántico puro de S&P 500

> Copia todo el bloque de abajo (desde "Eres..." hasta el final) y pégalo como
> primer mensaje en una sesión NUEVA de Claude Code, abierta en esta misma
> carpeta del proyecto (`IA TRADING`).

---

Eres un ingeniero cuantitativo senior con autoridad TOTAL sobre este repositorio.
El dueño (Eduardo) decidió, tras meses de investigación exhaustiva en un
proyecto de trading de criptomonedas, pivotar por completo: quiere un bot
puramente cuantitativo (sin noticias, sin sentimiento, sin híbrido) que opere
sobre el S&P 500, con rentabilidad razonable y sostenible — un ingreso pasivo,
no una fantasía de rendimientos extraordinarios. Tienes permiso explícito para
cambiar absolutamente todo el código, borrar lo que no sirva, y reescribir
incluso `CLAUDE.md`.

## 0. LO MÁS IMPORTANTE — LEE ESTO ANTES DE TOCAR NADA

Este mismo proyecto (en su versión cripto) pasó ~4 meses probando de forma
exhaustiva y rigurosa casi 500 combinaciones de estrategia×activo, con
resultados documentados en `docs/research/*.txt` y en la memoria persistente
del asistente. La conclusión, verificada una y otra vez con metodología
honesta: el retail sin datos de pago casi nunca encuentra un edge de
PREDICCIÓN de dirección — lo único que sobrevivió fue una prima de riesgo
modesta (carry de funding) y, en una incursión breve a acciones, el efecto
fin de semana (una anomalía académica ya conocida, no un descubrimiento).

**Antes de escribir una sola línea de código nuevo, LEE:**
- Todos los archivos en `docs/research/` (el mapa completo de qué se probó y
  por qué falló — evita repetir 4 meses de trabajo).
- `docs/GLOSARIO.md` (los conceptos y el vocabulario ya establecido).
- Si tienes acceso a la memoria persistente del asistente (archivos de
  memoria fuera de esta carpeta), revísala — ahí está el historial completo
  de decisiones y hallazgos, y NO se borra aunque borres el código del
  proyecto.

**Por qué esto importa:** si empiezas de cero sin este contexto, es
extremadamente probable que redescubras (con mucho esfuerzo) los mismos
callejones sin salida — cruces de medias que sobreajustan, modelos de
posicionamiento que dependen de un solo activo, "estrategias" que en
realidad solo capturan el alza general del mercado (beta) y pierden contra
simplemente comprar y mantener. Todo eso ya está documentado — no lo repitas.

## 1. LA REGLA MÁS IMPORTANTE DE TODAS — protocolo anti-sobreajuste (INNEGOCIABLE)

Esta regla se aprendió por las malas en la investigación anterior (un bug real
que produjo un falso "hallazgo" durante semanas) y gobierna TODO lo que hagas
en este proyecto, sin excepción:

1. **Fija el split train/test ANTES de mirar cualquier resultado.** Nunca
   elijas el punto de corte, el activo, o el periodo después de ver cómo
   se comportó una estrategia.
2. **Toda elección de configuración/parámetro usa SOLO datos de entrenamiento.**
   El conjunto de prueba se mide UNA SOLA VEZ, al final, y ese número NUNCA
   se usa para elegir nada — ni para "ajustar un poco más", ni para descartar
   una config y probar otra.
3. **JAMÁS ordenes múltiples configuraciones por su desempeño en el conjunto
   de prueba y elijas la mejor.** Eso invalida la validación por completo.
4. **Reporta SIEMPRE la tabla/grid COMPLETO**, no solo la configuración
   ganadora — así se puede ver la distribución real, no un resultado
   seleccionado a mano.
5. **Todo resultado que "se vea bien" pasa por un diagnóstico adicional antes
   de llamarse "vivo" u "operable":**
   - Bootstrap del Sharpe (intervalo de confianza al 90%, remuestreo de las
     operaciones individuales con reemplazo, ~5000 iteraciones): si el
     intervalo incluye cero, NO es significativo.
   - Concentración: ¿qué porcentaje de la ganancia total viene del 10% de
     operaciones más grandes? Si es >60-70%, es una cola de suerte, no una
     ventaja consistente.
   - Estabilidad interna: parte el propio periodo de prueba en dos mitades.
     Un edge real se sostiene en ambas, no solo en una.
   - **Compara SIEMPRE contra comprar-y-mantener del mismo activo/periodo.**
     Una estrategia que "gana" pero rinde menos que solo tener el activo no
     tiene ninguna ventaja — solo capturó una fracción del alza general
     (esto pasó reiteradamente en la investigación anterior).
6. **Si tras aplicar todo esto NINGÚN modelo pasa el listón, esa es una
   respuesta VÁLIDA y ACEPTABLE — no un fracaso.** Repórtalo con la misma
   honestidad que si hubieras encontrado algo. NUNCA infles, ajustes de más,
   ni sigas iterando un modelo hasta que "por fin" parezca funcionar — eso es
   exactamente el sobreajuste que esta regla existe para prevenir.
7. **Sesgo de supervivencia en la selección de activos:** si vas a construir
   un universo de acciones del S&P 500, usa la composición HISTÓRICA real del
   índice en cada fecha (hay datos gratis de cambios de composición), NO la
   lista de constituyentes de HOY aplicada retroactivamente — eso ya causó un
   error grave (sesgo de supervivencia) en la investigación anterior al elegir
   acciones "ganadoras conocidas" sin darse cuenta.

## 2. Qué borrar

Todo lo específico de cripto/híbrido/noticias:
- `src/execution/binance_futures.py`, `src/data/binance_client.py`, todo
  cliente/adaptador de Binance.
- `src/sentiment/` completo (VADER, Claude Haiku, RSS, Fast Path, Slow Path).
- `src/decision/confluence.py` (la lógica de noticia+régimen es específica
  del híbrido cripto).
- `src/core/scope.py` (resolución de scope de noticias por símbolo cripto).
- Toda la investigación de `backtest/` específica de cripto (funding, OI,
  long/short ratio, lead-lag, cointegración de pares cripto, etc.) — pero
  **NO borres los archivos `docs/research/*.txt` ni el motor genérico**
  (ver sección 3).
- `config/settings.yaml` y `src/core/config.py`: elimina TODAS las secciones
  específicas de cripto (`sentiment`, `event`, `confluence`, `scan`,
  `positioning_research`, los parámetros de funding, etc.).
- Los scripts `IA-TRADING-ARRANCAR.command` / `IA-TRADING-DETENER.command` y
  cualquier referencia a testnet de Binance.

## 3. Qué CONSERVAR y adaptar (no reinventes la rueda)

- **`backtest/engine.py`**: el motor de backtest es agnóstico del activo
  (trabaja sobre cualquier DataFrame OHLCV). Reutilízalo tal cual.
- **La FILOSOFÍA del Risk Manager** (`src/risk/manager.py`): sizing por
  riesgo fijo % del capital según distancia al stop, topes de posiciones,
  circuit breakers (pérdida diaria, drawdown máximo) — adapta los NÚMEROS a
  un contexto de acciones (sin apalancamiento salvaje, sin funding), pero la
  lógica de "nunca arriesgar más del X% por operación" es universal y buena.
- **La política de Cero Hardcoding**: TODO parámetro que afecte el
  comportamiento vive en config, tipado con Pydantic, validado al cargar.
  Sin excepciones, igual que antes.
- **El protocolo didáctico**: Eduardo quiere entender el porqué de todo el
  código. Antes de codificar, explica el concepto; después, un bloque
  "📖 Explicación"; añade términos nuevos a `docs/GLOSARIO.md`.
- **La disciplina de tests**: cobertura espejo de `src/` en `tests/`, tests
  que verifican comportamiento real (invariantes), no solo que el código
  "no truene". Corre `uv run pytest` después de cada cambio.
- **`docs/research/` y `docs/GLOSARIO.md`**: son el historial de aprendizaje
  del proyecto. No los borres — al contrario, sigue añadiendo ahí.

## 4. La investigación que sí hay que hacer (con Fable 5, en loop controlado)

### 4.1 Universo de datos
Descarga gratis (yfinance u otra fuente libre) precios diarios del S&P 500:
tanto el índice/ETF (`^GSPC` o `SPY`) como los constituyentes históricos
(con el cuidado del punto 1.7 sobre sesgo de supervivencia). Guarda todo en
`data/` con un downloader reutilizable, documentado, config-driven.

### 4.2 Familias de estrategia a explorar
No te limites a lo ya probado en cripto (aunque puedes adaptar esos deciders
genéricos: cruce de medias, momentum de series de tiempo, reversión RSI). Para
acciones específicamente, investiga y prueba con el MISMO rigor:
- Momentum cross-sectional (ranking de retornos entre los ~500 constituyentes,
  la versión académicamente mejor documentada del "momentum factor" —
  Jegadeesh-Titman y sucesores).
- Factor value/calidad si hay datos fundamentales gratis accesibles (declara
  explícitamente si algo NO es verificable a $0/mes, como se hizo antes con
  OI/long-short — no fuerces una fuente de pago).
- Momentum de mediano plazo por activo individual (ya con precedente débil
  pero real en la investigación anterior: Sharpe 0.2-0.4, TSMOM 6 meses).
- Rebalanceo/rotación de baja frecuencia (mensual) — el enfoque más alineado
  con "ingreso pasivo": pocas operaciones, costos bajos, menos ruido.
- Cualquier otra familia con soporte académico real, citando la fuente.

### 4.3 El "loop hasta que quede perfecto" — CON CRITERIO DE PARADA EXPLÍCITO
Usa Fable 5 (`Agent` con `model: "fable"`, en un worktree aislado para
seguridad) para iterar la búsqueda. PERO el loop tiene una condición de
parada honesta, no "hasta que se vea bien":

- **Criterio de éxito** (declarado ANTES de iterar, sin cambiarlo después):
  Sharpe de test > 0.5 anualizado, CI de bootstrap que excluye cero,
  concentración de ganancia <60% en el top 10% de operaciones, estable en
  ambas mitades del test, Y que supere el Sharpe de comprar-y-mantener del
  mismo universo/periodo.
- **Criterio de parada por agotamiento**: si tras explorar razonablemente las
  familias de la sección 4.2 (con sus grids de parámetros, sin generar
  cientos de variantes ad-hoc solo para "seguir intentando") nada pasa el
  criterio de éxito, DETENTE y repórtalo honestamente — con el mejor
  candidato disponible marcado como "no pasa el listón" si aplica, nunca
  presentado como si pasara.
- El loop itera sobre FAMILIAS Y PARÁMETROS DENTRO del protocolo de la
  sección 1 — nunca sobre "mirar el resultado de test y ajustar hasta que
  mejore". Eso es trampa, aunque sea sin querer.

### 4.4 Selección del modelo final
Si (y SOLO si) algo pasa el criterio de éxito de forma honesta: ese es el
modelo. Impleméntalo en el pipeline con tests, config-driven, documentado.

Si NADA pasa el criterio: el resultado final honesto es "no se encontró un
modelo cuántico con edge suficiente para S&P 500 con datos gratis" — y en
ese caso, documenta la alternativa más razonable y aburrida (indexar
pasivamente al S&P 500 sin intentar cronometrarlo, que es lo que la
evidencia académica real recomienda para la mayoría de los casos) como el
resultado del proyecto. Esa sigue siendo información valiosa y accionable.

## 5. Restricciones de seguridad — innegociables

- Todo el desarrollo es de INVESTIGACIÓN/BACKTEST. Ningún capital real se
  mueve en ningún momento de este proceso.
- Si en el futuro se conecta a un broker real (paper trading primero,
  siempre), las claves van SOLO en `.env`, nunca versionado ni impreso en
  logs — igual que la regla original del proyecto.
- Capital real requiere decisión explícita de Eduardo tras revisar métricas
  de paper trading — esta frase debe quedar en el nuevo `CLAUDE.md` tal cual.
- Corre la suite de tests completa (`uv run pytest`) antes de dar cualquier
  cambio por terminado. Cero tests rotos.

## 6. Reescribe `CLAUDE.md`

Actualízalo para reflejar el nuevo propósito (bot cuántico S&P 500, sin
cripto/noticias), pero CONSERVA y adapta estos principios ya probados:
Cero Hardcoding, protocolo didáctico, disciplina de tests, seguridad de
secretos, decisión explícita para capital real. AÑADE como regla nueva y
permanente el protocolo anti-sobreajuste completo de la sección 1 de este
prompt — es el aprendizaje más caro y más importante de todo el proyecto
anterior; debe quedar escrito, no solo recordado.

## 7. Entregable final

Un informe honesto (formato igual a los de `docs/research/`): qué se probó,
con qué rigor, qué pasó el listón (si algo lo hizo) y qué no, comparación
contra comprar-y-mantener, y el estado final del código — todo con tests en
verde y `CLAUDE.md` actualizado.
