# IA TRADING — Bot cuantitativo S&P 500

Bot puramente cuantitativo (sin noticias, sin sentimiento, sin híbrido) sobre
el S&P 500, con velas DIARIAS. Objetivo: rentabilidad razonable y sostenible —
un ingreso pasivo, no una fantasía de rendimientos extraordinarios. Presupuesto
de datos: $0/mes (yfinance + fuentes públicas).

**Historia**: este proyecto fue un bot híbrido cripto (técnico + noticias).
Tras ~4 meses y ~500 combinaciones estrategia×activo probadas con rigor, la
conclusión verificada fue: el retail sin datos de pago casi nunca encuentra un
edge de PREDICCIÓN de dirección. Todo está documentado en `docs/research/` y
`docs/archive/` — **leer antes de proponer una estrategia "nueva"**: lo más
probable es que ya se haya probado y muerto.

## Estado del proyecto (2026-07-11)

La ronda pre-registrada de 5 familias (momentum cross-sectional, TSMOM,
MA-timing, RSI-2, dual momentum) se corrió y **ninguna pasó el listón** —
todas quedaron por debajo del buy-and-hold de SPY en el test (ver
`docs/research/2026-07-11_sp500_resultados.txt`). El veredicto vigente del
proyecto es **indexación pasiva** (aportes periódicos a un ETF del S&P 500,
sin timing). NO hay estrategia activa "viva" ni bot en producción. Cualquier
experimento nuevo exige un pre-registro NUEVO fechado en `docs/research/`
ANTES de correrlo.

## Comandos

```bash
uv sync                            # instalar/actualizar dependencias
uv run pytest                      # correr tests (siempre antes de dar algo por terminado)
uv run python -m src.main --check  # smoke test: config + imports
```

## Protocolo anti-sobreajuste (LA REGLA MÁS IMPORTANTE — innegociable)

Aprendida por las malas: un bug de selección sobre el test set produjo un falso
"hallazgo" durante semanas (ver `finding-architecture-audit`, 2026-07-06).
Gobierna TODO en este repo, sin excepción:

1. **El split train/test se fija ANTES de mirar cualquier resultado.** Nunca se
   elige el punto de corte, el activo o el periodo después de ver cómo se
   comportó una estrategia. El split vigente está pre-registrado en
   `config/settings.yaml` (`research.test_start_date`) y congelado por test
   (`tests/test_config.py`).
2. **Toda elección de configuración usa SOLO datos de entrenamiento.** El test
   se mide UNA SOLA VEZ, al final, y ese número NUNCA se usa para elegir nada —
   ni para "ajustar un poco más", ni para descartar una config y probar otra.
3. **JAMÁS ordenar configuraciones por su desempeño en test y elegir la mejor.**
   Eso invalida la validación por completo.
4. **Reportar SIEMPRE el grid COMPLETO**, no solo la configuración ganadora —
   así se ve la distribución real, no un resultado seleccionado a mano.
5. **Todo resultado que "se vea bien" pasa diagnóstico antes de llamarse vivo:**
   - *Bootstrap del Sharpe*: CI al 90% (remuestreo de operaciones con
     reemplazo, ~5000 iteraciones). Si el intervalo incluye cero → NO significativo.
   - *Concentración*: si >60% de la ganancia viene del top 10% de operaciones,
     es una cola de suerte, no una ventaja.
   - *Estabilidad interna*: el periodo de test partido en dos mitades; un edge
     real se sostiene en ambas.
6. **Comparar SIEMPRE contra comprar-y-mantener** del mismo activo/periodo. Una
   estrategia que "gana" pero rinde menos que tener el activo no tiene ventaja:
   solo capturó una fracción de beta (pasó reiteradamente en cripto Y acciones).
7. **"Nada pasa el listón" es una respuesta VÁLIDA.** Se reporta con la misma
   honestidad que un hallazgo. Nunca inflar, ni seguir iterando hasta que "por
   fin" parezca funcionar — eso ES el sobreajuste.
8. **Sesgo de supervivencia**: cualquier universo de acciones usa la composición
   HISTÓRICA real del índice en cada fecha (punto-en-el-tiempo), nunca la lista
   de hoy aplicada retroactivamente. La cobertura de precios de empresas
   deslistadas se MIDE y se reporta (yfinance no tiene todas: la dirección del
   sesgo residual se declara).

Ver el protocolo completo pre-registrado en
`docs/research/2026-07-11_protocolo_sp500.md`.

## Convenciones

- Python 3.13. Proyecto de investigación/backtest: sin I/O en vivo hoy.
- Los contratos de datos viven en `src/core/models.py` (Pydantic). No se
  modifican sin actualizar `tests/test_models.py` en el mismo cambio.
- Parámetros en `config/settings.yaml`, tipados en `src/core/config.py`;
  secretos solo en `.env`. Nunca hardcodear símbolos, umbrales ni claves.
- Toda orden (cuando haya ejecución) pasa por `risk/manager.py`. Ningún módulo
  llamará a un broker directamente.
- Tests en `tests/`, espejo de `src/` y `backtest/`. Tests de COMPORTAMIENTO
  (invariantes), no de "no truena".
- Datos descargados en `data/` (parquet, regenerables, fuera de git).
- Informes de investigación en `docs/research/` con fecha en el nombre —
  NUNCA se borran: son la memoria del proyecto.

## Política de Cero Hardcoding (Cero Umbrales Ocultos)

**Regla inquebrantable, sin excepciones.** Ningún parámetro, multiplicador,
umbral o "número mágico" en la lógica de negocio (`src/` o `backtest/`). Todo
valor que modifique el comportamiento vive en `config/settings.yaml` y está
tipado y acotado en `src/core/config.py`. Si una feature nueva necesita un
parámetro, la única vía legal es: `settings.yaml` + `config.py` + su test en
`test_config.py`. Los grids de investigación también son config: un grid
hardcodeado en un runner es una violación.

## Protocolo didáctico (obligatorio)

Eduardo quiere entender el porqué de absolutamente todo el código. En cada
tarea de implementación:

1. **Antes de codificar**: explicar el concepto (problema que resuelve y
   matemática si aplica, con fórmulas).
2. **Después de codificar**: bloque "📖 Explicación" recorriendo el código
   sección por sección — el *porqué* de cada decisión y qué pasaría con la
   alternativa.
3. **Glosario**: todo término técnico nuevo va a `docs/GLOSARIO.md`.
4. **Cierre de sesión**: resumen de conceptos cubiertos y preguntas abiertas.

## Seguridad (no negociable)

- `.env` jamás se versiona ni se imprime en logs.
- Todo el desarrollo actual es INVESTIGACIÓN/BACKTEST: ningún capital real se
  mueve en ningún momento de este proceso.
- Si algún día se conecta un broker: paper trading primero, siempre; claves
  solo en `.env`, con permisos mínimos.
- Capital real requiere decisión explícita de Eduardo tras revisar métricas de
  paper trading.
