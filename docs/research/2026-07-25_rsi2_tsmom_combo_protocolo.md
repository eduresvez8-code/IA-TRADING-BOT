# Pre-registro: RSI-2 + filtro de régimen TSMOM (2026-07-25)

Fijado ANTES de correr un solo backtest de esta combinación.

**Idea:** RSI-2 ya combina dos plazos (rebote de 2 días + filtro de tendencia
SMA200). Se añade un TERCER plazo, intermedio: solo tomar la señal de RSI-2
si el momentum de 12 meses del índice (familia TSMOM, ya implementada) es
también favorable. Es una técnica real (filtro de régimen multi-plazo), no
una variante ad-hoc.

**Config (NO se re-tunea RSI-2 propio — se usa el ya elegido y publicado):**
- RSI-2: entry<10, exit>70 (config ya seleccionada por train el 2026-07-11;
  NO se vuelve a barrer, evita ensanchar el grid de una sola vez).
- Filtro de régimen: TSMOM(L) > 0, L barrido sobre el MISMO grid ya declarado
  en `research.tsmom_index.lookback_months_grid` = {3, 6, 12} — no es un
  grid nuevo, es el que ya existía para la familia TSMOM.
- Selección: SOLO por Sharpe de TRAIN entre los 3 valores de L. Test medido
  una vez con el L ganador.

**Split y costos:** idénticos al protocolo madre (TRAIN < 2015-01-01 ≤ TEST,
2 pb/lado, estrés 5 pb, cash devenga T-bill).

**Criterios de éxito:** los mismos 5 de siempre (Sharpe test > 0.5, bootstrap
CI excluye 0, concentración < 60%, ambas mitades > 0, supera Sharpe B&H SPY).

**Nota de honestidad sobre multiplicidad:** esta es la SEGUNDA extensión
sobre el mismo periodo 2015-2026 desde el veredicto original (la primera fue
el reencuadre Calmar, que no contaba como experimento nuevo por reusar
configs ya fijas). Esta SÍ es una estrategia nueva. Si tampoco pasa, el
protocolo recomienda no generar una tercera variante sobre esta misma
ventana — seguir así, aunque cada intento individual sea disciplinado,
empieza a acumular el mismo riesgo de sobreajuste por búsqueda repetida que
el protocolo entero existe para evitar.
