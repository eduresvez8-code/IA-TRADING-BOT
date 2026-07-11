# Resultados: RSI-2 + filtro de régimen TSMOM (2026-07-25)

Ejecuta `docs/research/2026-07-25_rsi2_tsmom_combo_protocolo.md` (leer antes
que esto — fija el grid, la regla de selección y los 5 criterios ANTES de
correr nada). Runner: `backtest/run_sp500_rsi_tsmom_combo.py`.

## Verificación de reproducibilidad

Antes de combinar, el runner recalcula la selección de RSI-2 ya publicada
el 2026-07-11 (mismo grid `entry_grid × exit_grid`, misma regla de máximo
Sharpe de train) — NO se hardcodea el resultado como literal nuevo. Salida:

```
entry<10.0, exit>70.0
```

Coincide con lo publicado. Confirma que el punto de partida es exactamente
la misma config congelada, no una nueva selección disfrazada.

## Grid del filtro de régimen (L, sobre el grid ya existente {3,6,12})

| L (régimen TSMOM) | Sh train | Sh test | n_tr | n_te |
|---|---|---|---|---|
| 3  | +1.17 | +0.79 | 5522 | 2895 |
| 6  | +1.25 | +0.79 | 5522 | 2895 |
| 12 | +1.18 | +0.75 | 5522 | 2895 |

Elegido por TRAIN (máximo Sharpe train, nunca por test): **L=6** (Sh train +1.25).

## Gate de 5 criterios sobre el test (L=6, medido una sola vez)

| Criterio | Valor | Umbral | Resultado |
|---|---|---|---|
| Sharpe test | +0.79 | > 0.5 | SÍ |
| Bootstrap CI 90% | [+0.20, +1.41] | excluye 0 | SÍ |
| Concentración top10% | 59% | < 60% | SÍ |
| Mitades del test | +0.38 / +1.24 | ambas > 0 | SÍ |
| vs B&H SPY | +0.79 vs +0.85 | superarlo | **no** |

**PASA 4 de 5 — no pasa el listón** (falla el único criterio comparativo).

Trades en test: 85. MaxDD test: 12.0%. Calmar test: +0.43 (referencia:
RSI-2 solo 10.9% DD / +0.52 Calmar; B&H 23.3% DD / +0.59 Calmar).

## Interpretación honesta

El filtro de régimen TSMOM **no añadió edge real**: el Sharpe de test
(+0.79) es esencialmente el mismo que el de RSI-2 solo (+0.81, publicado el
2026-07-11) — dentro del ruido, no una mejora. Tampoco mejoró el Calmar
(+0.43 vs +0.52 de RSI-2 solo): el filtro recortó algo de drawdown pero
también recortó retorno en proporción similar, sin ganancia neta de
"suavidad de viaje". El resultado confirma, no abre, el veredicto: RSI-2 ya
capturaba casi toda la señal disponible en estos datos, y una tercera capa
de régimen sobre el mismo periodo 2015-2026 no aporta información nueva —
es la misma señal vista con más pasos.

## Nota de multiplicidad (cumpliendo lo pre-registrado)

Esta fue la SEGUNDA extensión sobre la ventana 2015-2026 desde el veredicto
original del 2026-07-11 (la primera fue el reencuadre Calmar). Ambas
extensiones fueron disciplinadas (config/grid ya congelados, selección solo
por train, test medido una vez) pero **ninguna cambió el veredicto**. Por
la propia nota de honestidad del pre-registro: no se recomienda generar una
tercera variante sobre esta misma ventana. Seguir intentando combinaciones
sobre el mismo tramo histórico, aunque cada intento sea individualmente
riguroso, empieza a acumular el mismo riesgo de sobreajuste por búsqueda
repetida que todo el protocolo existe para evitar.

## Veredicto del proyecto (sin cambios)

Indexación pasiva. La única puerta honesta que queda es forward/paper
trading real de RSI-2 — sin sesgo posible porque nadie ve el futuro de
antemano, pero exige meses/años de muestra real; no hay atajo.
