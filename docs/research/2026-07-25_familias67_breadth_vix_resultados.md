# Resultados: Familias 6-7 (amplitud, VIX) + combos — ÚLTIMA ronda (2026-07-25)

Ejecuta `docs/research/2026-07-25_familias67_breadth_vix_protocolo.md` (leer
antes que esto). Runner: `backtest/run_sp500_breadth_vix.py`.

## RSI-2 reproducido

`entry<10.0, exit>70.0` — coincide con lo publicado el 2026-07-11. Punto de
partida verificado, no hardcodeado.

## Familia 6 — Amplitud de mercado (standalone)

| N (SMA) | umbral | Sh train | Sh test |
|---|---|---|---|
| 100 | 0.40 | +0.65 | +0.74 |
| 100 | 0.50 | +0.87 | +0.63 |
| 100 | 0.60 | +0.91 | +0.50 |
| 200 | 0.40 | +0.67 | +0.68 |
| 200 | 0.50 | +0.83 | +0.73 |
| 200 | 0.60 | +0.86 | +0.68 |

Elegida por TRAIN: N=100, umbral=0.60. Test: Sharpe +0.50 (roza el umbral
mínimo, no lo supera estrictamente), bootstrap CI cruza cero, concentración
199% (pérdida neta — la fórmula da esto por convención, ver `diagnostics.py`),
no supera B&H. **No pasa.**

## Familia 7 — Régimen de VIX (standalone)

| N (SMA) | dirección | Sh train | Sh test |
|---|---|---|---|
| 50 | below | +0.38 | +0.89 |
| 50 | above | +0.51 | +0.48 |
| 100 | below | +0.34 | +0.85 |
| 100 | above | +0.56 | +0.52 |
| 200 | below | +0.47 | +0.86 |
| 200 | above | +0.49 | +0.54 |

Elegida por TRAIN: N=100, dirección=above. Test: Sharpe +0.52 (pasa el
umbral y el bootstrap), pero concentración 74% (>60%) y no supera B&H.
**No pasa.** Dato curioso para el registro: la dirección "below" (calma)
hubiera dado Sharpe test +0.85-0.89 en las 3 ventanas — pero NINGUNA fue
elegida por train (su Sharpe de train era menor). Mirar esto en retrospectiva
sería exactamente el error de seleccionar por test que el protocolo prohíbe;
se deja anotado solo como nota de color, no como resultado válido.

## Combo RSI-2 + gate de amplitud

| N (SMA) | umbral | Sh train | Sh test |
|---|---|---|---|
| 100 | 0.40 | +1.13 | +0.78 |
| 100 | 0.50 | +1.39 | +0.64 |
| 100 | 0.60 | +1.84 | +0.86 |
| 200 | 0.40 | +1.16 | +0.84 |
| 200 | 0.50 | +1.10 | +0.81 |
| 200 | 0.60 | +1.40 | +0.68 |

Elegida por TRAIN: N=100, umbral=0.60 (Sh train +1.84 — el más alto del día).
Test: **Sharpe +0.86 vs B&H +0.85 — PASA los 5 criterios del gate oficial**
(bootstrap CI [+0.16,+1.44] excluye cero, concentración 49%<60%, ambas
mitades positivas, supera a B&H por 0.01).

### Diagnóstico extra obligatorio (protocolo punto 5)

Un Sharpe test de +0.86 contra un B&H de +0.85 pasa el gate por un margen de
apenas **0.01** — con solo 50 trades en el test. El gate de 5 criterios
compara dos números puntuales; no dice si esa ventaja de 0.01 es distinguible
del ruido. Se corrió un bootstrap PAREADO de la diferencia de Sharpe
(remuestreo conjunto de los mismos días para estrategia y B&H, 5000
iteraciones, mismo CI 90% — función nueva `paired_bootstrap_sharpe_diff_ci`
en `backtest/diagnostics.py`, testeada con valores a mano):

```
Bootstrap pareado de la ventaja vs B&H: [-0.59, +0.62]  (excluir 0) → NO
```

El intervalo cruza el cero de sobra. La "ventaja" de 0.01 Sharpe es
indistinguible del ruido de muestreo — **no sobrevive el diagnóstico extra**,
pese a que el gate de 5 criterios lo marcó como "pasa". Con 50 trades y una
ventaja de esa magnitud, esto es exactamente el perfil esperado de un falso
positivo por multiplicidad: esta es aproximadamente la 10ª configuración
evaluada contra la misma ventana de test 2015-2026 en todo el proyecto (5
familias originales + Calmar + TSMOM-combo + estas 4) — con suficientes
intentos, alguno roza el umbral por puro azar. **No se cuenta como hallazgo.**

## Combo RSI-2 + gate de régimen VIX

| N (SMA) | dirección | Sh train | Sh test |
|---|---|---|---|
| 50 | below | +1.25 | +1.46 |
| 50 | above | +1.22 | +0.69 |
| 100 | below | +1.21 | +1.23 |
| 100 | above | +1.20 | +0.70 |
| 200 | below | +1.32 | +0.84 |
| 200 | above | +1.13 | +0.79 |

Elegida por TRAIN: N=200, dirección=below (Sh train +1.32). Test: Sharpe
+0.84 (pasa umbral, bootstrap, concentración 60% exacto, ambas mitades) pero
**no supera a B&H** (+0.84 vs +0.85). **No pasa** — ni siquiera llega a
activar el diagnóstico extra (ya falla el gate de 5 por sí solo).

## Veredicto de la ronda

| Config | Sh test | vs B&H | Gate 5 | Diagnóstico extra |
|---|---|---|---|---|
| Amplitud (standalone) | +0.50 | +0.85 | no | — |
| Régimen VIX (standalone) | +0.52 | +0.85 | no | — |
| RSI-2 + gate amplitud | +0.86 | +0.85 | **SÍ** | **NO** (ruido) |
| RSI-2 + gate VIX | +0.84 | +0.85 | no | — |

**Ninguna de las 4 sobrevive el gate de 5 criterios MÁS el diagnóstico
extra de significancia.** El caso más cercano (RSI-2+amplitud) pasó el gate
formal pero se cayó exactamente en la prueba diseñada para esto: la
"ventaja" no distingue de la suerte.

## Cierre de la búsqueda (cumpliendo lo pre-registrado)

Esta era, por diseño, la ÚLTIMA ronda de búsqueda de estrategias sobre la
ventana 2015-2026. Con esta ronda se han evaluado en total ~10
configuraciones distintas contra el mismo periodo de test a lo largo del
proyecto (5 familias originales, el reencuadre Calmar, el combo RSI-2+TSMOM,
y estas 4). El proyecto **no genera más variantes sobre este periodo** —
seguir buscando ya no sería información nueva, sería solo aumentar la
probabilidad de un falso positivo por azar (como casi ocurrió aquí).

**Veredicto final del proyecto: indexación pasiva.** El único paso que queda,
si Eduardo quiere seguir explorando el enfoque activo, es forward/paper
trading real de RSI-2 (el candidato más cercano en todas las rondas) — sin
sesgo posible porque nadie ve el futuro de antemano, pero exige meses o años
de muestra real. No hay atajo de backtest que sustituya eso.
