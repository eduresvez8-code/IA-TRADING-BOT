# Resultados: ¿vale la pena pagar por Value/Quality? — proxy ETF (2026-07-25)

Ejecuta `docs/research/2026-07-25_factor_etf_proxy_protocolo.md`. Runner:
`backtest/run_factor_etf_proxy.py`.

## Resultado

| ETF | Categoría | Sh test | vs B&H SPY (+0.90) | Concentración | Gate |
|---|---|---|---|---|---|
| VTV | Value (Vanguard, 2004) | +0.79 | no supera | 79% (>60%) | NO pasa |
| QUAL | Quality (iShares, 2013) | +0.87 | no supera | 77% (>60%) | NO pasa |

Ningún ETF profesional de Value ni de Quality superó a comprar-y-mantener
SPY en el periodo de test (2015-2026), pese a estar gestionados
profesionalmente durante años. Ambos fallan además el criterio de
concentración: más del 75% de su ganancia salió del 10% de meses más
fuertes — probablemente la recuperación post-2020 y la rotación a valor de
2022, no una ventaja sostenida mes a mes.

## Interpretación

Esta prueba no requirió ninguna selección por train (cero grados de
libertad: cada ETF es un producto ya gestionado, elegido por ser el mayor/
más antiguo de su categoría antes de mirar resultados). El resultado
responde directamente la pregunta de gasto: **si ni un fondo profesional
con equipo de gestión, escala institucional y décadas de experiencia logra
superar al índice con este factor en este periodo, construir nuestra propia
versión con datos de pago difícilmente lo haría mejor.**

## Veredicto

**No se recomienda pagar por un proveedor de fundamentales point-in-time
(Sharadar/FMP de pago) para Familia 8 (Value/Quality) en este momento.** La
verificación de point-in-time de FMP (`finding-fmp-point-in-time-verified`)
había despejado la duda de VIABILIDAD TÉCNICA del dato — este resultado
despeja la duda de si vale la pena: no, no en 2015-2026. El dinero y las
semanas de ingeniería que costaría construir el pipeline propio no están
justificados por la evidencia disponible.

Esto refuerza, no cambia, el veredicto general del proyecto: indexación
pasiva. La única puerta activa que sigue abierta es forward/paper trading
real de RSI-2.
