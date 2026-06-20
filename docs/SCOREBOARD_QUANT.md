# Scoreboard de la matriz cuantitativa — Slow Path (cierre de 5 familias)

**Estado: MANDATO COMPLETO.** Las 5 familias de la matriz quant del Slow Path están
implementadas y evaluadas bajo el mismo embudo anti-overfit de 2 etapas. Ninguna
produce un edge direccional operable con ejecución taker. El Slow Path NO aporta alpha
direccional propio; su rol queda confinado a (a) carry estructural delta-neutral
(Familia E, piso modesto) y (b) régimen/sizing como overlay del Core Event-Driven
(ver `plan-v2-dual-core`). Fecha de cierre: 2026-06-20.

## El embudo (idéntico para las 5 familias)

- **ETAPA 1 — gate de significancia (barato).** Señales direccionales (C/D): IC de
  Spearman + t-stat con corrección `n_eff = n·(1−ρ₁)/(1+ρ₁)`. Yield/spread (B/E): t-stat
  de la media del retorno neto. Descarta si no supera la Regla de Oro (|t|≥2).
- **ETAPA 2 — P&L (caro).** Máquina de estados con costos REALES (taker 0.05% + slippage
  %+k·ATR), equity curve honesta (costos dentro), walk-forward 4 folds del mismo signo,
  PF>1.15. Descomposición GROSS vs NETO para separar "no hay señal" de "señal cara".

## Tabla resumen

| Familia | Hipótesis | Timeframe | Veredicto | Razón del null |
|---|---|---|---|---|
| **B** · Cointegración de pares | spread revierte (IC<0) | 1h | ❌ MUERTA | IC>0 (momentum); spreads cripto son I(1) (random walk), no cointegrados a 1h |
| **C** · Reversión a VWAP intradía | desviación al VWAP revierte (IC<0) | 5m | ⚠️ REAL pero ANTIECONÓMICA | gross +34%/año real y consistente (5/5, XRP t=−2.68, bounce-robust), pero costo taker ~149%/año → neto −115%. Solo viable maker (no validable sin L2) |
| **D** · Squeeze de volatilidad → ruptura | la ruptura continúa (IC>0) | 1h | ❌ MUERTA | IC≈0/leve neg (sin momentum); premisa de expansión INVERTIDA (vol clustering/GARCH); gross de signo mixto → null "no hay señal" |
| **E** · Cash-and-carry de funding | yield estructural delta-neutral | 8h | ✅ PISO ESTABLE (modesto) | 3/5 pasan; yield neto ~3–4%/año, Sharpe alto pero engañoso (MaxDD bajo por construcción delta-neutral, no por edge direccional) |
| (cross-sec) · Reversión cross-sectional | reversión entre perps (IC<0) | — | ⚠️ LEAD débil | IC negativo significativo en 518 perps, pero bloqueado por cola/skew + survivorship; falta backtest robusto a la cola |

(F&G overlay quedó muerta antes de la matriz: anti-predice, IC<0 sobre la señal quant.)

## Lecturas transversales (lo que aprendimos, más allá de cada familia)

1. **Corrección `n_eff` es decisiva.** El z-score de un spread/desviación cripto tiene
   ρ₁≈0.99 → n_eff de decenas aunque haya 25k barras. El t-stat naive infla ~20×. Sin
   esta corrección, B/C/D habrían "coronado" señales fantasma. Es el filtro que más
   candidatas mató honestamente.
2. **GROSS vs NETO diagnostica el TIPO de null.** Gross fuerte+consistente con neto<0
   (C) = señal real enterrada por fricción, rescatable con maker → es un *lead*. Gross
   de signo mixto (D) = no hay señal, irrescatable. La descomposición no mide solo
   "cuánto cuesta": dice si siquiera existe edge bruto.
3. **El bid-ask bounce simula reversión.** El skip-1 (`shift(1)` en la señal: observar
   en t, actuar en t+1) es lookahead-free Y bounce-robust a la vez. Sin él, toda señal
   de reversión a 5m parece real por puro rebote mecánico no capturable.
4. **El folclore técnico no sobrevive a majors líquidos.** Pares (B), breakout de
   squeeze (D) y reversión barata (C-neto) fallan donde el libro está arbitrado. La
   premisa del squeeze ("baja vol precede expansión") está empíricamente invertida: la
   volatilidad PERSISTE (clustering/GARCH), no rebota en su nivel.
5. **Auditoría-primero paga.** Cada familia empezó con un probe de IC ANTES del módulo.
   En B, D y F&G el probe ya mostró el null → no se construyó sobre una hipótesis muerta.

## Consecuencia para el diseño (decisión vigente)

El pivote a **V2 Dual-Core (Event-Driven)** queda RATIFICADO por evidencia: tras barrer
con rigor 5 familias quant + cross-sectional + F&G, no aparece alpha direccional barato
en el Slow Path. Las noticias ORIGINAN las decisiones; el quant se demota a régimen y
sizing. La Familia E (carry) puede correr como piso delta-neutral independiente. C queda
archivada como lead para un eventual forward-test con ejecución maker (requiere datos L2,
fuera del presupuesto $0/mes actual).

Detalle por familia en las memorias `finding-*` y en `docs/GLOSARIO.md`. Runner:
`uv run python -m backtest.run_quant_matrix`.
