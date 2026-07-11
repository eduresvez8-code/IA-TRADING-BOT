# Pre-registro: ¿vale la pena pagar por datos Value/Quality? — proxy con ETFs (2026-07-25)

Fijado ANTES de descargar un solo dato. Contexto: Eduardo preguntó si vale
la pena pagar por un proveedor de fundamentales point-in-time (ver
`finding-fmp-point-in-time-verified`) para construir factores Value/Quality
propios. Antes de gastar dinero y semanas de ingeniería, se prueba la
hipótesis central GRATIS: si el factor Value/Quality tuviera una ventaja
real y accesible, un ETF profesional que YA lo implementa (con equipos de
gestión, décadas de experiencia, y sin nuestros problemas de datos) debería
reflejarla. Si ni el ETF profesional le gana al índice, construir nuestra
propia versión desde cero difícilmente lo haría mejor.

## Esto NO es una reapertura de la búsqueda de estrategias

Es una verificación de una PREMISA de gasto (¿pagar datos?), no una nueva
variante de timing sobre SPY. No hay grid, no hay parámetro que elegir por
train: cada candidato es un ETF real, ya gestionado, comprado una vez y
sostenido — cero grados de libertad de nuestra parte. Por eso no cuenta
como una tercera extensión de búsqueda de estrategias en el sentido del
protocolo original.

## Candidatos (elegidos ANTES de mirar cualquier resultado)

Regla de selección: el ETF más grande/antiguo de su categoría en EE.UU.
(evita elegir a mano el que "se ve mejor" después de ver resultados):
- **Value**: VTV (Vanguard Value ETF, inception 2004-01-30) — el mayor ETF
  de valor en AUM, benchmark CRSP US Large Cap Value.
- **Quality**: QUAL (iShares MSCI USA Quality Factor ETF, inception
  2013-07-18) — el ETF de calidad más establecido, benchmark MSCI USA
  Sector Neutral Quality.

Ambos con historia completa vía yfinance cubriendo TODO el periodo de test
(2015-01-01 en adelante), verificado antes de escribir este documento.

## Mecánica

- Comprar-y-mantener puro (retorno total, `auto_adjust=True`) — SIN costos
  aplicados, igual que el benchmark B&H SPY oficial (una compra amortizada
  en años es costo despreciable, mismo tratamiento que el proyecto ya usa
  para SPY).
- Split, ventana y comparador: idénticos al protocolo madre — TEST desde
  2015-01-01, comparado contra el MISMO B&H SPY test ya publicado
  (Sharpe diario +0.85, mensual +0.90).
- Los 5 criterios de siempre (Sharpe test > 0.5, bootstrap CI 90% excluye 0
  sobre retornos mensuales, concentración del top 10% de meses < 60%, ambas
  mitades del test > 0, supera el Sharpe de B&H SPY).
- Sin entrenamiento: no hay nada que seleccionar por train (cero grados de
  libertad). Se reporta igual el desempeño en el tramo pre-2015 por
  transparencia, pero NO participa en ninguna decisión.

## Criterio de interpretación

- Si NINGUNO de los 2 ETFs pasa el gate → evidencia fuerte de que pagar por
  datos Value/Quality propios no se justifica en este periodo: ETFs
  profesionales con toda su ventaja de escala/gestión tampoco lo logran.
- Si alguno pasa con margen SÓLIDO (no al filo, como el falso positivo de
  hoy con RSI-2+amplitud) → justifica evaluar el gasto en datos propios,
  sabiendo que el factor SÍ tuvo tracción real en este periodo específico.
