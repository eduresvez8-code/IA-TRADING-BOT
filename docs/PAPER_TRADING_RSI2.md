# Paper trading RSI-2 — forward real (2026-07-25)

Único camino activo que queda tras el cierre de la búsqueda de estrategias
sobre 2015-2026 (ver `CLAUDE.md`). Config **ya seleccionada por train el
2026-07-11** (`entry<10, exit>70, SMA200`) — no se re-tunea aquí, solo se
opera hacia adelante.

## Por qué esto es distinto de todo lo demás en el proyecto

Todo backtest mide el pasado — con el riesgo, ya materializado varias veces
en este proyecto (`finding-architecture-audit`, el falso positivo de
amplitud+RSI-2 de hoy mismo), de confundir ruido con ventaja real. El
forward trading no tiene ese problema: nadie puede ver el futuro de
antemano. El precio es que exige TIEMPO REAL — meses o años de muestra — no
hay atajo de backtest que lo sustituya.

**100% simulado.** Jamás toca un broker ni dinero real (`CLAUDE.md
§Seguridad`). No hay claves de broker en ningún lado de este sistema.

## Cómo corre

GitHub Actions (`.github/workflows/paper_trading_rsi2.yml`), no una tarea
local ni el agente programado de Claude Code — ambos dependen de que una
máquina/app esté encendida; GitHub Actions corre en los servidores de
GitHub, independiente de si el Mac de Eduardo está prendido, y es gratis
para este volumen de uso (una corrida de segundos, una vez al día).

- Horario: 22:00 UTC todos los días (margen holgado tras el cierre del
  mercado en EDT y EST). Si el mercado estuvo cerrado ese día (fin de
  semana, feriado), el script no encuentra un día nuevo y no hace nada — no
  se mantiene a mano un calendario de feriados bursátiles.
- Cada corrida empieza con un checkout LIMPIO del repo: `data/` está fuera
  de git (parquet regenerable), así que cada vez se descarga el historial
  de SPY completo desde yfinance — no hace falta lógica de actualización
  incremental, y evita cualquier riesgo de estado de precios corrupto entre
  corridas.
- El resultado (`paper_trading/rsi2/daily_log.csv`) SÍ está versionado en
  git — es la memoria del experimento, no un dato regenerable. El workflow
  lo commitea automáticamente si hay filas nuevas.

## Formato del log

Un renglón por día de trading, APPEND-ONLY (nunca se reescribe):

| columna | significado |
|---|---|
| `date` | fecha del día (índice) |
| `close` | cierre de SPY |
| `rsi2` | RSI de 2 días |
| `sma_trend` | SMA200 |
| `above_trend` | `close > sma_trend` |
| `position` | señal cruda 0/1 — **decidida con el CIERRE de este día** |
| `action` | `"ENTER"` / `"EXIT"` / vacío |

**Convención causal, idéntica al backtest** (`backtest/sp500_families.py`):
`position` en la fila del día t es la señal decidida al cierre de t. La
exposición real de mercado llega un día después, a la apertura de t+1 (ver
`daily_strategy_returns`). Este log no calcula esa exposición ni un P&L —
solo registra la señal cruda; el análisis de desempeño (cuando haya
suficiente muestra) debe reusar `daily_strategy_returns` sobre este mismo
log, no reinventar la mecánica.

## Diseño a prueba de corridas perdidas

Si el workflow no corre uno o varios días (falla, GitHub tiene un
incidente, etc.), la siguiente corrida NO inventa nada con datos viejos:
recalcula la señal completa (`compute_daily_rows` en
`src/paper_trading/rsi2.py`) sobre el historial de precios ya actualizado
y solo le faltan por registrar los días entre el último renglón del log y
hoy — reconstruye las transiciones reales que hubo en ese lapso, con sus
fechas correctas, en vez de solo mirar "el estado de hoy".

## Primer arranque — nunca retroactivo

El primer renglón del log es el día en que este sistema se activó
(2026-07-10, primera corrida real). El log **nunca** se llena hacia atrás
con la ventana 2015-2026 del backtest — eso sería mezclar backtest con
forward test y perdería la propiedad que hace valioso al forward trading.

## Para correrlo a mano (verificación, no reemplaza el schedule)

```bash
uv run python -m src.paper_trading.rsi2
```

Idempotente: correrlo dos veces el mismo día no duplica el renglón.
