# Handoff — Arranque en testnet (Fase B) + preparación del dashboard

Estado a 2026-06-21. Resume qué falta para operar en testnet y qué hay/falta para el
dashboard en tiempo real de la próxima sesión.

---

## A. Lo que YA está listo (no hay que construir)

El entrypoint `src/main.py` tiene los 4 comandos del ciclo de vida:

| Comando | Qué hace | Opera? |
|---|---|---|
| `uv run python -m src.main --check` | Valida config + imports + .env | No |
| `uv run python -m src.main --preflight` | Conecta a testnet, valida claves/saldo/hedge mode/filtros | No (read-only + set hedge) |
| `uv run python -m src.main --status` | Foto: últimas órdenes (SQLite) + saldo y posiciones (Binance) | No |
| `uv run python -m src.main --live` | Lazo en vivo: datos spot mainnet (público) → órdenes a futuros testnet | **Sí** |

Cableado en `live()` COMPLETO: data client, exchange (testnet), storage, executor,
orchestrator, `event_fetch` (Fast Path) y `sentiment_fetch` (Slow Path). Los stops
(SL/TP) se colocan **server-side en Binance** (`STOP_MARKET`/`TAKE_PROFIT_MARKET` con
`closePosition`), así que protegen la posición aunque el bot/PC se apague.

Persistencia SQLite (`data/trading.db`, se crea al primer `--live`/`--status`):
`candles`, `orders`, `session_state`, `news`, `sentiment_scores`. El kill-switch y la
pérdida diaria sobreviven a reinicios (`session_state`).

Gates de seguridad (hoy `false` en `settings.yaml`): `event.enabled`, `sentiment.enabled`.
Con ambos en false, `--live` opera con **señal quant pura** (línea base de paper trading).

---

## B. Checklist para arrancar en testnet

1. **Claves de testnet en `.env`.** Ya hay `BINANCE_API_KEY`/`SECRET` (64 chars) y
   `BINANCE_TESTNET=true`. ⚠️ Deben ser de **testnet.binancefuture.com** (NO mainnet) y
   con permiso de Futuros. `ANTHROPIC_API_KEY` ya cargada (108 chars).

2. **Preflight** (no envía órdenes; confirma que todo conecta):
   ```bash
   uv run python -m src.main --preflight
   ```
   Debe mostrar saldo > 0 (si es 0, pide fondos en el **faucet** de la testnet), hedge
   mode (o one-way si el testnet lo veta) y los filtros de cada símbolo.

3. **Elegir el modo de arranque:**
   - **Conservador (recomendado primero):** dejar `event.enabled=false` y
     `sentiment.enabled=false` → `--live` opera solo con quant. Valida ejecución,
     reconciliación y stops sin gastar Claude ni meter la variable noticias.
   - **Fase B noticias:** poner `event.enabled=true` (Fast Path origina por shocks) y/o
     `sentiment.enabled=true` (Slow Path confirma/modula). Requiere `ANTHROPIC_API_KEY`
     (ya está). El gate abierto sin clave hace fail-fast (ya implementado).

4. **Lanzar** y observar logs:
   ```bash
   uv run python -m src.main --live      # Ctrl-C para detener
   ```
   En otra terminal, `--status` para ver qué hizo.

5. **Dejarlo correr** y revisar métricas de paper trading antes de cualquier decisión de
   capital real (decisión explícita de Eduardo; ver memoria `paper-trading-testnet`).

> El bot corre como proceso local en el Mac: **si apagas el PC o pierdes red, deja de
> operar** (no abre nada nuevo, no reconcilia). Las posiciones abiertas siguen protegidas
> por sus SL/TP server-side en Binance. Para 24/7 real → VPS siempre encendido.

---

## C. Preparación del dashboard (próxima sesión)

**Objetivo:** ver operaciones, posiciones, PnL y señales en tiempo real.

**Datos que YA existen (leer de `data/trading.db` + Binance):**
- `orders` — toda orden enviada (entrada + SL/TP), con `decision_reason`, status, ts.
- `sentiment_scores` + `news` — qué noticias se analizaron y con qué score/scope.
- `session_state` — peak wallet, day-start wallet, kill-switch (estado puntual).
- En vivo desde Binance: saldo, posiciones abiertas, **uPnL** (`acct.positions`).

**GAP identificado (lo que el dashboard tendrá que añadir):**
1. **No hay serie temporal de equity/PnL.** `session_state` es una sola fila. Para una
   curva de capital hay que (a) añadir una tabla `equity_snapshots(ts, wallet, upnl)` que
   el `_cycle` del engine escriba cada N, o (b) reconstruirla desde el income history de
   Binance (`futures_income_history`).
2. **No hay PnL realizado por trade.** `orders` guarda envíos, no resultados. Para una
   tabla "operaciones cerradas con resultado" hay que cruzar con fills/income de Binance
   o registrar el realized_pnl al cerrar.
3. **No hay registro de señales que NO cuajaron.** Solo se guardan órdenes enviadas; las
   decisiones `decide()` que quedaron en HOLD no se persisten. Útil para depurar "por qué
   no operó".

**Arquitectura sugerida (a decidir la próxima sesión):**
- Backend ligero (FastAPI) que sirve JSON desde `Storage` + un cliente Binance read-only,
  con polling o WebSocket al frontend. Mantiene la regla async/httpx/aiosqlite del repo.
- O un dashboard de terminal (rich/textual) si se quiere $0 infra y sin navegador.
- Cero Hardcoding: cualquier umbral/intervalo del dashboard → `settings.yaml`.
- El dashboard es **read-only**: nunca envía órdenes (la regla de que toda orden pasa por
  `risk/manager.py` → `execution/` se mantiene; el dashboard solo observa).

**Primer paso recomendado para el dashboard:** añadir la tabla `equity_snapshots` y que el
engine la escriba en cada ciclo — sin eso no hay curva de capital, que es lo primero que
se quiere ver. Es un cambio pequeño y aislado (Vía B para el intervalo).
