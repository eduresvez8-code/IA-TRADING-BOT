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

## C. Dashboard de observabilidad — CONSTRUIDO (2026-06-21)

**Objetivo cumplido:** ver equity, posiciones, PnL, decisiones y noticias en tiempo real.

```bash
uv run python -m src.main --dashboard      # http://127.0.0.1:8787 (Ctrl-C para parar)
```

Se puede correr en paralelo a `--live` (otra terminal) o sobre una BD ya existente.

**Arquitectura (READ-ONLY, $0):** proceso APARTE del trading. Stdlib `http.server` +
`sqlite3` en modo `ro` (`file:...?mode=ro`) → físicamente incapaz de escribir o enviar
órdenes. Cero dependencias nuevas (no FastAPI/uvicorn). Solo GET (`/` y `/api/snapshot`),
enlazado a loopback (no se expone a la red). El frontend (`src/dashboard/index.html`) es una
página única autocontenida (vanilla JS + SVG, sin CDN) que repolla `/api/snapshot` cada
`dashboard.refresh_seconds`.

**Archivos:** `src/dashboard/{queries.py,server.py,index.html}`; config en
`DashboardConfig` (`config.py` + `settings.yaml` sección `dashboard`); comando `--dashboard`
en `main.py`. Tests: `tests/dashboard/test_queries.py` + `tests/data/test_storage.py`
(tablas nuevas) + `tests/test_config.py` (DashboardConfig). 627 tests verdes.

**GAPs resueltos (tablas nuevas que el engine ya escribe):**
1. ✅ **Serie temporal de equity** → tabla `equity_snapshots(ts, wallet, equity, upnl,
   positions)`, escrita por `Orchestrator._record_equity` cada ciclo (desde el `get_account`
   del propio ciclo, sin llamadas extra al exchange). De aquí salen la curva, el uPnL, las
   posiciones y el PnL del día.
2. ✅ **Registro de señales en HOLD** → tabla `decisions(ts, symbol, action, reason,
   quant_score, sentiment_score, size_factor, source)`, escrita por
   `Orchestrator._record_decision` en CADA decisión (Slow y Fast Path). Alimenta el panel
   "¿por qué (no) operó?".
3. ⏳ **PnL realizado por trade** — sigue pendiente (no bloqueante): `orders` guarda envíos,
   no resultados. El PnL del día se deriva de `equity − day_start_wallet`, que es suficiente
   para paper trading. Para "operaciones cerradas con resultado" habría que cruzar con
   `futures_income_history` de Binance o registrar `realized_pnl` al cerrar. Mejora futura.

**Liveness:** el dashboard es un proceso aparte, no ve `halted` en memoria; deriva la salud
del último `equity_snapshots.ts` (obsoleto si > `stale_after_intervals` velas → el bot
probablemente está caído/halted). Robusto sin acoplar procesos.
