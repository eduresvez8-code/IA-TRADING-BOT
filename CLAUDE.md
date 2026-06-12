# IA TRADING — Bot híbrido (técnico + sentimiento)

Bot de trading de criptomonedas que combina señales de análisis técnico con
análisis de sentimiento de noticias (Claude Haiku). Mercado: Binance (testnet
primero). Presupuesto: $0/mes en datos. Ver `PLAN_MAESTRO.md` para la
arquitectura completa y el roadmap por sprints.

## Comandos

```bash
uv sync                          # instalar/actualizar dependencias
uv run pytest                    # correr tests
uv run python -m src.main --check  # smoke test: config + imports
```

## Convenciones

- Python 3.13, asyncio en todo el pipeline en vivo. Nada de I/O bloqueante
  dentro del event loop (BD → `aiosqlite`, HTTP → `httpx`).
- Los contratos de datos viven en `src/core/models.py` (Pydantic). No se
  modifican sin actualizar sus tests en el mismo cambio.
- Parámetros de trading en `config/settings.yaml`; secretos solo en `.env`.
  Nunca hardcodear símbolos, umbrales ni API keys en el código.
- SQLite siempre en modo WAL (se activa en `storage.py` al inicializar).
- Toda orden pasa por `risk/manager.py` antes de `execution/`. Ningún módulo
  llama al executor directamente.
- Una sesión de trabajo = un módulo. Validar aislado (pytest + demo) antes de
  integrar en `main.py`.
- Tests en `tests/`, espejo de `src/` (ej. `src/quant/indicators.py` →
  `tests/quant/test_indicators.py`).

## Protocolo didáctico (obligatorio)

Eduardo quiere entender el porqué de absolutamente todo el código. En cada
tarea de implementación:

1. **Antes de codificar**: explicar el concepto que se va a implementar
   (problema que resuelve y matemática si aplica, con fórmulas).
2. **Después de codificar**: incluir un bloque "📖 Explicación" que recorra el
   código sección por sección explicando el *porqué* de cada decisión de
   diseño y qué pasaría con la alternativa.
3. **Glosario**: añadir todo término técnico nuevo a `docs/GLOSARIO.md`.
4. **Cierre de sprint**: resumen de conceptos cubiertos y preguntas abiertas.

## Seguridad (no negociable)

- `.env` jamás se versiona ni se imprime en logs.
- Claves de Binance reales: con permiso de trade, SIN permiso de retiro.
- Sprints 0–5 operan solo contra testnet/backtest. Capital real requiere
  decisión explícita de Eduardo tras revisar métricas de paper trading.
