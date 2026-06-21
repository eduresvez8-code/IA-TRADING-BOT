"""Fase A — Demo observable del Fast Path (RSS shocks → Claude → originación).

Valida la CALIDAD de la señal de noticias de punta a punta SIN tocar el engine
ni Binance: sondea los feeds RSS REALES, corre el filtro de shock y, si hay
ANTHROPIC_API_KEY, escala los shocks frescos a Claude con la MISMA función de
producción que usará el engine (`fetch_events`). Para cada shock muestra el
veredicto de Claude (score/confidence/scope/rationale) y aplica el gate de
originación del `EventConfig` (min_impact_score ∧ min_confidence) para decir si
ORIGINARÍA un trade y en qué dirección.

Por qué este demo ANTES de habilitar `event.enabled`:
    El Fast Path está implementado pero apagado por gate de seguridad. Antes de
    dejar que ORIGINE trades en testnet, hay que confirmar que (1) los feeds
    entregan titulares, (2) el filtro VADER+dict los clasifica como shock cuando
    toca, y (3) Claude les pone un signo y una confianza sensatos. Esta es la
    capa de señal; el motor de órdenes (Fase B, testnet) va después.

Dos vistas:
    [1] Diagnóstico del feed (GRATIS, sin Claude): TODOS los titulares crudos con
        su clasificación de filtro (relevante / event_kind / score local / edad).
        Sirve para ver qué hay AHORA en los feeds y por qué cada uno pasa o no la
        frescura del Fast Path.
    [2] Fast Path real (si key): corre `fetch_events` sobre los MISMOS items (un
        solo fetch de red, cacheado) con la ventana de frescura de producción.
        Cada shock fresco y no-visto se escala a Claude y se evalúa el gate de
        originación.

Uso:
    uv run python -m src.sentiment.fast_path_demo                # ventana de prod (30 min)
    uv run python -m src.sentiment.fast_path_demo --age-hours 24 # ampliar para ver shocks
    uv run python -m src.sentiment.fast_path_demo --no-claude    # solo filtro (cero tokens)

Notas:
    - Cero gasto si no hay shocks frescos: el cliente Claude solo se construye si
      el filtro (gratis) detecta al menos un shock dentro de la ventana.
    - `--limit` es un tope DURO de seguridad sobre las llamadas a Claude (default
      10). Con la ventana de producción (30 min) los shocks suelen ser 0-3.
    - DEUDA_TICKER (documentada en slow_path.py): Claude devuelve scope como
      ["BTC"] pero operamos ["BTCUSDT"], así que el scope no-wildcard rara vez
      machea símbolos. El demo lo señala en la columna Scope.
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timedelta, timezone

from src.core.config import load_secrets, load_settings
from src.core.models import NewsItem, SentimentScore
from src.core.scope import resolve_scope
from src.sentiment.analyzer import analyze
from src.sentiment.events import fetch_events
from src.sentiment.feeds import fetch_feeds
from src.sentiment.filter import filter_news


def _fmt_age(delta: timedelta) -> str:
    """Antigüedad legible: '5m', '2h13m', '<1m'."""
    secs = int(delta.total_seconds())
    if secs < 60:
        return "<1m"
    mins = secs // 60
    if mins < 60:
        return f"{mins}m"
    return f"{mins // 60}h{mins % 60:02d}m"


def _dir(score: float) -> str:
    """Dirección de la apuesta a partir del signo del score."""
    return "LONG ↑" if score > 0 else ("SHORT ↓" if score < 0 else "—")


# ---------------------------------------------------------------------------
# Vista 1 — diagnóstico del feed (gratis, solo filtro)
# ---------------------------------------------------------------------------

def diagnose_feed(items: list[NewsItem], settings, now: datetime) -> dict[str, int]:
    """Imprime la clasificación de filtro de TODOS los titulares. Devuelve conteos."""
    hw = settings.sentiment.heuristic_weight
    counts = {"total": len(items), "relevant": 0, "shock": 0, "scheduled": 0}

    print(f"\n[1] DIAGNÓSTICO DEL FEED — {len(items)} titulares ingeridos")
    print("-" * 100)
    print(f"{'Edad':>6}  {'Clase':>9}  {'sLocal':>7}  {'Términos':<22}  Titular")
    print("-" * 100)

    # Más recientes primero: lo operable está arriba.
    for item in sorted(items, key=lambda it: it.published_at, reverse=True):
        fr = filter_news(item, heuristic_weight=hw)
        if fr.is_relevant:
            counts["relevant"] += 1
        if fr.event_kind == "shock":
            counts["shock"] += 1
        elif fr.event_kind == "scheduled":
            counts["scheduled"] += 1

        # Solo listamos lo relevante (el ruido no-cripto satura la tabla).
        if not fr.is_relevant:
            continue
        age = _fmt_age(now - item.published_at)
        terms = ",".join(fr.matched_terms[:3])[:22]
        kind = fr.event_kind if fr.event_kind != "none" else "·"
        print(f"{age:>6}  {kind:>9}  {fr.local_score:>+7.2f}  {terms:<22}  {item.title[:42]}")

    print("-" * 100)
    print(f"  Relevantes (cripto): {counts['relevant']}/{counts['total']}  |  "
          f"shocks: {counts['shock']}  |  scheduled (macro): {counts['scheduled']}")
    return counts


# ---------------------------------------------------------------------------
# Vista 2 — Fast Path real (fetch_events de producción + gate de originación)
# ---------------------------------------------------------------------------

async def run_fast_path(
    items: list[NewsItem], settings, secrets, *, age_seconds: int, limit: int,
    now: datetime,
) -> None:
    """Corre `fetch_events` (producción) sobre los items cacheados y evalúa originación."""
    ev = settings.event
    calls = {"n": 0}

    async def _analyze(item: NewsItem) -> SentimentScore:
        # Tope duro de presupuesto: nunca más de `limit` llamadas a Claude.
        if calls["n"] >= limit:
            raise RuntimeError("límite de llamadas a Claude alcanzado (--limit)")
        calls["n"] += 1
        return await analyze(item, settings.sentiment, secrets)

    async def _cached_feeds(_cfg):
        # Un solo fetch de red: Vista 1 y Vista 2 ven EXACTAMENTE los mismos items.
        return items

    print(f"\n[2] FAST PATH REAL — fetch_events (frescura ≤ {age_seconds // 60} min, "
          f"gate: |score|≥{ev.min_impact_score} ∧ conf≥{ev.min_confidence})")
    print("-" * 100)

    scores = await fetch_events(
        settings.sentiment,
        analyze_fn=_analyze,
        seen={},
        max_age_seconds=age_seconds,
        fetch_feeds_fn=_cached_feeds,
        now=now,
    )

    if not scores:
        print("  Sin shocks frescos y no-vistos en la ventana → el Fast Path no originaría nada.")
        print("  (Amplía con --age-hours N si quieres ver la pipeline de Claude con noticias más viejas.)")
        return

    symbols = settings.market.symbols
    quotes = settings.market.quote_assets
    print(f"{'Score':>6}  {'Conf':>5}  {'Scope':<10}  {'→Símbolos':<18}  {'Origina':>8}  {'Dir':>7}  Rationale")
    print("-" * 100)
    n_orig = 0
    for sc in sorted(scores, key=lambda s: abs(s.score), reverse=True):
        originates = abs(sc.score) >= ev.min_impact_score and sc.confidence >= ev.min_confidence
        if originates:
            n_orig += 1
        scope = ",".join(sc.symbol_scope)[:10]
        # Resolución real (fix DEUDA_TICKER): a qué símbolos del universo entra.
        resolved = resolve_scope(sc.symbol_scope, symbols, quotes)
        resolved_txt = (",".join(resolved) or "(ninguno)")[:18]
        flag = "✅ SÍ" if originates else "—"
        print(f"{sc.score:>+6.2f}  {sc.confidence:>5.2f}  {scope:<10}  {resolved_txt:<18}  "
              f"{flag:>8}  {_dir(sc.score):>7}  {sc.rationale[:40]}")

    print("-" * 100)
    print(f"  Shocks analizados por Claude: {len(scores)}  |  originarían trade: {n_orig}")
    print(f"  Llamadas a Claude gastadas: {calls['n']}")
    print(f"  (→Símbolos resuelve el scope por activo base: 'BTC'→'BTCUSDT'. Vacío = "
          f"shock de un activo que NO operamos.)")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

async def run_demo(age_hours: float | None, limit: int, no_claude: bool) -> None:
    settings = load_settings()
    secrets = load_secrets()
    now = datetime.now(timezone.utc)

    # Ventana de frescura: por defecto la de PRODUCCIÓN (event.max_headline_age_seconds);
    # --age-hours la amplía solo para este demo (no toca config).
    if age_hours is not None:
        age_seconds = int(age_hours * 3600)
    else:
        age_seconds = settings.event.max_headline_age_seconds

    print("=" * 100)
    print("FAST PATH DEMO (Fase A) — RSS shocks → Claude → señal de originación")
    print("=" * 100)
    print(f"Feeds: {len(settings.sentiment.rss_feeds)} RSS  |  modelo: {settings.sentiment.claude_model}")
    print(f"Frescura del Fast Path: ≤ {age_seconds // 60} min  |  reloj: {now:%Y-%m-%d %H:%M UTC}")

    items = await fetch_feeds(settings.sentiment)
    if not items:
        print("\n  ⚠ Los feeds no devolvieron titulares (¿sin red o feeds caídos?).")
        return

    counts = diagnose_feed(items, settings, now)

    # ¿Cuántos shocks FRESCOS hay? (precuenta gratis para decidir si vale la pena Claude.)
    fresh_shocks = [
        it for it in items
        if (now - it.published_at).total_seconds() <= age_seconds
        and filter_news(it, heuristic_weight=settings.sentiment.heuristic_weight).event_kind == "shock"
    ]

    has_key = bool(secrets.anthropic_api_key)
    if no_claude or not has_key:
        reason = "--no-claude" if no_claude else "sin ANTHROPIC_API_KEY"
        print(f"\n[2] FAST PATH REAL — OMITIDO ({reason}).")
        print(f"  Shocks frescos que SE escalarían a Claude: {len(fresh_shocks)}")
        for it in fresh_shocks[:limit]:
            print(f"    · ({_fmt_age(now - it.published_at)}) {it.title[:70]}")
        if not no_claude:
            print("  Añade ANTHROPIC_API_KEY al .env para ver el veredicto de Claude y la originación.")
        return

    if not fresh_shocks:
        print("\n[2] FAST PATH REAL — sin shocks frescos en la ventana → cero llamadas a Claude.")
        print("  (Amplía con --age-hours N para escalar noticias más viejas a Claude.)")
        return

    print(f"\n  → {len(fresh_shocks)} shock(s) fresco(s) detectado(s); escalando a Claude Haiku "
          f"(tope --limit={limit})…")
    await run_fast_path(items, settings, secrets, age_seconds=age_seconds, limit=limit, now=now)


def main() -> None:
    p = argparse.ArgumentParser(description="Fase A — demo del Fast Path (RSS shocks → Claude)")
    p.add_argument("--age-hours", type=float, default=None,
                   help="Ventana de frescura en horas (default: la de producción, ~0.5h).")
    p.add_argument("--limit", type=int, default=10,
                   help="Tope duro de llamadas a Claude (default 10).")
    p.add_argument("--no-claude", action="store_true",
                   help="No llamar a Claude; solo el filtro local (cero tokens).")
    args = p.parse_args()
    asyncio.run(run_demo(args.age_hours, args.limit, args.no_claude))


if __name__ == "__main__":
    main()
