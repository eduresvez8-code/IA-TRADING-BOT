"""Tests de fetch_events (Fase 2.5(ii)): solo shocks, dedup, frescura, resiliencia.

Todo con dobles inyectados (fetch_feeds_fn + analyze_fn): cero red, cero Claude.
"""

from datetime import datetime, timedelta, timezone

from src.core.config import load_settings
from src.core.models import NewsItem, SentimentScore
from src.sentiment.events import fetch_events

NOW = datetime(2026, 6, 19, 12, 0, tzinfo=timezone.utc)
CFG = load_settings().sentiment
MAX_AGE = 1800  # 30 min (= settings.event.max_headline_age_seconds del repo)


def make_news(title: str, *, published_at: datetime = NOW) -> NewsItem:
    return NewsItem(id=title[:16], title=title, source="t", url="u",
                    published_at=published_at, summary="")


def make_analyze(calls: list, *, raise_on: str | None = None):
    async def analyze_fn(item: NewsItem) -> SentimentScore:
        calls.append(item.id)
        if raise_on is not None and item.id == raise_on:
            raise RuntimeError("claude caído")
        # event_kind="none" a propósito: fetch_events debe forzarlo a "shock".
        return SentimentScore(news_id=item.id, symbol_scope=["BTC"], score=0.8,
                              confidence=0.9, high_impact=True, event_kind="none",
                              rationale="claude", analyzed_at=NOW)
    return analyze_fn


def make_feeds(items: list[NewsItem]):
    async def fetch_feeds_fn(config):
        return items
    return fetch_feeds_fn


# Títulos calibrados al filtro: "hacked"→shock, "steady"→none, "Apple"→irrelevante.
SHOCK = "Major exchange hacked, funds stolen"
NOISE = "Bitcoin trading volume steady today"
IRRELEVANT = "Apple earnings beat expectations"


async def test_solo_shocks_se_emiten_y_solo_ellos_llaman_a_claude():
    calls: list[str] = []
    out = await fetch_events(
        CFG, analyze_fn=make_analyze(calls), seen={}, max_age_seconds=MAX_AGE,
        fetch_feeds_fn=make_feeds([make_news(SHOCK), make_news(NOISE),
                                   make_news(IRRELEVANT)]),
        now=NOW,
    )
    assert len(out) == 1
    assert out[0].news_id == SHOCK[:16]
    assert out[0].symbol_scope == ["BTC"]      # passthrough del análisis
    assert calls == [SHOCK[:16]]               # Claude SOLO sobre el shock


async def test_event_kind_se_fuerza_a_shock():
    # Aunque el analyze_fn devuelva event_kind="none", fetch_events lo fija a "shock"
    # (etiqueta determinista del filtro, no juicio de Claude).
    out = await fetch_events(
        CFG, analyze_fn=make_analyze([]), seen={}, max_age_seconds=MAX_AGE,
        fetch_feeds_fn=make_feeds([make_news(SHOCK)]), now=NOW,
    )
    assert out[0].event_kind == "shock"


async def test_dedup_no_reanaliza_el_mismo_titular():
    calls: list[str] = []
    seen: dict[str, datetime] = {}
    analyze = make_analyze(calls)
    feeds = make_feeds([make_news(SHOCK)])

    first = await fetch_events(CFG, analyze_fn=analyze, seen=seen,
                               max_age_seconds=MAX_AGE, fetch_feeds_fn=feeds, now=NOW)
    second = await fetch_events(CFG, analyze_fn=analyze, seen=seen,
                                max_age_seconds=MAX_AGE, fetch_feeds_fn=feeds, now=NOW)
    assert len(first) == 1 and len(second) == 0   # el 2º poll no re-emite
    assert calls == [SHOCK[:16]]                   # Claude se llamó UNA sola vez


async def test_frescura_descarta_titular_viejo_sin_llamar_a_claude():
    calls: list[str] = []
    viejo = make_news(SHOCK, published_at=NOW - timedelta(hours=2))  # > 30 min
    out = await fetch_events(
        CFG, analyze_fn=make_analyze(calls), seen={}, max_age_seconds=MAX_AGE,
        fetch_feeds_fn=make_feeds([viejo]), now=NOW,
    )
    assert out == [] and calls == []   # ni se emite ni se gasta Claude


async def test_seen_se_purga_por_la_ventana_de_frescura():
    # Un id visto hace 2h se purga (ya no pasaría frescura): acota la memoria del set.
    seen = {"viejo_id": NOW - timedelta(hours=2)}
    await fetch_events(CFG, analyze_fn=make_analyze([]), seen=seen,
                       max_age_seconds=MAX_AGE, fetch_feeds_fn=make_feeds([]), now=NOW)
    assert "viejo_id" not in seen


async def test_fallo_de_claude_se_salta_y_se_reintenta():
    # analyze que revienta: el batch no cae, el id NO entra en `seen` (reintento),
    # y no se emite nada en este poll.
    calls: list[str] = []
    seen: dict[str, datetime] = {}
    out = await fetch_events(
        CFG, analyze_fn=make_analyze(calls, raise_on=SHOCK[:16]), seen=seen,
        max_age_seconds=MAX_AGE, fetch_feeds_fn=make_feeds([make_news(SHOCK)]), now=NOW,
    )
    assert out == []
    assert calls == [SHOCK[:16]]        # se intentó
    assert SHOCK[:16] not in seen       # pero NO se marca como visto → reintento
