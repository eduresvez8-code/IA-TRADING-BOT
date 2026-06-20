"""Tests de fetch_sentiment (productor del Slow Path): mapeo por símbolo, scope,
last-write-wins, dedup, frescura, resiliencia y emisión de score local.

Todo con dobles inyectados (fetch_feeds_fn + analyze_fn): cero red, cero Claude.
"""

from datetime import datetime, timedelta, timezone

from src.core.config import load_settings
from src.core.models import NewsItem, SentimentScore
from src.sentiment.slow_path import fetch_sentiment, resolve_scope

NOW = datetime(2026, 6, 19, 12, 0, tzinfo=timezone.utc)
CFG = load_settings().sentiment
SYMBOLS = ["BTCUSDT", "ETHUSDT"]

# Títulos calibrados al filtro (mismos que test_events): "hacked"→escala a Claude,
# "steady"→relevante-no-escala (score local), "Apple"→irrelevante (None).
SHOCK = "Major exchange hacked, funds stolen"
LOCAL = "Bitcoin trading volume steady today"
IRRELEVANT = "Apple earnings beat expectations"


def make_news(title: str, *, id: str | None = None,
              published_at: datetime = NOW) -> NewsItem:
    # id explícito: dos titulares con el mismo texto-shock pero distinto id (para
    # probar last-write-wins) no deben colisionar en el hash truncado del título.
    return NewsItem(id=id or title[:16], title=title, source="t", url="u",
                    published_at=published_at, summary="")


def make_analyze(calls: list, *, scope=("*",), scores=None, raise_on=None):
    async def analyze_fn(item: NewsItem) -> SentimentScore:
        calls.append(item.id)
        if raise_on is not None and item.id == raise_on:
            raise RuntimeError("claude caído")
        sc = (scores or {}).get(item.id, 0.8)
        return SentimentScore(news_id=item.id, symbol_scope=list(scope), score=sc,
                              confidence=0.9, high_impact=True, event_kind="none",
                              rationale="claude", analyzed_at=NOW)
    return analyze_fn


def make_feeds(items: list[NewsItem]):
    async def fetch_feeds_fn(config):
        return items
    return fetch_feeds_fn


def test_resolve_scope_wildcard_y_exacto():
    assert resolve_scope(["*"], SYMBOLS) == SYMBOLS          # mercado → todos
    assert resolve_scope(["BTCUSDT"], SYMBOLS) == ["BTCUSDT"]  # exacto
    assert resolve_scope(["BTC"], SYMBOLS) == []             # "BTC" != "BTCUSDT"
    assert resolve_scope(["DOGEUSDT"], SYMBOLS) == []        # no lo seguimos


async def test_wildcard_se_expande_a_todos_los_simbolos():
    calls: list[str] = []
    out = await fetch_sentiment(
        CFG, SYMBOLS, analyze_fn=make_analyze(calls), seen={},
        fetch_feeds_fn=make_feeds([make_news(SHOCK), make_news(IRRELEVANT)]), now=NOW,
    )
    assert set(out) == {"BTCUSDT", "ETHUSDT"}    # "*" llega a ambos
    assert out["BTCUSDT"].news_id == SHOCK[:16]
    assert calls == [SHOCK[:16]]                 # Claude SOLO sobre el shock


async def test_last_write_wins_por_published_at():
    calls: list[str] = []
    viejo = make_news(SHOCK, id="old", published_at=NOW - timedelta(hours=1))
    nuevo = make_news(SHOCK, id="new", published_at=NOW)
    out = await fetch_sentiment(
        CFG, SYMBOLS, analyze_fn=make_analyze(calls, scores={"old": 0.2, "new": 0.9}),
        seen={}, fetch_feeds_fn=make_feeds([nuevo, viejo]), now=NOW,   # orden mezclado
    )
    assert out["BTCUSDT"].score == 0.9 and out["ETHUSDT"].score == 0.9  # gana el nuevo


async def test_score_local_se_emite_sin_llamar_a_claude():
    # A diferencia del Fast Path (solo shocks), el Slow Path emite también lo
    # relevante-no-escalado: score local, sin gastar Claude.
    calls: list[str] = []
    out = await fetch_sentiment(
        CFG, SYMBOLS, analyze_fn=make_analyze(calls), seen={},
        fetch_feeds_fn=make_feeds([make_news(LOCAL)]), now=NOW,
    )
    assert set(out) == {"BTCUSDT", "ETHUSDT"}   # score local entra (scope ["*"])
    assert calls == []                          # nunca escaló a Claude


async def test_irrelevante_no_entra_pero_se_marca_visto():
    seen: dict[str, datetime] = {}
    out = await fetch_sentiment(
        CFG, SYMBOLS, analyze_fn=make_analyze([]), seen=seen,
        fetch_feeds_fn=make_feeds([make_news(IRRELEVANT)]), now=NOW,
    )
    assert out == {}                            # no relevante → fuera del store
    assert IRRELEVANT[:16] in seen              # pero visto: no re-evaluar cada poll


async def test_dedup_no_repuntua_el_mismo_titular():
    calls: list[str] = []
    seen: dict[str, datetime] = {}
    analyze = make_analyze(calls)
    feeds = make_feeds([make_news(SHOCK)])
    first = await fetch_sentiment(CFG, SYMBOLS, analyze_fn=analyze, seen=seen,
                                  fetch_feeds_fn=feeds, now=NOW)
    second = await fetch_sentiment(CFG, SYMBOLS, analyze_fn=analyze, seen=seen,
                                   fetch_feeds_fn=feeds, now=NOW)
    assert set(first) == {"BTCUSDT", "ETHUSDT"} and second == {}  # 2º poll no re-emite
    assert calls == [SHOCK[:16]]                                  # Claude una vez


async def test_frescura_descarta_titular_viejo_sin_llamar_a_claude():
    calls: list[str] = []
    viejo_h = CFG.max_news_age_hours + 1
    viejo = make_news(SHOCK, published_at=NOW - timedelta(hours=viejo_h))
    out = await fetch_sentiment(
        CFG, SYMBOLS, analyze_fn=make_analyze(calls), seen={},
        fetch_feeds_fn=make_feeds([viejo]), now=NOW,
    )
    assert out == {} and calls == []            # ni se emite ni se gasta Claude


async def test_seen_se_purga_por_la_ventana_de_frescura():
    viejo_h = CFG.max_news_age_hours + 1
    seen = {"viejo_id": NOW - timedelta(hours=viejo_h)}
    await fetch_sentiment(CFG, SYMBOLS, analyze_fn=make_analyze([]), seen=seen,
                          fetch_feeds_fn=make_feeds([]), now=NOW)
    assert "viejo_id" not in seen


async def test_fallo_de_claude_se_salta_y_se_reintenta():
    calls: list[str] = []
    seen: dict[str, datetime] = {}
    out = await fetch_sentiment(
        CFG, SYMBOLS, analyze_fn=make_analyze(calls, raise_on=SHOCK[:16]), seen=seen,
        fetch_feeds_fn=make_feeds([make_news(SHOCK)]), now=NOW,
    )
    assert out == {}
    assert calls == [SHOCK[:16]]                # se intentó
    assert SHOCK[:16] not in seen               # NO se marca visto → reintento
