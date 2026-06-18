"""Tests del scoring offline: relevancia, escalación a Claude y score local."""

from datetime import datetime, timezone

from src.core.config import load_settings
from src.core.models import NewsItem, SentimentScore

from src.sentiment.scoring import score_item

NOW = datetime(2025, 6, 1, 12, 0, tzinfo=timezone.utc)
CFG = load_settings().sentiment


def make_news(title: str) -> NewsItem:
    return NewsItem(id=title[:16], title=title, source="t", url="u",
                    published_at=NOW, summary="")


def make_analyze(calls: list):
    async def analyze_fn(item):
        calls.append(item.id)
        return SentimentScore(news_id=item.id, symbol_scope=["BTC"], score=0.5,
                              confidence=0.9, high_impact=True, rationale="claude",
                              analyzed_at=NOW)
    return analyze_fn


async def test_no_relevante_devuelve_none():
    calls = []
    out = await score_item(make_news("Apple earnings beat expectations"),
                           CFG, analyze_fn=make_analyze(calls))
    assert out is None and calls == []


async def test_no_escalado_usa_score_local_sin_claude():
    calls = []
    out = await score_item(make_news("Bitcoin trading volume steady today"),
                           CFG, analyze_fn=make_analyze(calls))
    assert out is not None
    assert calls == []                       # NO se llamó a Claude
    assert out.rationale.startswith("score local")
    assert out.confidence == abs(out.score)  # confianza local = |score|


async def test_high_impact_escala_a_claude():
    calls = []
    out = await score_item(make_news("Major exchange hacked, funds stolen"),
                           CFG, analyze_fn=make_analyze(calls))
    assert calls == [out.news_id]            # se escaló a Claude
    assert out.rationale == "claude"


async def test_score_local_fuerte_escala_a_claude():
    # "bullish breakout rally" → |local| alto (≥ umbral) pero NO high-impact:
    # aísla la escalación por magnitud del score.
    calls = []
    await score_item(make_news("Bitcoin bullish breakout rally accelerates"),
                     CFG, analyze_fn=make_analyze(calls))
    assert len(calls) == 1


async def test_event_kind_shock_se_propaga_en_escalada():
    # La etiqueta event_kind la pone el FILTRO (determinista por términos), no
    # Claude: aunque el analyze_fn no la fije, el score escalado sale "shock".
    out = await score_item(make_news("Major exchange hacked, funds stolen"),
                           CFG, analyze_fn=make_analyze([]))
    assert out.event_kind == "shock"


async def test_event_kind_scheduled_se_propaga_en_escalada():
    out = await score_item(make_news("FOMC decision shakes Bitcoin market"),
                           CFG, analyze_fn=make_analyze([]))
    assert out.event_kind == "scheduled"


async def test_event_kind_none_en_score_local():
    out = await score_item(make_news("Bitcoin trading volume steady today"),
                           CFG, analyze_fn=make_analyze([]))
    assert out.event_kind == "none"
