"""Tests de escenario de la matriz de confluencia (PLAN_MAESTRO §1).

Un test por fila de la matriz, más la simetría LONG/SHORT y los casos borde.
Los umbrales se inyectan vía settings reales del repo (0.5 / 0.3 / 0.5) para
que el test también vigile que el settings.yaml sigue siendo coherente.
"""

from datetime import datetime, timezone

from src.core.config import load_settings
from src.core.models import Action, SentimentScore, Signal
from src.decision.confluence import decide

NOW = datetime(2026, 6, 13, 12, 0, tzinfo=timezone.utc)
CFG = load_settings()


def make_signal(score: float, symbol: str = "BTCUSDT") -> Signal:
    return Signal(symbol=symbol, score=score, strategy="ema_cross_rsi", timestamp=NOW)


def make_sentiment(score: float, *, high_impact: bool = False,
                   confidence: float = 0.8) -> SentimentScore:
    return SentimentScore(
        news_id="n1", symbol_scope=["BTCUSDT"], score=score,
        confidence=confidence, high_impact=high_impact, analyzed_at=NOW,
    )


# ---- Fila 0: high-impact bloquea TODA entrada ----

def test_high_impact_bloquea_aunque_quant_y_sentimiento_confirmen():
    # Quant fuerte + sentimiento confirma, pero hay evento de alto impacto.
    d = decide(make_signal(0.9), make_sentiment(0.8, high_impact=True), CFG)
    assert d.action == Action.HOLD
    assert d.size_factor == 0.0
    assert d.reason == "high_impact_block"


# ---- Fila 1: quant débil → HOLD (circuit breaker b incluido) ----

def test_quant_debil_es_hold():
    d = decide(make_signal(0.2), make_sentiment(0.1), CFG)
    assert d.action == Action.HOLD
    assert d.reason == "quant_weak"


def test_sentimiento_extremo_sin_quant_no_abre():
    # Circuit breaker (b): titular extremo pero el precio no se ha movido.
    d = decide(make_signal(0.1), make_sentiment(0.95), CFG)
    assert d.action == Action.HOLD
    assert d.reason == "quant_weak"


# ---- Fila 2: quant fuerte + sentimiento opuesto fuerte → HOLD ----

def test_long_con_sentimiento_opuesto_es_hold():
    d = decide(make_signal(0.8), make_sentiment(-0.6), CFG)
    assert d.action == Action.HOLD
    assert d.reason == "sentiment_conflict"


def test_short_con_sentimiento_opuesto_es_hold():
    d = decide(make_signal(-0.8), make_sentiment(0.6), CFG)
    assert d.action == Action.HOLD
    assert d.reason == "sentiment_conflict"


# ---- Fila 3: quant fuerte + sentimiento confirma → tamaño pleno ----

def test_long_confirmado_tamano_pleno():
    d = decide(make_signal(0.8), make_sentiment(0.5), CFG)
    assert d.action == Action.LONG
    assert d.size_factor == 1.0
    assert d.reason == "sentiment_confirms"


def test_short_confirmado_tamano_pleno():
    d = decide(make_signal(-0.8), make_sentiment(-0.5), CFG)
    assert d.action == Action.SHORT
    assert d.size_factor == 1.0
    assert d.reason == "sentiment_confirms"


# ---- Fila 4: quant fuerte + sentimiento neutro/ausente → tamaño reducido ----

def test_long_sentimiento_neutro_tamano_reducido():
    d = decide(make_signal(0.8), make_sentiment(0.1), CFG)
    assert d.action == Action.LONG
    assert d.size_factor == CFG.confluence.reduced_size_factor
    assert d.reason == "sentiment_neutral"


def test_long_sin_sentimiento_tamano_reducido():
    # Sin noticia relevante → operamos la técnica con tamaño reducido.
    d = decide(make_signal(0.8), None, CFG)
    assert d.action == Action.LONG
    assert d.size_factor == CFG.confluence.reduced_size_factor
    assert d.sentiment_score == 0.0


def test_short_sin_sentimiento_tamano_reducido():
    d = decide(make_signal(-0.8), None, CFG)
    assert d.action == Action.SHORT
    assert d.size_factor == CFG.confluence.reduced_size_factor


# ---- Bordes y auditoría ----

def test_quant_justo_en_el_umbral_es_fuerte():
    # |quant| == umbral cuenta como fuerte (>=), no como débil.
    thr = CFG.confluence.quant_strong_threshold
    d = decide(make_signal(thr), None, CFG)
    assert d.action == Action.LONG


def test_decision_registra_scores_para_auditoria():
    d = decide(make_signal(0.7), make_sentiment(0.4), CFG)
    assert d.quant_score == 0.7
    assert d.sentiment_score == 0.4
    assert d.symbol == "BTCUSDT"
