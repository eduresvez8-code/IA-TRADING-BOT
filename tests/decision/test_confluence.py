"""Tests de escenario de la matriz de confluencia (PLAN_MAESTRO §1).

Un test por fila de la matriz, más la simetría LONG/SHORT y los casos borde.
Venue en vivo = Futuros USD-M (allow_short=true): los cortos fluyen simétricos.
El gate de cortos (config) se verifica con una copia que lo desactiva.
"""

from datetime import datetime, timedelta, timezone

from src.core.config import load_settings
from src.core.models import Action, SentimentScore, Signal
from src.decision.confluence import decide, decide_event

NOW = datetime(2026, 6, 13, 12, 0, tzinfo=timezone.utc)
CFG = load_settings()  # repo real: allow_short=true (Futuros)


def cfg_sin_cortos():
    """Copia de la config con el gate de cortos desactivado (long-only)."""
    s = CFG.model_copy(deep=True)
    s.confluence.allow_short = False
    return s


def cfg_sin_impulso():
    """Copia con el gate de impulso desactivado (confirm_impulse_bps=0, ablación)."""
    s = CFG.model_copy(deep=True)
    s.event.confirm_impulse_bps = 0.0
    return s


def make_signal(score: float, symbol: str = "BTCUSDT") -> Signal:
    return Signal(symbol=symbol, score=score, strategy="ema_cross_rsi", timestamp=NOW)


def make_sentiment(score: float, *, high_impact: bool = False,
                   event_kind: str = "none", confidence: float = 0.8) -> SentimentScore:
    return SentimentScore(
        news_id="n1", symbol_scope=["BTCUSDT"], score=score,
        confidence=confidence, high_impact=high_impact, event_kind=event_kind,
        analyzed_at=NOW,
    )


# ---- Fila 0: macro PROGRAMADO bloquea (tiene prioridad sobre todo) ----

def test_scheduled_macro_bloquea_aunque_haya_noticia_y_regimen():
    # FOMC/CPI: resultado incierto → no abrir hacia el dato, diga lo que diga el
    # régimen. El bloqueo tiene prioridad sobre la originación por noticia.
    d = decide(make_signal(0.9), make_sentiment(0.8, event_kind="scheduled"), CFG)
    assert d.action == Action.HOLD
    assert d.size_factor == 0.0
    assert d.reason == "scheduled_macro_block"


# ---- Fila 1: la INVERSIÓN — sin noticia significativa NO se abre (Opción 2) ----

def test_regimen_fuerte_sin_noticia_no_origina():
    # El cambio de fondo: aunque el quant (régimen) sea fortísimo, sin sentimiento
    # NO se abre nada. El quant ya no tiene gatillo (su EMA-cross en 5m perdía).
    d = decide(make_signal(0.9), None, CFG)
    assert d.action == Action.HOLD
    assert d.size_factor == 0.0
    assert d.reason == "no_news_origination"


def test_sentimiento_debil_no_origina():
    # Sentimiento por debajo del umbral de originación → tampoco abre.
    d = decide(make_signal(0.9), make_sentiment(0.1), CFG)
    assert d.action == Action.HOLD
    assert d.reason == "no_news_origination"


def test_sentimiento_justo_en_umbral_si_origina():
    thr = CFG.confluence.sentiment_confirm_threshold
    d = decide(make_signal(0.8), make_sentiment(thr), CFG)
    assert d.action == Action.LONG  # >= umbral origina


# ---- La NOTICIA pone la dirección; régimen FUERTE y alineado → tamaño pleno ----

def test_noticia_alcista_con_regimen_a_favor_long_pleno():
    d = decide(make_signal(0.8), make_sentiment(0.6), CFG)
    assert d.action == Action.LONG
    assert d.size_factor == 1.0
    assert d.reason == "regime_confirms"


def test_noticia_bajista_con_regimen_a_favor_short_pleno():
    d = decide(make_signal(-0.8), make_sentiment(-0.6), CFG)
    assert d.action == Action.SHORT
    assert d.size_factor == 1.0
    assert d.reason == "regime_confirms"


def test_shock_significativo_origina_como_noticia():
    # Un shock direccional con score significativo origina por el Slow Path normal
    # (la originación sub-vela vive en el Fast Path; aquí cae a la matriz).
    d = decide(make_signal(0.8), make_sentiment(0.6, event_kind="shock"), CFG)
    assert d.action == Action.LONG
    assert d.reason == "regime_confirms"


# ---- Régimen NEUTRO/débil → opera la noticia con tamaño reducido ----

def test_regimen_neutro_tamano_reducido():
    # Noticia alcista origina, pero la tendencia HTF es débil (0.1 < umbral) → no
    # confirma ni contradice → tamaño reducido.
    d = decide(make_signal(0.1), make_sentiment(0.6), CFG)
    assert d.action == Action.LONG
    assert d.size_factor == CFG.confluence.reduced_size_factor
    assert d.reason == "regime_neutral"


def test_sin_quant_pero_con_noticia_opera_reducido():
    # Régimen exactamente neutro (0.0, p.ej. HTF aún sin calentar) + noticia → la
    # noticia ORIGINA igual, con tamaño reducido. (El engine pasa régimen 0 así.)
    d = decide(make_signal(0.0), make_sentiment(0.6), CFG)
    assert d.action == Action.LONG
    assert d.size_factor == CFG.confluence.reduced_size_factor
    assert d.reason == "regime_neutral"


def test_regimen_debil_opuesto_no_veta():
    # Tendencia HTF bajista pero DÉBIL (−0.2, bajo umbral) contra noticia alcista:
    # no es lo bastante fuerte para vetar → opera reducido, no HOLD.
    d = decide(make_signal(-0.2), make_sentiment(0.6), CFG)
    assert d.action == Action.LONG
    assert d.size_factor == CFG.confluence.reduced_size_factor
    assert d.reason == "regime_neutral"


# ---- Régimen FUERTE y OPUESTO a la noticia → veta (espejo del viejo conflicto) ----

def test_regimen_fuerte_opuesto_veta():
    # Noticia alcista pero tendencia 1h marcadamente bajista → no peleamos la
    # tendencia con un titular puntual. HOLD por regime_conflict.
    d = decide(make_signal(-0.8), make_sentiment(0.6), CFG)
    assert d.action == Action.HOLD
    assert d.size_factor == 0.0
    assert d.reason == "regime_conflict"


def test_regimen_fuerte_opuesto_veta_simetrico_short():
    # Noticia bajista pero tendencia 1h alcista fuerte → veta el SHORT.
    d = decide(make_signal(0.8), make_sentiment(-0.6), CFG)
    assert d.action == Action.HOLD
    assert d.reason == "regime_conflict"


# ---- Gate de cortos (config) ----

def test_short_bloqueado_si_gate_desactivado():
    d = decide(make_signal(-0.8), make_sentiment(-0.6), cfg_sin_cortos())
    assert d.action == Action.HOLD
    assert d.reason == "short_disabled"


def test_short_gate_tiene_prioridad_sobre_regimen():
    # Noticia bajista (→SHORT) + régimen alcista (conflicto) PERO shorts off: el
    # gate de venue es restricción dura y se evalúa antes → short_disabled, no
    # regime_conflict.
    d = decide(make_signal(0.8), make_sentiment(-0.6), cfg_sin_cortos())
    assert d.action == Action.HOLD
    assert d.reason == "short_disabled"


# ---- Bordes y auditoría ----

def test_regimen_justo_en_el_umbral_es_fuerte():
    thr = CFG.confluence.quant_strong_threshold
    d = decide(make_signal(thr), make_sentiment(0.6), CFG)
    assert d.action == Action.LONG
    assert d.size_factor == 1.0
    assert d.reason == "regime_confirms"


def test_decision_registra_scores_para_auditoria():
    # quant_score guarda el RÉGIMEN; sentiment_score, la noticia.
    d = decide(make_signal(0.7), make_sentiment(0.4), CFG)
    assert d.quant_score == 0.7
    assert d.sentiment_score == 0.4
    assert d.symbol == "BTCUSDT"


def test_as_of_fija_el_timestamp_de_forma_determinista():
    # Inyectar as_of hace la matriz pura: misma entrada → mismo timestamp. Sin él,
    # la decisión usaría el reloj actual (no reproducible). El TTL NO vive aquí
    # (lo aplica el orquestador): pasar un sentimiento "viejo" no lo invalida.
    old = make_sentiment(0.5)  # analyzed_at = NOW
    later = NOW + timedelta(hours=5)
    d = decide(make_signal(0.8), old, CFG, as_of=later)
    assert d.timestamp == later
    assert d.action == Action.LONG  # la matriz ignora la edad: la noticia origina
    assert d.reason == "regime_confirms"


# ===========================================================================
# Fast Path — decide_event (Plan V2 Fase 2.2): la noticia ORIGINA
# ===========================================================================
# settings.yaml: min_impact_score=0.6, min_confidence=0.7, ttl=180s,
# cooldown=900s, confirm_impulse_bps=8, size_factor=0.5.


def shock(score: float, *, confidence: float = 0.8) -> SentimentScore:
    """Un SentimentScore de clase shock (lo que el Fast Path puede originar)."""
    return make_sentiment(score, event_kind="shock", confidence=confidence)


# ---- Originación: todas las puertas pasan ----

def test_event_origina_long_con_todas_las_puertas():
    # shock alcista fuerte + confiado + fresco + sin cooldown + impulso alcista ≥8.
    d = decide_event(shock(0.7), "BTCUSDT", price_impulse_bps=10.0, settings=CFG, as_of=NOW)
    assert d.action == Action.LONG
    assert d.size_factor == CFG.event.size_factor   # tamaño base de evento (0.5)
    assert d.reason == "event_originate_long"
    assert d.quant_score == 0.0                       # el quant NO originó esto
    assert d.sentiment_score == 0.7


def test_event_origina_short_simetrico():
    d = decide_event(shock(-0.7), "BTCUSDT", price_impulse_bps=-10.0, settings=CFG, as_of=NOW)
    assert d.action == Action.SHORT
    assert d.reason == "event_originate_short"


# ---- Puerta 0: solo shock origina ----

def test_event_scheduled_no_origina():
    sent = make_sentiment(0.7, event_kind="scheduled")
    d = decide_event(sent, "BTCUSDT", price_impulse_bps=10.0, settings=CFG, as_of=NOW)
    assert d.action == Action.HOLD
    assert d.reason == "event_not_shock"


def test_event_none_no_origina():
    sent = make_sentiment(0.7, event_kind="none")
    d = decide_event(sent, "BTCUSDT", price_impulse_bps=10.0, settings=CFG, as_of=NOW)
    assert d.reason == "event_not_shock"


# ---- Puerta 1: frescura (TTL propio del evento) ----

def test_event_caducado_no_origina():
    # analyzed_at=NOW; a NOW+181s supera el ttl de 180s → stale.
    later = NOW + timedelta(seconds=181)
    d = decide_event(shock(0.7), "BTCUSDT", price_impulse_bps=10.0, settings=CFG, as_of=later)
    assert d.action == Action.HOLD
    assert d.reason == "event_stale"


def test_event_justo_dentro_del_ttl_si_origina():
    # Frontera: a NOW+180s (== ttl) sigue fresco (la condición es > ttl).
    later = NOW + timedelta(seconds=180)
    d = decide_event(shock(0.7), "BTCUSDT", price_impulse_bps=10.0, settings=CFG, as_of=later)
    assert d.action == Action.LONG


# ---- Puerta 2: cooldown por símbolo ----

def test_event_en_cooldown_no_origina():
    # Último trade de evento hace 100s; cooldown=900s → aún enfriando.
    last = NOW - timedelta(seconds=100)
    d = decide_event(shock(0.7), "BTCUSDT", price_impulse_bps=10.0, settings=CFG,
                     as_of=NOW, last_event_trade_at=last)
    assert d.action == Action.HOLD
    assert d.reason == "event_cooldown"


def test_event_pasado_el_cooldown_si_origina():
    last = NOW - timedelta(seconds=901)  # > 900s
    d = decide_event(shock(0.7), "BTCUSDT", price_impulse_bps=10.0, settings=CFG,
                     as_of=NOW, last_event_trade_at=last)
    assert d.action == Action.LONG


# ---- Puertas 3 y 4: magnitud y confianza ----

def test_event_score_debil_no_origina():
    d = decide_event(shock(0.5), "BTCUSDT", price_impulse_bps=10.0, settings=CFG, as_of=NOW)
    assert d.reason == "event_weak_score"   # 0.5 < min_impact_score 0.6


def test_event_baja_confianza_no_origina():
    d = decide_event(shock(0.7, confidence=0.5), "BTCUSDT", price_impulse_bps=10.0,
                     settings=CFG, as_of=NOW)
    assert d.reason == "event_low_confidence"   # 0.5 < min_confidence 0.7


# ---- Puerta 5: gate de cortos ----

def test_event_short_bloqueado_si_gate_desactivado():
    d = decide_event(shock(-0.7), "BTCUSDT", price_impulse_bps=-10.0,
                     settings=cfg_sin_cortos(), as_of=NOW)
    assert d.action == Action.HOLD
    assert d.reason == "short_disabled"


# ---- Puerta 6: confirmación de impulso (circuit breaker b) ----

def test_event_sin_impulso_alineado_no_origina():
    # Titular alcista pero el precio cayó: el mercado NO respalda → no operamos.
    d = decide_event(shock(0.7), "BTCUSDT", price_impulse_bps=-10.0, settings=CFG, as_of=NOW)
    assert d.action == Action.HOLD
    assert d.reason == "event_no_impulse"


def test_event_impulso_alineado_pero_debil_no_origina():
    # Dirección correcta pero magnitud < 8 bps: confirmación insuficiente.
    d = decide_event(shock(0.7), "BTCUSDT", price_impulse_bps=5.0, settings=CFG, as_of=NOW)
    assert d.reason == "event_no_impulse"


def test_event_gate_de_impulso_desactivado_origina_sin_confirmacion():
    # confirm_impulse_bps=0 (ablación A/B): el gate se salta → origina aunque el
    # impulso sea opuesto. Es el brazo "B" del experimento de los kill criteria.
    d = decide_event(shock(0.7), "BTCUSDT", price_impulse_bps=-50.0,
                     settings=cfg_sin_impulso(), as_of=NOW)
    assert d.action == Action.LONG
    assert d.reason == "event_originate_long"


# ---- Orden de puertas (qué razón gana cuando varias fallan) ----

def test_event_kind_tiene_prioridad_sobre_score_debil():
    # No-shock Y score débil: el motivo es event_not_shock (puerta 0 primero).
    sent = make_sentiment(0.3, event_kind="none")
    d = decide_event(sent, "BTCUSDT", price_impulse_bps=0.0, settings=CFG, as_of=NOW)
    assert d.reason == "event_not_shock"


def test_event_short_gate_tiene_prioridad_sobre_impulso():
    # Bearish shock, shorts OFF y sin impulso: gana short_disabled (puerta 5 < 6).
    d = decide_event(shock(-0.7), "BTCUSDT", price_impulse_bps=0.0,
                     settings=cfg_sin_cortos(), as_of=NOW)
    assert d.reason == "short_disabled"


# ---- Determinismo del timestamp ----

def test_event_as_of_fija_el_timestamp():
    d = decide_event(shock(0.7), "BTCUSDT", price_impulse_bps=10.0, settings=CFG, as_of=NOW)
    assert d.timestamp == NOW
