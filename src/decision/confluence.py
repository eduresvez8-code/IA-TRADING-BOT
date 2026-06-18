"""Decisores del Dual-Core: dos funciones puras, un mismo contrato `Decision`.

Conviven dos caminos (Plan V2 §2), ambos convergen en el mismo Risk Manager:

- `decide` (**Slow Path / estratégico**): por vela cerrada (5m). El quant manda
  la DIRECCIÓN; el sentimiento solo CONFIRMA (tamaño pleno), guarda silencio
  (tamaño reducido) o VETA. Nunca abre por sentimiento solo: sin movimiento de
  precio que lo respalde, un titular extremo puede ser falso (circuit breaker (b)).

- `decide_event` (**Fast Path / originación por evento**): por LLEGADA de un
  shock de noticia. Aquí la NOTICIA origina y el quant ya NO es condición
  necesaria — corrige la causalidad invertida del v1 (PLAN_MAESTRO_V2 §1). El
  circuit breaker (b) se preserva como **confirmación de impulso**: el precio
  debe respaldar el titular antes de operar.

Ambas son funciones puras: misma entrada → misma salida, sin estado ni I/O. El
reloj (`as_of`) y los estados temporales (TTL del store, cooldown) los inyecta el
orquestador, que es quien posee el reloj. Así cada regla se valida aislada.
"""

from __future__ import annotations

from datetime import datetime, timezone

from src.core.config import Settings, load_settings
from src.core.models import Action, Decision, SentimentScore, Signal


def _sign(x: float) -> int:
    """Signo de un score: +1 alcista, -1 bajista, 0 neutro."""
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0


def decide(
    signal: Signal,
    sentiment: SentimentScore | None = None,
    settings: Settings | None = None,
    *,
    as_of: datetime | None = None,
) -> Decision:
    """Combina señal técnica y sentimiento en una Decision auditable.

    Args:
        signal:    salida del Quant Engine (score técnico en [-1, +1]).
        sentiment: salida del Sentiment Engine, o None si no hay noticia
                   relevante para este símbolo en la ventana actual. La
                   CADUCIDAD (TTL) del sentimiento la aplica quien posee el reloj
                   y el store (el orquestador, vía `_fresh_sentiment`): aquí el
                   sentimiento que llega ya se asume vigente. Así esta matriz
                   sigue siendo agnóstica al tiempo y el backtest (que caduca a
                   escala de horas con max_news_age_hours) no se ve afectado.
        settings:  inyectable en tests; por defecto carga settings.yaml.
        as_of:     instante de evaluación. Fija `Decision.timestamp` de forma
                   determinista (misma entrada → misma salida). Por defecto, el
                   reloj actual. Inyectarlo hace la función realmente pura y
                   prepara el camino de eventos (Plan V2 Fase 2).

    Returns:
        Decision con action (LONG/SHORT/HOLD), size_factor en [0,1] y la regla
        de la matriz que disparó (campo `reason`, para auditoría).
    """
    cfg = (settings or load_settings()).confluence
    now = as_of or datetime.now(timezone.utc)
    quant = signal.score
    sent = sentiment.score if sentiment is not None else 0.0
    event_kind = sentiment.event_kind if sentiment is not None else "none"

    def _decision(action: Action, size_factor: float, reason: str) -> Decision:
        return Decision(
            symbol=signal.symbol,
            action=action,
            quant_score=quant,
            sentiment_score=sent,
            size_factor=size_factor,
            reason=reason,
            timestamp=now,
        )

    # (0) Macro PROGRAMADO de resultado incierto (FOMC, CPI): no abrir HACIA un
    #     dato que puede ir a cualquier lado. Solo los `scheduled` bloquean; tiene
    #     prioridad sobre todo. Un `shock` direccional (hack/ETF/depeg) ya NO
    #     bloquea: cae a la matriz normal (Plan V2 Fase 1.2, crítica #2 — corrige
    #     la lógica invertida del v1 que tiraba los eventos más operables). El
    #     refinamiento por ventana temporal (bloquear solo cerca del dato) y la
    #     ORIGINACIÓN por shock viven en el Fast Path (Fase 2).
    if event_kind == "scheduled":
        return _decision(Action.HOLD, 0.0, "scheduled_macro_block")

    # (1) Sin señal técnica fuerte no se abre. Esto encarna el circuit breaker
    #     (b): el sentimiento, por extremo que sea, no entra solo al mercado.
    if abs(quant) < cfg.quant_strong_threshold:
        return _decision(Action.HOLD, 0.0, "quant_weak")

    direction = Action.LONG if quant > 0 else Action.SHORT
    sent_aligned = _sign(sent) == _sign(quant)
    sent_significant = abs(sent) >= cfg.sentiment_confirm_threshold

    # (2) Sentimiento significativo y OPUESTO al quant → la noticia puede
    #     invalidar el patrón técnico. Mejor no operar (HOLD), no apostar a ciegas.
    if sent_significant and not sent_aligned:
        return _decision(Action.HOLD, 0.0, "sentiment_conflict")

    # Gate de cortos (config, no venue hardcodeado). En Futuros USD-M va activo
    # (allow_short=true) y los SHORT fluyen simétricos; ponerlo en false vuelve
    # el bot long-only sin tocar código.
    if direction == Action.SHORT and not cfg.allow_short:
        return _decision(Action.HOLD, 0.0, "short_disabled")

    # (3) Sentimiento confirma la dirección técnica → convicción plena.
    if sent_significant and sent_aligned:
        return _decision(direction, 1.0, "sentiment_confirms")

    # (4) Sentimiento neutro (o ausente) → operamos la técnica con tamaño
    #     reducido: hay señal de precio pero nadie la respalda con noticias.
    return _decision(direction, cfg.reduced_size_factor, "sentiment_neutral")


def decide_event(
    sentiment: SentimentScore,
    symbol: str,
    price_impulse_bps: float,
    settings: Settings | None = None,
    *,
    as_of: datetime | None = None,
    last_event_trade_at: datetime | None = None,
) -> Decision:
    """Fast Path: ¿este shock de noticia ORIGINA un trade? (Plan V2 §2.2).

    A diferencia de `decide` (Slow Path), aquí la NOTICIA origina y el quant NO es
    condición necesaria. Devuelve una `Decision` LONG/SHORT solo si se cumplen
    TODAS las puertas; si alguna falla, `HOLD` con la razón de la puerta (auditoría).

    Función pura: el orquestador inyecta el reloj (`as_of`), el impulso de precio
    ya medido sobre el buffer (`price_impulse_bps`, con signo, en bps) y el instante
    del último trade de evento del símbolo (`last_event_trade_at`) para el cooldown.
    El gate maestro `event.enabled` lo aplica el engine ANTES de llamar aquí (esto
    DECIDE, no opera): no se re-chequea.

    Args:
        sentiment:           score del shock (score, confidence, event_kind, analyzed_at).
        symbol:              símbolo ya resuelto desde symbol_scope (lo hace el engine, §2.3).
        price_impulse_bps:   movimiento del precio en confirm_window, con signo (bps).
        settings:            inyectable; por defecto carga settings.yaml.
        as_of:               instante de evaluación (TTL + cooldown). Por defecto ahora.
        last_event_trade_at: último trade de evento de este símbolo, o None.

    Returns:
        Decision con action LONG/SHORT (origina, size_factor=event.size_factor) o
        HOLD (size_factor=0) con la `reason` de la puerta que cerró el paso.
    """
    s = settings or load_settings()
    ev = s.event
    now = as_of or datetime.now(timezone.utc)
    score = sentiment.score

    def _hold(reason: str) -> Decision:
        return Decision(
            symbol=symbol, action=Action.HOLD, quant_score=0.0,
            sentiment_score=score, size_factor=0.0, reason=reason, timestamp=now,
        )

    # (0) Solo los SHOCK direccionales originan. `scheduled` (FOMC/CPI) y `none` no
    #     abren por el Fast Path: el shock tiene signo conocido, el macro no.
    if sentiment.event_kind != "shock":
        return _hold("event_not_shock")

    # (1) Frescura: un shock más viejo que el TTL de evento ya está descontado por
    #     el mercado. TTL propio (≈180s), distinto del TTL del Slow Path (≈300s).
    if (now - sentiment.analyzed_at).total_seconds() > ev.ttl_seconds:
        return _hold("event_stale")

    # (2) Cooldown por símbolo: un mismo suceso dispara titulares correlacionados en
    #     cadena; sin esto reentraríamos varias veces sobre la misma información.
    if (last_event_trade_at is not None
            and (now - last_event_trade_at).total_seconds() < ev.cooldown_seconds):
        return _hold("event_cooldown")

    # (3) Magnitud: el shock debe ser suficientemente fuerte para arriesgar.
    if abs(score) < ev.min_impact_score:
        return _hold("event_weak_score")

    # (4) Confianza: no originamos sobre un titular dudoso (parseo/ambigüedad).
    if sentiment.confidence < ev.min_confidence:
        return _hold("event_low_confidence")

    direction = Action.LONG if score > 0 else Action.SHORT

    # (5) Gate de cortos (config de venue, no hardcode). Aplica a TODA originación,
    #     igual que en el Slow Path: ponerlo en false vuelve el bot long-only.
    if direction == Action.SHORT and not s.confluence.allow_short:
        return _hold("short_disabled")

    # (6) Confirmación de impulso = núcleo legítimo del circuit breaker (b): el
    #     precio ya debe haberse movido EN LA DIRECCIÓN del shock y con suficiente
    #     magnitud. Con confirm_impulse_bps=0 el gate se DESACTIVA a propósito
    #     (ablación A/B de los kill criteria §B: ¿discrimina el gate o es ruido?).
    cap = ev.confirm_impulse_bps
    if cap > 0:
        aligned = _sign(price_impulse_bps) == _sign(score)
        strong = abs(price_impulse_bps) >= cap
        if not (aligned and strong):
            return _hold("event_no_impulse")

    # Origina: tamaño BASE de evento (más pequeño). El Risk Manager lo afina en modo
    # event (§2.4) y el contexto estratégico puede reducirlo más (§2.3).
    return Decision(
        symbol=symbol, action=direction, quant_score=0.0, sentiment_score=score,
        size_factor=ev.size_factor, reason=f"event_originate_{direction.value.lower()}",
        timestamp=now,
    )
