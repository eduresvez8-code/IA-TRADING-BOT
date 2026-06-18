"""Matriz de confluencia: cruza la señal técnica con el sentimiento → Decision.

Filosofía (PLAN_MAESTRO §1): el motor cuantitativo manda la DIRECCIÓN; el
sentimiento solo CONFIRMA (tamaño pleno), guarda silencio (tamaño reducido) o
VETA (la noticia contradice el patrón técnico). Nunca abrimos por sentimiento
solo: sin movimiento de precio que lo confirme, un titular extremo puede ser
falso o estar mal parseado (circuit breaker (b) del plan).

Es una función pura: misma entrada → misma salida, sin estado ni I/O. Así cada
fila de la matriz se valida aislada con un test de escenario.
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
    high_impact = sentiment.high_impact if sentiment is not None else False

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

    # (0) Evento de alto impacto pendiente (FOMC, CPI, hack) → bloqueo total de
    #     entradas, gane lo que gane la técnica. Tiene prioridad sobre todo.
    if high_impact:
        return _decision(Action.HOLD, 0.0, "high_impact_block")

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
