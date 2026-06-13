"""Tests de carga de configuración: el settings.yaml real del repo debe ser válido."""

import pytest
from pydantic import ValidationError

from src.core.config import BacktestConfig, RiskConfig, SentimentConfig, load_settings


def _valid_sentiment_kwargs(**overrides):
    base = dict(
        rss_feeds=["https://coindesk.com/rss"],
        poll_interval_seconds=120,
        claude_model="claude-haiku-4-5-20251001",
        heuristic_weight=0.7,
        escalate_score_threshold=0.3,
        max_news_age_hours=24,
    )
    base.update(overrides)
    return base


def test_settings_yaml_del_repo_es_valido():
    s = load_settings()
    assert "BTCUSDT" in s.market.symbols
    assert s.risk.risk_per_trade_pct <= 2.0
    assert len(s.sentiment.rss_feeds) >= 1
    assert s.backtest.initial_capital > 0
    # Sprint 4: nuevos campos de SentimentConfig
    assert 0.0 <= s.sentiment.heuristic_weight <= 1.0
    assert 0.0 < s.sentiment.escalate_score_threshold < 1.0
    assert s.sentiment.max_news_age_hours >= 1


def test_sentiment_config_valido():
    sc = SentimentConfig(**_valid_sentiment_kwargs())
    assert sc.heuristic_weight == 0.7
    assert sc.escalate_score_threshold == 0.3
    assert sc.max_news_age_hours == 24


def test_sentiment_heuristic_weight_fuera_de_rango():
    with pytest.raises(ValidationError):
        SentimentConfig(**_valid_sentiment_kwargs(heuristic_weight=1.5))


def test_sentiment_escalate_threshold_en_cero_es_rechazado():
    # Un threshold de 0 escalaría TODO a Claude — sin sentido y costoso.
    with pytest.raises(ValidationError):
        SentimentConfig(**_valid_sentiment_kwargs(escalate_score_threshold=0.0))


def test_sentiment_max_news_age_cero_es_rechazado():
    with pytest.raises(ValidationError):
        SentimentConfig(**_valid_sentiment_kwargs(max_news_age_hours=0))


def _valid_backtest_kwargs(**overrides):
    base = dict(
        initial_capital=10000.0, commission_pct=0.04, slippage_pct=0.02,
        slippage_atr_multiplier=0.1,
        entry_threshold=0.5, exit_threshold=0.1, take_profit_rr=2.0,
        allow_short=True,
    )
    base.update(overrides)
    return base


def test_backtest_config_valido():
    bt = BacktestConfig(**_valid_backtest_kwargs())
    assert bt.commission_pct == 0.04
    assert bt.slippage_atr_multiplier == 0.1


def test_slippage_multiplier_negativo_es_rechazado():
    # Un slippage negativo "pagaría" por operar — imposible (ge=0).
    with pytest.raises(ValidationError):
        BacktestConfig(**_valid_backtest_kwargs(slippage_atr_multiplier=-0.5))


def test_comision_absurda_es_rechazada():
    # commission_pct: 40 sería un 40% por lado — claramente un typo (le=1.0).
    with pytest.raises(ValidationError):
        BacktestConfig(**_valid_backtest_kwargs(commission_pct=40.0))


def test_exit_threshold_no_puede_superar_entry():
    # Salir con |score| ≥ el umbral de entrada cerraría en la misma vela.
    with pytest.raises(ValidationError):
        BacktestConfig(**_valid_backtest_kwargs(entry_threshold=0.3, exit_threshold=0.5))


def _valid_risk_kwargs(**overrides):
    base = dict(
        risk_per_trade_pct=1.0, max_open_positions=3,
        max_daily_loss_pct=3.0, max_drawdown_pct=10.0,
        atr_stop_multiplier=1.5, atr_period=14,
        take_profit_rr=2.0, low_confidence_threshold=0.4,
        low_confidence_size_factor=0.5, stale_feed_seconds=30,
    )
    base.update(overrides)
    return base


def test_riesgo_absurdo_es_rechazado():
    # 10% de riesgo por trade es un typo, no una estrategia.
    with pytest.raises(ValidationError):
        RiskConfig(**_valid_risk_kwargs(risk_per_trade_pct=10.0))


def test_risk_config_sprint5_valido():
    rc = RiskConfig(**_valid_risk_kwargs())
    assert rc.take_profit_rr == 2.0
    assert rc.low_confidence_threshold == 0.4
    assert rc.low_confidence_size_factor == 0.5
    assert rc.stale_feed_seconds == 30


def test_take_profit_rr_cero_es_rechazado():
    # Un RR de 0 pondría el take-profit en la propia entrada — sin sentido (gt=0).
    with pytest.raises(ValidationError):
        RiskConfig(**_valid_risk_kwargs(take_profit_rr=0.0))


def test_low_confidence_size_factor_fuera_de_rango():
    # Un factor > 1 AUMENTARÍA el tamaño con baja confianza — al revés (le=1.0).
    with pytest.raises(ValidationError):
        RiskConfig(**_valid_risk_kwargs(low_confidence_size_factor=1.5))


def test_stale_feed_seconds_cero_es_rechazado():
    # 0 s vetaría siempre (cualquier precio "ya es viejo"). gt=0.
    with pytest.raises(ValidationError):
        RiskConfig(**_valid_risk_kwargs(stale_feed_seconds=0))


def test_settings_yaml_risk_sprint5():
    # El settings.yaml real del repo debe traer los campos del Sprint 5.
    s = load_settings()
    assert s.risk.take_profit_rr > 0
    assert 0.0 < s.risk.low_confidence_threshold < 1.0
    assert 0.0 < s.risk.low_confidence_size_factor <= 1.0
    assert s.risk.stale_feed_seconds > 0
