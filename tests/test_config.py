"""Tests de carga de configuración: el settings.yaml real del repo debe ser válido."""

import pytest
from pydantic import ValidationError

from src.core.config import (
    BacktestConfig,
    ExecutionConfig,
    RiskConfig,
    SentimentConfig,
    load_settings,
)


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
        max_leverage=3, max_portfolio_margin_pct=85.0,
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


def test_leverage_de_casino_es_rechazado():
    # 20x sería un typo en este bot (le=10): nada de apalancamiento de casino.
    with pytest.raises(ValidationError):
        RiskConfig(**_valid_risk_kwargs(max_leverage=20))


def test_leverage_cero_es_rechazado():
    with pytest.raises(ValidationError):
        RiskConfig(**_valid_risk_kwargs(max_leverage=0))


def test_margen_mayor_que_100_es_rechazado():
    # Comprometer >100% del wallet como margen no tiene sentido (le=100).
    with pytest.raises(ValidationError):
        RiskConfig(**_valid_risk_kwargs(max_portfolio_margin_pct=120.0))


def test_settings_yaml_risk_futuros():
    # El settings.yaml real del repo debe traer los campos de Futuros USD-M.
    s = load_settings()
    assert s.risk.take_profit_rr > 0
    assert 0.0 < s.risk.low_confidence_threshold < 1.0
    assert 0.0 < s.risk.low_confidence_size_factor <= 1.0
    assert s.risk.stale_feed_seconds > 0
    # Futuros: apalancamiento auto-limitado, margen agregado acotado, cortos ON.
    assert 1 <= s.risk.max_leverage <= 10
    assert 0.0 < s.risk.max_portfolio_margin_pct <= 100.0
    assert s.confluence.allow_short is True


def _valid_execution_kwargs(**overrides):
    base = dict(reconcile_position_tolerance=0.001, stop_working_type="MARK_PRICE")
    base.update(overrides)
    return base


def test_execution_config_valido():
    e = ExecutionConfig(**_valid_execution_kwargs())
    assert e.reconcile_position_tolerance == 0.001
    assert e.stop_working_type == "MARK_PRICE"


def test_working_type_invalido_es_rechazado():
    # Solo MARK_PRICE / CONTRACT_PRICE; un valor libre es un typo (Literal).
    with pytest.raises(ValidationError):
        ExecutionConfig(**_valid_execution_kwargs(stop_working_type="LAST_PRICE"))


def test_tolerancia_de_reconciliacion_fuera_de_rango():
    # 0 marcaría todo como discrepancia; ≥1 nunca detectaría una (0<tol<1).
    with pytest.raises(ValidationError):
        ExecutionConfig(**_valid_execution_kwargs(reconcile_position_tolerance=0.0))


def test_settings_yaml_execution():
    s = load_settings()
    assert 0.0 < s.execution.reconcile_position_tolerance < 1.0
    assert s.execution.stop_working_type in ("MARK_PRICE", "CONTRACT_PRICE")


def test_settings_yaml_orchestrator():
    s = load_settings()
    assert 20 <= s.orchestrator.warmup_candles <= 1000
    assert 1 <= s.orchestrator.reconcile_grace_cycles <= 20


def test_warmup_demasiado_corto_es_rechazado():
    # Un buffer < 20 velas no daría datos a los indicadores (ge=20).
    from src.core.config import OrchestratorConfig
    with pytest.raises(ValidationError):
        OrchestratorConfig(warmup_candles=5, reconcile_grace_cycles=3)


def test_gracia_cero_es_rechazada():
    # Una gracia de 0 dispararía el HALT a la primera observación (ge=1).
    from src.core.config import OrchestratorConfig
    with pytest.raises(ValidationError):
        OrchestratorConfig(warmup_candles=60, reconcile_grace_cycles=0)
