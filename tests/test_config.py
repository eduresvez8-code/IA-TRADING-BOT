"""Tests de carga de configuración: el settings.yaml real del repo debe ser válido."""

import pytest
from pydantic import ValidationError

from src.core.config import (
    BacktestConfig,
    BreakoutConfig,
    ConfluenceConfig,
    CrossSectionalConfig,
    EdgeConfig,
    EventConfig,
    ExecutionConfig,
    FundingEdgeConfig,
    MeanReversionConfig,
    QuantHypothesesConfig,
    QuantMatrixConfig,
    RiskConfig,
    ScanConfig,
    SentimentConfig,
    load_settings,
)


def _valid_sentiment_kwargs(**overrides):
    base = dict(
        enabled=False,
        rss_feeds=["https://coindesk.com/rss"],
        poll_interval_seconds=120,
        fetch_timeout_seconds=10,
        claude_model="claude-haiku-4-5-20251001",
        heuristic_weight=0.7,
        escalate_score_threshold=0.3,
        max_news_age_hours=24,
    )
    base.update(overrides)
    return base


def test_market_quote_assets_requerido_no_vacio():
    # quote_assets gobierna la derivación del activo base al resolver scope de
    # noticias (src/core/scope.py). min_length=1: una lista vacía no permitiría
    # macher 'BTC'→'BTCUSDT' y dejaría la DEUDA_TICKER sin arreglar.
    from src.core.config import MarketConfig
    with pytest.raises(ValidationError):
        MarketConfig(symbols=["BTCUSDT"], timeframe="5m", htf_timeframe="1h", quote_assets=[])
    mc = MarketConfig(symbols=["BTCUSDT"], timeframe="5m", htf_timeframe="1h", quote_assets=["USDT"])
    assert mc.quote_assets == ["USDT"]


def test_settings_yaml_del_repo_es_valido():
    s = load_settings()
    assert "BTCUSDT" in s.market.symbols
    assert s.market.quote_assets == ["USDT"]
    assert s.risk.risk_per_trade_pct <= 2.0
    assert len(s.sentiment.rss_feeds) >= 1
    assert s.backtest.initial_capital > 0
    # Sprint 4: nuevos campos de SentimentConfig
    assert 0.0 <= s.sentiment.heuristic_weight <= 1.0
    assert 0.0 < s.sentiment.escalate_score_threshold < 1.0
    assert s.sentiment.max_news_age_hours >= 1
    # Gate de sentimiento: booleano válido (True en testnet live, False por defecto).
    assert isinstance(s.sentiment.enabled, bool)
    assert 5 <= s.sentiment.fetch_timeout_seconds <= 60


def test_sentiment_config_valido():
    sc = SentimentConfig(**_valid_sentiment_kwargs())
    assert sc.enabled is False
    assert sc.heuristic_weight == 0.7
    assert sc.escalate_score_threshold == 0.3
    assert sc.max_news_age_hours == 24


def test_sentiment_enabled_es_obligatorio():
    # Sin `enabled` explícito, la config no valida: el gate no puede quedar implícito
    # (un descuido no debe encender el overlay de Claude por omisión).
    kwargs = _valid_sentiment_kwargs()
    del kwargs["enabled"]
    with pytest.raises(ValidationError):
        SentimentConfig(**kwargs)


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


def _valid_confluence_kwargs(**overrides):
    base = dict(
        quant_veto_threshold=0.15, quant_confirm_threshold=0.35,
        sentiment_confirm_threshold=0.3,
        reduced_size_factor=0.5, allow_short=True, sentiment_ttl_seconds=300,
        quant_regime_enabled=True,
    )
    base.update(overrides)
    return base


def test_confluence_confirm_debe_ser_ge_veto():
    # El confirm (ofensivo) no puede ser más laxo que el veto (defensivo): si se
    # invierte, un régimen débil subiría a tamaño pleno y uno fuerte no vetaría.
    from src.core.config import ConfluenceConfig
    with pytest.raises(ValidationError):
        ConfluenceConfig(**_valid_confluence_kwargs(
            quant_veto_threshold=0.40, quant_confirm_threshold=0.20))


def test_confluence_confirm_igual_a_veto_es_valido():
    # Frontera: confirm == veto es válido (umbral simétrico clásico, sin zona muerta).
    from src.core.config import ConfluenceConfig
    cfg = ConfluenceConfig(**_valid_confluence_kwargs(
        quant_veto_threshold=0.3, quant_confirm_threshold=0.3))
    assert cfg.quant_confirm_threshold == cfg.quant_veto_threshold


def _valid_quant_matrix_kwargs(**overrides):
    base = dict(
        taker_commission_pct=0.05,
        carry_capital_multiplier=2.0,
        carry_maintenance_bps_per_period=0.0,
        golden_min_tstat=2.0,
        golden_min_profit_factor=1.15,
        pairs_lookback_hours=720,
        pairs_z_entry=2.0,
        pairs_z_exit=0.5,
        slippage_pct=0.02,
        slippage_atr_mult=0.1,
        atr_period=14,
        vwap_z_window=288,
        vwap_z_entry=2.0,
        vwap_z_exit=0.5,
        vwap_forward_horizon=6,
        squeeze_bb_period=20,
        squeeze_bb_std=2.0,
        squeeze_keltner_atr_mult=1.5,
        squeeze_forward_horizon=4,
        squeeze_breakout_threshold=1.0,
    )
    base.update(overrides)
    return base


def test_quant_matrix_config_del_repo_es_valido():
    qm = load_settings().quant_matrix
    assert qm.taker_commission_pct == 0.05
    assert qm.carry_capital_multiplier >= 1.0
    assert qm.golden_min_profit_factor > 1.0


def test_quant_matrix_capital_multiplier_minimo_es_uno():
    # No puedes desplegar menos capital que el notional spot que compras (ge=1.0).
    with pytest.raises(ValidationError):
        QuantMatrixConfig(**_valid_quant_matrix_kwargs(carry_capital_multiplier=0.5))


def test_quant_matrix_pf_debe_superar_uno():
    # Un PF de corte ≤ 1 no exigiría rentabilidad (gt=1.0).
    with pytest.raises(ValidationError):
        QuantMatrixConfig(**_valid_quant_matrix_kwargs(golden_min_profit_factor=1.0))


def _valid_quant_hyp_kwargs(**overrides):
    base = dict(
        atr_stop_mult=2.0,
        tsmom_lookback_days_grid=[126, 252],
        funding_extreme_neg_ann_pct=-20.0,
        funding_extreme_pos_ann_pct=40.0,
        funding_normal_low_ann_pct=-5.0,
        funding_normal_high_ann_pct=10.0,
        funding_trend_ma_days=200,
        donchian_entry_period=20,
        donchian_exit_ema=10,
        donchian_funding_min_8h_pct=0.01,
        donchian_funding_max_8h_pct=0.05,
        donchian_take_profit_rr=3.0,
        donchian_max_hold_bars=30,
        ma_cross_pairs=[[9, 21], [50, 200]],
        ma_cross_types=["ema", "sma"],
        ma_cross_timeframes=["1h", "4h", "1d"],
        ma_cross_allow_short=True,
        leadlag_leader="BTCUSDT",
        leadlag_target_assets=["ETHUSDT", "SOLUSDT"],
        leadlag_lag_hours_grid=[6, 24, 72],
        leadlag_regime_sma=200,
        leadlag_allow_short=True,
        seasonality_entry_open_hour_utc=22,
        seasonality_hold_hours=2,
        dow_entry_weekday=0,
        dow_hold_days=1,
        rsi_reversion_period=10,
        rsi_reversion_oversold=30.0,
        rsi_reversion_overbought=70.0,
        rsi_reversion_trend_sma=200,
    )
    base.update(overrides)
    return base


def test_leadlag_lag_fuera_de_rango_es_rechazado():
    # Un lag de 0h (o >336h) no tiene sentido para el lead-lag (validador de campo).
    with pytest.raises(ValidationError):
        QuantHypothesesConfig(**_valid_quant_hyp_kwargs(leadlag_lag_hours_grid=[0]))
    with pytest.raises(ValidationError):
        QuantHypothesesConfig(**_valid_quant_hyp_kwargs(leadlag_lag_hours_grid=[500]))


def test_quant_hypotheses_config_del_repo_es_valido():
    qh = load_settings().quant_hypotheses
    assert qh.atr_stop_mult == 2.0
    # Grid de robustez: no vacío y dentro del rango válido (el contenido exacto puede
    # variar al explorar; lo importante es que cargue y respete los límites).
    assert len(qh.tsmom_lookback_days_grid) >= 1
    assert all(10 <= d <= 500 for d in qh.tsmom_lookback_days_grid)
    assert qh.funding_extreme_pos_ann_pct > qh.funding_extreme_neg_ann_pct


def test_quant_hyp_lookback_fuera_de_rango_es_rechazado():
    # Un lookback de 5 días no es "momentum", es ruido (rango [10, 500]).
    with pytest.raises(ValidationError):
        QuantHypothesesConfig(**_valid_quant_hyp_kwargs(tsmom_lookback_days_grid=[5]))


def test_quant_hyp_grid_no_puede_estar_vacio():
    with pytest.raises(ValidationError):
        QuantHypothesesConfig(**_valid_quant_hyp_kwargs(tsmom_lookback_days_grid=[]))


def test_quant_hyp_funding_extremo_pos_debe_superar_neg():
    # El umbral positivo de euforia debe estar por encima del negativo de capitulación.
    with pytest.raises(ValidationError):
        QuantHypothesesConfig(**_valid_quant_hyp_kwargs(
            funding_extreme_neg_ann_pct=-20.0, funding_extreme_pos_ann_pct=-30.0))


def test_quant_hyp_donchian_funding_band_coherente():
    # max de la banda de funding debe ser > min.
    with pytest.raises(ValidationError):
        QuantHypothesesConfig(**_valid_quant_hyp_kwargs(
            donchian_funding_min_8h_pct=0.05, donchian_funding_max_8h_pct=0.01))


def test_quant_hyp_exit_ema_menor_que_canal_de_entrada():
    # La EMA de salida debe ser más rápida que el canal Donchian de entrada.
    with pytest.raises(ValidationError):
        QuantHypothesesConfig(**_valid_quant_hyp_kwargs(
            donchian_entry_period=10, donchian_exit_ema=20))


def test_quant_hyp_ma_cross_del_repo_es_valido():
    qh = load_settings().quant_hypotheses
    assert len(qh.ma_cross_pairs) >= 1
    assert all(len(p) == 2 and p[0] < p[1] for p in qh.ma_cross_pairs)
    assert set(qh.ma_cross_types) <= {"ema", "sma"}
    assert set(qh.ma_cross_timeframes) <= {"1h", "4h", "1d"}


def test_quant_hyp_ma_cross_par_invertido_es_rechazado():
    # fast >= slow no es un cruce (par mal formado).
    with pytest.raises(ValidationError):
        QuantHypothesesConfig(**_valid_quant_hyp_kwargs(ma_cross_pairs=[[50, 20]]))


def test_quant_hyp_ma_cross_tipo_invalido_es_rechazado():
    with pytest.raises(ValidationError):
        QuantHypothesesConfig(**_valid_quant_hyp_kwargs(ma_cross_types=["wma"]))


def test_quant_hyp_ma_cross_timeframe_sub_hora_es_rechazado():
    # Nada por debajo de 1h: foros y nuestro hallazgo coinciden (ruido + costo).
    with pytest.raises(ValidationError):
        QuantHypothesesConfig(**_valid_quant_hyp_kwargs(ma_cross_timeframes=["5m"]))


def test_pairs_lookback_minimo_24h():
    # Menos de 1 día de lookback no tiene sentido para estimar OLS y z-score.
    with pytest.raises(ValidationError):
        QuantMatrixConfig(**_valid_quant_matrix_kwargs(pairs_lookback_hours=23))


def test_pairs_z_exit_debe_ser_menor_que_z_entry():
    # Si z_exit >= z_entry, entraríamos y saldríamos en el mismo bar.
    with pytest.raises(ValidationError):
        QuantMatrixConfig(**_valid_quant_matrix_kwargs(pairs_z_entry=2.0, pairs_z_exit=2.0))


def test_pairs_z_exit_igual_a_z_entry_tambien_falla():
    with pytest.raises(ValidationError):
        QuantMatrixConfig(**_valid_quant_matrix_kwargs(pairs_z_entry=1.5, pairs_z_exit=1.5))


def test_pairs_config_del_repo_es_valido():
    qm = load_settings().quant_matrix
    assert qm.pairs_lookback_hours == 720
    assert qm.pairs_z_exit < qm.pairs_z_entry


def test_vwap_z_exit_debe_ser_menor_que_z_entry():
    # Si z_exit >= z_entry, abriría y cerraría en la misma barra.
    with pytest.raises(ValidationError):
        QuantMatrixConfig(**_valid_quant_matrix_kwargs(vwap_z_entry=2.0, vwap_z_exit=2.0))


def test_vwap_z_window_minimo_una_hora():
    # Menos de 12 barras de 5m (1h) no da media/desv estables (ge=12).
    with pytest.raises(ValidationError):
        QuantMatrixConfig(**_valid_quant_matrix_kwargs(vwap_z_window=11))


def test_slippage_atr_mult_absurdo_es_rechazado():
    # Un k enorme inflaría el slippage sin sentido (le=5.0).
    with pytest.raises(ValidationError):
        QuantMatrixConfig(**_valid_quant_matrix_kwargs(slippage_atr_mult=10.0))


def test_vwap_config_del_repo_es_valido():
    qm = load_settings().quant_matrix
    assert qm.vwap_z_window == 288
    assert qm.vwap_z_exit < qm.vwap_z_entry
    assert qm.atr_period == 14


# --- Familia D — squeeze de volatilidad ---

def test_squeeze_bb_period_minimo():
    # Menos de 10 barras da σ/ATR demasiado ruidosos para hablar de régimen (ge=10).
    with pytest.raises(ValidationError):
        QuantMatrixConfig(**_valid_quant_matrix_kwargs(squeeze_bb_period=9))


def test_squeeze_bb_std_fuera_de_rango_falla():
    # Por debajo de 1σ la banda es tan estrecha que casi todo cierre la rompe (ge=1.0).
    with pytest.raises(ValidationError):
        QuantMatrixConfig(**_valid_quant_matrix_kwargs(squeeze_bb_std=0.5))
    # Más de 4σ casi nunca se rompe (le=4.0).
    with pytest.raises(ValidationError):
        QuantMatrixConfig(**_valid_quant_matrix_kwargs(squeeze_bb_std=5.0))


def test_squeeze_keltner_mult_fuera_de_rango_falla():
    with pytest.raises(ValidationError):
        QuantMatrixConfig(**_valid_quant_matrix_kwargs(squeeze_keltner_atr_mult=0.5))


def test_squeeze_breakout_threshold_fuera_de_rango_falla():
    # Por debajo de media banda no es ruptura (ge=0.5); más de 3 anchos no dispara (le=3.0).
    with pytest.raises(ValidationError):
        QuantMatrixConfig(**_valid_quant_matrix_kwargs(squeeze_breakout_threshold=0.4))
    with pytest.raises(ValidationError):
        QuantMatrixConfig(**_valid_quant_matrix_kwargs(squeeze_breakout_threshold=3.5))


def test_squeeze_forward_horizon_minimo_uno():
    with pytest.raises(ValidationError):
        QuantMatrixConfig(**_valid_quant_matrix_kwargs(squeeze_forward_horizon=0))


def test_squeeze_config_del_repo_es_valido():
    qm = load_settings().quant_matrix
    assert qm.squeeze_bb_period == 20
    assert qm.squeeze_bb_std == 2.0
    assert qm.squeeze_keltner_atr_mult == 1.5
    assert qm.squeeze_forward_horizon == 4
    assert qm.squeeze_breakout_threshold == 1.0


def test_confluence_config_valido():
    c = ConfluenceConfig(**_valid_confluence_kwargs())
    assert c.sentiment_ttl_seconds == 300
    assert c.allow_short is True


def test_confluence_ttl_cero_es_rechazado():
    # Un TTL de 0 caducaría el sentimiento al instante: nunca se usaría (ge=1).
    with pytest.raises(ValidationError):
        ConfluenceConfig(**_valid_confluence_kwargs(sentiment_ttl_seconds=0))


def test_confluence_ttl_absurdo_es_rechazado():
    # Más de un día no es "noticia fresca": un valor enorme es un typo (le=86400).
    with pytest.raises(ValidationError):
        ConfluenceConfig(**_valid_confluence_kwargs(sentiment_ttl_seconds=200_000))


def test_settings_yaml_confluence():
    s = load_settings()
    assert 0.0 < s.confluence.quant_veto_threshold < 1.0
    assert 0.0 < s.confluence.quant_confirm_threshold < 1.0
    assert s.confluence.quant_confirm_threshold >= s.confluence.quant_veto_threshold
    assert s.confluence.allow_short is True
    assert 1 <= s.confluence.sentiment_ttl_seconds <= 86400


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
        max_same_direction_positions=3,
        max_daily_loss_pct=3.0, max_drawdown_pct=10.0,
        atr_stop_multiplier=1.5, atr_period=14,
        take_profit_rr=2.0, min_confidence_to_trade=0.5,
        low_confidence_threshold=0.7,
        low_confidence_size_factor=0.5, stale_feed_seconds=30,
        stale_feed_intervals=2.0,
        max_leverage=3, max_portfolio_margin_pct=85.0,
        event_risk_per_trade_pct=0.5, event_atr_stop_multiplier=2.5,
        vol_regime_lookback=20, vol_expansion_cap=2.0,
    )
    base.update(overrides)
    return base


def test_riesgo_absurdo_es_rechazado():
    # 10% de riesgo por trade es un typo, no una estrategia.
    with pytest.raises(ValidationError):
        RiskConfig(**_valid_risk_kwargs(risk_per_trade_pct=10.0))


def test_cap_direccional_no_puede_superar_max_open_positions():
    # Capar a 3 con un máximo de 2 abiertas no tiene sentido (validador cruzado).
    with pytest.raises(ValidationError):
        RiskConfig(**_valid_risk_kwargs(
            max_open_positions=2, max_same_direction_positions=3))


def test_time_stop_del_repo_y_rango():
    # El settings.yaml del repo carga el time-stop; un negativo se rechaza (ge=0).
    from src.core.config import OrchestratorConfig
    o = load_settings().orchestrator
    assert o.max_position_hold_candles >= 0
    data = o.model_dump()
    data["max_position_hold_candles"] = -1   # aísla la restricción del resto de campos
    with pytest.raises(ValidationError):
        OrchestratorConfig(**data)


def test_risk_config_sprint5_valido():
    rc = RiskConfig(**_valid_risk_kwargs())
    assert rc.take_profit_rr == 2.0
    assert rc.min_confidence_to_trade == 0.5
    assert rc.low_confidence_threshold == 0.7
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


def test_piso_de_confianza_no_puede_superar_el_techo():
    # min_confidence_to_trade > low_confidence_threshold dejaría la banda reducida
    # vacía/invertida (validador cruzado).
    with pytest.raises(ValidationError):
        RiskConfig(**_valid_risk_kwargs(
            min_confidence_to_trade=0.8, low_confidence_threshold=0.7))


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
    assert s.risk.stale_feed_intervals > 0
    # Futuros: apalancamiento auto-limitado, margen agregado acotado, cortos ON.
    assert 1 <= s.risk.max_leverage <= 10
    assert 0.0 < s.risk.max_portfolio_margin_pct <= 100.0
    assert s.confluence.allow_short is True


# ----- RiskConfig: sizing de evento (Fase 2.4) -----

def test_risk_event_sizing_config_valido():
    rc = RiskConfig(**_valid_risk_kwargs())
    assert rc.event_risk_per_trade_pct == 0.5
    assert rc.event_atr_stop_multiplier == 2.5
    assert rc.vol_regime_lookback == 20
    assert rc.vol_expansion_cap == 2.0


def test_event_risk_pct_fuera_de_rango_es_rechazado():
    # gt=0 y le=2.0.
    with pytest.raises(ValidationError):
        RiskConfig(**_valid_risk_kwargs(event_risk_per_trade_pct=0.0))
    with pytest.raises(ValidationError):
        RiskConfig(**_valid_risk_kwargs(event_risk_per_trade_pct=3.0))


def test_event_atr_stop_fuera_de_rango_es_rechazado():
    # gt=0 y le=10.
    with pytest.raises(ValidationError):
        RiskConfig(**_valid_risk_kwargs(event_atr_stop_multiplier=0.0))
    with pytest.raises(ValidationError):
        RiskConfig(**_valid_risk_kwargs(event_atr_stop_multiplier=15.0))


def test_vol_regime_lookback_fuera_de_rango_es_rechazado():
    # ge=2 y le=500.
    with pytest.raises(ValidationError):
        RiskConfig(**_valid_risk_kwargs(vol_regime_lookback=1))
    with pytest.raises(ValidationError):
        RiskConfig(**_valid_risk_kwargs(vol_regime_lookback=600))


def test_vol_expansion_cap_uno_es_rechazado():
    # gt=1.0: un cap de 1.0 recortaría ante cualquier expansión mínima (sin holgura).
    with pytest.raises(ValidationError):
        RiskConfig(**_valid_risk_kwargs(vol_expansion_cap=1.0))


def test_vol_expansion_cap_absurdo_es_rechazado():
    # le=10: un cap de 50x no tiene sentido operativo.
    with pytest.raises(ValidationError):
        RiskConfig(**_valid_risk_kwargs(vol_expansion_cap=50.0))


def test_event_risk_mayor_que_base_es_rechazado():
    # event_risk_per_trade_pct debe ser ≤ risk_per_trade_pct: el evento no puede
    # arriesgar más que el Slow Path (sería una ampliación, no reducción).
    with pytest.raises(ValidationError):
        RiskConfig(**_valid_risk_kwargs(risk_per_trade_pct=0.5, event_risk_per_trade_pct=1.0))


def test_event_stop_menor_que_base_es_rechazado():
    # event_atr_stop_multiplier debe ser ≥ atr_stop_multiplier: el stop de evento
    # no puede ser más estrecho que el del Slow Path (se llenaría de ruido post-noticia).
    with pytest.raises(ValidationError):
        RiskConfig(**_valid_risk_kwargs(atr_stop_multiplier=3.0, event_atr_stop_multiplier=2.0))


def test_settings_yaml_risk_evento():
    s = load_settings()
    r = s.risk
    assert 0.0 < r.event_risk_per_trade_pct <= r.risk_per_trade_pct
    assert r.event_atr_stop_multiplier >= r.atr_stop_multiplier
    assert 2 <= r.vol_regime_lookback <= 500
    assert r.vol_expansion_cap > 1.0


def _valid_execution_kwargs(**overrides):
    base = dict(reconcile_position_tolerance=0.001, stop_working_type="MARK_PRICE",
                fill_confirm_retries=5, fill_confirm_delay_seconds=0.3,
                slippage_cap_bps=10, aggressive_entry_tif="IOC")
    base.update(overrides)
    return base


def test_execution_config_valido():
    e = ExecutionConfig(**_valid_execution_kwargs())
    assert e.reconcile_position_tolerance == 0.001
    assert e.stop_working_type == "MARK_PRICE"
    assert e.slippage_cap_bps == 10
    assert e.aggressive_entry_tif == "IOC"


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
    assert s.execution.fill_confirm_retries >= 1
    assert s.execution.fill_confirm_delay_seconds > 0
    # Fase 1.3: tope de slippage y modo de entrada
    assert 0.0 < s.execution.slippage_cap_bps <= 100
    assert s.execution.aggressive_entry_tif in ("IOC", "GTC")


def test_fill_confirm_retries_cero_es_rechazado():
    with pytest.raises(ValidationError):
        ExecutionConfig(**_valid_execution_kwargs(fill_confirm_retries=0))


def test_slippage_cap_cero_es_rechazado():
    # 0 bps = sin límite de precio: equivalente a MARKET (gt=0).
    with pytest.raises(ValidationError):
        ExecutionConfig(**_valid_execution_kwargs(slippage_cap_bps=0))


def test_slippage_cap_absurdo_es_rechazado():
    # 100 bps = 1% es el máximo; más es un typo que inflaría el precio límite
    # hasta permitir entradas muy fuera del mercado (le=100).
    with pytest.raises(ValidationError):
        ExecutionConfig(**_valid_execution_kwargs(slippage_cap_bps=150))


def test_aggressive_tif_invalido_es_rechazado():
    # Solo "IOC" y "GTC" son válidos; cualquier otro valor es un typo (Literal).
    with pytest.raises(ValidationError):
        ExecutionConfig(**_valid_execution_kwargs(aggressive_entry_tif="FOK"))


def test_edge_config_valido():
    e = EdgeConfig(forward_horizons=[1, 4, 12, 24], n_quantiles=5)
    assert e.forward_horizons == [1, 4, 12, 24]
    assert e.n_quantiles == 5


def test_edge_horizons_vacio_es_rechazado():
    # Sin horizontes no hay nada que medir.
    with pytest.raises(ValidationError):
        EdgeConfig(forward_horizons=[], n_quantiles=5)


def test_edge_horizonte_cero_es_rechazado():
    # Un horizonte de 0 velas no mira al futuro (cada horizonte ≥ 1).
    with pytest.raises(ValidationError):
        EdgeConfig(forward_horizons=[0, 4], n_quantiles=5)


def test_edge_un_solo_cuantil_es_rechazado():
    # Un solo cubo no discrimina la señal (ge=2).
    with pytest.raises(ValidationError):
        EdgeConfig(forward_horizons=[1], n_quantiles=1)


def test_settings_yaml_edge():
    s = load_settings()
    assert len(s.edge.forward_horizons) >= 1
    assert all(h >= 1 for h in s.edge.forward_horizons)
    assert 2 <= s.edge.n_quantiles <= 20


def test_scan_config_valido():
    sc = ScanConfig(symbols=["BTCUSDT"], history_days=1095,
                    walk_forward_folds=4, edge_profit_factor_min=1.15)
    assert sc.symbols == ["BTCUSDT"]
    assert sc.edge_profit_factor_min == 1.15


def test_scan_symbols_vacio_es_rechazado():
    with pytest.raises(ValidationError):
        ScanConfig(symbols=[], history_days=1095,
                   walk_forward_folds=4, edge_profit_factor_min=1.15)


def test_scan_pf_min_no_mayor_que_uno_es_rechazado():
    # Un PF≤1 ya es "sin edge"; el umbral de edge debe ser >1 (gt=1.0).
    with pytest.raises(ValidationError):
        ScanConfig(symbols=["BTCUSDT"], history_days=1095,
                   walk_forward_folds=4, edge_profit_factor_min=1.0)


def test_scan_un_solo_tramo_es_rechazado():
    # Un solo fold no testea consistencia entre periodos (ge=2).
    with pytest.raises(ValidationError):
        ScanConfig(symbols=["BTCUSDT"], history_days=1095,
                   walk_forward_folds=1, edge_profit_factor_min=1.15)


def test_mean_reversion_config_valido():
    mr = MeanReversionConfig(bb_period=20, bb_num_std=2.0,
                             rsi_oversold=30.0, rsi_overbought=70.0)
    assert mr.bb_period == 20 and mr.bb_num_std == 2.0


def test_mean_reversion_overbought_bajo_oversold_es_rechazado():
    # Si la zona de sobrecompra ≤ la de sobreventa, comprar y vender se cruzan.
    with pytest.raises(ValidationError):
        MeanReversionConfig(bb_period=20, bb_num_std=2.0,
                            rsi_oversold=70.0, rsi_overbought=30.0)


def test_mean_reversion_bandas_degeneradas_es_rechazado():
    with pytest.raises(ValidationError):
        MeanReversionConfig(bb_period=20, bb_num_std=0.0,
                            rsi_oversold=30.0, rsi_overbought=70.0)


def _valid_breakout_kwargs(**overrides):
    base = dict(donchian_period=20, exit_donchian_period=10, volume_ma_period=20,
                volume_multiplier=1.0, atr_filter_period=20, atr_expansion_mult=1.0)
    base.update(overrides)
    return base


def test_breakout_config_valido():
    bo = BreakoutConfig(**_valid_breakout_kwargs())
    assert bo.donchian_period == 20 and bo.exit_donchian_period == 10
    assert bo.atr_expansion_mult == 1.0


def test_breakout_periodo_demasiado_corto_es_rechazado():
    with pytest.raises(ValidationError):
        BreakoutConfig(**_valid_breakout_kwargs(donchian_period=1))


def test_breakout_salida_mas_ancha_que_entrada_es_rechazada():
    # Un canal de salida más ancho que el de entrada no tiene sentido (Turtle: M<N).
    with pytest.raises(ValidationError):
        BreakoutConfig(**_valid_breakout_kwargs(donchian_period=20, exit_donchian_period=30))


def test_settings_yaml_laboratorio_estrategia():
    s = load_settings()
    assert len(s.scan.symbols) == 5
    assert "SOLUSDT" in s.scan.symbols
    assert s.scan.edge_profit_factor_min > 1.0
    assert s.mean_reversion.rsi_overbought > s.mean_reversion.rsi_oversold
    assert s.breakout.donchian_period >= 2


def test_funding_edge_config_valido():
    fe = FundingEdgeConfig(premium_interval="1h",
                           forward_horizons_hours=[8, 24, 72, 168], n_quantiles=5)
    assert fe.premium_interval == "1h"
    assert fe.forward_horizons_hours == [8, 24, 72, 168]


def test_funding_edge_horizontes_vacios_es_rechazado():
    with pytest.raises(ValidationError):
        FundingEdgeConfig(premium_interval="1h", forward_horizons_hours=[], n_quantiles=5)


def test_funding_edge_horizonte_cero_es_rechazado():
    with pytest.raises(ValidationError):
        FundingEdgeConfig(premium_interval="1h", forward_horizons_hours=[0, 8], n_quantiles=5)


def _valid_xs_kwargs(**overrides):
    base = dict(history_days=1100, min_history_days=60, momentum_lookback_days=30,
                momentum_skip_days=0, vol_adjust=False, vol_lookback_days=30,
                forward_days=7, rebalance_days=7, n_quantiles=5, min_assets=10,
                liquidity_drop_pct=0.25, winsorize_quantile=0.02, max_weight=0.10,
                xs_liquid_lookback_days_grid=[3, 7, 14, 30],
                xs_liquid_n_side_grid=[1, 2])
    base.update(overrides)
    return base


def test_cross_sectional_config_valido():
    x = CrossSectionalConfig(**_valid_xs_kwargs())
    assert x.momentum_lookback_days == 30 and x.forward_days == 7


def test_cross_sectional_min_assets_uno_es_rechazado():
    # Una cross-section de 1 activo no rankea nada (ge=2).
    with pytest.raises(ValidationError):
        CrossSectionalConfig(**_valid_xs_kwargs(min_assets=1))


def test_cross_sectional_lookback_cero_es_rechazado():
    with pytest.raises(ValidationError):
        CrossSectionalConfig(**_valid_xs_kwargs(momentum_lookback_days=0))


def test_cross_sectional_max_weight_invalido_es_rechazado():
    # Un peso máximo > 1 (más del 100% en un activo) no tiene sentido.
    with pytest.raises(ValidationError):
        CrossSectionalConfig(**_valid_xs_kwargs(max_weight=1.5))


def test_cross_sectional_winsorize_fuera_de_rango_es_rechazado():
    # winsorize_quantile ≥ 0.5 recortaría todo contra la mediana.
    with pytest.raises(ValidationError):
        CrossSectionalConfig(**_valid_xs_kwargs(winsorize_quantile=0.5))


def test_xs_liquid_n_side_fuera_de_rango_es_rechazado():
    # Con 5 activos, ir 3 por lado (6 posiciones) no cabe: n_side ∈ [1,2].
    with pytest.raises(ValidationError):
        CrossSectionalConfig(**_valid_xs_kwargs(xs_liquid_n_side_grid=[3]))


def test_settings_yaml_portfolio_robustez():
    s = load_settings()
    assert 0.0 <= s.cross_sectional.liquidity_drop_pct < 1.0
    assert 0.0 <= s.cross_sectional.winsorize_quantile < 0.5
    assert 0.0 < s.cross_sectional.max_weight <= 1.0


def test_sentiment_regime_config_valido():
    from src.core.config import SentimentRegimeConfig
    sr = SentimentRegimeConfig(ext_fear_below=25, fear_below=45, greed_above=55,
                               ext_greed_above=75, forward_days=7, mr_lookback_days=5,
                               extreme_abs_threshold=25, vol_scale_min=0.3)
    assert sr.greed_above == 55


def test_sentiment_regime_umbrales_desordenados_es_rechazado():
    from src.core.config import SentimentRegimeConfig
    with pytest.raises(ValidationError):
        SentimentRegimeConfig(ext_fear_below=50, fear_below=45, greed_above=55,
                              ext_greed_above=75, forward_days=7, mr_lookback_days=5,
                              extreme_abs_threshold=25, vol_scale_min=0.3)


def test_settings_yaml_sentiment_regime():
    s = load_settings()
    sr = s.sentiment_regime
    assert sr.ext_fear_below < sr.fear_below < sr.greed_above < sr.ext_greed_above
    assert sr.forward_days >= 1 and 0 < sr.vol_scale_min <= 1.0


def test_settings_yaml_cross_sectional():
    s = load_settings()
    assert s.cross_sectional.momentum_lookback_days >= 1
    assert s.cross_sectional.forward_days >= 1
    assert s.cross_sectional.min_assets >= 2
    assert s.storage.universe_dir


def test_settings_yaml_funding_edge_y_storage():
    s = load_settings()
    assert s.storage.funding_dir  # ruta de almacenamiento de funding/basis
    assert s.funding_edge.premium_interval in ("1h", "5m", "15m", "4h")
    assert len(s.funding_edge.forward_horizons_hours) >= 1
    assert all(h >= 1 for h in s.funding_edge.forward_horizons_hours)


def test_settings_yaml_orchestrator():
    s = load_settings()
    assert 20 <= s.orchestrator.warmup_candles <= 1000
    assert 1 <= s.orchestrator.reconcile_grace_cycles <= 20
    # Régimen HTF (Opción 2): debe cubrir el mínimo del quant (ema_slow + rsi).
    assert s.orchestrator.regime_htf_bars >= s.quant.ema_slow_period + s.quant.rsi_period
    # Sondeo de PnL realizado (observabilidad del dashboard).
    assert 10 <= s.orchestrator.realized_pnl_poll_seconds <= 3600


def test_realized_pnl_poll_fuera_de_rango_es_rechazado():
    from src.core.config import OrchestratorConfig
    with pytest.raises(ValidationError):
        OrchestratorConfig(warmup_candles=60, reconcile_grace_cycles=3,
                           regime_htf_bars=50, realized_pnl_poll_seconds=5)


def test_clock_retry_delay_en_settings_y_en_rango():
    # Espera tras -1021 (reloj desincronizado) en el sondeo de PnL.
    s = load_settings()
    assert 0 < s.orchestrator.clock_retry_delay_seconds <= 300


def test_clock_retry_delay_fuera_de_rango_es_rechazado():
    from src.core.config import OrchestratorConfig
    with pytest.raises(ValidationError):
        OrchestratorConfig(warmup_candles=60, reconcile_grace_cycles=3,
                           regime_htf_bars=50, clock_retry_delay_seconds=0)


def test_settings_yaml_universo_sin_doge():
    # Universo (2026-06-26): DOGE retirado (meme de cola gorda, peor fricción, nunca
    # validado en la matriz quant). Quedan 5 perps USD-M; tope de posiciones acorde.
    s = load_settings()
    assert "DOGEUSDT" not in s.market.symbols
    assert {"BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT"} == set(s.market.symbols)
    assert s.risk.max_open_positions <= len(s.market.symbols)


def test_settings_yaml_quant_regime_hibrido():
    # 2026-07-01 (último intento): régimen SMA 50/200 @ 4h ENCENDIDO + noticias.
    s = load_settings()
    assert s.confluence.quant_regime_enabled is True
    assert s.quant.ma_type == "sma"
    assert s.quant.ema_fast_period == 50 and s.quant.ema_slow_period == 200
    assert s.quant.ema_weight == 1.0        # 100% cruce de medias, RSI pesa 0
    assert s.market.timeframe == "1h" and s.market.htf_timeframe == "4h"


def test_warmup_demasiado_corto_es_rechazado():
    # Un buffer < 20 velas no daría datos a los indicadores (ge=20).
    from src.core.config import OrchestratorConfig
    with pytest.raises(ValidationError):
        OrchestratorConfig(warmup_candles=5, reconcile_grace_cycles=3, regime_htf_bars=50)


def test_gracia_cero_es_rechazada():
    # Una gracia de 0 dispararía el HALT a la primera observación (ge=1).
    from src.core.config import OrchestratorConfig
    with pytest.raises(ValidationError):
        OrchestratorConfig(warmup_candles=60, reconcile_grace_cycles=0, regime_htf_bars=50)


def test_regime_htf_bars_por_debajo_del_minimo_del_quant_es_rechazado():
    # El régimen necesita ema_slow(21) + rsi(14) = 35 velas HTF para tener señal;
    # ge=35 lo garantiza. Menos sería un quant que nunca confirma.
    from src.core.config import OrchestratorConfig
    with pytest.raises(ValidationError):
        OrchestratorConfig(warmup_candles=60, reconcile_grace_cycles=3, regime_htf_bars=34)


# ------------------------- DashboardConfig (visor read-only) -------------------------


def _valid_dashboard_kwargs(**overrides):
    base = dict(host="127.0.0.1", port=8787, refresh_seconds=5.0, equity_points=500,
                decisions_rows=60, orders_rows=40, news_rows=25, stale_after_intervals=3.0)
    base.update(overrides)
    return base


def test_settings_yaml_dashboard_valido():
    s = load_settings()
    assert s.dashboard.host == "127.0.0.1"
    assert 1024 <= s.dashboard.port <= 65535
    assert s.dashboard.refresh_seconds > 0
    assert s.dashboard.equity_points >= 10


def test_dashboard_puerto_privilegiado_rechazado():
    # ge=1024: no permitir puertos privilegiados (requieren root, mala práctica).
    from src.core.config import DashboardConfig
    with pytest.raises(ValidationError):
        DashboardConfig(**_valid_dashboard_kwargs(port=80))


def test_dashboard_refresh_no_positivo_rechazado():
    from src.core.config import DashboardConfig
    with pytest.raises(ValidationError):
        DashboardConfig(**_valid_dashboard_kwargs(refresh_seconds=0))


def test_dashboard_stale_intervals_minimo():
    # ge=1.0: al menos una vela de gracia antes de marcar el feed obsoleto.
    from src.core.config import DashboardConfig
    with pytest.raises(ValidationError):
        DashboardConfig(**_valid_dashboard_kwargs(stale_after_intervals=0.5))


# ------------------------- EventConfig (Plan V2 Fase 2.1) -------------------------


def _valid_event_kwargs(**overrides):
    base = dict(
        enabled=False, poll_interval_seconds=15, min_impact_score=0.6,
        min_confidence=0.7, ttl_seconds=180, cooldown_seconds=900,
        confirm_impulse_bps=8, confirm_window_seconds=60, size_factor=0.5,
        macro_block_minutes_before=30, macro_block_minutes_after=5,
        markprice_buffer_seconds=180, markprice_stale_seconds=5,
        markprice_min_ticks=5, max_headline_age_seconds=1800,
    )
    base.update(overrides)
    return base


def test_event_config_valido():
    e = EventConfig(**_valid_event_kwargs())
    assert e.enabled is False           # arranca apagado (Fast Path no cableado)
    assert e.poll_interval_seconds == 15
    assert e.min_impact_score == 0.6
    assert e.confirm_impulse_bps == 8


def test_event_poll_demasiado_rapido_es_rechazado():
    # <5s martillaría el RSS gratis y arriesga un baneo de IP (ge=5).
    with pytest.raises(ValidationError):
        EventConfig(**_valid_event_kwargs(poll_interval_seconds=2))


def test_event_poll_demasiado_lento_es_rechazado():
    # >60s ya no compite con la vela de 5m: deja de ser "fast" (le=60).
    with pytest.raises(ValidationError):
        EventConfig(**_valid_event_kwargs(poll_interval_seconds=120))


def test_event_min_impact_score_uno_es_rechazado():
    # Un umbral de exactamente 1 exigiría el score máximo perfecto: nunca dispara.
    with pytest.raises(ValidationError):
        EventConfig(**_valid_event_kwargs(min_impact_score=1.0))


def test_event_min_impact_score_cero_es_rechazado():
    # Un 0 originaría con cualquier ruido (gt=0).
    with pytest.raises(ValidationError):
        EventConfig(**_valid_event_kwargs(min_impact_score=0.0))


def test_event_confirm_impulse_cero_es_valido():
    # 0 bps DESACTIVA el gate de impulso a propósito (ablación A/B de kill §B): no
    # es un typo, es una configuración legítima (ge=0, no gt=0).
    e = EventConfig(**_valid_event_kwargs(confirm_impulse_bps=0.0))
    assert e.confirm_impulse_bps == 0.0


def test_event_confirm_impulse_absurdo_es_rechazado():
    # 1000 bps = 10% de impulso exigido es un typo (le=1000).
    with pytest.raises(ValidationError):
        EventConfig(**_valid_event_kwargs(confirm_impulse_bps=1500))


def test_event_size_factor_mayor_que_uno_es_rechazado():
    # Un factor >1 AMPLIFICARÍA el tamaño en eventos arriesgados — al revés (le=1).
    with pytest.raises(ValidationError):
        EventConfig(**_valid_event_kwargs(size_factor=1.5))


def test_event_ttl_absurdo_es_rechazado():
    # Más de un día no es un evento "fresco" (le=86400).
    with pytest.raises(ValidationError):
        EventConfig(**_valid_event_kwargs(ttl_seconds=200_000))


def test_event_macro_block_negativo_es_rechazado():
    # Minutos de bloqueo negativos no tienen sentido (ge=0).
    with pytest.raises(ValidationError):
        EventConfig(**_valid_event_kwargs(macro_block_minutes_before=-10))


# ---------------- Plano de datos markPrice@1s (Fase 2.5(i)) ----------------


def test_event_markprice_config_valido():
    e = EventConfig(**_valid_event_kwargs())
    assert e.markprice_buffer_seconds == 180
    assert e.markprice_stale_seconds == 5
    assert e.markprice_min_ticks == 5


def test_event_markprice_stale_cero_es_rechazado():
    # 0s marcaría stale cualquier tick al instante: nunca operaría (gt=0).
    with pytest.raises(ValidationError):
        EventConfig(**_valid_event_kwargs(markprice_stale_seconds=0.0))


def test_event_markprice_min_ticks_uno_es_rechazado():
    # Con <2 ticks no hay retorno medible (ge=2).
    with pytest.raises(ValidationError):
        EventConfig(**_valid_event_kwargs(markprice_min_ticks=1))


def test_event_markprice_buffer_menor_que_ventana_es_rechazado():
    # Validador cruzado: si el buffer no retiene la ventana de impulso, la
    # comprobación "ventana cubierta" nunca pasaría → el Fast Path nunca operaría.
    with pytest.raises(ValidationError):
        EventConfig(**_valid_event_kwargs(
            markprice_buffer_seconds=30, confirm_window_seconds=60))


def test_event_markprice_buffer_igual_a_ventana_es_valido():
    # Frontera: buffer == ventana es legal (≥), aunque el repo deja holgura.
    e = EventConfig(**_valid_event_kwargs(
        markprice_buffer_seconds=60, confirm_window_seconds=60))
    assert e.markprice_buffer_seconds == 60


def test_event_max_headline_age_valido():
    e = EventConfig(**_valid_event_kwargs())
    assert e.max_headline_age_seconds == 1800


def test_event_max_headline_age_cero_es_rechazado():
    # 0s descartaría cualquier titular al instante: nunca originaría (ge=1).
    with pytest.raises(ValidationError):
        EventConfig(**_valid_event_kwargs(max_headline_age_seconds=0))


def test_event_max_headline_age_absurdo_es_rechazado():
    # Más de un día no es un "shock" fresco (le=86400).
    with pytest.raises(ValidationError):
        EventConfig(**_valid_event_kwargs(max_headline_age_seconds=200_000))


def test_settings_yaml_event():
    s = load_settings()
    # 2026-07-01: Fast Path APAGADO para el híbrido limpio (Slow news + régimen SMA).
    assert s.event.enabled is False
    # El poll de eventos debe ser estrictamente más rápido que el del Slow Path.
    assert s.event.poll_interval_seconds < s.sentiment.poll_interval_seconds
    assert 0.0 < s.event.min_impact_score < 1.0
    assert 0.0 < s.event.min_confidence < 1.0
    assert 0.0 < s.event.size_factor <= 1.0
    assert s.event.confirm_impulse_bps >= 0.0
    # Fase 2.5(i): plano de datos markPrice. El buffer debe cubrir la ventana.
    assert s.event.markprice_buffer_seconds >= s.event.confirm_window_seconds
    assert s.event.markprice_stale_seconds > 0
    assert s.event.markprice_min_ticks >= 2
    # Fase 2.5(ii): guardia de frescura del event_fetch.
    assert s.event.max_headline_age_seconds >= 1


# --- Auditoría 2026-07: validadores cruzados nuevos + score_squash_factor ---

def test_htf_no_multiplo_del_base_es_rechazado():
    # El engine deriva el ratio HTF/base con división ENTERA: un HTF que no sea
    # múltiplo exacto (1h → 90m daría ratio 1 truncado) rompería el resampleo
    # del régimen EN SILENCIO. Debe morir al cargar la config.
    from src.core.config import MarketConfig
    with pytest.raises(ValidationError):
        MarketConfig(symbols=["BTCUSDT"], timeframe="1h", htf_timeframe="90m",
                     quote_assets=["USDT"])


def test_htf_igual_o_menor_que_base_es_rechazado():
    from src.core.config import MarketConfig
    with pytest.raises(ValidationError):
        MarketConfig(symbols=["BTCUSDT"], timeframe="1h", htf_timeframe="1h",
                     quote_assets=["USDT"])
    with pytest.raises(ValidationError):
        MarketConfig(symbols=["BTCUSDT"], timeframe="4h", htf_timeframe="1h",
                     quote_assets=["USDT"])


def test_htf_multiplo_valido():
    from src.core.config import MarketConfig
    mc = MarketConfig(symbols=["BTCUSDT"], timeframe="1h", htf_timeframe="4h",
                      quote_assets=["USDT"])
    assert mc.htf_timeframe == "4h"


def test_regime_htf_bars_insuficiente_es_rechazado():
    # Con menos velas HTF que ema_slow+rsi, compute_signal devolvería None SIEMPRE
    # y el régimen quedaría neutro en silencio (degradación invisible). El
    # validador de Settings lo convierte en un fallo ruidoso al arrancar.
    s = load_settings()
    raw = s.model_dump()
    raw["orchestrator"]["regime_htf_bars"] = (
        s.quant.ema_slow_period + s.quant.rsi_period - 1)
    from src.core.config import Settings
    with pytest.raises(ValidationError):
        Settings.model_validate(raw)


def test_regime_htf_bars_del_repo_cubre_el_quant():
    s = load_settings()
    assert (s.orchestrator.regime_htf_bars
            >= s.quant.ema_slow_period + s.quant.rsi_period)


def test_score_squash_factor_en_settings_yaml():
    # Antes era un 50 hardcodeado en strategy.py (violación de Cero Hardcoding).
    s = load_settings()
    assert s.quant.score_squash_factor == 50.0


def test_score_squash_factor_invalido_es_rechazado():
    from src.core.config import QuantConfig
    with pytest.raises(ValidationError):
        QuantConfig(ema_fast_period=9, ema_slow_period=21, rsi_period=14,
                    ema_weight=0.6, score_squash_factor=0.0)
    with pytest.raises(ValidationError):
        QuantConfig(ema_fast_period=9, ema_slow_period=21, rsi_period=14,
                    ema_weight=0.6, score_squash_factor=5000.0)


# --- PositioningResearchConfig (investigación de posicionamiento 2026-07) ---

def _valid_positioning_kwargs(**overrides):
    base = dict(
        zscore_window_bars=240,
        zscore_window_days=60,
        imbalance_ma_bars_grid=[4, 24, 72],
        entry_threshold_grid=[0.5, 1.0],
        cost_per_side_pct=0.06,
        train_test_split_date="2024-12-15T00:00:00Z",
        metrics_ffill_limit_bars=12,
    )
    base.update(overrides)
    return base


def test_positioning_research_en_settings_yaml():
    # La sección debe existir y cargar tipada: es la Vía B del estudio E1/E2/E2b.
    s = load_settings()
    pr = s.positioning_research
    assert pr.zscore_window_bars == 240
    assert pr.imbalance_ma_bars_grid == [4, 24, 72]
    assert pr.cost_per_side_pct == 0.06
    # El corte train/test lleva zona horaria explícita (anti-corte silencioso).
    assert pr.train_test_split_date.endswith("Z")
    # Nuevo dir de storage para las métricas de Binance Vision.
    assert s.storage.metrics_dir


def test_positioning_research_valida():
    from src.core.config import PositioningResearchConfig
    pr = PositioningResearchConfig(**_valid_positioning_kwargs())
    assert pr.zscore_window_days == 60


def test_positioning_umbral_cero_es_rechazado():
    # Umbral 0 = "siempre en mercado": deja de ser un test de umbral y además
    # multiplica el turnover; el validador lo ataja como typo.
    from src.core.config import PositioningResearchConfig
    with pytest.raises(ValidationError):
        PositioningResearchConfig(
            **_valid_positioning_kwargs(entry_threshold_grid=[0.0]))


def test_positioning_ma_bars_fuera_de_rango_es_rechazado():
    from src.core.config import PositioningResearchConfig
    with pytest.raises(ValidationError):
        PositioningResearchConfig(
            **_valid_positioning_kwargs(imbalance_ma_bars_grid=[0]))
    with pytest.raises(ValidationError):
        PositioningResearchConfig(
            **_valid_positioning_kwargs(imbalance_ma_bars_grid=[501]))


def test_positioning_split_date_naive_es_rechazada():
    # Sin tz el corte se compararía mal contra índices UTC → partiría el dataset
    # en otro punto del que se cree. Mejor morir al cargar la config.
    from src.core.config import PositioningResearchConfig
    with pytest.raises(ValidationError):
        PositioningResearchConfig(
            **_valid_positioning_kwargs(train_test_split_date="2024-12-15"))
    with pytest.raises(ValidationError):
        PositioningResearchConfig(
            **_valid_positioning_kwargs(train_test_split_date="no-es-fecha"))
