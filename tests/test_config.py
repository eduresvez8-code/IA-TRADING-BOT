"""Tests de carga y validación de configuración.

Dos frentes:
    1. La config REAL (config/settings.yaml) carga y respeta los invariantes
       del protocolo pre-registrado.
    2. Los validadores rechazan configs malformadas (typos de rango, splits
       sin timezone, grids vacíos) — la primera línea de defensa del repo.
"""

import pytest
from pydantic import ValidationError

from src.core.config import (
    BacktestConfig,
    BreadthConfig,
    DataConfig,
    DualMomentumConfig,
    MaTimingConfig,
    MarketConfig,
    PaperTradingRsi2Config,
    QuantConfig,
    ResearchConfig,
    RiskConfig,
    RsiReversionConfig,
    Settings,
    TsmomIndexConfig,
    VixRegimeConfig,
    XsMomentumConfig,
    load_settings,
)


# ---------- la config real ----------

def test_settings_reales_cargan():
    cfg = load_settings()
    assert cfg.market.benchmark_symbol
    assert cfg.market.timeframe == "1d"
    assert cfg.research.success_sharpe_min > 0


def test_protocolo_preregistrado_es_el_declarado():
    """El protocolo NO se cambia tras mirar resultados. Este test congela los
    valores pre-registrados el 2026-07-11: si alguien los toca, el test falla
    y obliga a justificarlo en docs/research/ con un documento NUEVO."""
    cfg = load_settings()
    assert cfg.research.test_start_date == "2015-01-01T00:00:00Z"
    assert cfg.research.success_sharpe_min == 0.5
    assert cfg.research.bootstrap_ci == 0.90
    assert cfg.research.concentration_max == 0.60
    assert cfg.research.bootstrap_iterations == 5000


def test_cuenta_cash_sin_cortos_en_config_real():
    cfg = load_settings()
    assert cfg.risk.allow_short is False
    assert cfg.backtest.allow_short is False


# ---------- market ----------

def test_market_rechaza_timeframe_intradia():
    with pytest.raises(ValidationError, match="1d"):
        MarketConfig(benchmark_symbol="SPY", index_symbol="^GSPC",
                     tbill_symbol="^IRX", vix_symbol="^VIX", timeframe="1h")


# ---------- data ----------

def _data_kwargs(**overrides):
    base = dict(dir="data/sp500", start_date="1990-01-01",
                constituents_repo="fja05680/sp500", batch_size=50,
                pause_seconds=1.0)
    base.update(overrides)
    return base


def test_data_valida():
    d = DataConfig(**_data_kwargs())
    assert d.extra_symbols == []


def test_data_rechaza_fecha_malformada():
    with pytest.raises(ValidationError, match="YYYY-MM-DD"):
        DataConfig(**_data_kwargs(start_date="01/01/1990"))


def test_data_rechaza_repo_sin_owner():
    with pytest.raises(ValidationError):
        DataConfig(**_data_kwargs(constituents_repo="sp500"))


# ---------- risk ----------

def _risk_kwargs(**overrides):
    base = dict(risk_per_trade_pct=0.5, max_open_positions=10,
                max_daily_loss_pct=2.0, max_drawdown_pct=15.0,
                atr_stop_multiplier=2.0, atr_period=14, take_profit_rr=2.0,
                let_winners_run=True, allow_short=False,
                allow_fractional_shares=False)
    base.update(overrides)
    return base


def test_risk_valido():
    r = RiskConfig(**_risk_kwargs())
    assert r.risk_per_trade_pct == 0.5


def test_risk_rechaza_riesgo_de_casino():
    # 10% por trade es un typo, no una política.
    with pytest.raises(ValidationError):
        RiskConfig(**_risk_kwargs(risk_per_trade_pct=10.0))


def test_risk_rechaza_drawdown_absurdo():
    with pytest.raises(ValidationError):
        RiskConfig(**_risk_kwargs(max_drawdown_pct=50.0))


# ---------- quant ----------

def test_quant_slow_debe_superar_fast():
    with pytest.raises(ValidationError, match="mayor que"):
        QuantConfig(ema_fast_period=50, ema_slow_period=20, rsi_period=14,
                    ema_weight=1.0)


def test_quant_rechaza_ma_type_invalido():
    with pytest.raises(ValidationError, match="ma_type"):
        QuantConfig(ema_fast_period=9, ema_slow_period=21, rsi_period=14,
                    ema_weight=0.6, ma_type="wma")


# ---------- backtest ----------

def _bt_kwargs(**overrides):
    base = dict(initial_capital=10000.0, commission_pct=0.0, slippage_pct=0.02,
                slippage_atr_multiplier=0.0, entry_threshold=0.5,
                exit_threshold=0.1, take_profit_rr=2.0, allow_short=False)
    base.update(overrides)
    return base


def test_backtest_exit_debe_ser_menor_que_entry():
    with pytest.raises(ValidationError, match="menor que"):
        BacktestConfig(**_bt_kwargs(exit_threshold=0.6))


def test_backtest_rechaza_comision_typo():
    # 40 se interpretaría como 40% por lado.
    with pytest.raises(ValidationError):
        BacktestConfig(**_bt_kwargs(commission_pct=40.0))


# ---------- research (el pre-registro) ----------

def _research_kwargs(**overrides):
    base = dict(
        test_start_date="2015-01-01T00:00:00Z",
        success_sharpe_min=0.5, bootstrap_iterations=5000, bootstrap_ci=0.90,
        concentration_max=0.60, slippage_stress_pct=0.05, cash_earns_tbill=True,
        xs_momentum=dict(lookback_months_grid=[3, 6, 12], skip_months_grid=[0, 1],
                         top_n_grid=[30, 50], min_coverage=0.6,
                         min_history_days=280),
        tsmom_index=dict(lookback_months_grid=[3, 6, 12]),
        ma_timing=dict(sma_days_grid=[100, 200], cross_pairs=[[50, 200]]),
        rsi_reversion=dict(rsi_period=2, entry_grid=[5.0, 10.0],
                           exit_grid=[50.0, 70.0], trend_sma_days=200),
        dual_momentum=dict(lookback_months=12, bond_symbol="VUSTX"),
        breadth=dict(sma_days_grid=[100, 200], threshold_grid=[0.40, 0.50, 0.60]),
        vix_regime=dict(sma_days_grid=[50, 100, 200], directions=["below", "above"]),
    )
    base.update(overrides)
    return base


def test_research_valido():
    r = ResearchConfig(**_research_kwargs())
    assert r.dual_momentum.lookback_months == 12


def test_research_split_sin_timezone_no_carga():
    # Un corte naive partiría el dataset en otro punto del que se cree.
    with pytest.raises(ValidationError, match="zona horaria"):
        ResearchConfig(**_research_kwargs(test_start_date="2015-01-01"))


def test_xs_momentum_rechaza_grid_vacio():
    with pytest.raises(ValidationError):
        XsMomentumConfig(lookback_months_grid=[], skip_months_grid=[0],
                         top_n_grid=[50], min_coverage=0.6, min_history_days=280)


def test_xs_momentum_rechaza_lookback_fuera_de_rango():
    with pytest.raises(ValidationError, match="fuera de"):
        XsMomentumConfig(lookback_months_grid=[36], skip_months_grid=[0],
                         top_n_grid=[50], min_coverage=0.6, min_history_days=280)


def test_tsmom_rechaza_lookback_fuera_de_rango():
    with pytest.raises(ValidationError, match="fuera de"):
        TsmomIndexConfig(lookback_months_grid=[0])


def test_ma_timing_rechaza_par_invertido():
    with pytest.raises(ValidationError, match="fast < slow"):
        MaTimingConfig(sma_days_grid=[200], cross_pairs=[[200, 50]])


def test_rsi_reversion_rechaza_entrada_imposible():
    with pytest.raises(ValidationError, match="fuera de"):
        RsiReversionConfig(rsi_period=2, entry_grid=[60.0], exit_grid=[70.0],
                           trend_sma_days=200)


def test_dual_momentum_sin_grid_por_diseno():
    # La familia más simple NO tiene grid: un solo lookback canónico.
    d = DualMomentumConfig(lookback_months=12, bond_symbol="VUSTX")
    assert d.lookback_months == 12


def test_breadth_rechaza_umbral_fuera_de_rango():
    with pytest.raises(ValidationError, match="fuera de"):
        BreadthConfig(sma_days_grid=[200], threshold_grid=[1.5])


def test_breadth_rechaza_sma_fuera_de_rango():
    with pytest.raises(ValidationError, match="fuera de"):
        BreadthConfig(sma_days_grid=[5], threshold_grid=[0.5])


def test_vix_regime_rechaza_direccion_invalida():
    with pytest.raises(ValidationError, match="below.*above"):
        VixRegimeConfig(sma_days_grid=[100], directions=["sideways"])


# ---------- paper trading (forward real, sin capital) ----------

def test_paper_trading_rsi2_valido():
    p = PaperTradingRsi2Config(entry_below=10.0, exit_above=70.0, trend_sma_days=200,
                               rsi_period=2, log_dir="paper_trading/rsi2")
    assert p.entry_below == 10.0


def test_paper_trading_rsi2_rechaza_entry_fuera_de_rango():
    with pytest.raises(ValidationError):
        PaperTradingRsi2Config(entry_below=60.0, exit_above=70.0, trend_sma_days=200,
                               rsi_period=2, log_dir="paper_trading/rsi2")


def test_paper_trading_real_carga_config_ya_publicada():
    cfg = load_settings()
    assert cfg.paper_trading.rsi2.entry_below == 10.0
    assert cfg.paper_trading.rsi2.exit_above == 70.0


# ---------- coherencia cruzada ----------

def test_backtest_no_puede_shortear_si_risk_lo_prohibe():
    cfg = load_settings()
    raw = cfg.model_dump()
    raw["backtest"]["allow_short"] = True     # risk.allow_short sigue false
    with pytest.raises(ValidationError, match="allow_short"):
        Settings.model_validate(raw)
