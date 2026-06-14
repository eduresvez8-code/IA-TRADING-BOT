"""Demo del camino de decisión: Signal + SentimentScore → Confluencia → Risk.

Recorre varios escenarios y muestra cómo la matriz de confluencia y el Risk
Manager colaboran en **Binance Futuros USD-M**: la confluencia decide dirección
(LONG/SHORT) y tamaño; el Risk Manager veta, dimensiona por riesgo, valida el
margen contra el saldo disponible y ajusta la orden a la microestructura.

    uv run python -m src.risk.risk_demo
"""

from datetime import datetime, timezone

from src.core.config import load_settings
from src.core.models import SentimentScore, Signal, SymbolFilters
from src.decision.confluence import decide
from src.risk.manager import PortfolioState, RiskManager

NOW = datetime.now(timezone.utc)
PRICE = 1000.0
ATR = 50.0

FILTERS = SymbolFilters(symbol="BTCUSDT", tick_size="0.01", step_size="0.0001",
                        min_qty="0.0001", min_notional="5")


def _sig(score: float) -> Signal:
    return Signal(symbol="BTCUSDT", score=score, strategy="ema_cross_rsi", timestamp=NOW)


def _sent(score: float, *, high_impact: bool = False, confidence: float = 0.8):
    return SentimentScore(news_id="demo", symbol_scope=["BTCUSDT"], score=score,
                          confidence=confidence, high_impact=high_impact, analyzed_at=NOW)


SCENARIOS = [
    ("Quant alcista + noticia confirma (LONG)", _sig(0.8), _sent(0.6)),
    ("Quant BAJISTA + hack/FUD confirma (SHORT)", _sig(-0.8), _sent(-0.7)),
    ("Quant alcista + sin noticias (neutro)", _sig(0.8), None),
    ("Quant alcista + confirma pero baja confianza", _sig(0.8), _sent(0.6, confidence=0.2)),
    ("Quant alcista + noticia OPUESTA fuerte", _sig(0.8), _sent(-0.6)),
    ("Sentimiento extremo SIN quant (no abre)", _sig(0.1), _sent(0.95)),
    ("Evento de alto impacto pendiente", _sig(0.8), _sent(0.6, high_impact=True)),
]


def main() -> int:
    cfg = load_settings()
    rm = RiskManager(cfg)
    L = cfg.risk.max_leverage
    # Cuenta sana: todo el wallet libre como margen, nada comprometido.
    state = PortfolioState(wallet_balance=10_000.0, available_balance=10_000.0,
                           committed_margin=0.0, peak_wallet_balance=10_000.0,
                           day_start_wallet_balance=10_000.0, open_positions=0)

    print(f"FUTUROS USD-M · wallet {state.wallet_balance:,.0f} USDT · precio {PRICE} · "
          f"ATR {ATR} · riesgo {cfg.risk.risk_per_trade_pct}%/trade · "
          f"leverage máx {L}x · margen máx {cfg.risk.max_portfolio_margin_pct:.0f}%\n")

    for name, sig, sent in SCENARIOS:
        d = decide(sig, sent, cfg)
        conf = sent.confidence if sent is not None else 1.0
        a = rm.assess(d, price=PRICE, atr=ATR, state=state, filters=FILTERS, confidence=conf)
        print(f"▶ {name}")
        print(f"    confluencia: {d.action.value:5s} size={d.size_factor:.2f} "
              f"({d.reason})")
        if a.approved:
            o = a.order
            notional = o.quantity * o.entry_price
            print(f"    risk:        APROBADA {o.side.value} {o.leverage}x "
                  f"qty={o.quantity:.4f} SL={o.stop_loss:.2f} TP={o.take_profit:.2f} "
                  f"nocional={notional:.2f} margen={notional / o.leverage:.2f}\n")
        else:
            print(f"    risk:        VETADA ({a.reason})\n")

    # Techo de margen agregado: ATR bajo dispara una qty enorme; el nocional se
    # topa en margen_máx (85% wallet) × leverage.
    print("▶ Techo de margen agregado (ATR bajo, SHORT)")
    a = rm.assess(decide(_sig(-0.8), _sent(-0.7), cfg), price=PRICE, atr=1.0,
                  state=state, filters=FILTERS)
    o = a.order
    print(f"    risk:        APROBADA {o.side.value} {o.leverage}x "
          f"nocional={o.quantity * o.entry_price:.2f} "
          f"margen={o.quantity * o.entry_price / o.leverage:.2f} (= 85%·wallet)\n")

    # available_balance bajo: el techo físico manda aunque el wallet sea grande.
    print("▶ Margen libre escaso: available 1k, wallet 10k (ATR bajo)")
    tight = PortfolioState(wallet_balance=10_000.0, available_balance=1_000.0,
                           committed_margin=2_000.0, peak_wallet_balance=10_000.0,
                           day_start_wallet_balance=10_000.0, open_positions=2)
    a = rm.assess(decide(_sig(0.8), _sent(0.6), cfg), price=PRICE, atr=1.0,
                  state=tight, filters=FILTERS)
    o = a.order
    print(f"    risk:        APROBADA nocional={o.quantity * o.entry_price:.2f} "
          f"margen={o.quantity * o.entry_price / o.leverage:.2f} (≤ available 1000)\n")

    # Kill switch por drawdown del 11% del wallet.
    print("▶ Kill switch: drawdown del 11% del wallet sobre una señal válida")
    breached = PortfolioState(wallet_balance=8_900.0, available_balance=8_900.0,
                              committed_margin=0.0, peak_wallet_balance=10_000.0,
                              day_start_wallet_balance=10_000.0, open_positions=0)
    a = rm.assess(decide(_sig(0.8), _sent(0.6), cfg), price=PRICE, atr=ATR,
                  state=breached, filters=FILTERS)
    print(f"    risk:        VETADA ({a.reason}) · kill_switch_active="
          f"{rm.kill_switch_active}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
