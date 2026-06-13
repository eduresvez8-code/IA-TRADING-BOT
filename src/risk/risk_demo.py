"""Demo del camino de decisión: Signal + SentimentScore → Confluencia → Risk.

Recorre varios escenarios y muestra cómo la matriz de confluencia y el Risk
Manager colaboran en un entorno **Binance Spot, long-only**: la confluencia
decide dirección y tamaño, el Risk Manager veta, dimensiona sobre el saldo libre
y ajusta la orden a los filtros de microestructura.

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

# Filtros típicos de un par USDT (valores ilustrativos al estilo exchangeInfo).
FILTERS = SymbolFilters(symbol="BTCUSDT", tick_size="0.01", step_size="0.0001",
                        min_qty="0.0001", min_notional="5")


def _sig(score: float) -> Signal:
    return Signal(symbol="BTCUSDT", score=score, strategy="ema_cross_rsi", timestamp=NOW)


def _sent(score: float, *, high_impact: bool = False, confidence: float = 0.8):
    return SentimentScore(news_id="demo", symbol_scope=["BTCUSDT"], score=score,
                          confidence=confidence, high_impact=high_impact, analyzed_at=NOW)


SCENARIOS = [
    ("Quant alcista + noticia confirma", _sig(0.8), _sent(0.6)),
    ("Quant alcista + sin noticias (neutro)", _sig(0.8), None),
    ("Quant alcista + noticia OPUESTA fuerte", _sig(0.8), _sent(-0.6)),
    ("Quant alcista + confirma pero baja confianza", _sig(0.8), _sent(0.6, confidence=0.2)),
    ("Quant BAJISTA fuerte (Spot: no se abre corto)", _sig(-0.8), _sent(-0.6)),
    ("Sentimiento extremo SIN quant (no abre)", _sig(0.1), _sent(0.95)),
    ("Evento de alto impacto pendiente", _sig(0.8), _sent(0.6, high_impact=True)),
]


def main() -> int:
    cfg = load_settings()
    rm = RiskManager(cfg)
    # Caja sana: todo el capital libre, nada comprometido.
    state = PortfolioState(equity=10_000.0, free_balance=10_000.0, committed_notional=0.0,
                           peak_equity=10_000.0, day_start_equity=10_000.0, open_positions=0)

    print(f"SPOT long-only · capital {state.equity:,.0f} USDT · precio {PRICE} · "
          f"ATR {ATR} · riesgo {cfg.risk.risk_per_trade_pct}%/trade · "
          f"exposición máx {cfg.risk.max_portfolio_exposure_pct:.0f}%\n")

    for name, sig, sent in SCENARIOS:
        d = decide(sig, sent, cfg)
        conf = sent.confidence if sent is not None else 1.0
        a = rm.assess(d, price=PRICE, atr=ATR, state=state, filters=FILTERS, confidence=conf)
        print(f"▶ {name}")
        print(f"    confluencia: {d.action.value:5s} size={d.size_factor:.2f} "
              f"({d.reason})")
        if a.approved:
            o = a.order
            print(f"    risk:        APROBADA {o.side.value} qty={o.quantity:.4f} "
                  f"SL={o.stop_loss:.2f} TP={o.take_profit:.2f} "
                  f"nocional={o.quantity * o.entry_price:.2f}\n")
        else:
            print(f"    risk:        VETADA ({a.reason})\n")

    # Dinero fantasma: equity 10k pero solo 2k libres y 8k comprometidos.
    print("▶ Dinero fantasma: equity 10k, free 2k, comprometido 8k (ATR bajo)")
    phantom = PortfolioState(equity=10_000.0, free_balance=2_000.0, committed_notional=8_000.0,
                             peak_equity=10_000.0, day_start_equity=10_000.0, open_positions=2)
    a = rm.assess(decide(_sig(0.8), _sent(0.6), cfg), price=PRICE, atr=1.0,
                  state=phantom, filters=FILTERS)
    if a.approved:
        o = a.order
        print(f"    risk:        APROBADA qty={o.quantity:.4f} "
              f"nocional={o.quantity * o.entry_price:.2f} (≤ free 2000 y ≤ 95%·eq−comprometido)\n")
    else:
        print(f"    risk:        VETADA ({a.reason})\n")

    # Kill switch por drawdown del 11%.
    print("▶ Kill switch: drawdown del 11% sobre una señal alcista válida")
    breached = PortfolioState(equity=8_900.0, free_balance=8_900.0, committed_notional=0.0,
                              peak_equity=10_000.0, day_start_equity=10_000.0, open_positions=0)
    a = rm.assess(decide(_sig(0.8), _sent(0.6), cfg), price=PRICE, atr=ATR,
                  state=breached, filters=FILTERS)
    print(f"    risk:        VETADA ({a.reason}) · kill_switch_active="
          f"{rm.kill_switch_active}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
