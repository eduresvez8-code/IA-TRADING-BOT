"""Demo del camino de decisión: Signal + SentimentScore → Confluencia → Risk.

Recorre varios escenarios y muestra cómo la matriz de confluencia y el Risk
Manager colaboran: la confluencia decide dirección y tamaño, el Risk Manager
veta o construye la orden con stop-loss obligatorio.

    uv run python -m src.risk.risk_demo
"""

from datetime import datetime, timezone

from src.core.config import load_settings
from src.core.models import Action, SentimentScore, Signal
from src.decision.confluence import decide
from src.risk.manager import PortfolioState, RiskManager

NOW = datetime.now(timezone.utc)
PRICE = 1000.0
ATR = 50.0


def _sig(score: float) -> Signal:
    return Signal(symbol="BTCUSDT", score=score, strategy="ema_cross_rsi", timestamp=NOW)


def _sent(score: float, *, high_impact: bool = False, confidence: float = 0.8):
    return SentimentScore(news_id="demo", symbol_scope=["BTCUSDT"], score=score,
                          confidence=confidence, high_impact=high_impact, analyzed_at=NOW)


SCENARIOS = [
    ("Quant alcista + noticia confirma", _sig(0.8), _sent(0.6)),
    ("Quant alcista + sin noticias (neutro)", _sig(0.8), None),
    ("Quant alcista + noticia OPUESTA fuerte", _sig(0.8), _sent(-0.6)),
    ("Quant alcista + noticia confirma pero baja confianza", _sig(0.8), _sent(0.6, confidence=0.2)),
    ("Quant bajista + noticia confirma (short)", _sig(-0.8), _sent(-0.6)),
    ("Sentimiento extremo SIN quant (no abre)", _sig(0.1), _sent(0.95)),
    ("Evento de alto impacto pendiente", _sig(0.8), _sent(0.6, high_impact=True)),
]


def main() -> int:
    cfg = load_settings()
    rm = RiskManager(cfg)
    state = PortfolioState(equity=10_000.0, peak_equity=10_000.0,
                           day_start_equity=10_000.0, open_positions=0)

    print(f"Capital {state.equity:,.0f} USDT · precio {PRICE} · ATR {ATR} · "
          f"riesgo {cfg.risk.risk_per_trade_pct}%/trade\n")

    for name, sig, sent in SCENARIOS:
        d = decide(sig, sent, cfg)
        conf = sent.confidence if sent is not None else 1.0
        a = rm.assess(d, price=PRICE, atr=ATR, state=state, confidence=conf)
        print(f"▶ {name}")
        print(f"    confluencia: {d.action.value:5s} size={d.size_factor:.2f} "
              f"({d.reason})")
        if a.approved:
            o = a.order
            print(f"    risk:        APROBADA {o.side.value} qty={o.quantity:.4f} "
                  f"SL={o.stop_loss:.1f} TP={o.take_profit:.1f}\n")
        else:
            print(f"    risk:        VETADA ({a.reason})\n")

    # Escenario de límites duros: drawdown que dispara el kill switch.
    print("▶ Kill switch: drawdown del 11% sobre una señal alcista válida")
    breached = PortfolioState(equity=8_900.0, peak_equity=10_000.0,
                              day_start_equity=10_000.0, open_positions=0)
    a = rm.assess(decide(_sig(0.8), _sent(0.6), cfg), price=PRICE, atr=ATR, state=breached)
    print(f"    risk:        VETADA ({a.reason}) · kill_switch_active="
          f"{rm.kill_switch_active}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
