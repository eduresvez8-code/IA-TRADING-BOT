"""¿Vale la pena pagar por datos Value/Quality? — proxy con ETFs (2026-07-25).

Pre-registrado en docs/research/2026-07-25_factor_etf_proxy_protocolo.md.
Prueba GRATIS, antes de gastar en un proveedor de fundamentales, si el
factor Value/Quality tiene ventaja real en 2015-2026 usando ETFs
profesionales ya existentes (comprar-y-mantener puro, cero grados de
libertad nuestros — no hay nada que "seleccionar por train").

    uv run python -m backtest.run_factor_etf_proxy
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.core.config import load_settings
from src.data.sp500 import download_symbols, load_prices
from backtest.diagnostics import calmar_ratio, max_drawdown, sharpe
from backtest.run_sp500_research import MONTHS_PER_YEAR, _fmt_gate, _gate_monthly, _split_monthly
from backtest.sp500_families import monthly_hold_returns

# Elegidos ANTES de mirar un solo resultado: el ETF más grande/antiguo de su
# categoría (evita elegir a mano el que "se ve mejor" después de ver números).
CANDIDATES = {
    "VTV": "Value (Vanguard Value ETF, 2004)",
    "QUAL": "Quality (iShares MSCI USA Quality Factor, 2013)",
}


def main() -> int:
    cfg = load_settings()
    rc = cfg.research
    cut = pd.Timestamp(rc.test_start_date)

    print("=" * 86)
    print("¿VALE LA PENA PAGAR POR VALUE/QUALITY? — proxy con ETFs profesionales")
    print("pre-registrado 2026-07-25 — NO es una reapertura de la búsqueda de estrategias")
    print("=" * 86)

    ok, missing = download_symbols(list(CANDIDATES), cfg.data)
    print(f"\ndescarga: {len(ok)} ok, {len(missing)} sin datos {missing or ''}")

    data_dir = Path(cfg.data.dir)
    spy = load_prices(data_dir, cfg.market.benchmark_symbol)
    spy_hold_m = monthly_hold_returns(spy)
    _, bh_m_te = _split_monthly(spy_hold_m, cut)
    bh_te_sh_m = sharpe(bh_m_te, MONTHS_PER_YEAR)
    print(f"\nB&H SPY (referencia oficial, ya publicada 2026-07-11): "
          f"Sh test mensual {bh_te_sh_m:+.2f}")

    verdicts: list[tuple[str, str, object]] = []
    for ticker, label in CANDIDATES.items():
        if ticker in missing:
            print(f"\n{ticker} ({label}): SIN DATOS, se salta")
            continue
        print(f"\n{'-' * 86}")
        print(f"{ticker} — {label}")
        df = load_prices(data_dir, ticker)
        hold_m = monthly_hold_returns(df)
        tr, te = _split_monthly(hold_m, cut)
        print(f"  pre-2015 (informativo, NO participa en ninguna decisión — cero grados de")
        print(f"  libertad: no hay nada que seleccionar por train): "
              f"Sh {sharpe(tr, MONTHS_PER_YEAR):+.2f} (n={len(tr)})")
        g = _gate_monthly(te, cfg, bh_te_sh_m)
        print(_fmt_gate(g))
        dd = max_drawdown(te.to_numpy())
        cal = calmar_ratio(te.to_numpy(), MONTHS_PER_YEAR)
        print(f"    MaxDD test: {dd:.1%}  |  Calmar test: {cal:+.2f}  "
              f"(referencia B&H SPY: 23.3% DD / +0.59 Calmar)")
        verdicts.append((ticker, label, g))

    print("\n" + "=" * 86)
    print("VEREDICTO (proxy ETF — informa la decisión de pagar, no es un backtest propio):")
    any_pass = False
    for ticker, label, g in verdicts:
        status = "PASA los 5 criterios" if g.passes_all else "no pasa"
        any_pass = any_pass or g.passes_all
        print(f"  {ticker:6s} {label:38s} Sh test {g.sharpe_test:+.2f} "
              f"vs B&H {g.sharpe_buyhold:+.2f} → {status}")
    if not any_pass:
        print("\n  NINGÚN ETF profesional de Value/Quality superó a B&H SPY en este periodo,")
        print("  ni siquiera con toda su ventaja de escala y gestión. Evidencia fuerte de que")
        print("  pagar por datos propios de Value/Quality NO se justifica en 2015-2026 — el")
        print("  problema no es la calidad del dato, es que el factor no tuvo tracción aquí.")
    else:
        print("\n  Al menos un ETF profesional SÍ superó a B&H SPY con rigor completo en este")
        print("  periodo — revisar el margen (¿sólido o al filo, como el falso positivo de hoy")
        print("  con RSI-2+amplitud?) antes de decidir si esto justifica pagar por datos propios.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
