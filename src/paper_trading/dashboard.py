"""Dashboard estático del paper trading RSI-2 — sin servidor, sin JS externo.

El dashboard viejo del bot de cripto (`src/dashboard/`, borrado en el pivote
2026-07-11) era un servidor `http.server` leyendo SQLite en modo read-only,
porque el bot vivo escribía continuamente a una base local. Ese modelo no
aplica aquí: no hay proceso vivo ni base de datos — hay una corrida diaria
de GitHub Actions que commitea un CSV append-only
(`paper_trading/rsi2/daily_log.csv`). El equivalente correcto no es un
servidor: es una página ESTÁTICA que se regenera en cada corrida y se
publica en GitHub Pages (gratis, mismo espíritu $0/mes del proyecto).

`build_snapshot` (lee CSV + descarga precio de referencia) y `render_html`
(HTML puro a partir del snapshot) están deliberadamente separados, igual
que el `queries.py` viejo separaba la lectura de la presentación — así
`render_html` es testeable sin red ni disco.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.sp500_families import daily_hold_returns, daily_strategy_returns
from src.core.config import Settings, load_settings
from src.data.sp500 import download_symbols, load_prices, tbill_daily_return

# Bajo esta cantidad de días de muestra, cualquier número de desempeño es
# ruido — mismo umbral de "no saques conclusiones todavía" en espíritu que
# el resto del proyecto (aquí no hay un test estadístico que aplicar: con
# tan pocas observaciones ni el bootstrap tendría sentido).
MIN_DAYS_FOR_PERFORMANCE = 60


def _naive(ts: pd.Timestamp) -> pd.Timestamp:
    return ts.tz_convert("UTC").tz_localize(None) if ts.tzinfo else ts


def build_snapshot(cfg: Settings, *, now: pd.Timestamp | None = None) -> dict:
    """Ensambla todo lo que la plantilla necesita en una sola lectura.

    `now` inyectable (determinista en tests) — por defecto, el momento REAL
    de la corrida. Se guarda en el snapshot como "generado el" para que el
    dashboard distinga "no hubo día de mercado nuevo" de "el sistema dejó
    de correr": la fecha del log solo cambia con datos nuevos, pero esta
    marca de tiempo cambia SIEMPRE que la corrida se ejecuta con éxito.
    """
    now = now or pd.Timestamp.now(tz="UTC")
    pc = cfg.paper_trading.rsi2
    log_path = Path(pc.log_dir) / "daily_log.csv"
    if not log_path.exists():
        return {"has_data": False}

    log = pd.read_csv(log_path, parse_dates=["date"], index_col="date").sort_index()
    if log.empty:
        return {"has_data": False}
    # Mismo fix que _load_existing_log en rsi2.py: un campo vacío del CSV se
    # lee como NaN, no "" — sin esto, `action != ""` marca TODOS los días
    # como transacción.
    log["action"] = log["action"].fillna("")

    start_date = log.index.min()
    last_date = log.index.max()
    n_days = len(log)
    current_position = float(log["position"].iloc[-1])
    trades = log[log["action"] != ""]
    n_entries = int((log["action"] == "ENTER").sum())
    n_exits = int((log["action"] == "EXIT").sum())

    data_dir = Path(cfg.data.dir)
    download_symbols([cfg.market.benchmark_symbol, cfg.market.tbill_symbol], cfg.data)
    spy = load_prices(data_dir, cfg.market.benchmark_symbol)
    irx = load_prices(data_dir, cfg.market.tbill_symbol)

    hold = daily_hold_returns(spy)
    hold = hold[hold.index >= start_date]
    irx_close = irx.set_index("open_time")["close"]
    irx_close.index = irx_close.index.tz_convert("UTC").tz_localize(None)
    tbill_d = tbill_daily_return(irx_close)

    per_side = (cfg.backtest.commission_pct + cfg.backtest.slippage_pct) / 100.0
    strat_r = daily_strategy_returns(log["position"], hold, tbill_d, per_side)
    strat_r = strat_r[strat_r.index.isin(log.index)]  # solo días ya logueados

    strat_cum = (1.0 + strat_r).cumprod()
    bh_cum = (1.0 + hold[hold.index.isin(strat_r.index)]).cumprod()

    kpis = {
        "current_position": current_position,
        "start_date": start_date,
        "last_date": last_date,
        "n_days": n_days,
        "n_entries": n_entries,
        "n_exits": n_exits,
        "last_close": float(log["close"].iloc[-1]),
        "last_rsi2": float(log["rsi2"].iloc[-1]),
        "strat_cum_return": float(strat_cum.iloc[-1] - 1.0) if len(strat_cum) else None,
        "bh_cum_return": float(bh_cum.iloc[-1] - 1.0) if len(bh_cum) else None,
        "enough_sample": n_days >= MIN_DAYS_FOR_PERFORMANCE,
        "generated_at": now,
    }

    return {
        "has_data": True,
        "kpis": kpis,
        "log": log,
        "trades": trades,
        "strat_cum": strat_cum,
        "bh_cum": bh_cum,
        "config": {"entry_below": pc.entry_below, "exit_above": pc.exit_above,
                   "trend_sma_days": pc.trend_sma_days},
    }


# ---------------------------------------------------------------------------
# Render — SVG a mano, cero dependencias/CDN nuevas (mismo espíritu $0 que
# el dashboard viejo: "cero dependencias nuevas").
# ---------------------------------------------------------------------------

def _svg_line(series: pd.Series, *, width: int = 760, height: int = 200,
             color: str = "#2563eb", y_fmt: str = "{:.1f}",
             shade_where: pd.Series | None = None) -> str:
    vals = series.dropna()
    if len(vals) < 2:
        return (f'<svg width="{width}" height="{height}" role="img">'
                f'<text x="10" y="{height // 2}" fill="#888">'
                f'datos insuficientes todavía</text></svg>')
    pad_l, pad_r, pad_t, pad_b = 46, 10, 14, 20
    xs = np.linspace(pad_l, width - pad_r, len(vals))
    lo, hi = float(vals.min()), float(vals.max())
    span = (hi - lo) or 1.0
    ys = height - pad_b - (vals.to_numpy() - lo) / span * (height - pad_t - pad_b)

    shading = ""
    if shade_where is not None:
        flags = shade_where.reindex(vals.index).fillna(0.0).to_numpy()
        in_run = False
        run_start = 0
        rects = []
        for i, f in enumerate(flags):
            if f == 1.0 and not in_run:
                in_run, run_start = True, i
            elif f != 1.0 and in_run:
                rects.append((run_start, i - 1))
                in_run = False
        if in_run:
            rects.append((run_start, len(flags) - 1))
        for a, b in rects:
            x0, x1 = xs[a], xs[b] if b < len(xs) else xs[-1]
            shading += (f'<rect x="{x0:.1f}" y="{pad_t}" width="{max(x1 - x0, 2):.1f}" '
                       f'height="{height - pad_t - pad_b}" fill="#2563eb" opacity="0.10"/>')

    points = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
    path = f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>'
    labels = (f'<text x="4" y="{height - pad_b + 4}" font-size="10" fill="#667085">'
             f'{y_fmt.format(lo)}</text>'
             f'<text x="4" y="{pad_t + 8}" font-size="10" fill="#667085">'
             f'{y_fmt.format(hi)}</text>')
    return (f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
           f'role="img">{shading}{labels}{path}</svg>')


def _fmt_pct(x: float | None) -> str:
    return "—" if x is None else f"{x:+.1%}"


def render_html(snap: dict) -> str:
    if not snap.get("has_data"):
        return (
            "<!doctype html><html><head><meta charset='utf-8'>"
            "<title>Paper trading RSI-2</title></head>"
            "<body style='font-family:system-ui;padding:40px;max-width:640px;margin:auto'>"
            "<h1>Paper trading RSI-2</h1>"
            "<p>Todavía no hay datos registrados — el sistema arranca con la "
            "primera corrida exitosa de GitHub Actions.</p></body></html>"
        )

    k = snap["kpis"]
    log = snap["log"]
    trades = snap["trades"]
    pos_label = "EN POSICIÓN (long SPY)" if k["current_position"] == 1.0 else "FUERA (en cash/T-bill)"
    pos_color = "#16a34a" if k["current_position"] == 1.0 else "#64748b"

    caveat = ""
    if not k["enough_sample"]:
        caveat = (
            '<div style="background:#fef3c7;border:1px solid #f59e0b;border-radius:8px;'
            'padding:12px 16px;margin:16px 0;color:#78350f;font-size:14px">'
            f'⚠️ Solo {k["n_days"]} día(s) de muestra (mínimo {MIN_DAYS_FOR_PERFORMANCE} '
            'para que cualquier número de desempeño empiece a significar algo). '
            'Los retornos de abajo son informativos, NO una conclusión.</div>'
        )

    price_chart = _svg_line(log["close"], y_fmt="${:.0f}", shade_where=log["position"])
    rsi_chart = _svg_line(log["rsi2"], color="#dc2626", y_fmt="{:.0f}")
    perf_chart = ""
    if len(snap["strat_cum"]) >= 2:
        combined = pd.DataFrame({
            "estrategia": snap["strat_cum"] * 100.0,
            "B&H SPY": snap["bh_cum"] * 100.0,
        })
        perf_chart = (
            '<div style="display:flex;gap:24px;font-size:13px;margin-bottom:6px">'
            '<span style="color:#2563eb">■ RSI-2 (paper)</span>'
            '<span style="color:#9333ea">■ Comprar-y-mantener SPY</span></div>'
            + _svg_line(combined["estrategia"], color="#2563eb", y_fmt="{:.0f}")
            + _svg_line(combined["B&H SPY"], color="#9333ea", y_fmt="{:.0f}")
        )

    trade_rows = "".join(
        f'<tr><td>{dt.date()}</td><td>{row["action"]}</td><td>${row["close"]:.2f}</td></tr>'
        for dt, row in trades.iloc[::-1].iterrows()
    ) or '<tr><td colspan="3" style="color:#888">sin operaciones todavía</td></tr>'

    recent_rows = "".join(
        f'<tr><td>{dt.date()}</td><td>${row["close"]:.2f}</td>'
        f'<td>{row["rsi2"]:.1f}</td><td>{"SÍ" if row["above_trend"] else "no"}</td>'
        f'<td>{"dentro" if row["position"] == 1.0 else "fuera"}</td>'
        f'<td>{row["action"] or "—"}</td></tr>'
        for dt, row in log.tail(20).iloc[::-1].iterrows()
    )

    return f"""<!doctype html>
<html lang="es"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Paper trading RSI-2 — IA TRADING</title>
<style>
  body {{ font-family: -apple-system, system-ui, sans-serif; max-width: 820px;
         margin: 0 auto; padding: 24px 16px 60px; color: #1e293b; background: #fff; }}
  h1 {{ font-size: 22px; margin-bottom: 4px; }}
  .sub {{ color: #667085; font-size: 14px; margin-bottom: 20px; }}
  .kpis {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
          gap: 12px; margin: 20px 0; }}
  .kpi {{ border: 1px solid #e2e8f0; border-radius: 10px; padding: 12px 14px; }}
  .kpi .label {{ font-size: 12px; color: #667085; }}
  .kpi .value {{ font-size: 20px; font-weight: 600; margin-top: 2px; }}
  section {{ margin: 28px 0; }}
  h2 {{ font-size: 16px; border-bottom: 1px solid #e2e8f0; padding-bottom: 6px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  th, td {{ text-align: left; padding: 6px 8px; border-bottom: 1px solid #f1f5f9; }}
  th {{ color: #667085; font-weight: 500; }}
  footer {{ margin-top: 40px; font-size: 12px; color: #94a3b8; }}
  a {{ color: #2563eb; }}
</style>
</head><body>
<h1>Paper trading RSI-2 — S&amp;P 500</h1>
<p class="sub">100% simulado, sin broker, sin capital real. Config ya seleccionada por
train el 2026-07-11 (entry&lt;{snap['config']['entry_below']:.0f},
exit&gt;{snap['config']['exit_above']:.0f}, SMA{snap['config']['trend_sma_days']}) —
no se re-tunea. Arrancó {k['start_date'].date()}.</p>

{caveat}

<div class="kpis">
  <div class="kpi"><div class="label">Estado actual</div>
    <div class="value" style="color:{pos_color}">{pos_label}</div></div>
  <div class="kpi"><div class="label">Último cierre SPY</div>
    <div class="value">${k['last_close']:.2f}</div></div>
  <div class="kpi"><div class="label">RSI-2 actual</div>
    <div class="value">{k['last_rsi2']:.1f}</div></div>
  <div class="kpi"><div class="label">Días de muestra</div>
    <div class="value">{k['n_days']}</div></div>
  <div class="kpi"><div class="label">Entradas / salidas</div>
    <div class="value">{k['n_entries']} / {k['n_exits']}</div></div>
  <div class="kpi"><div class="label">Retorno RSI-2 (desde arranque)</div>
    <div class="value">{_fmt_pct(k['strat_cum_return'])}</div></div>
  <div class="kpi"><div class="label">Retorno B&amp;H SPY (mismo periodo)</div>
    <div class="value">{_fmt_pct(k['bh_cum_return'])}</div></div>
  <div class="kpi"><div class="label">Última actualización</div>
    <div class="value">{k['last_date'].date()}</div></div>
</div>

<section>
  <h2>Precio SPY (sombreado = en posición)</h2>
  {price_chart}
</section>

<section>
  <h2>RSI-2</h2>
  {rsi_chart}
</section>

<section>
  <h2>Retorno acumulado — estrategia vs comprar-y-mantener (base 100)</h2>
  {perf_chart or '<p style="color:#888">Hacen falta al menos 2 días para graficar esto.</p>'}
</section>

<section>
  <h2>Operaciones</h2>
  <table><thead><tr><th>Fecha</th><th>Acción</th><th>Precio</th></tr></thead>
  <tbody>{trade_rows}</tbody></table>
</section>

<section>
  <h2>Últimos 20 días</h2>
  <table><thead><tr><th>Fecha</th><th>Cierre</th><th>RSI-2</th>
    <th>&gt;SMA</th><th>Posición</th><th>Acción</th></tr></thead>
  <tbody>{recent_rows}</tbody></table>
</section>

<footer>
  Generado automáticamente por GitHub Actions el
  {k['generated_at'].strftime('%Y-%m-%d %H:%M UTC')}
  (esta marca cambia en CADA corrida exitosa, aunque no haya día de mercado
  nuevo — si deja de moverse, el sistema dejó de correr).
  <a href="https://github.com/eduresvez8-code/IA-TRADING-BOT">Código y metodología</a> ·
  <a href="https://github.com/eduresvez8-code/IA-TRADING-BOT/actions/workflows/paper_trading_rsi2.yml">
    Historial de corridas</a> ·
  <a href="https://github.com/eduresvez8-code/IA-TRADING-BOT/blob/main/paper_trading/rsi2/daily_log.csv">
    Log crudo (CSV)</a>
</footer>
</body></html>"""


def main() -> int:
    cfg = load_settings()
    snap = build_snapshot(cfg)
    html = render_html(snap)
    out_dir = Path("docs/dashboard")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "index.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"Dashboard generado: {out_path} ({len(html)} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
