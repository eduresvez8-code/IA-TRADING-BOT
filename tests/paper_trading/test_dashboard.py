"""Tests de la parte PURA del dashboard (sin red, sin disco).

`build_snapshot` toca red (descarga SPY/IRX) y disco (lee el CSV) — igual
que `download_symbols`, no se testea unitariamente (convención ya
establecida en el proyecto: "Parte de RED" vs "parte PURA").
"""

import numpy as np
import pandas as pd

from src.paper_trading.dashboard import MIN_DAYS_FOR_PERFORMANCE, _fmt_pct, _svg_line, render_html


def test_fmt_pct_formatea_con_signo():
    assert _fmt_pct(0.0523) == "+5.2%"
    assert _fmt_pct(-0.0523) == "-5.2%"
    assert _fmt_pct(None) == "—"


def test_svg_line_datos_insuficientes_no_truena():
    out = _svg_line(pd.Series([1.0], index=[pd.Timestamp("2020-01-01")]))
    assert "datos insuficientes" in out
    assert "<svg" in out


def test_svg_line_produce_svg_valido_con_datos():
    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    s = pd.Series(np.linspace(100, 110, 10), index=dates)
    out = _svg_line(s)
    assert out.startswith("<svg")
    assert "<polyline" in out
    assert out.count(",") >= 10  # al menos un par de coordenadas por punto


def test_svg_line_sombrea_donde_shade_where_es_1():
    dates = pd.date_range("2020-01-01", periods=5, freq="D")
    s = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0], index=dates)
    shade = pd.Series([0.0, 1.0, 1.0, 0.0, 0.0], index=dates)
    out = _svg_line(s, shade_where=shade)
    assert "<rect" in out


def _fake_snapshot(n_days: int, current_position: float = 0.0) -> dict:
    dates = pd.date_range("2026-07-10", periods=n_days, freq="B")
    log = pd.DataFrame({
        "close": np.linspace(500.0, 510.0, n_days),
        "rsi2": np.full(n_days, 50.0),
        "sma_trend": np.full(n_days, 480.0),
        "above_trend": np.full(n_days, True),
        "position": np.full(n_days, current_position),
        "action": [""] * n_days,
    }, index=dates)
    log.index.name = "date"
    trades = log.iloc[0:0]
    strat_cum = pd.Series(np.linspace(1.0, 1.01, n_days), index=dates)
    bh_cum = pd.Series(np.linspace(1.0, 1.02, n_days), index=dates)
    return {
        "has_data": True,
        "kpis": {
            "current_position": current_position, "start_date": dates[0],
            "last_date": dates[-1], "n_days": n_days, "n_entries": 0, "n_exits": 0,
            "last_close": float(log["close"].iloc[-1]), "last_rsi2": 50.0,
            "strat_cum_return": float(strat_cum.iloc[-1] - 1.0),
            "bh_cum_return": float(bh_cum.iloc[-1] - 1.0),
            "enough_sample": n_days >= MIN_DAYS_FOR_PERFORMANCE,
            "generated_at": pd.Timestamp("2026-07-11T22:05:00Z"),
        },
        "log": log, "trades": trades, "strat_cum": strat_cum, "bh_cum": bh_cum,
        "config": {"entry_below": 10.0, "exit_above": 70.0, "trend_sma_days": 200},
    }


def test_render_html_sin_datos_no_truena():
    html = render_html({"has_data": False})
    assert "<html" in html
    assert "Todavía no hay datos" in html


def test_render_html_muestra_advertencia_con_poca_muestra():
    html = render_html(_fake_snapshot(5))
    assert "⚠️" in html
    assert "5 día(s)" in html


def test_render_html_sin_advertencia_con_muestra_suficiente():
    html = render_html(_fake_snapshot(MIN_DAYS_FOR_PERFORMANCE))
    assert "⚠️" not in html


def test_render_html_refleja_posicion_actual():
    dentro = render_html(_fake_snapshot(3, current_position=1.0))
    fuera = render_html(_fake_snapshot(3, current_position=0.0))
    assert "EN POSICIÓN" in dentro
    assert "FUERA" in fuera


def test_render_html_muestra_marca_de_generado_el():
    # Distingue "no hubo día nuevo" de "el sistema dejó de correr": esta
    # marca debe reflejar el momento de la CORRIDA, no la última fecha del log.
    html = render_html(_fake_snapshot(3))
    assert "2026-07-11 22:05 UTC" in html
