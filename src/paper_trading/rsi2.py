"""Forward/paper trading real de RSI-2 — único camino activo que queda tras
el cierre de la búsqueda de estrategias (2026-07-25, ver CLAUDE.md).

Por qué esto es distinto de todo lo demás en el proyecto: cada backtest de
`backtest/` mide el pasado, con el riesgo (ya materializado varias veces en
este proyecto) de encontrar patrones que son ruido y no ventaja real. El
forward trading no tiene ese problema — nadie puede ver el futuro de
antemano — pero exige TIEMPO REAL, no atajos. Este módulo registra, día a
día, qué habría hecho la config de RSI-2 YA seleccionada por train el
2026-07-11 (entry<10, exit>70, SMA200), sin re-tunear nada.

100% SIMULADO — jamás toca un broker ni dinero real (CLAUDE.md §Seguridad).

Corre vía GitHub Actions (.github/workflows/paper_trading_rsi2.yml), una vez
al día, independiente de si el Mac de Eduardo está encendido. Cada corrida
empieza con un checkout LIMPIO del repo (sin `data/`, que está fuera de git
por diseño) — por eso siempre descarga el historial de precios completo, no
hace falta lógica de actualización incremental.

El log (`paper_trading/rsi2/daily_log.csv`) es APPEND-ONLY y versionado en
git: es la memoria del experimento. Cada fila usa la MISMA convención causal
que el backtest (`backtest/sp500_families.py`): `position` en la fila del
día t es la señal decidida con el CIERRE de t — la exposición real de
mercado llega un día después, a la apertura de t+1 (ver
`daily_strategy_returns`). Este módulo no calcula esa exposición: solo
registra la señal cruda, día a día, para que el análisis de desempeño
(cuando haya suficiente muestra) reutilice las mismas funciones ya
probadas del backtest sobre este mismo log.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from backtest.sp500_families import daily_close_series, rsi_reversion_daily_position
from src.core.config import Settings, load_settings
from src.data.sp500 import download_symbols, load_prices
from src.quant.indicators import rsi as _rsi

LOG_COLUMNS = ["close", "rsi2", "sma_trend", "above_trend", "position", "action"]


def compute_daily_rows(daily_df: pd.DataFrame, *, rsi_period: int, entry_below: float,
                       exit_above: float, trend_sma_days: int,
                       since: pd.Timestamp) -> pd.DataFrame:
    """Estado crudo de la señal RSI-2, un renglón por día, desde `since`.

    `since` es INCLUSIVO. La posición y la acción se calculan sobre la serie
    COMPLETA (para detectar correctamente una transición justo en el primer
    día de `since`) y solo se recorta al final — así una corrida que retoma
    tras días perdidos reconstruye las transiciones reales, no inventa nada.
    """
    close = daily_close_series(daily_df)
    r = _rsi(close, rsi_period)
    sma = close.rolling(trend_sma_days).mean()
    pos = rsi_reversion_daily_position(
        daily_df, rsi_period=rsi_period, entry_below=entry_below,
        exit_above=exit_above, trend_sma_days=trend_sma_days)

    prev_pos = pos.shift(1).fillna(0.0)
    action = pd.Series("", index=pos.index, dtype=object)
    action[(pos == 1.0) & (prev_pos == 0.0)] = "ENTER"
    action[(pos == 0.0) & (prev_pos == 1.0)] = "EXIT"

    df = pd.DataFrame({
        "close": close, "rsi2": r, "sma_trend": sma,
        "above_trend": close > sma, "position": pos, "action": action,
    })
    return df[df.index >= since]


def _load_existing_log(log_path: Path) -> pd.DataFrame:
    if not log_path.exists():
        return pd.DataFrame(columns=LOG_COLUMNS)
    df = pd.read_csv(log_path, parse_dates=["date"], index_col="date")
    # pandas lee un campo vacío del CSV como NaN, no como "" — sin esto,
    # cualquier comparación `action != ""` se vuelve True para TODOS los
    # días (NaN nunca es igual a nada), marcando cada día como si fuera
    # una transacción real. Se normaliza aquí, en la frontera de lectura,
    # no en cada lugar que consume el log.
    df["action"] = df["action"].fillna("")
    return df


def run(cfg: Settings) -> pd.DataFrame:
    """Descarga precios frescos, calcula filas nuevas, actualiza el log en
    disco. Devuelve SOLO las filas nuevas añadidas en esta corrida (vacío si
    no había nada que registrar — mercado cerrado hoy, o ya se corrió hoy)."""
    pc = cfg.paper_trading.rsi2
    log_dir = Path(pc.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "daily_log.csv"

    data_dir = Path(cfg.data.dir)
    download_symbols([cfg.market.benchmark_symbol], cfg.data)
    spy = load_prices(data_dir, cfg.market.benchmark_symbol)
    last_available = daily_close_series(spy).index.max()

    existing = _load_existing_log(log_path)
    if existing.empty:
        since = last_available  # primer arranque: NUNCA retroactivo al backtest
    else:
        since = existing.index.max() + pd.Timedelta(days=1)

    new_rows = compute_daily_rows(
        spy, rsi_period=pc.rsi_period, entry_below=pc.entry_below,
        exit_above=pc.exit_above, trend_sma_days=pc.trend_sma_days, since=since)
    if not existing.empty:
        new_rows = new_rows[~new_rows.index.isin(existing.index)]
    if new_rows.empty:
        return new_rows

    combined = pd.concat([existing, new_rows]).sort_index()
    combined.index.name = "date"
    combined.to_csv(log_path)
    return new_rows


def main() -> int:
    cfg = load_settings()
    print("=" * 86)
    print("PAPER TRADING RSI-2 — forward real, 100% simulado (sin broker, sin capital real)")
    print(f"config: entry<{cfg.paper_trading.rsi2.entry_below}, "
          f"exit>{cfg.paper_trading.rsi2.exit_above}, "
          f"SMA{cfg.paper_trading.rsi2.trend_sma_days} "
          f"(ya seleccionada por train el 2026-07-11, no re-tuneada)")
    print("=" * 86)

    new_rows = run(cfg)
    if new_rows.empty:
        print("\nSin días nuevos que registrar (mercado cerrado hoy, o ya se corrió hoy).")
        return 0

    print(f"\n{len(new_rows)} día(s) nuevo(s) registrado(s):")
    for dt, row in new_rows.iterrows():
        tag = f"  <- {row['action']}" if row["action"] else ""
        print(f"  {dt.date()}  close={row['close']:.2f}  rsi2={row['rsi2']:.1f}  "
              f"posición={row['position']:.0f}{tag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
