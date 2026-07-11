"""Capa de datos S&P 500: precios diarios (yfinance) + membresía histórica.

Dos problemas que esta capa resuelve, y por qué importan:

1. **Sesgo de supervivencia** (la lección cara de 2026-07-08): si eliges hoy
   las acciones a backtestear, eliges — sin querer — las que sobrevivieron y
   ganaron. La cura es la membresía PUNTO-EN-EL-TIEMPO: qué tickers estaban en
   el índice en CADA fecha (CSV gratis del repo GitHub `fja05680/sp500`,
   formato `date,tickers`). El ranking de cada mes solo ve los miembros de ese
   mes, incluidos los que luego quebraron (AAMRQ, ENRNQ...).

   Limitación DECLARADA: la lista es histórica real, pero yfinance no tiene
   precios de muchas empresas deslistadas. Esa cobertura se MIDE
   (`coverage_report`) y se reporta; el sesgo residual (faltan justo los
   muertos) infla al alza los resultados long-only y así se declara en el
   protocolo (docs/research/2026-07-11_protocolo_sp500.md §4.1).

2. **Retorno total vs retorno de precio**: comparar una estrategia contra el
   índice de PRECIO (^GSPC) regala ~2%/año de ventaja ficticia (los dividendos
   no están). Por eso el benchmark es SPY con `auto_adjust=True` (dividendos
   reinvertidos) y ^GSPC solo sirve para señales de historia larga.

Uso (config-driven, Cero Hardcoding):

    uv run python -m src.data.sp500 --constituents   # CSV de membresía
    uv run python -m src.data.sp500 --core           # SPY, ^GSPC, ^IRX, extras
    uv run python -m src.data.sp500 --universe       # todos los miembros históricos
    uv run python -m src.data.sp500 --all

Los parquet quedan en `data.dir` (regenerables, fuera de git). La descarga es
IDEMPOTENTE: lo ya descargado se salta (borrar el parquet fuerza re-descarga).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import httpx
import pandas as pd

from src.core.config import DataConfig, Settings, load_settings

# Días de trading por año en bolsa EE.UU. (anualizaciones y conversión T-bill).
TRADING_DAYS_PER_YEAR = 252

_PRICE_COLS = ["open", "high", "low", "close", "volume"]


# ---------------------------------------------------------------------------
# Parte PURA (testeable sin red)
# ---------------------------------------------------------------------------

def normalize_ticker_for_yahoo(ticker: str) -> str:
    """'BF.B' → 'BF-B': Yahoo usa guion donde el CSV usa punto (clases de acción)."""
    return ticker.strip().replace(".", "-")


def load_membership(csv_path: Path) -> pd.DataFrame:
    """CSV `date,tickers` → DataFrame ordenado con columna `tickers` (lista).

    Cada fila es la composición VIGENTE desde esa fecha (snapshot completo, no
    un delta): la membresía en una fecha t es la última fila con date ≤ t.
    """
    raw = pd.read_csv(csv_path)
    if not {"date", "tickers"}.issubset(raw.columns):
        raise ValueError(
            f"CSV de membresía malformado: esperaba columnas date,tickers, "
            f"llegaron {list(raw.columns)}"
        )
    out = pd.DataFrame({
        "date": pd.to_datetime(raw["date"], utc=True),
        "tickers": raw["tickers"].map(
            lambda s: [t.strip() for t in str(s).split(",") if t.strip()]
        ),
    })
    return out.sort_values("date").reset_index(drop=True)


def members_asof(membership: pd.DataFrame, when: pd.Timestamp) -> list[str]:
    """Miembros del índice vigentes en `when` (última fila con date ≤ when).

    Devuelve [] si `when` es anterior al primer snapshot: mejor universo vacío
    (la fecha se descarta por cobertura) que un universo inventado.
    """
    when = pd.Timestamp(when)
    if when.tzinfo is None:
        when = when.tz_localize("UTC")
    eligible = membership[membership["date"] <= when]
    if eligible.empty:
        return []
    return list(eligible.iloc[-1]["tickers"])


def all_tickers_ever(membership: pd.DataFrame) -> list[str]:
    """Unión de todos los tickers que alguna vez estuvieron en el índice."""
    seen: set[str] = set()
    for row in membership["tickers"]:
        seen.update(row)
    return sorted(seen)


def tbill_daily_return(irx_close: pd.Series) -> pd.Series:
    """^IRX (yield anualizado en %) → retorno diario simple del cash.

    Convención declarada (simple, no compuesta): r_d = (y/100) / 252. A los
    niveles de tasas relevantes (0-6%) la diferencia con la capitalización
    exacta es <0.1 pb/día — irrelevante frente a los 2 pb de slippage.
    El yield vigente se propaga a días sin dato (ffill).
    """
    return (irx_close.ffill() / 100.0) / TRADING_DAYS_PER_YEAR


def prices_path(data_dir: Path, symbol: str) -> Path:
    """Ruta canónica del parquet de un símbolo (^GSPC → ^GSPC.parquet)."""
    return data_dir / "prices" / f"{symbol}.parquet"


def load_prices(data_dir: Path, symbol: str) -> pd.DataFrame:
    """Lee el parquet de un símbolo: open_time UTC + open/high/low/close/volume."""
    df = pd.read_parquet(prices_path(data_dir, symbol))
    return df.sort_values("open_time").reset_index(drop=True)


def save_prices(data_dir: Path, symbol: str, df: pd.DataFrame) -> Path:
    """Normaliza y guarda un DataFrame OHLCV en el layout canónico."""
    dest = prices_path(data_dir, symbol)
    dest.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    out["open_time"] = pd.to_datetime(out["open_time"], utc=True)
    out = out[["open_time", *_PRICE_COLS]].sort_values("open_time")
    out = out.dropna(subset=["close"]).reset_index(drop=True)
    out.to_parquet(dest, index=False)
    return dest


def coverage_report(membership: pd.DataFrame, available: set[str],
                    dates: list[pd.Timestamp]) -> pd.DataFrame:
    """Cobertura punto-en-el-tiempo: en cada fecha, ¿qué fracción de los
    miembros del índice tiene precio descargado?

    Es EL número que acota cuánto sesgo de supervivencia residual queda en el
    universo (los tickers sin precio son, en su mayoría, los deslistados).
    """
    rows = []
    for d in dates:
        members = members_asof(membership, d)
        if not members:
            rows.append({"date": d, "members": 0, "with_price": 0, "coverage": 0.0})
            continue
        have = sum(1 for t in members if normalize_ticker_for_yahoo(t) in available)
        rows.append({"date": d, "members": len(members), "with_price": have,
                     "coverage": have / len(members)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Parte de RED (thin wrappers; la lógica de arriba no toca la red)
# ---------------------------------------------------------------------------

def resolve_constituents_url(repo: str) -> str:
    """Resuelve la URL raw del CSV de membresía más completo del repo.

    El nombre exacto del archivo cambia con las actualizaciones del autor
    (incluye "(Updated)"), así que se lista el repo vía la API de GitHub y se
    elige el CSV de componentes históricos más reciente — nada hardcodeado.
    """
    api = f"https://api.github.com/repos/{repo}/contents/"
    r = httpx.get(api, timeout=30.0,
                  headers={"Accept": "application/vnd.github+json"})
    r.raise_for_status()
    candidates = [
        item for item in r.json()
        if item["name"].lower().endswith(".csv")
        and "historical components" in item["name"].lower()
    ]
    if not candidates:
        raise RuntimeError(f"no encontré CSV de componentes históricos en {repo}")
    # Preferir el "(Updated)" si existe; si no, el primero.
    candidates.sort(key=lambda it: ("updated" not in it["name"].lower(), it["name"]))
    return candidates[0]["download_url"]


def download_constituents(cfg: DataConfig) -> Path:
    """Descarga el CSV de membresía histórica a `data.dir`/constituents.csv."""
    dest = Path(cfg.dir) / "constituents.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = resolve_constituents_url(cfg.constituents_repo)
    r = httpx.get(url, timeout=120.0, follow_redirects=True)
    r.raise_for_status()
    dest.write_bytes(r.content)
    # Validación inmediata: si el formato cambió, fallar aquí, no en el backtest.
    load_membership(dest)
    return dest


def download_symbols(symbols: list[str], cfg: DataConfig) -> tuple[list[str], list[str]]:
    """Descarga histórico diario (auto_adjust) de `symbols` en lotes.

    Idempotente: los símbolos con parquet existente se saltan. Devuelve
    (descargados_ok, sin_datos). Los sin_datos se anotan en missing_tickers.txt
    para el reporte de cobertura.
    """
    import yfinance as yf

    data_dir = Path(cfg.dir)
    todo = [s for s in symbols if not prices_path(data_dir, s).exists()]
    ok: list[str] = [s for s in symbols if s not in todo]
    missing: list[str] = []

    for i in range(0, len(todo), cfg.batch_size):
        batch = todo[i:i + cfg.batch_size]
        raw = yf.download(
            batch, start=cfg.start_date, auto_adjust=True, progress=False,
            group_by="ticker", threads=True,
        )
        for sym in batch:
            try:
                sub = raw[sym] if len(batch) > 1 else raw
            except KeyError:
                missing.append(sym)
                continue
            sub = sub.dropna(subset=["Close"]) if "Close" in sub.columns else pd.DataFrame()
            if sub.empty:
                missing.append(sym)
                continue
            df = pd.DataFrame({
                "open_time": pd.to_datetime(sub.index, utc=True),
                "open": sub["Open"].to_numpy(dtype=float),
                "high": sub["High"].to_numpy(dtype=float),
                "low": sub["Low"].to_numpy(dtype=float),
                "close": sub["Close"].to_numpy(dtype=float),
                "volume": sub["Volume"].to_numpy(dtype=float),
            })
            save_prices(data_dir, sym, df)
            ok.append(sym)
        if i + cfg.batch_size < len(todo):
            time.sleep(cfg.pause_seconds)

    if missing:
        miss_file = data_dir / "missing_tickers.txt"
        existing = set()
        if miss_file.exists():
            existing = set(miss_file.read_text().split())
        miss_file.write_text("\n".join(sorted(existing | set(missing))) + "\n")
    return ok, missing


def download_core(settings: Settings) -> None:
    """Series núcleo: benchmark (SPY), índice (^GSPC), T-bill (^IRX) y extras."""
    core = [settings.market.benchmark_symbol, settings.market.index_symbol,
            settings.market.tbill_symbol, *settings.data.extra_symbols]
    ok, missing = download_symbols(core, settings.data)
    print(f"core: {len(ok)} ok, {len(missing)} sin datos {missing or ''}")


def download_universe(settings: Settings) -> None:
    """Todos los tickers que alguna vez fueron miembros del índice."""
    csv_path = Path(settings.data.dir) / "constituents.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} no existe: corre primero --constituents")
    membership = load_membership(csv_path)
    tickers = [normalize_ticker_for_yahoo(t) for t in all_tickers_ever(membership)]
    ok, missing = download_symbols(tickers, settings.data)
    print(f"universo: {len(ok)} con datos, {len(missing)} sin datos "
          f"(deslistados sin histórico en yfinance — se miden en cobertura)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Descarga de datos S&P 500")
    parser.add_argument("--constituents", action="store_true")
    parser.add_argument("--core", action="store_true")
    parser.add_argument("--universe", action="store_true")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args(argv)

    settings = load_settings()
    if args.all or args.constituents:
        path = download_constituents(settings.data)
        print(f"constituyentes → {path}")
    if args.all or args.core:
        download_core(settings)
    if args.all or args.universe:
        download_universe(settings)
    if not any([args.all, args.constituents, args.core, args.universe]):
        parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
