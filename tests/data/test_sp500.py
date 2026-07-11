"""Tests de la parte PURA de la capa de datos (sin red).

La membresía punto-en-el-tiempo es la defensa contra el sesgo de supervivencia:
estos tests verifican que la lógica as-of nunca mire al futuro y que la
cobertura se mida como se declara en el protocolo.
"""

from pathlib import Path

import pandas as pd
import pytest

from src.data.sp500 import (
    all_tickers_ever,
    coverage_report,
    load_membership,
    load_prices,
    members_asof,
    normalize_ticker_for_yahoo,
    save_prices,
    tbill_daily_return,
)


@pytest.fixture
def membership(tmp_path: Path) -> pd.DataFrame:
    csv = tmp_path / "constituents.csv"
    csv.write_text(
        'date,tickers\n'
        '1996-01-02,"AAA,BBB,CCC"\n'
        '2000-06-15,"AAA,BBB,DDD"\n'
        '2020-03-01,"AAA,EEE"\n'
    )
    return load_membership(csv)


# ---------- membresía punto-en-el-tiempo ----------

def test_load_membership_parsea_listas(membership):
    assert list(membership["tickers"].iloc[0]) == ["AAA", "BBB", "CCC"]
    assert membership["date"].is_monotonic_increasing


def test_members_asof_toma_el_snapshot_vigente(membership):
    # Entre 2000-06-15 y 2020-03-01 rige la fila del 2000 (CCC salió, DDD entró).
    assert members_asof(membership, pd.Timestamp("2010-01-01")) == ["AAA", "BBB", "DDD"]


def test_members_asof_no_mira_al_futuro(membership):
    # Un día ANTES del snapshot del 2000, rige el de 1996: DDD aún no existe.
    assert members_asof(membership, pd.Timestamp("2000-06-14")) == ["AAA", "BBB", "CCC"]


def test_members_asof_antes_del_primer_snapshot_es_vacio(membership):
    # Mejor universo vacío (la fecha se descarta) que un universo inventado.
    assert members_asof(membership, pd.Timestamp("1990-01-01")) == []


def test_all_tickers_ever_une_todo(membership):
    assert all_tickers_ever(membership) == ["AAA", "BBB", "CCC", "DDD", "EEE"]


def test_membership_rechaza_csv_malformado(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("fecha,simbolos\n2020-01-01,AAA\n")
    with pytest.raises(ValueError, match="malformado"):
        load_membership(bad)


# ---------- normalización de tickers ----------

def test_normalize_ticker_clases_de_accion():
    assert normalize_ticker_for_yahoo("BF.B") == "BF-B"
    assert normalize_ticker_for_yahoo(" BRK.B ") == "BRK-B"
    assert normalize_ticker_for_yahoo("AAPL") == "AAPL"


# ---------- T-bill → retorno diario del cash ----------

def test_tbill_daily_return_convencion_simple():
    # Yield 5.04% anual → 0.0504/252 = 2 pb/día.
    y = pd.Series([5.04, 5.04])
    r = tbill_daily_return(y)
    assert r.iloc[0] == pytest.approx(0.0002)


def test_tbill_daily_return_propaga_huecos():
    y = pd.Series([5.04, float("nan"), float("nan")])
    r = tbill_daily_return(y)
    assert r.iloc[2] == pytest.approx(0.0002)


# ---------- persistencia parquet ----------

def test_save_y_load_prices_roundtrip(tmp_path):
    df = pd.DataFrame({
        "open_time": pd.date_range("2020-01-01", periods=3, freq="D", tz="UTC"),
        "open": [1.0, 2.0, 3.0], "high": [1.1, 2.1, 3.1],
        "low": [0.9, 1.9, 2.9], "close": [1.05, 2.05, 3.05],
        "volume": [100.0, 200.0, 300.0],
    })
    save_prices(tmp_path, "TEST", df)
    back = load_prices(tmp_path, "TEST")
    assert len(back) == 3
    assert back["close"].iloc[-1] == 3.05
    assert str(back["open_time"].dt.tz) == "UTC"


def test_save_prices_descarta_filas_sin_cierre(tmp_path):
    df = pd.DataFrame({
        "open_time": pd.date_range("2020-01-01", periods=3, freq="D", tz="UTC"),
        "open": [1.0, 2.0, 3.0], "high": [1.1, 2.1, 3.1],
        "low": [0.9, 1.9, 2.9], "close": [1.05, float("nan"), 3.05],
        "volume": [100.0, 200.0, 300.0],
    })
    save_prices(tmp_path, "TEST", df)
    assert len(load_prices(tmp_path, "TEST")) == 2


# ---------- cobertura (el termómetro del sesgo de supervivencia) ----------

def test_coverage_report_mide_fraccion_con_precio(membership):
    # En 2010 los miembros son AAA,BBB,DDD; solo AAA y BBB tienen precio → 2/3.
    rep = coverage_report(membership, available={"AAA", "BBB"},
                          dates=[pd.Timestamp("2010-01-01", tz="UTC")])
    assert rep["members"].iloc[0] == 3
    assert rep["coverage"].iloc[0] == pytest.approx(2 / 3)


def test_coverage_report_fecha_sin_universo_da_cero(membership):
    rep = coverage_report(membership, available={"AAA"},
                          dates=[pd.Timestamp("1990-01-01", tz="UTC")])
    assert rep["coverage"].iloc[0] == 0.0
