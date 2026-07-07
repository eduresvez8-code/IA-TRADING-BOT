"""Tests de src/data/download_metrics.py: helpers puros (sin red)."""

import io
import zipfile
from datetime import date

import pandas as pd

from src.data.download_metrics import consolidate_days, metrics_url, parse_metrics_zip


def _zip_with_csv(csv_text: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("X-metrics-2024-01-01.csv", csv_text)
    return buf.getvalue()


def test_metrics_url_formato_binance_vision():
    url = metrics_url("BTCUSDT", date(2024, 6, 3))
    assert url == ("https://data.binance.vision/data/futures/um/daily/metrics/"
                   "BTCUSDT/BTCUSDT-metrics-2024-06-03.zip")


def test_parse_metrics_zip_timestamps_utc():
    csv = ("create_time,symbol,sum_open_interest\n"
           "2024-01-01 00:05:00,BTCUSDT,100.5\n"
           "2024-01-01 00:10:00,BTCUSDT,101.0\n")
    df = parse_metrics_zip(_zip_with_csv(csv))
    assert len(df) == 2
    # UTC explícito: sin tz el merge contra velas UTC correría el dato en silencio.
    assert str(df["create_time"].dt.tz) == "UTC"
    assert df["sum_open_interest"].tolist() == [100.5, 101.0]


def test_consolidate_days_deduplica_y_ordena():
    a = pd.DataFrame({
        "create_time": pd.to_datetime(["2024-01-02 00:00", "2024-01-01 23:55"], utc=True),
        "sum_open_interest": [2.0, 1.0],
    })
    b = pd.DataFrame({
        # El registro de medianoche viene repetido en el zip del día siguiente.
        "create_time": pd.to_datetime(["2024-01-02 00:00", "2024-01-02 00:05"], utc=True),
        "sum_open_interest": [2.0, 3.0],
    })
    out = consolidate_days([a, b])
    assert len(out) == 3  # el duplicado de medianoche se elimina
    assert out["create_time"].is_monotonic_increasing
    assert out["sum_open_interest"].tolist() == [1.0, 2.0, 3.0]
