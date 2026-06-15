"""Tests del índice Fear & Greed — sin red (cliente fake)."""

import pytest

from src.sentiment.fear_greed import fear_greed_to_score, fetch_fear_greed


class FakeResp:
    def __init__(self, data):
        self._data = data

    def json(self):
        return self._data


class FakeClient:
    def __init__(self, data):
        self._data = data
        self.calls = []

    async def get(self, url, params=None):
        self.calls.append((url, params))
        return FakeResp(self._data)


# ---------- mapeo a score ----------

def test_mapeo_extremos_y_centro():
    assert fear_greed_to_score(0) == -1.0    # miedo extremo → bajista
    assert fear_greed_to_score(50) == 0.0    # neutro
    assert fear_greed_to_score(100) == 1.0   # codicia extrema → alcista


def test_mapeo_intermedio():
    assert fear_greed_to_score(75) == pytest.approx(0.5)
    assert fear_greed_to_score(20) == pytest.approx(-0.6)


# ---------- fetch ----------

async def test_fetch_parsea_lecturas():
    data = {"name": "Fear and Greed Index", "data": [
        {"value": "75", "value_classification": "Greed", "timestamp": "1700000000"},
        {"value": "20", "value_classification": "Fear", "timestamp": "1699913600"},
    ]}
    out = await fetch_fear_greed(client=FakeClient(data))
    assert len(out) == 2
    ts, value, classification = out[0]
    assert value == 75 and classification == "Greed"
    assert ts.tzinfo is not None


async def test_fetch_ignora_filas_malformadas():
    data = {"data": [
        {"value": "60", "value_classification": "Greed", "timestamp": "1700000000"},
        {"value": "n/a", "timestamp": "bad"},   # malformada → se ignora
    ]}
    out = await fetch_fear_greed(client=FakeClient(data))
    assert len(out) == 1 and out[0][1] == 60
