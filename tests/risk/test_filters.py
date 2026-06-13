"""Tests de los helpers de microestructura (floor a stepSize, redondeo a tickSize).

El test estrella es `test_floor_evita_el_bug_de_float`: demuestra por qué se usa
Decimal y no float.
"""

from decimal import Decimal

from src.risk.filters import floor_to_step, round_to_tick


def test_floor_trunca_hacia_abajo():
    assert floor_to_step(1.3333, Decimal("0.001")) == Decimal("1.333")


def test_floor_nunca_redondea_hacia_arriba():
    # 1.9999 con paso 1 → 1 (no 2): subir violaría riesgo y saldo libre.
    assert floor_to_step(1.9999, Decimal("1")) == Decimal("1")


def test_floor_evita_el_bug_de_float():
    # En float, 0.3 // 0.1 == 2.0 (porque 0.3/0.1 == 2.9999…). Con Decimal, 3.
    assert (0.3 // 0.1) == 2.0                      # el bug que NO queremos
    assert floor_to_step(0.3, Decimal("0.1")) == Decimal("0.3")  # correcto


def test_round_to_tick_half_up():
    assert round_to_tick(925.017, Decimal("0.01")) == Decimal("925.02")


def test_round_to_tick_baja_cuando_corresponde():
    assert round_to_tick(100.04, Decimal("0.1")) == Decimal("100.0")


def test_round_to_tick_grueso():
    # Tick de 0.5: 949.99 → 950.0 (múltiplo de 0.5).
    r = round_to_tick(949.99, Decimal("0.5"))
    assert r == Decimal("950.0")
    assert r % Decimal("0.5") == 0
