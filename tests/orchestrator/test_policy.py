"""Tests de las políticas puras: gestión de una pierna y reconciliación."""

from src.core.models import PositionSide
from src.orchestrator.policy import (
    PositionAction,
    ReconVerdict,
    classify_reconciliation,
    decide_position_action,
)

LONG, SHORT = PositionSide.LONG, PositionSide.SHORT


# ---- decide_position_action (una pierna por símbolo) ----

def test_plano_y_sin_senal_no_hace_nada():
    assert decide_position_action(None, None) == PositionAction.NONE


def test_plano_con_senal_abre():
    assert decide_position_action(None, LONG) == PositionAction.OPEN
    assert decide_position_action(None, SHORT) == PositionAction.OPEN


def test_misma_direccion_no_duplica():
    assert decide_position_action(LONG, LONG) == PositionAction.NONE
    assert decide_position_action(SHORT, SHORT) == PositionAction.NONE


def test_direccion_opuesta_hace_flip():
    assert decide_position_action(LONG, SHORT) == PositionAction.FLIP
    assert decide_position_action(SHORT, LONG) == PositionAction.FLIP


def test_hold_con_pierna_abierta_no_cierra():
    # HOLD/vetada: los SL/TP gestionan la pierna; el orquestador no la toca.
    assert decide_position_action(LONG, None) == PositionAction.NONE


# ---- classify_reconciliation ----

def test_reconciliacion_ok():
    exp = {("BTCUSDT", LONG): 1.0}
    act = {("BTCUSDT", LONG): 1.0}
    assert classify_reconciliation(exp, act, 0.001) == ReconVerdict.OK


def test_pierna_esperada_ausente_es_resync():
    # Esperábamos LONG pero el exchange no lo tiene → SL/TP disparó (benigno).
    exp = {("BTCUSDT", LONG): 1.0}
    assert classify_reconciliation(exp, {}, 0.001) == ReconVerdict.RESYNC


def test_pierna_desconocida_es_halt():
    # Hay una pierna en el exchange que nosotros no abrimos → peligro.
    act = {("BTCUSDT", LONG): 1.0}
    assert classify_reconciliation({}, act, 0.001) == ReconVerdict.HALT


def test_cantidad_divergente_es_halt():
    exp = {("BTCUSDT", LONG): 1.0}
    act = {("BTCUSDT", LONG): 2.0}
    assert classify_reconciliation(exp, act, 0.001) == ReconVerdict.HALT


def test_halt_domina_sobre_resync():
    # Una pierna esperada ausente (resync) + una desconocida (halt) → HALT.
    exp = {("BTCUSDT", LONG): 1.0}
    act = {("ETHUSDT", SHORT): 3.0}
    assert classify_reconciliation(exp, act, 0.001) == ReconVerdict.HALT


def test_diferencia_dentro_de_tolerancia_es_ok():
    exp = {("BTCUSDT", LONG): 1.0}
    act = {("BTCUSDT", LONG): 1.0005}  # 0.05% < 0.1%
    assert classify_reconciliation(exp, act, 0.001) == ReconVerdict.OK
