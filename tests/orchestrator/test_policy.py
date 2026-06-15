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
    r = classify_reconciliation(exp, act, 0.001)
    assert r.verdict == ReconVerdict.OK
    assert r.resync_keys == () and r.suspect_keys == ()


def test_pierna_esperada_ausente_es_resync():
    # Esperábamos LONG pero el exchange no lo tiene → SL/TP disparó (benigno).
    exp = {("BTCUSDT", LONG): 1.0}
    r = classify_reconciliation(exp, {}, 0.001)
    assert r.verdict == ReconVerdict.RESYNC
    assert r.resync_keys == (("BTCUSDT", LONG),)


def test_pierna_desconocida_es_sospechosa():
    # Pierna en el exchange que no abrimos → sospechosa (el motor le da gracia).
    act = {("BTCUSDT", LONG): 1.0}
    r = classify_reconciliation({}, act, 0.001)
    assert r.verdict == ReconVerdict.SUSPECT
    assert r.suspect_keys == (("BTCUSDT", LONG),)


def test_cantidad_divergente_es_sospechosa():
    exp = {("BTCUSDT", LONG): 1.0}
    act = {("BTCUSDT", LONG): 2.0}
    r = classify_reconciliation(exp, act, 0.001)
    assert r.verdict == ReconVerdict.SUSPECT
    assert ("BTCUSDT", LONG) in r.suspect_keys


def test_pierna_en_vuelo_se_ignora_no_es_sospechosa():
    # Una pierna que acabamos de abrir (in_flight) y aún no aparece en la cuenta
    # no debe ni resync-ear ni HALT-ear: es latencia, no divergencia.
    act = {("BTCUSDT", LONG): 1.0}
    r = classify_reconciliation({}, act, 0.001, in_flight={("BTCUSDT", LONG)})
    assert r.verdict == ReconVerdict.OK


def test_en_vuelo_esperada_y_ausente_no_resync():
    # La abrimos (expected) pero el exchange aún no la reporta y está en vuelo:
    # no es un cierre por SL/TP, es lag → no resync.
    exp = {("BTCUSDT", LONG): 1.0}
    r = classify_reconciliation(exp, {}, 0.001, in_flight={("BTCUSDT", LONG)})
    assert r.verdict == ReconVerdict.OK


def test_diferencia_dentro_de_tolerancia_es_ok():
    exp = {("BTCUSDT", LONG): 1.0}
    act = {("BTCUSDT", LONG): 1.0005}  # 0.05% < 0.1%
    assert classify_reconciliation(exp, act, 0.001).verdict == ReconVerdict.OK
