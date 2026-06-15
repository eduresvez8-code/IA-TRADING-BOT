"""Decisiones de gestión de posición y reconciliación — funciones PURAS.

Aisladas del orquestador async para poder validar cada rama con un test de
tabla. Aquí viven dos políticas:

1. "Una pierna por símbolo": qué hacer con la pierna actual ante una dirección
   deseada (abrir, no hacer nada, o flip).
2. Clasificación de la reconciliación: distinguir un cierre benigno por SL/TP
   (resync) de una divergencia peligrosa (halt).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from src.core.models import PositionSide

# Clave de una pierna: (símbolo, lado de posición).
LegKey = tuple[str, PositionSide]


class PositionAction(str, Enum):
    NONE = "none"   # no tocar (HOLD/vetada, o ya en esa dirección)
    OPEN = "open"   # plano → abrir la pierna deseada
    FLIP = "flip"   # señal opuesta → cerrar la actual y abrir la nueva


def decide_position_action(
    held: PositionSide | None, want: PositionSide | None
) -> PositionAction:
    """Política de UNA pierna por símbolo.

    Args:
        held: pierna abierta ahora (LONG/SHORT) o None si estamos planos.
        want: dirección que el Risk Manager aprobó (LONG/SHORT) o None si la
              confluencia dijo HOLD o el riesgo vetó.

    Returns:
        NONE  → nada que hacer (los SL/TP gestionan la pierna abierta).
        OPEN  → estamos planos y hay dirección aprobada.
        FLIP  → tenemos una pierna y llega la dirección OPUESTA aprobada.
    """
    if want is None:
        return PositionAction.NONE
    if held is None:
        return PositionAction.OPEN
    if held == want:
        return PositionAction.NONE   # ya en posición: no escalamos ni duplicamos
    return PositionAction.FLIP


class ReconVerdict(str, Enum):
    OK = "ok"            # estado local y exchange coinciden
    RESYNC = "resync"    # benigno: una pierna esperada ya no está (SL/TP disparó)
    SUSPECT = "suspect"  # anomalía esta vela: necesita gracia antes de un HALT


@dataclass(frozen=True)
class ReconReport:
    """Resultado de clasificar la reconciliación de UNA vela.

    No decide el HALT (eso es estatal: el motor cuenta observaciones de gracia).
    Solo separa lo benigno (resync) de lo sospechoso (suspect).
    """

    verdict: ReconVerdict
    resync_keys: tuple[LegKey, ...] = ()    # piernas esperadas que el exchange ya no tiene
    suspect_keys: tuple[LegKey, ...] = ()   # piernas desconocidas o con cantidad divergente


def classify_reconciliation(
    expected: dict[LegKey, float],
    actual: dict[LegKey, float],
    tolerance: float,
    in_flight: set[LegKey] | frozenset[LegKey] = frozenset(),
) -> ReconReport:
    """Compara el modelo interno con lo que reporta el exchange.

    Las piernas EN VUELO (`in_flight`) se ignoran por completo: son operaciones
    en tránsito y la latencia del WebSocket puede mostrarlas o no de forma
    transitoria. No deben provocar ni resync ni HALT.

    Del resto:
    - Pierna esperada AUSENTE en el exchange → cierre por SL/TP (resync, benigno).
    - Pierna esperada con cantidad divergente (> tolerancia) → sospechosa.
    - Pierna en el exchange que NO abrimos ni tenemos en vuelo → sospechosa.

    El motor aplica la ventana de gracia sobre `suspect_keys` antes de detener.
    """
    resync: list[LegKey] = []
    suspect: list[LegKey] = []

    for key, eq in expected.items():
        if key in in_flight:
            continue
        aq = actual.get(key)
        if aq is None:
            resync.append(key)
        else:
            denom = max(abs(eq), abs(aq), 1e-12)
            if abs(aq - eq) / denom > tolerance:
                suspect.append(key)

    for key in actual:
        if key in expected or key in in_flight:
            continue
        suspect.append(key)

    if suspect:
        verdict = ReconVerdict.SUSPECT
    elif resync:
        verdict = ReconVerdict.RESYNC
    else:
        verdict = ReconVerdict.OK
    return ReconReport(verdict, tuple(resync), tuple(suspect))
