"""Decisiones de gestión de posición y reconciliación — funciones PURAS.

Aisladas del orquestador async para poder validar cada rama con un test de
tabla. Aquí viven dos políticas:

1. "Una pierna por símbolo": qué hacer con la pierna actual ante una dirección
   deseada (abrir, no hacer nada, o flip).
2. Clasificación de la reconciliación: distinguir un cierre benigno por SL/TP
   (resync) de una divergencia peligrosa (halt).
"""

from __future__ import annotations

from enum import Enum

from src.core.models import PositionSide


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
    OK = "ok"          # estado local y exchange coinciden
    RESYNC = "resync"  # benigno: una pierna esperada ya no está (SL/TP disparó)
    HALT = "halt"      # peligroso: pierna desconocida o cantidad divergente


def classify_reconciliation(
    expected: dict[tuple[str, PositionSide], float],
    actual: dict[tuple[str, PositionSide], float],
    tolerance: float,
) -> ReconVerdict:
    """Compara el modelo interno con lo que reporta el exchange.

    - Pierna esperada AUSENTE en el exchange → cierre por SL/TP (RESYNC, benigno).
    - Pierna presente pero con cantidad divergente (> tolerancia) → HALT.
    - Pierna en el exchange que NO abrimos → HALT (riesgo desconocido).

    HALT domina sobre RESYNC: ante cualquier señal peligrosa, detenerse.
    """
    halt = False
    resync = False

    for key, eq in expected.items():
        aq = actual.get(key)
        if aq is None:
            resync = True
        else:
            denom = max(abs(eq), abs(aq), 1e-12)
            if abs(aq - eq) / denom > tolerance:
                halt = True

    for key in actual:
        if key not in expected:
            halt = True

    if halt:
        return ReconVerdict.HALT
    if resync:
        return ReconVerdict.RESYNC
    return ReconVerdict.OK
