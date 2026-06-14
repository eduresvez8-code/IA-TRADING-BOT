"""Alertas del bot en vivo (hardening del Sprint 7).

Un `AlertSink` desacopla "ocurrió un evento que un humano debe ver" de CÓMO se
notifica. El default escribe al log; en producción se puede enchufar un webhook
(Telegram/Discord) sin tocar el orquestador. En tests, un sink que graba permite
afirmar qué alertas se dispararon.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Protocol, runtime_checkable

logger = logging.getLogger("ia_trading.alerts")


class AlertLevel(str, Enum):
    INFO = "INFO"          # apertura/cierre normal
    WARNING = "WARNING"    # resync, fallo de orden, reinicio de tarea
    CRITICAL = "CRITICAL"  # kill switch, halt por reconciliación, feed caído


@runtime_checkable
class AlertSink(Protocol):
    def alert(self, level: AlertLevel, event: str, detail: str) -> None: ...


class LoggingAlertSink:
    """Sink por defecto: vuelca la alerta al logger según su nivel."""

    _LEVELS = {
        AlertLevel.INFO: logging.INFO,
        AlertLevel.WARNING: logging.WARNING,
        AlertLevel.CRITICAL: logging.CRITICAL,
    }

    def alert(self, level: AlertLevel, event: str, detail: str) -> None:
        logger.log(self._LEVELS[level], "[%s] %s — %s", level.value, event, detail)


class RecordingAlertSink:
    """Sink de tests: guarda las alertas para inspeccionarlas."""

    def __init__(self) -> None:
        self.alerts: list[tuple[AlertLevel, str, str]] = []

    def alert(self, level: AlertLevel, event: str, detail: str) -> None:
        self.alerts.append((level, event, detail))

    def events(self) -> list[str]:
        return [e for _, e, _ in self.alerts]
