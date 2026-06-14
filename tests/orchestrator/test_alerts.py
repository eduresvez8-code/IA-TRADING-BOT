"""Tests de los sinks de alertas."""

import logging

from src.orchestrator.alerts import AlertLevel, LoggingAlertSink, RecordingAlertSink


def test_recording_sink_guarda_y_lista():
    sink = RecordingAlertSink()
    sink.alert(AlertLevel.INFO, "open", "BTCUSDT LONG")
    sink.alert(AlertLevel.CRITICAL, "kill_switch_drawdown", "BTCUSDT")
    assert sink.alerts[0] == (AlertLevel.INFO, "open", "BTCUSDT LONG")
    assert sink.events() == ["open", "kill_switch_drawdown"]


def test_logging_sink_emite_en_el_nivel_correcto(caplog):
    sink = LoggingAlertSink()
    with caplog.at_level(logging.WARNING, logger="ia_trading.alerts"):
        sink.alert(AlertLevel.INFO, "open", "no debería verse en WARNING")
        sink.alert(AlertLevel.CRITICAL, "halt", "esto sí")
    # El INFO queda por debajo del umbral WARNING; el CRITICAL aparece.
    assert any("halt" in r.message for r in caplog.records)
    assert not any("no debería verse" in r.message for r in caplog.records)
