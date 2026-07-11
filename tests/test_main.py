"""Smoke test del punto de entrada: la config real carga y los imports viven."""

from src.main import main, run_check


def test_check_devuelve_cero():
    assert run_check() == 0


def test_main_check_flag():
    assert main(["--check"]) == 0


def test_main_sin_argumentos_no_revienta():
    assert main([]) == 0
