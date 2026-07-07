"""Tests de los gates de seguridad de src/main.py (sin red).

Auditoría 2026-07: `--preflight` exigía BINANCE_TESTNET=true pero `--live` no lo
verificaba — con claves reales y el flag en false habría operado capital real sin
ninguna barrera, violando la política de seguridad del repo (solo testnet/backtest).
Estos tests fijan el contrato: `live()` muere ANTES de tocar la red si el entorno
no es testnet o si faltan claves.
"""

import src.core.config as config_module
from src.core.config import Secrets
from src.main import live


def _secrets(**overrides) -> Secrets:
    base = dict(binance_api_key="k", binance_api_secret="s", binance_testnet=True,
                anthropic_api_key="a", cryptopanic_token="")
    base.update(overrides)
    # kwargs explícitos tienen precedencia sobre .env/entorno en pydantic-settings.
    return Secrets(**base)


async def test_live_rechaza_binance_testnet_false(monkeypatch):
    # Gate no negociable: --live solo opera contra testnet. Debe salir con 1
    # ANTES de crear cliente alguno (este test correría sin red).
    monkeypatch.setattr(config_module, "load_secrets",
                        lambda: _secrets(binance_testnet=False))
    assert await live() == 1


async def test_live_rechaza_sin_claves(monkeypatch):
    monkeypatch.setattr(config_module, "load_secrets",
                        lambda: _secrets(binance_api_key="", binance_api_secret=""))
    assert await live() == 1
