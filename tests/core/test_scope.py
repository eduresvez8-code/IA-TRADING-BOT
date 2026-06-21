"""Tests de la resolución de scope de noticias → símbolos (src/core/scope.py).

Función pura compartida por Fast Path (engine._resolve_scope) y Slow Path
(fetch_sentiment). Cubre el fix de la DEUDA_TICKER: 'BTC' (como devuelve Claude)
debe machear 'BTCUSDT' por activo base.
"""

from src.core.scope import base_asset, normalize_ticker, resolve_scope

SYMS = ["BTCUSDT", "ETHUSDT"]
QUOTES = ["USDT"]


def test_normalize_ticker():
    assert normalize_ticker(" btc ") == "BTC"
    assert normalize_ticker("Eth") == "ETH"


def test_base_asset_quita_el_quote():
    assert base_asset("BTCUSDT", QUOTES) == "BTC"
    assert base_asset("ethusdt", QUOTES) == "ETH"


def test_base_asset_sin_quote_conocido_devuelve_el_simbolo():
    # Si ningún quote casa, no inventamos base: devuelve el símbolo completo.
    assert base_asset("BTCBUSD", QUOTES) == "BTCBUSD"


def test_base_asset_prefiere_el_quote_mas_largo():
    # Con quotes solapados, recorta el MÁS LARGO (USDT, no USD → sin 'T' colgando).
    assert base_asset("BTCUSDT", ["USD", "USDT"]) == "BTC"


def test_wildcard_expande_a_todos():
    assert resolve_scope(["*"], SYMS, QUOTES) == SYMS
    # "*" gana aunque venga mezclado con otros tickers.
    assert resolve_scope(["BTC", "*"], SYMS, QUOTES) == SYMS


def test_base_ticker_machea_simbolo_completo():
    # EL FIX: 'BTC' (lo que devuelve Claude) ahora machea 'BTCUSDT'.
    assert resolve_scope(["BTC"], SYMS, QUOTES) == ["BTCUSDT"]
    assert resolve_scope(["ETH"], SYMS, QUOTES) == ["ETHUSDT"]
    assert resolve_scope(["BTC", "ETH"], SYMS, QUOTES) == ["BTCUSDT", "ETHUSDT"]


def test_simbolo_completo_tambien_machea():
    assert resolve_scope(["BTCUSDT"], SYMS, QUOTES) == ["BTCUSDT"]


def test_case_insensitive():
    assert resolve_scope(["btc", "eth"], SYMS, QUOTES) == ["BTCUSDT", "ETHUSDT"]


def test_ticker_no_seguido_se_ignora():
    assert resolve_scope(["DOGE"], SYMS, QUOTES) == []
    assert resolve_scope(["DOGEUSDT"], SYMS, QUOTES) == []
    # Mezcla: solo entra lo que seguimos.
    assert resolve_scope(["BTC", "DOGE"], SYMS, QUOTES) == ["BTCUSDT"]


def test_scope_vacio_no_machea_nada():
    assert resolve_scope([], SYMS, QUOTES) == []


def test_preserva_orden_y_formato_de_symbols():
    # Devuelve los símbolos TAL CUAL están en `symbols`, en su orden, no en el del scope.
    syms = ["ETHUSDT", "BTCUSDT"]
    assert resolve_scope(["BTC", "ETH"], syms, QUOTES) == ["ETHUSDT", "BTCUSDT"]
