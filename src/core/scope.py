"""Resolución de scope de noticias → símbolos operados (compartido Fast/Slow Path).

Claude devuelve el `symbol_scope` de una noticia como tickers de ACTIVO BASE
(`["BTC", "ETH"]`) o `["*"]` (todo el mercado), pero los símbolos que operamos son
pares completos de Binance (`["BTCUSDT", "ETHUSDT"]`). La intersección EXACTA del v1
nunca macheaba `"BTC"` con `"BTCUSDT"` → casi todo el overlay de noticias entraba
solo por `"*"` y los shocks idiosincráticos por símbolo (hack de UN activo, ETF de
UN activo) se perdían. Esta era la DEUDA_TICKER documentada en slow_path.py.

La resolución correcta machea por ACTIVO BASE: el ticker `"BTC"` machea el símbolo
cuyo base asset (el par menos su quote) es `"BTC"`. Los quotes válidos viven en
`market.quote_assets` (Cero Hardcoding: no asumimos "USDT" en el código).

Única fuente de verdad para AMBOS paths: `Orchestrator._resolve_scope` (Fast Path)
y `fetch_sentiment` (Slow Path) delegan aquí, así tratan el scope idéntico.
"""

from __future__ import annotations


def normalize_ticker(ticker: str) -> str:
    """Normaliza un ticker de scope: sin espacios y en mayúsculas ('btc ' → 'BTC')."""
    return ticker.strip().upper()


def base_asset(symbol: str, quote_assets: list[str]) -> str:
    """Activo base de un símbolo quitando el quote conocido.

    'BTCUSDT' con quotes ['USDT'] → 'BTC'. Si ningún quote casa, devuelve el símbolo
    completo (no inventamos una base). Probamos los quotes MÁS LARGOS primero para
    no recortar de menos (p.ej. 'USDT' antes que un hipotético 'USD' que dejaría una
    'T' colgando).
    """
    s = normalize_ticker(symbol)
    for q in sorted((normalize_ticker(qa) for qa in quote_assets), key=len, reverse=True):
        if s.endswith(q) and len(s) > len(q):
            return s[: -len(q)]
    return s


def resolve_scope(
    scope: list[str], symbols: list[str], quote_assets: list[str]
) -> list[str]:
    """Resuelve el `symbol_scope` de una noticia a los símbolos que operamos.

    - `"*"` (en cualquier posición) → TODOS los símbolos configurados (market-wide).
    - En otro caso, machea cada símbolo cuyo NOMBRE COMPLETO ('BTCUSDT') o ACTIVO
      BASE ('BTC') esté en el scope. Case-insensitive. Ignora tickers que no
      seguimos. Preserva el orden y el formato original de `symbols`.

    Ejemplos (symbols=['BTCUSDT','ETHUSDT'], quote_assets=['USDT']):
        ['*']        → ['BTCUSDT','ETHUSDT']
        ['BTC']      → ['BTCUSDT']          # ← lo que arregla la DEUDA_TICKER
        ['BTCUSDT']  → ['BTCUSDT']          # símbolo completo también vale
        ['btc','eth']→ ['BTCUSDT','ETHUSDT'] # case-insensitive
        ['DOGE']     → []                    # no lo seguimos
    """
    wanted = {normalize_ticker(t) for t in scope}
    if "*" in wanted:
        return list(symbols)
    out: list[str] = []
    for sym in symbols:
        if normalize_ticker(sym) in wanted or base_asset(sym, quote_assets) in wanted:
            out.append(sym)
    return out
