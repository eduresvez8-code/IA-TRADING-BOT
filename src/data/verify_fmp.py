"""Verificación de point-in-time de Financial Modeling Prep (FMP), gratis.

Antes de pagar CUALQUIER proveedor de fundamentales/earnings, hay que
confirmar que el dato distingue la fecha de PUBLICACIÓN del reporte
(`fillingDate`/`acceptedDate`) de la fecha de FIN DE PERIODO (`date`). Si
solo existe la fecha de fin de periodo, usarla como si fuera la fecha en que
la información estuvo disponible es exactamente el sesgo de look-ahead que
mató el hallazgo de 2026-07-06 (`finding-architecture-audit`) — el reporte
del Q4 2020 no existía el 31-dic-2020, existió cuando la empresa lo PRESENTÓ
semanas después.

Este script NO decide nada por sí solo: imprime la evidencia (fechas crudas,
el lag en días) para que la decisión de comprar o no un proveedor de pago se
tome viendo los datos reales, no una promesa de marketing.

Uso (requiere FMP_API_KEY en .env — tier gratis, 250 llamadas/día, $0):

    uv run python -m src.data.verify_fmp
"""

from __future__ import annotations

import sys
from datetime import date

import httpx

from src.core.config import load_secrets

BASE_URL = "https://financialmodelingprep.com/api/v3"
# Tickers de muestra: sin relación con el universo del protocolo SP500 (esto
# es solo verificación de la FORMA del dato, no un experimento de estrategia
# — no hay resultado de test que contaminar aquí).
SAMPLE_TICKERS = ["AAPL", "MSFT", "JPM"]


def fetch_income_statements(ticker: str, api_key: str, limit: int = 8) -> list[dict]:
    url = f"{BASE_URL}/income-statement/{ticker}"
    r = httpx.get(url, params={"period": "quarter", "limit": limit, "apikey": api_key},
                  timeout=30.0)
    r.raise_for_status()
    return r.json()


def fetch_earnings_surprises(ticker: str, api_key: str, limit: int = 8) -> list[dict]:
    url = f"{BASE_URL}/earnings-surprises/{ticker}"
    r = httpx.get(url, params={"apikey": api_key}, timeout=30.0)
    r.raise_for_status()
    data = r.json()
    return data[:limit] if isinstance(data, list) else []


def _lag_days(period_end: str, filed: str) -> int | None:
    try:
        d0 = date.fromisoformat(period_end)
        d1 = date.fromisoformat(filed[:10])
        return (d1 - d0).days
    except (ValueError, TypeError):
        return None


def main() -> int:
    secrets = load_secrets()
    if not secrets.fmp_api_key:
        print("FMP_API_KEY vacío en .env — crea una cuenta gratis en")
        print("https://site.financialmodelingprep.com y copia tu API key a .env")
        print("(ver .env.example). No se hace ninguna llamada sin la clave.")
        return 1

    print("=" * 86)
    print("VERIFICACIÓN point-in-time — Financial Modeling Prep (tier gratis)")
    print("=" * 86)

    any_suspicious = False
    for ticker in SAMPLE_TICKERS:
        print(f"\n--- {ticker}: estados financieros trimestrales ---")
        try:
            rows = fetch_income_statements(ticker, secrets.fmp_api_key)
        except httpx.HTTPStatusError as e:
            print(f"  ERROR HTTP: {e}")
            continue
        if not rows:
            print("  (sin datos — plan gratis puede no cubrir este endpoint)")
            continue
        print(f"  {'fin de periodo':<14} {'fillingDate':<14} {'acceptedDate':<22} {'lag (días)'}")
        for row in rows:
            period = row.get("date", "?")
            filed = row.get("fillingDate", "?")
            accepted = row.get("acceptedDate", "?")
            lag = _lag_days(period, filed) if filed != "?" else None
            flag = ""
            if lag is None:
                flag = "  <- SIN fillingDate: sospechoso"
                any_suspicious = True
            elif lag <= 0:
                flag = "  <- lag <= 0: SOSPECHOSO (publicado antes del fin de periodo?)"
                any_suspicious = True
            elif lag > 120:
                flag = "  <- lag > 120 días: revisar (¿10-K anual con lag normal, o error?)"
            print(f"  {period:<14} {filed:<14} {accepted:<22} {lag if lag is not None else '?':<10}{flag}")

        print(f"\n--- {ticker}: earnings surprises ---")
        try:
            surprises = fetch_earnings_surprises(ticker, secrets.fmp_api_key)
        except httpx.HTTPStatusError as e:
            print(f"  ERROR HTTP: {e}")
            continue
        if not surprises:
            print("  (sin datos — plan gratis puede no cubrir este endpoint)")
        for s in surprises[:4]:
            print(f"  fecha={s.get('date', '?')}  actual={s.get('actualEarningResult', '?')}  "
                  f"estimado={s.get('estimatedEarning', '?')}")

    print("\n" + "=" * 86)
    if any_suspicious:
        print("VEREDICTO: al menos un registro sin fillingDate válido o con lag <= 0.")
        print("NO comprar el plan de pago sin resolver esto primero — point-in-time no está")
        print("garantizado con lo visto aquí.")
    else:
        print("VEREDICTO: fillingDate presente y con lag positivo y razonable en la muestra.")
        print("Buena señal para point-in-time — igual conviene revisar más tickers/periodos")
        print("antes de comprometer presupuesto, esto es una muestra pequeña.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
