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

# FMP retiró los endpoints /api/v3/ ("Legacy Endpoint") el 31-ago-2025 para
# cuentas sin suscripción previa a esa fecha — la API vigente es /stable/,
# con el ticker como query param `symbol=`, no como segmento de la URL.
BASE_URL = "https://financialmodelingprep.com/stable"
# Tickers de muestra: sin relación con el universo del protocolo SP500 (esto
# es solo verificación de la FORMA del dato, no un experimento de estrategia
# — no hay resultado de test que contaminar aquí).
SAMPLE_TICKERS = ["AAPL", "MSFT", "JPM"]


class FetchError(Exception):
    """Error de red/HTTP SIN la URL completa (que llevaría la API key) — el
    mensaje por defecto de httpx.HTTPStatusError incluye la query string
    entera, y `apikey` viaja ahí. Nunca se debe dejar que ese mensaje llegue
    a un print/log."""


def _get(url: str, params: dict) -> object:
    r = httpx.get(url, params=params, timeout=30.0)
    if r.status_code >= 400:
        # Cuerpo de la respuesta (explica el motivo real: plan insuficiente,
        # endpoint movido, etc.) SIN la URL ni la query string de la petición.
        try:
            body = r.json()
        except ValueError:
            body = r.text[:300]
        raise FetchError(f"HTTP {r.status_code} — respuesta del servidor: {body}")
    return r.json()


def fetch_income_statements(ticker: str, api_key: str, limit: int = 4) -> list[dict]:
    url = f"{BASE_URL}/income-statement"
    data = _get(url, {"symbol": ticker, "period": "quarter", "limit": limit, "apikey": api_key})
    return data if isinstance(data, list) else []


def fetch_earnings_surprises(ticker: str, api_key: str, limit: int = 8) -> list[dict]:
    url = f"{BASE_URL}/earnings-surprises"
    data = _get(url, {"symbol": ticker, "apikey": api_key})
    return data[:limit] if isinstance(data, list) else []


def fetch_company_profile(ticker: str, api_key: str) -> dict:
    """Endpoint casi siempre incluido en el tier gratis — sirve para aislar
    si el problema es la CLAVE (inválida/sin activar) o el ENDPOINT
    concreto (movido a un plan de pago)."""
    url = f"{BASE_URL}/profile"
    data = _get(url, {"symbol": ticker, "apikey": api_key})
    return data[0] if isinstance(data, list) and data else {}


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

    # ---- Prueba de aislamiento: ¿la clave funciona en absoluto? ----
    print("\n--- prueba de aislamiento: /profile (casi siempre en el tier gratis) ---")
    try:
        profile = fetch_company_profile("AAPL", secrets.fmp_api_key)
        key_works = bool(profile)
        print(f"  clave funciona: {'SÍ' if key_works else 'respuesta vacía, dudoso'} "
              f"(companyName={profile.get('companyName', '?')})" if profile else "  respuesta vacía")
    except FetchError as e:
        key_works = False
        print(f"  {e}")

    # Contadores fail-closed: el veredicto final exige haber visto datos
    # REALES (rows_ok > 0) — nunca asume "bien" solo porque nada falló.
    rows_ok = 0
    rows_suspicious = 0
    any_fetch_failed = False

    for ticker in SAMPLE_TICKERS:
        print(f"\n--- {ticker}: estados financieros trimestrales ---")
        try:
            rows = fetch_income_statements(ticker, secrets.fmp_api_key)
        except FetchError as e:
            print(f"  {e}")
            any_fetch_failed = True
            continue
        if not rows:
            print("  (sin datos — plan gratis puede no cubrir este endpoint)")
            any_fetch_failed = True
            continue
        print(f"  {'fin de periodo':<14} {'fillingDate':<14} {'acceptedDate':<22} {'lag (días)'}")
        for row in rows:
            period = row.get("date", "?")
            filling = row.get("fillingDate") or row.get("filingDate")
            accepted = row.get("acceptedDate", "?")
            # La API "stable" ya no expone fillingDate en algunos planes —
            # acceptedDate (timestamp real de aceptación SEC) sirve igual de
            # bien como fecha point-in-time; se usa como respaldo, no como
            # ausencia de dato.
            filed = filling or accepted
            lag = _lag_days(period, filed) if filed and filed != "?" else None
            flag = ""
            if lag is None:
                flag = "  <- SIN fillingDate: sospechoso"
                rows_suspicious += 1
            elif lag <= 0:
                flag = "  <- lag <= 0: SOSPECHOSO (publicado antes del fin de periodo?)"
                rows_suspicious += 1
            else:
                rows_ok += 1
                if lag > 120:
                    flag = "  <- lag > 120 días: revisar (¿10-K anual con lag normal, o error?)"
            print(f"  {period:<14} {filed:<14} {accepted:<22} {lag if lag is not None else '?':<10}{flag}")

        print(f"\n--- {ticker}: earnings surprises ---")
        try:
            surprises = fetch_earnings_surprises(ticker, secrets.fmp_api_key)
        except FetchError as e:
            print(f"  {e}")
            continue
        if not surprises:
            print("  (sin datos — plan gratis puede no cubrir este endpoint)")
        for s in surprises[:4]:
            print(f"  fecha={s.get('date', '?')}  actual={s.get('actualEarningResult', '?')}  "
                  f"estimado={s.get('estimatedEarning', '?')}")

    print("\n" + "=" * 86)
    if not key_works:
        print("VEREDICTO: INCONCLUSO — ni el endpoint más básico (/profile) respondió. La")
        print("clave puede estar mal copiada, sin activar, o el plan gratis cambió. Revisar")
        print("el dashboard de FMP antes de sacar cualquier conclusión sobre point-in-time.")
    elif rows_ok == 0:
        print("VEREDICTO: INCONCLUSO — la clave funciona (perfil OK) pero NINGÚN estado")
        print("financiero se pudo verificar (0 filas válidas). El tier gratis probablemente")
        print("NO incluye income-statement/earnings-surprises. No se puede afirmar nada sobre")
        print("point-in-time sin ver datos reales — no comprar el plan de pago basándose en")
        print("promesas de marketing sin haber visto esto funcionar.")
    elif rows_suspicious > 0:
        print(f"VEREDICTO: {rows_suspicious} registro(s) sin fillingDate válido o con lag <= 0")
        print(f"de {rows_ok + rows_suspicious} vistos. NO comprar el plan de pago sin resolver")
        print("esto primero — point-in-time no está garantizado con lo visto aquí.")
    else:
        print(f"VEREDICTO: {rows_ok} registros con fillingDate presente y lag positivo y")
        print("razonable. Buena señal para point-in-time — igual conviene revisar más")
        print("tickers/periodos antes de comprometer presupuesto, esto es una muestra pequeña.")
    if any_fetch_failed and rows_ok > 0:
        print("(nota: algún ticker falló al descargar pese a que otros sí funcionaron — revisar")
        print(" arriba cuál y por qué antes de generalizar el veredicto a todo el universo.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
