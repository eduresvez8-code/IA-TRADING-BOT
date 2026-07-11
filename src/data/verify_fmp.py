"""Verificación de point-in-time de Financial Modeling Prep (FMP), gratis.

Antes de pagar CUALQUIER proveedor de fundamentales/earnings, hay que
confirmar que el dato distingue la fecha de PUBLICACIÓN del reporte
(`filingDate`/`acceptedDate`) de la fecha de FIN DE PERIODO (`date`). Si solo
existe la fecha de fin de periodo, usarla como si fuera la fecha en que la
información estuvo disponible es exactamente el sesgo de look-ahead que mató
el hallazgo de 2026-07-06 (`finding-architecture-audit`) — el reporte del Q4
2020 no existía el 31-dic-2020, existió cuando la empresa lo PRESENTÓ semanas
después.

Cubre 5 endpoints (todos accesibles en el tier gratis, verificado a mano):
    - income-statement, balance-sheet-statement, cash-flow-statement: SÍ
      traen su propia filingDate/acceptedDate.
    - key-metrics, ratios (P/E, P/B, ROE...): NO traen su propia fecha de
      filing — su `date` es el fin de año fiscal. Hay que CRUZARLOS con la
      filingDate del income-statement del MISMO periodo, o se cuela el mismo
      look-ahead que este script existe para detectar. Este script hace ese
      cruce explícitamente y lo marca si falla.
    - earnings (reemplaza a earnings-surprises, que está retirado/404): SÍ
      trae fecha de anuncio real, pero su campo `lastUpdated` sugiere que FMP
      puede revisar valores después — riesgo residual menor, declarado.
    - key-metrics/ratios SOLO responden con period="annual" en el tier
      gratis (period="quarter" da 402). Todos los endpoints están limitados
      a 5 periodos por respuesta (5 años en anual, ~15 meses en trimestral)
      — el tier gratis alcanza para VERIFICAR la estructura del dato, no
      para un backtest real (que necesita décadas de historia).

Este script NO decide nada por sí solo: imprime la evidencia (fechas crudas,
el lag en días, si el cruce con filingDate funcionó) para que la decisión de
pagar un proveedor se tome viendo datos reales, no promesas de marketing.

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
# Tope real del tier gratis en TODOS los endpoints probados (verificado a
# mano: limit=8 y limit=30 devuelven 402 "must be between 0 and 5").
FREE_TIER_MAX_LIMIT = 5
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


def fetch_statement(endpoint: str, ticker: str, api_key: str, *,
                    period: str = "quarter", limit: int = FREE_TIER_MAX_LIMIT) -> list[dict]:
    """income-statement / balance-sheet-statement / cash-flow-statement — las
    3 comparten forma y traen filingDate/acceptedDate propios."""
    url = f"{BASE_URL}/{endpoint}"
    data = _get(url, {"symbol": ticker, "period": period, "limit": limit, "apikey": api_key})
    return data if isinstance(data, list) else []


def fetch_key_metrics(ticker: str, api_key: str, limit: int = FREE_TIER_MAX_LIMIT) -> list[dict]:
    """SOLO period='annual' funciona en el tier gratis — 'quarter' da 402."""
    url = f"{BASE_URL}/key-metrics"
    data = _get(url, {"symbol": ticker, "period": "annual", "limit": limit, "apikey": api_key})
    return data if isinstance(data, list) else []


def fetch_ratios(ticker: str, api_key: str, limit: int = FREE_TIER_MAX_LIMIT) -> list[dict]:
    url = f"{BASE_URL}/ratios"
    data = _get(url, {"symbol": ticker, "period": "annual", "limit": limit, "apikey": api_key})
    return data if isinstance(data, list) else []


def fetch_earnings(ticker: str, api_key: str, limit: int = FREE_TIER_MAX_LIMIT) -> list[dict]:
    """Reemplaza a earnings-surprises (retirado, devuelve 404 vacío)."""
    url = f"{BASE_URL}/earnings"
    data = _get(url, {"symbol": ticker, "limit": limit, "apikey": api_key})
    return data if isinstance(data, list) else []


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


def _check_statement(label: str, endpoint: str, ticker: str, api_key: str,
                     counters: dict) -> None:
    print(f"\n--- {ticker}: {label} ---")
    try:
        rows = fetch_statement(endpoint, ticker, api_key)
    except FetchError as e:
        print(f"  {e}")
        counters["any_fetch_failed"] = True
        return
    if not rows:
        print("  (sin datos)")
        counters["any_fetch_failed"] = True
        return
    print(f"  {'fin de periodo':<14} {'filingDate':<12} {'acceptedDate':<22} {'lag (días)'}")
    for row in rows:
        period = row.get("date", "?")
        filing = row.get("filingDate")
        accepted = row.get("acceptedDate", "?")
        filed = filing or accepted
        lag = _lag_days(period, filed) if filed and filed != "?" else None
        flag = ""
        if lag is None:
            flag = "  <- SIN filingDate/acceptedDate: sospechoso"
            counters["rows_suspicious"] += 1
        elif lag <= 0:
            flag = "  <- lag <= 0: SOSPECHOSO (publicado antes del fin de periodo?)"
            counters["rows_suspicious"] += 1
        else:
            counters["rows_ok"] += 1
            if lag > 120:
                flag = "  <- lag > 120 días: revisar (¿10-K anual, o error?)"
        print(f"  {period:<14} {str(filing):<12} {str(accepted):<22} "
              f"{lag if lag is not None else '?':<10}{flag}")


def _check_key_metrics_and_ratios(ticker: str, api_key: str, counters: dict) -> None:
    """key-metrics/ratios NO traen su propia fecha de filing — se cruzan con
    el income-statement anual del MISMO fiscalYear para obtener la fecha
    point-in-time real. Si el cruce falla, se marca sospechoso: usar el
    `date` (fin de año fiscal) de estos endpoints directamente SERÍA
    exactamente el look-ahead que este script existe para atajar."""
    print(f"\n--- {ticker}: key-metrics + ratios (anual, cruzados con filingDate) ---")
    try:
        income = fetch_statement("income-statement", ticker, api_key, period="annual")
        metrics = fetch_key_metrics(ticker, api_key)
        ratios = fetch_ratios(ticker, api_key)
    except FetchError as e:
        print(f"  {e}")
        counters["any_fetch_failed"] = True
        return
    filing_by_year = {row.get("fiscalYear"): row.get("filingDate") for row in income}
    if not metrics and not ratios:
        print("  (sin datos)")
        counters["any_fetch_failed"] = True
        return
    print(f"  {'fiscalYear':<11} {'fin FY':<12} {'filingDate (cruzado)':<20} "
          f"{'ROE':<8} {'P/E':<8} {'P/B'}")
    for m, r in zip(metrics, ratios):
        fy = m.get("fiscalYear")
        filing = filing_by_year.get(fy)
        if filing is None:
            print(f"  {fy!s:<11} {m.get('date', '?'):<12} {'SIN CRUCE — sospechoso':<20} "
                  f"{'?':<8} {'?':<8} ?")
            counters["rows_suspicious"] += 1
            continue
        counters["rows_ok"] += 1
        roe = m.get("returnOnEquity")
        pe = r.get("priceToEarningsRatio")
        pb = r.get("priceToBookRatio")
        print(f"  {fy!s:<11} {m.get('date', '?'):<12} {filing:<20} "
              f"{roe:<8.2f} {pe:<8.2f} {pb:.2f}" if all(v is not None for v in (roe, pe, pb))
              else f"  {fy!s:<11} {m.get('date', '?'):<12} {filing:<20} (algún valor None)")


def _check_earnings(ticker: str, api_key: str, counters: dict) -> None:
    print(f"\n--- {ticker}: earnings (reemplaza earnings-surprises, retirado) ---")
    try:
        rows = fetch_earnings(ticker, api_key)
    except FetchError as e:
        print(f"  {e}")
        return
    if not rows:
        print("  (sin datos)")
        return
    for row in rows:
        revised = " <- lastUpdated distinto de la fecha de anuncio: posible revisión retroactiva" \
            if row.get("lastUpdated") and row.get("lastUpdated") != row.get("date") else ""
        print(f"  fecha={row.get('date')}  epsActual={row.get('epsActual')}  "
              f"epsEstimated={row.get('epsEstimated')}  lastUpdated={row.get('lastUpdated')}{revised}")


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
        if profile:
            print(f"  clave funciona: SÍ (companyName={profile.get('companyName', '?')})")
        else:
            print("  respuesta vacía")
    except FetchError as e:
        key_works = False
        print(f"  {e}")

    # Contadores fail-closed: el veredicto final exige haber visto datos
    # REALES (rows_ok > 0) — nunca asume "bien" solo porque nada falló.
    counters = {"rows_ok": 0, "rows_suspicious": 0, "any_fetch_failed": False}

    if key_works:
        for ticker in SAMPLE_TICKERS:
            _check_statement("income-statement (trimestral)", "income-statement",
                             ticker, secrets.fmp_api_key, counters)
            _check_statement("balance-sheet-statement (trimestral)", "balance-sheet-statement",
                             ticker, secrets.fmp_api_key, counters)
            _check_statement("cash-flow-statement (trimestral)", "cash-flow-statement",
                             ticker, secrets.fmp_api_key, counters)
            _check_key_metrics_and_ratios(ticker, secrets.fmp_api_key, counters)
            _check_earnings(ticker, secrets.fmp_api_key, counters)

    print("\n" + "=" * 86)
    rows_ok, rows_suspicious = counters["rows_ok"], counters["rows_suspicious"]
    if not key_works:
        print("VEREDICTO: INCONCLUSO — ni el endpoint más básico (/profile) respondió. La")
        print("clave puede estar mal copiada, sin activar, o el plan gratis cambió. Revisar")
        print("el dashboard de FMP antes de sacar cualquier conclusión sobre point-in-time.")
    elif rows_ok == 0:
        print("VEREDICTO: INCONCLUSO — la clave funciona (perfil OK) pero NINGÚN registro se")
        print("pudo verificar. No se puede afirmar nada sobre point-in-time sin ver datos")
        print("reales — no comprar el plan de pago basándose en promesas de marketing.")
    elif rows_suspicious > 0:
        print(f"VEREDICTO: {rows_suspicious} registro(s) sospechoso(s) de {rows_ok + rows_suspicious}")
        print("vistos (sin fecha de filing, lag imposible, o sin cruce ratios<->filingDate).")
        print("NO comprar el plan de pago sin resolver esto primero.")
    else:
        print(f"VEREDICTO: {rows_ok} registros verificados con fecha de filing real y lag")
        print("positivo/razonable (income-statement, balance-sheet, cash-flow, Y el cruce")
        print("key-metrics/ratios<->filingDate). Buena señal para point-in-time. El tier")
        print("gratis alcanza para verificar la ESTRUCTURA del dato, pero tope de 5 periodos")
        print("por endpoint — NO alcanza para un backtest real (hacen falta décadas). Los")
        print("valores de 'earnings' pueden revisarse retroactivamente (ver lastUpdated) —")
        print("riesgo residual menor a declarar en cualquier pre-registro futuro.")
    if counters["any_fetch_failed"] and rows_ok > 0:
        print("(nota: algún endpoint/ticker falló pese a que otros sí funcionaron — revisar")
        print(" arriba cuál y por qué antes de generalizar el veredicto.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
