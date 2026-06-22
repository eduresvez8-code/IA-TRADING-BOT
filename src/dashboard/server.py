"""Servidor del dashboard: stdlib `http.server`, cero dependencias nuevas ($0).

Sirve dos cosas y nada más:
  GET /              → la página única (index.html)
  GET /api/snapshot  → el JSON de `build_snapshot` (una lectura READ-ONLY de SQLite)

Solo responde a GET; no hay ninguna ruta de escritura. Se enlaza a `host`
(loopback por defecto), así que no queda expuesto a la red. Corre como proceso
independiente del bot: si el dashboard cae, el trading sigue, y viceversa.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from src.core.config import Settings, load_secrets, load_settings
from src.dashboard.queries import build_snapshot

logger = logging.getLogger("ia_trading.dashboard")

_INDEX_HTML = Path(__file__).resolve().parent / "index.html"


def _make_handler(settings: Settings, html: bytes, testnet: bool):
    class Handler(BaseHTTPRequestHandler):
        # Silencia el log por defecto (ruidoso); enruta por nuestro logger.
        def log_message(self, fmt, *args):  # noqa: N802
            logger.debug("dashboard %s - %s", self.address_string(), fmt % args)

        def _send(self, code: int, body: bytes, content_type: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def do_GET(self):  # noqa: N802
            if self.path == "/" or self.path.startswith("/index"):
                self._send(200, html, "text/html; charset=utf-8")
                return
            if self.path.startswith("/api/snapshot"):
                try:
                    snap = build_snapshot(
                        settings, now=datetime.now(timezone.utc), testnet=testnet)
                    body = json.dumps(snap, default=str).encode("utf-8")
                    self._send(200, body, "application/json")
                except Exception as exc:  # nunca tirar el servidor por un fallo de lectura
                    logger.exception("error construyendo el snapshot")
                    body = json.dumps({"error": str(exc)}).encode("utf-8")
                    self._send(500, body, "application/json")
                return
            self._send(404, b'{"error":"not found"}', "application/json")

        do_HEAD = do_GET  # noqa: N815

    return Handler


def serve(settings: Settings | None = None) -> None:
    """Arranca el servidor (bloqueante). Ctrl-C para detener."""
    settings = settings or load_settings()
    testnet = load_secrets().binance_testnet  # solo el booleano, nunca las claves
    html = _INDEX_HTML.read_bytes()
    d = settings.dashboard

    httpd = ThreadingHTTPServer((d.host, d.port), _make_handler(settings, html, testnet))
    url = f"http://{d.host}:{d.port}"
    db = settings.storage.db_path
    print("=" * 70)
    print(f"  IA TRADING — dashboard READ-ONLY en {url}")
    print(f"  Lee (modo ro): {db}")
    print(f"  Modo: {'testnet' if testnet else 'MAINNET'} | refresco {d.refresh_seconds}s")
    print("  Ctrl-C para detener. (No envía órdenes: solo observa.)")
    print("=" * 70)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n  Dashboard detenido.")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    serve()
