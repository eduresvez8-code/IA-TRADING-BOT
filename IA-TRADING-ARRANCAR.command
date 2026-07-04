#!/bin/zsh
# Doble clic para ARRANCAR el bot + dashboard + vigía. No hay que escribir nada.
export PATH="/opt/homebrew/bin:$HOME/.local/bin:$PATH"
PROJECT="/Users/eduardorestrepo/Desktop/IA TRADING"
cd "$PROJECT" || exit 1

# Quita el freno de mano: reactiva la auto-recuperación del vigía.
rm -f "$HOME/.iatrading_paused"

echo "Deteniendo cualquier instancia previa..."
pkill -f "src.main --live" 2>/dev/null
pkill -f "src.main --dashboard" 2>/dev/null
pkill -f "ops/watchdog.sh" 2>/dev/null
sleep 2

echo "Arrancando el bot (live)..."
date +%s > "$HOME/.iatrading_last_start"   # marca de arranque (gracia de warmup del vigía)
printf '===== ARRANQUE %s =====\n' "$(date '+%Y-%m-%d %H:%M:%S')" > /tmp/ia_live.log
nohup uv run python -m src.main --live >> /tmp/ia_live.log 2>&1 &

echo "Arrancando el dashboard..."
nohup uv run python -m src.main --dashboard >> /tmp/ia_dash.log 2>&1 &

echo "Arrancando el vigía (auto-recuperación)..."
nohup /bin/zsh "$PROJECT/ops/watchdog.sh" >/dev/null 2>&1 &

sleep 12
echo ""
if pgrep -f "src.main --live" >/dev/null; then echo "✅ BOT en marcha"; else echo "❌ el bot no arrancó (revisa /tmp/ia_live.log)"; fi
if pgrep -f "src.main --dashboard" >/dev/null; then echo "✅ DASHBOARD en marcha → http://127.0.0.1:8787"; else echo "❌ el dashboard no arrancó"; fi
if pgrep -f "ops/watchdog.sh" >/dev/null; then echo "✅ VIGÍA activo (si se congela, lo reinicia solo)"; else echo "⚠️ el vigía no arrancó"; fi
echo ""
echo "Puedes cerrar esta ventana. Todo sigue corriendo en segundo plano."
