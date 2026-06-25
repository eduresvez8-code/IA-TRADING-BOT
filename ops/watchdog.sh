#!/bin/zsh
# Vigía de auto-recuperación del bot IA TRADING (bucle en segundo plano).
# Lo lanza el botón ARRANCAR. Cada INTERVAL revisa salud y, si el bot está caído O
# congelado (sin ciclar > STALE_MIN), lo reinicia solo. Sobrevive sleep/wake; tras
# despertar el Mac, recupera el bot en pocos minutos. Respeta un "freno de mano":
# si existe ~/.iatrading_paused (lo crea DETENER), no toca nada.
#
# NO toca la lógica de trading: solo arranca/reinicia procesos. Red de seguridad
# externa, a prueba de cualquier causa de congelamiento (sleep, red, etc.).

export PATH="/opt/homebrew/bin:$HOME/.local/bin:/usr/bin:/bin:$PATH"
PROJECT="/Users/eduardorestrepo/Desktop/IA TRADING"
DB="$PROJECT/data/trading.db"
PAUSE="$HOME/.iatrading_paused"
MARK="$HOME/.iatrading_last_start"   # epoch (s) del último arranque del bot
LOG="$HOME/.iatrading_watchdog.log"
STALE_MIN=15      # > 3 ciclos de 5m sin snapshot = congelado
INTERVAL=300      # revisa cada 5 min

log(){ echo "$(date '+%Y-%m-%d %H:%M:%S')  $1" >> "$LOG"; }

start_live(){
  date +%s > "$MARK"   # marca el arranque → activa la gracia de warmup
  printf '===== ARRANQUE (watchdog) %s =====\n' "$(date '+%F %T')" >> /tmp/ia_live.log
  ( cd "$PROJECT" && nohup uv run python -m src.main --live >> /tmp/ia_live.log 2>&1 & )
}
start_dash(){ ( cd "$PROJECT" && nohup uv run python -m src.main --dashboard >> /tmp/ia_dash.log 2>&1 & ); }

check_once(){
  [[ -f "$PAUSE" ]] && return   # freno de mano: el usuario lo apagó a propósito

  local alive last_start now now_ms last_ms age_min healthy=0
  alive=$(pgrep -f "src.main --live" | head -1)
  now=$(date +%s)
  if [[ -n "$alive" ]]; then
    last_start=$(cat "$MARK" 2>/dev/null)
    if [[ -n "$last_start" ]] && (( now - last_start < STALE_MIN * 60 )); then
      healthy=1   # recién arrancado: dale tiempo a cerrar su primera vela
    else
      last_ms=$(sqlite3 -readonly "$DB" "SELECT MAX(ts) FROM equity_snapshots;" 2>/dev/null)
      if [[ -n "$last_ms" ]]; then
        now_ms=$(( now * 1000 ))
        age_min=$(( (now_ms - last_ms) / 60000 ))
        (( age_min < STALE_MIN )) && healthy=1
      fi
    fi
  fi

  if (( healthy == 1 )); then
    log "OK (bot sano)"
  else
    log "BOT no sano (vivo=${alive:-no}) → reiniciando"
    pkill -f "src.main --live" 2>/dev/null
    sleep 3
    start_live
  fi

  pgrep -f "src.main --dashboard" >/dev/null || { log "dashboard caído → reinicio"; start_dash; }
}

# Modo "once": una sola revisión y salir (para tests/diagnóstico).
if [[ "$1" == "once" ]]; then check_once; exit 0; fi

log "===== vigía iniciado (pid $$) ====="
while true; do
  check_once
  sleep "$INTERVAL"
done
