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
CLOCK_SKEW_MS=900 # deriva (ms) vs Binance a partir de la cual se avisa (-1021 salta a >1000ms)

log(){ echo "$(date '+%Y-%m-%d %H:%M:%S')  $1" >> "$LOG"; }

# Aviso de reloj: tras despertar el Mac, el reloj local suele quedar adelantado
# y Binance rechaza los requests firmados con -1021. El bot YA se auto-cura en
# caliente (retry_with_backoff recalibra timestamp_offset), pero dejamos constancia
# de la deriva en el log y, si es grande, recordamos el arreglo manual del SO.
# Solo lectura: consulta la hora del servidor de Binance, no requiere sudo.
clock_note(){
  local server_ms local_ms drift
  server_ms=$(curl -s --max-time 5 https://api.binance.com/api/v3/time \
              | sed -n 's/.*"serverTime":\([0-9]*\).*/\1/p')
  [[ -z "$server_ms" ]] && { log "aviso de reloj: no se pudo leer serverTime de Binance"; return; }
  local_ms=$(( $(date +%s) * 1000 ))
  drift=$(( local_ms - server_ms ))   # >0 = reloj local ADELANTADO (causa del -1021)
  if (( drift > CLOCK_SKEW_MS || drift < -CLOCK_SKEW_MS )); then
    log "aviso de reloj: deriva ${drift}ms vs Binance (el bot recalibra solo; si el -1021 persiste corre: sudo sntp -sS time.apple.com)"
  else
    log "reloj OK (deriva ${drift}ms vs Binance)"
  fi
}

start_live(){
  clock_note          # deja constancia del estado del reloj en cada arranque
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
