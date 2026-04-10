#!/bin/bash
# shellcheck shell=bash
#
# RunPod training watchdog — independent safety-net for pod auto-stop.
#
# Runs as a detached process alongside the training script. Guarantees pod
# shutdown after any logical end of training — even if the parent bash is
# SIGKILL'd, Python hangs, or the pipeline monitor dies.
#
# MAIN GATE: Pipeline heartbeat. While the pipeline actively monitors the pod
# (touches /workspace/.pipeline_heartbeat every ~5s via SSH), the watchdog
# stays dormant and does NOT check GPU or kill anything. Once the heartbeat
# goes stale (pipeline gone), the watchdog wakes up.
#
# Stop reasons written to /workspace/STOPPED_BY_WATCHDOG:
#   LOGICAL_END  — training marker appeared (TRAINING_COMPLETE/FAILED)
#   GPU_IDLE     — GPU idle for a sustained period after startup grace
#   MAX_LIFETIME — hard 48h lifetime exceeded (ultimate safety)

set -uo pipefail

WORKSPACE="${WATCHDOG_WORKSPACE:-/workspace}"
PIPELINE_HEARTBEAT="$WORKSPACE/.pipeline_heartbeat"
WATCHDOG_HEARTBEAT="$WORKSPACE/.watchdog_heartbeat"
WATCHDOG_LOG="$WORKSPACE/watchdog.log"
STOPPED_REASON_FILE="$WORKSPACE/STOPPED_BY_WATCHDOG"
MARKER_COMPLETE="$WORKSPACE/TRAINING_COMPLETE"
MARKER_FAILED="$WORKSPACE/TRAINING_FAILED"
STOP_POD_LIB="$WORKSPACE/runpod_stop_pod.sh"

# Timing constants (seconds)
POLL_INTERVAL=30
STARTUP_GRACE=300              # 5 min — do not touch GPU idle detection before this
PIPELINE_HEARTBEAT_STALE=600   # 10 min — pipeline considered gone
IDLE_THRESHOLD=1200            # 20 min — continuous GPU idle required to stop
LOGICAL_END_GRACE=60           # grace after marker appears (let happy-path run)
MAX_LIFETIME=172800            # 48 h

# GPU idle thresholds (both must hold simultaneously)
IDLE_UTIL_MAX=5                # max util.gpu (%)
IDLE_MEM_MAX_PCT=30            # max memory.used / memory.total (%)

# --- helpers ---------------------------------------------------------------

log() {
  echo "[$(date -Iseconds)] $*" >> "$WATCHDOG_LOG" 2>/dev/null || true
}

record_stop() {
  local reason=$1
  {
    echo "reason=$reason"
    echo "timestamp=$(date -Iseconds)"
    echo "uptime_seconds=$((${NOW:-0} - START_TS))"
    echo "last_gpu_util=${LAST_UTIL:-n/a}"
    echo "last_gpu_mem_pct=${LAST_MEM_PCT:-n/a}"
    echo "pid=$$"
  } > "$STOPPED_REASON_FILE" 2>/dev/null || true
}

is_pipeline_active() {
  [ -f "$PIPELINE_HEARTBEAT" ] || return 1
  local mtime now age
  mtime=$(stat -c %Y "$PIPELINE_HEARTBEAT" 2>/dev/null || echo 0)
  now=$(date +%s)
  age=$((now - mtime))
  [ "$age" -lt "$PIPELINE_HEARTBEAT_STALE" ]
}

# Prints "util mem_pct" — max across all GPUs. Falls back to "0 0" on error.
gpu_metrics() {
  local raw util mem_used mem_total pct
  local max_util=0 max_mem_pct=0
  raw=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total \
    --format=csv,noheader,nounits 2>/dev/null) || { echo "0 0"; return; }
  [ -z "$raw" ] && { echo "0 0"; return; }
  while IFS=',' read -r util mem_used mem_total; do
    util=$(echo "$util" | tr -d ' ')
    mem_used=$(echo "$mem_used" | tr -d ' ')
    mem_total=$(echo "$mem_total" | tr -d ' ')
    [ -z "$util" ] && continue
    [ "$mem_total" -eq 0 ] 2>/dev/null && continue
    if [ "$util" -gt "$max_util" ] 2>/dev/null; then max_util=$util; fi
    pct=$((mem_used * 100 / mem_total))
    if [ "$pct" -gt "$max_mem_pct" ]; then max_mem_pct=$pct; fi
  done <<< "$raw"
  echo "$max_util $max_mem_pct"
}

stop_pod_and_exit() {
  local reason=$1
  log "STOP reason=$reason util=${LAST_UTIL:-n/a} mem=${LAST_MEM_PCT:-n/a}%"
  record_stop "$reason"
  if [ -f "$STOP_POD_LIB" ]; then
    # shellcheck source=/dev/null
    source "$STOP_POD_LIB"
    _runpod_stop_pod || log "_runpod_stop_pod returned non-zero"
  else
    log "FATAL: stop-pod library not found at $STOP_POD_LIB"
  fi
  exit 0
}

# --- main loop -------------------------------------------------------------

START_TS=$(date +%s)
IDLE_SINCE=0
LAST_UTIL=0
LAST_MEM_PCT=0

log "watchdog started pid=$$ workspace=$WORKSPACE"
# Initial heartbeat so start_training.sh can verify we launched
touch "$WATCHDOG_HEARTBEAT" 2>/dev/null || true

while true; do
  touch "$WATCHDOG_HEARTBEAT" 2>/dev/null || true
  NOW=$(date +%s)
  UPTIME=$((NOW - START_TS))

  # Hard max lifetime — ultimate safety
  if [ "$UPTIME" -gt "$MAX_LIFETIME" ]; then
    stop_pod_and_exit "MAX_LIFETIME"
  fi

  # Dormant while pipeline is actively monitoring
  if is_pipeline_active; then
    IDLE_SINCE=0
    sleep "$POLL_INTERVAL"
    continue
  fi

  # Pipeline is absent. Check training markers first.
  if [ -f "$MARKER_COMPLETE" ] || [ -f "$MARKER_FAILED" ]; then
    if [ -f "$MARKER_FAILED" ] && [ "${RUNPOD_KEEP_ON_ERROR:-false}" = "true" ]; then
      # Debug mode: keep pod alive until max lifetime
      sleep "$POLL_INTERVAL"
      continue
    fi
    log "logical end detected, grace ${LOGICAL_END_GRACE}s before stop"
    sleep "$LOGICAL_END_GRACE"
    stop_pod_and_exit "LOGICAL_END"
  fi

  # GPU idle detection (respects startup grace)
  if [ "$UPTIME" -gt "$STARTUP_GRACE" ]; then
    read -r LAST_UTIL LAST_MEM_PCT <<< "$(gpu_metrics)"
    if [ "$LAST_UTIL" -lt "$IDLE_UTIL_MAX" ] && [ "$LAST_MEM_PCT" -lt "$IDLE_MEM_MAX_PCT" ]; then
      if [ "$IDLE_SINCE" -eq 0 ]; then
        IDLE_SINCE=$NOW
        log "GPU idle started util=$LAST_UTIL mem=$LAST_MEM_PCT%"
      fi
      if [ $((NOW - IDLE_SINCE)) -gt "$IDLE_THRESHOLD" ]; then
        stop_pod_and_exit "GPU_IDLE"
      fi
    else
      if [ "$IDLE_SINCE" -ne 0 ]; then
        log "GPU active again util=$LAST_UTIL mem=$LAST_MEM_PCT%"
      fi
      IDLE_SINCE=0
    fi
  fi

  sleep "$POLL_INTERVAL"
done
