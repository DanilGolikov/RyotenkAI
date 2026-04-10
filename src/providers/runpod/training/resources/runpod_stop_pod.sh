#!/bin/bash
# shellcheck shell=bash
#
# RunPod stop-pod helper. Defines _runpod_stop_pod function that gracefully
# stops the current pod via RunPod GraphQL API (podStop mutation).
#
# Sourced from:
#   - start_training.sh post-python hook (happy path, fast)
#   - watchdog.sh (safety-net path)
#
# Required env vars: RUNPOD_API_KEY, RUNPOD_POD_ID, RUNPOD_AUTO_STOP
# Optional:          RUNPOD_KEEP_ON_ERROR (respected only when exit_code != 0)
#
# Callers may set `exit_code` in their scope — we honor it for KEEP_ON_ERROR.

_runpod_stop_pod() {
  local caller_exit_code=${exit_code:-0}

  if [ "${RUNPOD_AUTO_STOP:-}" != "true" ]; then
    return 0
  fi

  if [ "$caller_exit_code" -ne 0 ] && [ "${RUNPOD_KEEP_ON_ERROR:-false}" = "true" ]; then
    echo "[STOP_POD] Training failed and RUNPOD_KEEP_ON_ERROR=true — pod stays running for debug"
    return 0
  fi

  if [ -z "${RUNPOD_API_KEY:-}" ] || [ -z "${RUNPOD_POD_ID:-}" ]; then
    echo "[STOP_POD] Missing RUNPOD_API_KEY or RUNPOD_POD_ID — skipping"
    return 0
  fi

  local gql='mutation{podStop(input:{podId:"'"$RUNPOD_POD_ID"'"}){id}}'
  local attempt resp
  for attempt in 1 2 3; do
    echo "[STOP_POD] Stopping pod $RUNPOD_POD_ID (attempt $attempt/3)..."
    resp=$(curl -s --max-time 30 -X POST "https://api.runpod.io/graphql" \
      -H "Content-Type: application/json" \
      -H "Authorization: Bearer $RUNPOD_API_KEY" \
      -d "{\"query\":\"$gql\"}" 2>&1) || true
    if echo "$resp" | grep -q '"id"'; then
      echo "[STOP_POD] Pod stop requested successfully"
      return 0
    fi
    # podStop is idempotent: already-stopped pods may return an error that
    # still indicates success. Treat common "already stopped" markers as OK.
    if echo "$resp" | grep -qiE 'already.*(stop|exit)|not running'; then
      echo "[STOP_POD] Pod already stopped"
      return 0
    fi
    echo "[STOP_POD] Attempt $attempt failed: $resp"
    sleep $((attempt * 5))
  done
  echo "[STOP_POD] All attempts failed — pod stays running"
  return 1
}
