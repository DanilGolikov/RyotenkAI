#!/usr/bin/env bash
# Stop backend and/or frontend. Idempotent.
#
# Usage:
#   ./scripts/stop.sh             # both
#   ./scripts/stop.sh backend     # backend only
#   ./scripts/stop.sh frontend    # frontend only

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "${SCRIPT_DIR}/_common.sh"

stop_one() {
  local label="$1" file="$2"
  local pid
  pid="$(read_pid "$file")"
  if [[ -z "$pid" ]]; then
    info "${label} is not running (no pid file)"
    return 0
  fi
  if ! is_alive "$pid"; then
    warn "${label} pid ${pid} is stale, clearing"
    rm -f "$file"
    return 0
  fi
  info "stopping ${label} (pid=${pid})"
  kill_tree "$pid"
  rm -f "$file"
  ok "${label} stopped"
}

case "${TARGET}" in
  all)
    stop_one "frontend" "${FRONTEND_PID_FILE}"
    stop_one "backend"  "${BACKEND_PID_FILE}"
    ;;
  backend)  stop_one "backend"  "${BACKEND_PID_FILE}" ;;
  frontend) stop_one "frontend" "${FRONTEND_PID_FILE}" ;;
  *)
    err "unknown target: ${TARGET} (expected: all | backend | frontend)"
    exit 2
    ;;
esac
