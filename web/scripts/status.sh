#!/usr/bin/env bash
# Report whether backend/frontend are running and where.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "${SCRIPT_DIR}/_common.sh"

report() {
  local label="$1" file="$2" url="$3"
  local pid
  pid="$(read_pid "$file")"
  if [[ -n "$pid" ]] && is_alive "$pid"; then
    printf "  %-9s ${C_GREEN}running${C_RESET}  pid=%-7s  %s\n" "$label" "$pid" "$url"
  elif [[ -n "$pid" ]]; then
    printf "  %-9s ${C_YELLOW}stale  ${C_RESET}  pid=%-7s  (pid dead, remove with stop.sh)\n" "$label" "$pid"
  else
    printf "  %-9s ${C_DIM}stopped${C_RESET}\n" "$label"
  fi
}

printf "${C_BOLD}RyotenkAI web:${C_RESET}\n"
report "backend"  "${BACKEND_PID_FILE}"  "http://${API_HOST}:${API_PORT}/docs"
report "frontend" "${FRONTEND_PID_FILE}" "http://localhost:${WEB_PORT}"
printf "  ${C_DIM}logs:     %s${C_RESET}\n" "${RUN_DIR}"
