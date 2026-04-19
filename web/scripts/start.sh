#!/usr/bin/env bash
# Start backend (FastAPI via `ryotenkai serve`) and frontend (Vite dev server)
# as detached background processes. Idempotent: if a process is already alive
# its PID is reused and no second copy is spawned.
#
# Usage:
#   ./scripts/start.sh              # both
#   ./scripts/start.sh backend      # backend only
#   ./scripts/start.sh frontend     # frontend only

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "${SCRIPT_DIR}/_common.sh"

start_backend() {
  local existing
  existing="$(read_pid "${BACKEND_PID_FILE}")"
  if [[ -n "$existing" ]] && is_alive "$existing"; then
    ok "backend already running (pid=${existing}) on http://${API_HOST}:${API_PORT}"
    return 0
  fi
  clear_stale_pid "${BACKEND_PID_FILE}" || true

  local python
  python="$(resolve_python)"
  info "starting backend (python=${python}, port=${API_PORT}, runs_dir=${RUNS_DIR})"

  # Use setsid when available so SIGTERM to the PID reaches the whole group.
  local launcher=(nohup)
  if command -v setsid >/dev/null 2>&1; then
    launcher+=(setsid)
  fi

  : > "${BACKEND_LOG}"
  (
    cd "${REPO_ROOT}"
    "${launcher[@]}" "${python}" -m src.main serve \
      --host "${API_HOST}" \
      --port "${API_PORT}" \
      --runs-dir "${RUNS_DIR}" \
      --cors-origins "${CORS_ORIGINS}" \
      --log-level "${API_LOG_LEVEL}" \
      >> "${BACKEND_LOG}" 2>&1 &
    echo $! > "${BACKEND_PID_FILE}"
  )
  sleep 0.3
  local pid
  pid="$(read_pid "${BACKEND_PID_FILE}")"
  if ! is_alive "$pid"; then
    err "backend failed to start — tail of ${BACKEND_LOG}:"
    tail -n 30 "${BACKEND_LOG}" >&2 || true
    rm -f "${BACKEND_PID_FILE}"
    return 1
  fi
  ok "backend started (pid=${pid}) — http://${API_HOST}:${API_PORT}/docs"
}

start_frontend() {
  local existing
  existing="$(read_pid "${FRONTEND_PID_FILE}")"
  if [[ -n "$existing" ]] && is_alive "$existing"; then
    ok "frontend already running (pid=${existing}) on http://localhost:${WEB_PORT}"
    return 0
  fi
  clear_stale_pid "${FRONTEND_PID_FILE}" || true

  if ! command -v npm >/dev/null 2>&1; then
    err "npm not found in PATH — install Node.js 18+ or run backend only"
    return 1
  fi

  ensure_node_deps || return 1

  info "starting frontend (vite dev on port ${WEB_PORT})"
  local launcher=(nohup)
  if command -v setsid >/dev/null 2>&1; then
    launcher+=(setsid)
  fi

  : > "${FRONTEND_LOG}"
  (
    cd "${WEB_DIR}"
    "${launcher[@]}" npm run dev -- --host --port "${WEB_PORT}" \
      >> "${FRONTEND_LOG}" 2>&1 &
    echo $! > "${FRONTEND_PID_FILE}"
  )
  sleep 0.3
  local pid
  pid="$(read_pid "${FRONTEND_PID_FILE}")"
  if ! is_alive "$pid"; then
    err "frontend failed to start — tail of ${FRONTEND_LOG}:"
    tail -n 30 "${FRONTEND_LOG}" >&2 || true
    rm -f "${FRONTEND_PID_FILE}"
    return 1
  fi
  ok "frontend started (pid=${pid}) — http://localhost:${WEB_PORT}"
}

case "${TARGET}" in
  all)       start_backend && start_frontend ;;
  backend)   start_backend ;;
  frontend)  start_frontend ;;
  *)
    err "unknown target: ${TARGET} (expected: all | backend | frontend)"
    exit 2
    ;;
esac
