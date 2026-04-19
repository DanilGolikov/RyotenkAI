#!/usr/bin/env bash
# Shared helpers for start/stop/restart/status scripts.
# Sourced, not executed. Requires bash.

set -u

# Resolve repo root regardless of caller cwd.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WEB_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${WEB_DIR}/.." && pwd)"
RUN_DIR="${WEB_DIR}/.run"

BACKEND_PID_FILE="${RUN_DIR}/backend.pid"
FRONTEND_PID_FILE="${RUN_DIR}/frontend.pid"
BACKEND_LOG="${RUN_DIR}/backend.log"
FRONTEND_LOG="${RUN_DIR}/frontend.log"

# Tunables (env-overridable).
API_HOST="${RYOTENKAI_API_HOST:-127.0.0.1}"
API_PORT="${RYOTENKAI_API_PORT:-8000}"
WEB_PORT="${RYOTENKAI_WEB_PORT:-5173}"
RUNS_DIR="${RYOTENKAI_RUNS_DIR:-${REPO_ROOT}/runs}"
CORS_ORIGINS="${RYOTENKAI_API_CORS_ORIGINS:-http://localhost:${WEB_PORT}}"
API_LOG_LEVEL="${RYOTENKAI_API_LOG_LEVEL:-info}"

TARGET="${1:-all}"   # all | backend | frontend

# Color helpers (no-op when stdout is not a tty).
if [[ -t 1 ]]; then
  C_DIM='\033[2m'; C_BOLD='\033[1m'; C_RED='\033[31m'; C_GREEN='\033[32m'
  C_YELLOW='\033[33m'; C_CYAN='\033[36m'; C_RESET='\033[0m'
else
  C_DIM=''; C_BOLD=''; C_RED=''; C_GREEN=''; C_YELLOW=''; C_CYAN=''; C_RESET=''
fi

info() { printf "${C_CYAN}[%s]${C_RESET} %s\n" "web" "$*"; }
ok()   { printf "${C_GREEN}[%s]${C_RESET} %s\n" "web" "$*"; }
warn() { printf "${C_YELLOW}[%s]${C_RESET} %s\n" "web" "$*" >&2; }
err()  { printf "${C_RED}[%s]${C_RESET} %s\n" "web" "$*" >&2; }

mkdir -p "${RUN_DIR}"

# -------- pid helpers --------

read_pid() {
  local file="$1"
  [[ -f "$file" ]] || { echo ""; return; }
  local pid
  pid="$(cat "$file" 2>/dev/null | tr -d ' \t\n\r')"
  [[ -n "$pid" ]] || { echo ""; return; }
  echo "$pid"
}

is_alive() {
  local pid="$1"
  [[ -n "$pid" ]] || return 1
  kill -0 "$pid" 2>/dev/null
}

clear_stale_pid() {
  local file="$1"
  local pid
  pid="$(read_pid "$file")"
  if [[ -n "$pid" ]] && ! is_alive "$pid"; then
    rm -f "$file"
    return 0
  fi
  return 1
}

_collect_descendants() {
  # Recursively echo all descendant PIDs of $1, one per line.
  local parent="$1"
  local children
  children="$(pgrep -P "$parent" 2>/dev/null || true)"
  local child
  for child in $children; do
    _collect_descendants "$child"
    echo "$child"
  done
}

kill_tree() {
  # Graceful shutdown: collect descendants, SIGTERM whole tree,
  # wait up to 5s, then SIGKILL leftovers. We deliberately avoid
  # `kill -SIG -pgid` because nohup does not put the child into a
  # new process group on bash, so the pgid may coincide with the
  # caller's shell and we would signal ourselves.
  local pid="$1"
  is_alive "$pid" || return 0

  local descendants
  descendants="$(_collect_descendants "$pid")"
  local all_pids="$descendants $pid"

  # Deepest-first SIGTERM.
  local p
  for p in $all_pids; do
    kill -TERM "$p" 2>/dev/null || true
  done

  local _i
  for _i in $(seq 1 25); do
    if ! is_alive "$pid"; then return 0; fi
    sleep 0.2
  done

  warn "pid $pid did not exit on SIGTERM, sending SIGKILL"
  # Re-collect — children could have been replaced by reloaders etc.
  descendants="$(_collect_descendants "$pid")"
  all_pids="$descendants $pid"
  for p in $all_pids; do
    kill -KILL "$p" 2>/dev/null || true
  done
  return 0
}

# Find a free python that has uvicorn/fastapi available. We prefer the repo's
# .venv, then project env var, then system python.
resolve_python() {
  if [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
    echo "${REPO_ROOT}/.venv/bin/python"; return
  fi
  if [[ -x "${REPO_ROOT}/.venv/bin/python3" ]]; then
    echo "${REPO_ROOT}/.venv/bin/python3"; return
  fi
  if [[ -n "${PYTHON:-}" ]]; then
    echo "${PYTHON}"; return
  fi
  if command -v python3 >/dev/null 2>&1; then
    echo "python3"; return
  fi
  echo "python"
}

ensure_node_deps() {
  if [[ ! -d "${WEB_DIR}/node_modules" ]]; then
    info "node_modules missing — running \`npm install\`"
    (cd "${WEB_DIR}" && npm install --silent) || {
      err "npm install failed"
      return 1
    }
  fi
}
