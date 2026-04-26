#!/usr/bin/env bash
# ==============================================================================
# Training Runtime Container Entrypoint
# ==============================================================================
#
# Boot order:
#   1. Generate SSH host keys (idempotent).
#   2. Start /usr/sbin/sshd in the background — Mac control plane
#      reaches this container via `ssh root@<pod>` for rsync, exec,
#      and the local-port-forward `ssh -L 18080:127.0.0.1:8080`
#      that connects to the job server below.
#   3. If a custom command was passed (`docker run image bash …`),
#      exec it — preserves the legacy "shell into the runtime" path.
#   4. Otherwise exec uvicorn so the in-pod job server takes over PID
#      from this script. dumb-init (already PID 1) forwards SIGTERM
#      directly to uvicorn.
#
# The job server binds 127.0.0.1:8080 only — never publicly exposed.
# All traffic from Mac arrives over the SSH tunnel.
#
# Override via env:
#   RYOTENKAI_RUNNER_PORT     default 8080
#   RYOTENKAI_RUNNER_HOST     default 127.0.0.1 (DO NOT change to 0.0.0.0
#                             without setting up authentication)
#
# Usage:
#   docker run <image>                     # sshd + job server
#   docker run <image> bash                # interactive shell (no job server)
#   docker run <image> --help              # show this help
#
# ==============================================================================

set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  head -n 35 "$0" | grep "^#" | sed 's/^# \?//'
  exit 0
fi

# 1. SSH host keys.
if command -v ssh-keygen >/dev/null 2>&1; then
  ssh-keygen -A >/dev/null 2>&1 || true
fi

# 2. Background sshd. Best-effort — failure here doesn't block training.
if command -v /usr/sbin/sshd >/dev/null 2>&1; then
  mkdir -p /var/run/sshd
  /usr/sbin/sshd || true
fi

# 3. Custom command path (interactive debugging, integration tests).
if [[ $# -gt 0 ]]; then
  exec "$@"
fi

# 4. Default: launch the in-pod job server.
RYOTENKAI_RUNNER_HOST="${RYOTENKAI_RUNNER_HOST:-127.0.0.1}"
RYOTENKAI_RUNNER_PORT="${RYOTENKAI_RUNNER_PORT:-8080}"

PY_BIN=""
if command -v python3 >/dev/null 2>&1; then
  PY_BIN=python3
elif command -v python >/dev/null 2>&1; then
  PY_BIN=python
elif [ -x /opt/conda/bin/python3 ]; then
  PY_BIN=/opt/conda/bin/python3
elif [ -x /opt/conda/bin/python ]; then
  PY_BIN=/opt/conda/bin/python
else
  echo "PYTHON_NOT_FOUND" >&2
  exit 127
fi

# `exec` so dumb-init's child becomes uvicorn directly — bash doesn't
# stay around to swallow signals. SIGTERM from `docker stop` flows
# through dumb-init → uvicorn → graceful shutdown of the asyncio loop
# → the supervisor's SIGTERM-on-trainer code path (Phase 2).
exec "$PY_BIN" -m uvicorn src.runner.main:app \
  --host "$RYOTENKAI_RUNNER_HOST" \
  --port "$RYOTENKAI_RUNNER_PORT" \
  --no-access-log
