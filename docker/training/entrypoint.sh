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

# Pod-side log file. Captures uvicorn stdout/stderr from PID 1 of the
# Python process — including ImportError / SyntaxError that fire BEFORE
# any application logging is configured. Mac control plane rsyncs this
# file via existing LogManager.download() chain.
#
# Why direct redirect (and NOT `tee` or uvicorn's --log-config):
#   * `tee` via process-substitution (`> >(tee -a) 2>&1`) would put a
#     bash subshell between dumb-init and uvicorn — uvicorn becomes a
#     grandchild, SIGTERM stops propagating, and `docker stop`
#     escalates to SIGKILL after 10 s. Graceful shutdown lost,
#     in-flight checkpoints lost.
#   * `--log-config` only kicks in AFTER Python's logging dictConfig is
#     applied. Pre-import errors (the ones we just had with
#     ``ModuleNotFoundError: src.utils``) fire BEFORE that point and
#     never reach the file.
#   * Direct ``>>`` keeps the dumb-init → uvicorn parent-child link
#     intact and captures everything from the first byte of stderr.
#
# `stdbuf -oL -eL` forces line-buffered stdout/stderr so the file is
# useful for live `tail -f` (without it Python's default 4 KB block
# buffering makes the file appear empty for the first ~seconds).
#
# Trade-off accepted: `docker logs <ctr>` will be EMPTY for this
# container after the redirect. We don't use `docker logs` from the
# control plane anyway — the Mac pulls /workspace/runner.log over SSH
# via LogManager.
RUNNER_LOG="${RYOTENKAI_RUNNER_LOG:-/workspace/runner.log}"
mkdir -p "$(dirname "$RUNNER_LOG")" || true

# Probe writability — if /workspace is read-only or missing, fall back
# to /tmp (non-persistent across container restarts but at least the
# pod boots and we capture the bootstrap window). The fallback is
# logged to docker stderr so it shows up in provider logs even though
# /workspace mount is broken.
if ! : >> "$RUNNER_LOG" 2>/dev/null; then
  echo "WARNING: $RUNNER_LOG not writable, falling back to /tmp/runner.log" >&2
  RUNNER_LOG="/tmp/runner.log"
fi

# `exec` so dumb-init's child becomes uvicorn directly — bash doesn't
# stay around to swallow signals. SIGTERM from `docker stop` flows
# through dumb-init → uvicorn → graceful shutdown of the asyncio loop
# → the supervisor's SIGTERM-on-trainer code path (Phase 2).
exec stdbuf -oL -eL "$PY_BIN" -m uvicorn src.runner.main:app \
  --host "$RYOTENKAI_RUNNER_HOST" \
  --port "$RYOTENKAI_RUNNER_PORT" \
  --no-access-log \
  >> "$RUNNER_LOG" 2>&1
