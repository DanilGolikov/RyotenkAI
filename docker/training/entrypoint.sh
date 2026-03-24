#!/usr/bin/env bash
# ==============================================================================
# Training Runtime Container Entrypoint
# ==============================================================================
#
# Starts SSH daemon for remote access, then executes the provided command.
# If no command is given, keeps the container alive (tail -f /dev/null).
#
# Usage:
#   docker run <image>                     # keep alive (SSH-only access)
#   docker run <image> bash -c "train.sh"  # run a command
#   docker run <image> --help              # show this help
#
# ==============================================================================

set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  head -n 15 "$0" | grep "^#" | sed 's/^# \?//'
  exit 0
fi

if command -v ssh-keygen >/dev/null 2>&1; then
  ssh-keygen -A >/dev/null 2>&1 || true
fi

if command -v /usr/sbin/sshd >/dev/null 2>&1; then
  mkdir -p /var/run/sshd
  /usr/sbin/sshd || true
fi

if [[ $# -gt 0 ]]; then
  exec "$@"
fi

exec tail -f /dev/null
