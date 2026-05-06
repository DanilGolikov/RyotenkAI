#!/usr/bin/env bash
# ==============================================================================
# Inference Runtime Container Entrypoint
# ==============================================================================
#
# Configures SSH access (via PUBLIC_KEY env), starts sshd, then executes
# the provided command. If no command is given, keeps the container alive.
#
# Usage:
#   docker run -e PUBLIC_KEY="ssh-rsa ..." <image>                  # SSH-only
#   docker run <image> python -m vllm.entrypoints.openai.api_server # run vLLM
#   docker run <image> --help                                       # show help
#
# Environment:
#   PUBLIC_KEY    SSH public key to authorize (optional)
#
# ==============================================================================

set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  head -n 17 "$0" | grep "^#" | sed 's/^# \?//'
  exit 0
fi

if command -v ssh-keygen >/dev/null 2>&1; then
  ssh-keygen -A >/dev/null 2>&1 || true
fi

if [[ -n "${PUBLIC_KEY:-}" ]]; then
  mkdir -p /root/.ssh
  chmod 700 /root/.ssh
  touch /root/.ssh/authorized_keys
  chmod 600 /root/.ssh/authorized_keys
  if ! grep -qxF "${PUBLIC_KEY}" /root/.ssh/authorized_keys; then
    echo "${PUBLIC_KEY}" >> /root/.ssh/authorized_keys
  fi
fi

if command -v /usr/sbin/sshd >/dev/null 2>&1; then
  mkdir -p /var/run/sshd
  /usr/sbin/sshd || true
elif command -v sshd >/dev/null 2>&1; then
  mkdir -p /var/run/sshd
  sshd || true
fi

if [[ $# -gt 0 ]]; then
  exec "$@"
fi

exec tail -f /dev/null
