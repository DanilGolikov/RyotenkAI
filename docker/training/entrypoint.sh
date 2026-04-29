#!/usr/bin/env bash
# ==============================================================================
# Training Runtime Container Entrypoint
# ==============================================================================
#
# Boot order:
#   1. Generate SSH host keys (idempotent).
#   2. Inject PUBLIC_KEY env var into /root/.ssh/authorized_keys so the
#      Mac control plane can SSH in.
#   3. Start /usr/sbin/sshd in the background — Mac uses it for rsync,
#      exec, and the runner SSH tunnel.
#   4. If a custom command was passed (e.g. `docker run image bash`),
#      exec it. Used for integration tests and ad-hoc shells.
#   5. Otherwise: ``sleep infinity`` and wait for the Mac to launch
#      uvicorn via SSH-exec (see runner_launcher.py).
#
# The pod is INTENTIONALLY INERT at boot — the Mac orchestrates the
# in-pod uvicorn runner the same way it orchestrates the trainer
# subprocess. Symmetry, restartability, and provider-agnostic
# bootstrap (RunPod / single_node / future providers all use the
# same pattern).
#
# Usage:
#   docker run <image>                     # sshd + idle (Mac drives)
#   docker run <image> bash                # interactive shell
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

# 2. Inject PUBLIC_KEY into authorized_keys.
#
# Provider control planes (RunPod, single_node bootstrap) inject the
# Mac's SSH public key as the ``PUBLIC_KEY`` env var so the pod
# accepts our SSH connections for rsync, exec, and the in-pod runner
# tunnel. Doing this here — inside our entrypoint — instead of in a
# provider-specific ``docker_args`` shell command means the same
# bootstrap works for every provider: nobody has to remember to
# duplicate the logic in each provider's pod-creation kwargs.
#
# Idempotent: appending a duplicate key on a restarted pod is
# harmless — sshd deduplicates lines on auth.
if [[ -n "${PUBLIC_KEY:-}" ]]; then
  mkdir -p /root/.ssh
  chmod 700 /root/.ssh
  printf '%s\n' "$PUBLIC_KEY" >> /root/.ssh/authorized_keys
  chmod 600 /root/.ssh/authorized_keys
fi

# 3. Background sshd. Best-effort — failure here doesn't block training.
if command -v /usr/sbin/sshd >/dev/null 2>&1; then
  mkdir -p /var/run/sshd
  /usr/sbin/sshd || service ssh start || true
fi

# 4. Custom command path (interactive debugging, integration tests).
#
# Used by ``docker run image bash`` and the integration-test
# fixtures. Production providers leave CMD empty and fall through to
# the inert ``sleep infinity`` below.
if [[ $# -gt 0 ]]; then
  exec "$@"
fi

# Default path: keep the pod alive and wait for an SSH-launched
# runner.
#
# Architecture: pod is an INERT compute fabric. The Mac control plane
# (gpu_deployer → runner_launcher) SSH-execs the in-pod uvicorn after
# files are uploaded and dependencies are verified, redirecting its
# stdout/stderr to ``/workspace/runner.log`` from the Mac side. We
# DON'T launch uvicorn here for three reasons:
#
#   1. Symmetry with the trainer subprocess — the Mac orchestrates
#      both runner and trainer via SSH-exec. One pattern, one mental
#      model.
#   2. Restartability — a runner crash can be recovered by re-execing
#      uvicorn over the same pod, without redeploying or recreating
#      the pod. With auto-launch, a crash forces a full pod cycle.
#   3. Provider compatibility — RunPod historically passes a
#      ``docker_args`` CMD that ends in ``sleep infinity``, which
#      RunPod's CMD-override silently shadowed any auto-launch we
#      tried to do here. Making the pod intentionally inert removes
#      this ambiguity for any provider.
#
# The Mac always launches uvicorn with stdbuf+redirect to /workspace/
# runner.log, so capture semantics are identical to the
# previously-considered in-entrypoint redirect. See
# ``src/pipeline/stages/managers/deployment/runner_launcher.py``.
echo "[entrypoint] pod ready — sleeping until SSH commands arrive"
exec sleep infinity
