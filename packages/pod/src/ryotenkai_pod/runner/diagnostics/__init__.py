"""Pod-side diagnostic collectors (Phase 2 PR-2.1+).

Replaces the SSH-based ``ssh dmesg``, ``ssh nvidia-smi`` postmortem
probes the Mac orchestrator used to issue. The collectors here run
as ordinary subprocesses inside the pod (no SSH involved); the
``ryotenkai_pod.runner.api.diagnostics`` router exposes them over
HTTP.

Each collector:

1. Returns a typed Pydantic report from
   :mod:`ryotenkai_shared.contracts.runner_api.diagnostics`.
2. Surfaces failures via the report's ``error`` field instead of
   raising — the HTTP handler stays 200 unless EVERY block is
   unhealthy (RP2 — dmesg may be CAP_SYSLOG-restricted in RunPod;
   one collector failure doesn't kill the call).
3. Uses subprocess timeouts so a hung shell doesn't deadlock the
   request.
"""

from ryotenkai_pod.runner.diagnostics.collectors import (
    DEFAULT_DMESG_TAIL_LINES,
    DEFAULT_KERNEL_SIGNALS_TAIL_LINES,
    DEFAULT_SUBPROCESS_TIMEOUT_S,
    KERNEL_SIGNAL_PATTERN,
    collect_dmesg,
    collect_kernel_signals,
    collect_nvidia_smi,
)

__all__ = [
    "DEFAULT_DMESG_TAIL_LINES",
    "DEFAULT_KERNEL_SIGNALS_TAIL_LINES",
    "DEFAULT_SUBPROCESS_TIMEOUT_S",
    "KERNEL_SIGNAL_PATTERN",
    "collect_dmesg",
    "collect_kernel_signals",
    "collect_nvidia_smi",
]
