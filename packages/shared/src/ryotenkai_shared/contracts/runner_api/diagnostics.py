"""Diagnostics endpoint DTOs (Phase 2 PR-2.1).

Replaces the SSH-based postmortem probes (``dmesg``, ``nvidia-smi``,
kernel-signal grep) with a structured HTTP response. Each block has
its own typed report so the Mac-side caller can switch on
``response.<block>.error`` without dict-poking.

Endpoint contract (runner-side):

    GET /api/v1/diagnostics?include=dmesg,gpu,kernel_signals
        → 200 application/json
            DiagnosticsResponse
        → 422 application/problem+json (DIAGNOSTIC_INVALID_INCLUDE)
        → 502 application/problem+json (DIAGNOSTIC_FAILED) — only if
            EVERY collector failed; per-block failures are surfaced
            inside the response without an HTTP error.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field

from ._strict import _StrictModel


class DiagnosticsInclude(StrEnum):
    """Whitelist of include keys for ``GET /diagnostics?include=``.

    StrEnum so FastAPI auto-validates the query parameter and 422s
    on unknown values via the universal validation handler.
    """

    DMESG = "dmesg"
    GPU = "gpu"
    KERNEL_SIGNALS = "kernel_signals"


# Sentinels surfaced inside per-block ``error`` so the HTTP layer
# stays 200 even when one collector is unhealthy.
class DiagnosticsBlockError(StrEnum):
    PERMISSION_DENIED = "permission_denied"     # CAP_SYSLOG missing for dmesg
    TOOL_MISSING = "tool_missing"               # nvidia-smi not installed
    TIMEOUT = "timeout"                          # subprocess timeout
    UNKNOWN = "unknown"                          # any other subprocess error


class DmesgReport(_StrictModel):
    """``dmesg | tail -n N`` output with a ``truncated`` flag if the
    pod kernel buffer exceeded the response cap."""

    lines: list[str] = Field(
        default_factory=list,
        description="Tail of the kernel ring buffer, newest last.",
    )
    truncated: bool = Field(
        default=False,
        description="True if the kernel buffer was larger than the response cap.",
    )
    error: DiagnosticsBlockError | None = Field(
        default=None,
        description="Block-level failure indicator; HTTP stays 200.",
    )


class KernelSignalsReport(_StrictModel):
    """``dmesg | grep -iE 'oom|kill|memory|nvrm|xid|nvidia'`` —
    filtered view of the same source, intended for postmortem
    triage when ``dmesg`` itself is too noisy to scan."""

    matches: list[str] = Field(
        default_factory=list,
        description="Filtered kernel lines (OOM / NVRM / XID / nvidia).",
    )
    truncated: bool = Field(default=False)
    error: DiagnosticsBlockError | None = Field(default=None)


class GpuRow(_StrictModel):
    """One per-GPU row from ``nvidia-smi --query-gpu=...``."""

    name: str
    utilization_gpu_percent: int = Field(ge=0, le=100)
    memory_used_mib: int = Field(ge=0)
    memory_total_mib: int = Field(ge=0)


class GpuReport(_StrictModel):
    """``nvidia-smi`` snapshot — one row per GPU.

    No fancy temperature / power draw fields here; those go through
    ``/api/v1/resources`` (PR-2.2). This block is postmortem-shaped:
    enough to know "GPU was at 0% utilization" or "VRAM exhausted".
    """

    rows: list[GpuRow] = Field(default_factory=list)
    error: DiagnosticsBlockError | None = Field(default=None)


class DiagnosticsResponse(_StrictModel):
    """The composite envelope.

    One block per requested ``include`` key. Blocks NOT requested are
    set to ``None`` and stripped from the wire via
    ``model_dump(exclude_none=True)``.

    Example wire:

        {"dmesg": {"lines": ["..."]},
         "gpu": {"rows": [{"name": "RTX 4090", ...}]}}
    """

    dmesg: DmesgReport | None = None
    gpu: GpuReport | None = None
    kernel_signals: KernelSignalsReport | None = None


__all__ = [
    "DiagnosticsBlockError",
    "DiagnosticsInclude",
    "DiagnosticsResponse",
    "DmesgReport",
    "GpuReport",
    "GpuRow",
    "KernelSignalsReport",
]
