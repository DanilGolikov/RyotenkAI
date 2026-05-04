"""Subprocess-based diagnostic collectors.

Pure functions: each takes only the subprocess parameters it needs
and returns a typed report. No coupling to FastAPI / ``Depends``,
so unit tests are trivial.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from collections.abc import Callable

from ryotenkai_shared.contracts.runner_api.diagnostics import (
    DiagnosticsBlockError,
    DmesgReport,
    GpuReport,
    GpuRow,
    KernelSignalsReport,
)

# Patterns mirror the pre-Phase-2 SSH probe in
# ``training_monitor._postmortem_diagnostics``:
#     dmesg | grep -iE 'oom|kill|memory|nvrm|xid|nvidia'
KERNEL_SIGNAL_PATTERN = re.compile(
    r"oom|kill|memory|nvrm|xid|nvidia",
    re.IGNORECASE,
)

DEFAULT_SUBPROCESS_TIMEOUT_S: float = 10.0
DEFAULT_DMESG_TAIL_LINES: int = 200
DEFAULT_KERNEL_SIGNALS_TAIL_LINES: int = 30
DEFAULT_DMESG_RESPONSE_CAP_LINES: int = 200  # hard cap on the wire payload


# ---------------------------------------------------------------------------
# Subprocess shim — overridden in tests via dependency injection
# ---------------------------------------------------------------------------


SubprocessRunner = Callable[..., subprocess.CompletedProcess]


def _default_runner(*args, **kwargs) -> subprocess.CompletedProcess:  # type: ignore[no-untyped-def]
    return subprocess.run(*args, **kwargs)


# ---------------------------------------------------------------------------
# dmesg — kernel ring buffer
# ---------------------------------------------------------------------------


def _classify_subprocess_error(exc: BaseException) -> DiagnosticsBlockError:
    """Classify a subprocess exception into the canonical block-level
    error sentinel. Falls through to ``unknown`` for anything we
    haven't seen before — caller logs the raw exc separately."""
    if isinstance(exc, subprocess.TimeoutExpired):
        return DiagnosticsBlockError.TIMEOUT
    if isinstance(exc, PermissionError):
        return DiagnosticsBlockError.PERMISSION_DENIED
    if isinstance(exc, FileNotFoundError):
        return DiagnosticsBlockError.TOOL_MISSING
    return DiagnosticsBlockError.UNKNOWN


def _looks_like_permission_denied(stderr: str) -> bool:
    """``dmesg`` on locked-down kernels (CAP_SYSLOG missing) writes
    something like ``dmesg: read kernel buffer failed: Operation not
    permitted`` to stderr and exits with rc=1. Matching by stderr text
    is brittle but unavoidable — there's no dedicated exit code."""
    needle = stderr.lower()
    return (
        "operation not permitted" in needle
        or "permission denied" in needle
    )


def collect_dmesg(
    *,
    tail_lines: int = DEFAULT_DMESG_TAIL_LINES,
    response_cap_lines: int = DEFAULT_DMESG_RESPONSE_CAP_LINES,
    timeout_s: float = DEFAULT_SUBPROCESS_TIMEOUT_S,
    runner: SubprocessRunner = _default_runner,
) -> DmesgReport:
    """Tail of the kernel ring buffer.

    Returns a :class:`DmesgReport` even on failure — the
    ``error`` field carries a sentinel and the HTTP handler keeps a
    200 status so other collectors can still report normally (RP2).

    Truncation: ``response_cap_lines`` caps the on-wire payload. If
    ``tail_lines > response_cap_lines`` we ship the latter and set
    ``truncated=True``.
    """
    if shutil.which("dmesg") is None:
        return DmesgReport(error=DiagnosticsBlockError.TOOL_MISSING)

    try:
        completed = runner(
            ["dmesg"],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except BaseException as exc:  # subprocess can raise OSError variants
        return DmesgReport(error=_classify_subprocess_error(exc))

    if completed.returncode != 0:
        if _looks_like_permission_denied(completed.stderr or ""):
            return DmesgReport(error=DiagnosticsBlockError.PERMISSION_DENIED)
        return DmesgReport(error=DiagnosticsBlockError.UNKNOWN)

    all_lines = (completed.stdout or "").splitlines()
    requested_tail = all_lines[-tail_lines:] if tail_lines > 0 else all_lines
    truncated = len(requested_tail) > response_cap_lines
    capped = requested_tail[-response_cap_lines:] if truncated else requested_tail

    return DmesgReport(lines=capped, truncated=truncated)


def collect_kernel_signals(
    *,
    tail_lines: int = DEFAULT_KERNEL_SIGNALS_TAIL_LINES,
    timeout_s: float = DEFAULT_SUBPROCESS_TIMEOUT_S,
    runner: SubprocessRunner = _default_runner,
) -> KernelSignalsReport:
    """Filtered ``dmesg`` view — OOM / NVRM / XID / nvidia.

    Reuses :func:`collect_dmesg` to avoid a second subprocess
    invocation per request, then filters in-process. Cheaper and
    keeps the two reports consistent.
    """
    full = collect_dmesg(
        tail_lines=0,                    # don't pre-tail — filter then tail
        response_cap_lines=10**9,        # effectively no cap
        timeout_s=timeout_s,
        runner=runner,
    )
    if full.error is not None:
        # Same sentinel propagates so the operator sees one root cause.
        return KernelSignalsReport(error=full.error)

    matches = [line for line in full.lines if KERNEL_SIGNAL_PATTERN.search(line)]
    truncated = len(matches) > tail_lines
    capped = matches[-tail_lines:] if truncated else matches
    return KernelSignalsReport(matches=capped, truncated=truncated)


# ---------------------------------------------------------------------------
# nvidia-smi
# ---------------------------------------------------------------------------


def _parse_nvidia_smi_csv(csv_text: str) -> list[GpuRow]:
    """Parse ``nvidia-smi --query-gpu=name,utilization.gpu,memory.used,
    memory.total --format=csv,noheader`` output.

    Each row is comma-separated; numeric fields end with `` %`` /
    `` MiB``. We strip those suffixes and coerce via ``int``. Any
    malformed line is skipped (logged at DEBUG by the caller — not
    here, to keep the parser pure)."""
    rows: list[GpuRow] = []
    for raw in csv_text.splitlines():
        if not raw.strip():
            continue
        parts = [p.strip() for p in raw.split(",")]
        if len(parts) != 4:
            continue
        name, util_raw, used_raw, total_raw = parts
        try:
            util = int(util_raw.removesuffix("%").strip())
            used = int(used_raw.removesuffix("MiB").strip())
            total = int(total_raw.removesuffix("MiB").strip())
        except ValueError:
            continue
        try:
            rows.append(
                GpuRow(
                    name=name,
                    utilization_gpu_percent=util,
                    memory_used_mib=used,
                    memory_total_mib=total,
                )
            )
        except Exception:  # noqa: BLE001 — pydantic validation
            # Invalid (negative util, util > 100) — skip; better an
            # empty list than 502 the whole endpoint.
            continue
    return rows


def collect_nvidia_smi(
    *,
    timeout_s: float = DEFAULT_SUBPROCESS_TIMEOUT_S,
    runner: SubprocessRunner = _default_runner,
) -> GpuReport:
    """``nvidia-smi`` snapshot — postmortem-shaped (utilization, VRAM)."""
    if shutil.which("nvidia-smi") is None:
        return GpuReport(error=DiagnosticsBlockError.TOOL_MISSING)

    try:
        completed = runner(
            [
                "nvidia-smi",
                "--query-gpu=name,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except BaseException as exc:
        return GpuReport(error=_classify_subprocess_error(exc))

    if completed.returncode != 0:
        return GpuReport(error=DiagnosticsBlockError.UNKNOWN)

    rows = _parse_nvidia_smi_csv(completed.stdout or "")
    return GpuReport(rows=rows)
