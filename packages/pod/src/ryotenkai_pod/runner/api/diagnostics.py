"""``GET /api/v1/diagnostics`` — kernel + GPU postmortem snapshot.

Replaces the SSH probes the Mac orchestrator used to issue from
``training_monitor._postmortem_diagnostics``. The endpoint composes
the requested blocks (``?include=dmesg,gpu,kernel_signals``) into a
single :class:`DiagnosticsResponse`.

Failure shape (RP2):

* Per-block failures (``dmesg`` PERMISSION_DENIED on RunPod, missing
  ``nvidia-smi`` on a CPU pod, etc.) surface inside the response —
  HTTP stays 200 so the other blocks are still consumable.
* ``DIAGNOSTIC_INVALID_INCLUDE`` (422) — unknown enum value.
* ``DIAGNOSTIC_FAILED`` (502) is reserved for the catastrophic case
  where all collectors raised an exception **outside** their normal
  failure-mode envelopes; in practice this is unreachable under
  normal use.
"""

from __future__ import annotations

from collections.abc import Iterable

from fastapi import APIRouter, Query

from ryotenkai_shared.errors import DiagnosticInvalidIncludeError
from ryotenkai_pod.runner.diagnostics import (
    collect_dmesg,
    collect_kernel_signals,
    collect_nvidia_smi,
)
from ryotenkai_shared.contracts.runner_api.diagnostics import (
    DiagnosticsInclude,
    DiagnosticsResponse,
)

router = APIRouter(prefix="/diagnostics", tags=["diagnostics"])


def _normalise_include(raw: list[str] | None) -> set[DiagnosticsInclude]:
    """Convert a raw ``?include=`` list into the StrEnum set.

    Callers can pass ``include=dmesg,gpu`` (single comma-list) OR
    repeated ``include=dmesg&include=gpu`` — FastAPI surfaces both as
    ``list[str]``. Empty / missing means "all blocks" — this matches
    operator expectations (the SSH probe always emitted everything).
    """
    if not raw:
        return set(DiagnosticsInclude)

    flattened: list[str] = []
    for item in raw:
        flattened.extend(p.strip() for p in item.split(",") if p.strip())

    chosen: set[DiagnosticsInclude] = set()
    for item in flattened:
        try:
            chosen.add(DiagnosticsInclude(item))
        except ValueError:
            raise DiagnosticInvalidIncludeError(
                detail=(
                    f"unknown include={item!r}; valid: "
                    f"{sorted(v.value for v in DiagnosticsInclude)}"
                ),
                context={"include": item},
            ) from None
    return chosen or set(DiagnosticsInclude)


@router.get("", response_model=DiagnosticsResponse, response_model_exclude_none=True)
def get_diagnostics(
    include: list[str] | None = Query(  # noqa: B008 — FastAPI idiom
        default=None,
        description=(
            "Comma-separated or repeated list of blocks to include. "
            "Valid: dmesg, gpu, kernel_signals. Empty ⇒ all."
        ),
    ),
) -> DiagnosticsResponse:
    """Compose the requested blocks into a single envelope.

    Per-block failures (PERMISSION_DENIED for ``dmesg`` on CAP_SYSLOG-
    locked kernels, ``tool_missing`` for ``nvidia-smi``) appear in
    the response with their typed sentinel; HTTP stays 200.
    """
    chosen = _normalise_include(include)
    response = DiagnosticsResponse()
    if DiagnosticsInclude.DMESG in chosen:
        response.dmesg = collect_dmesg()
    if DiagnosticsInclude.GPU in chosen:
        response.gpu = collect_nvidia_smi()
    if DiagnosticsInclude.KERNEL_SIGNALS in chosen:
        response.kernel_signals = collect_kernel_signals()
    return response


__all__ = ["router"]
