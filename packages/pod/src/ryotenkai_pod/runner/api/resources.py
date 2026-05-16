"""``GET /api/v1/resources`` — instant GPU/CPU/RAM snapshot.

Reuses :func:`default_health_snapshot` already used by
:class:`HealthReporter` so the WS event ``health_snapshot`` and the
HTTP ``/resources`` endpoint return the SAME shape from the SAME
provider — no field drift between the two transports.

Why HTTP in addition to the existing WS event: the WS stream has
a chicken-and-egg problem with the status line. ``health_snapshot``
events are published every 30 s; if the trainer dies in the first
30 s of a run, the Mac client never sees a single snapshot and the
status line stays empty. The HTTP endpoint lets the Mac client
poll on its own cadence (15 s) and surface ``running | …`` from
second 15 onwards regardless of trainer state.
"""

from __future__ import annotations

from fastapi import APIRouter

from ryotenkai_shared.api.error_handlers import APIError
from ryotenkai_pod.runner.health_reporter import default_health_snapshot
from ryotenkai_shared.contracts.problem_details import ErrorCode
from ryotenkai_shared.contracts.runner_api.resources import ResourceSnapshot

router = APIRouter(prefix="/resources", tags=["resources"])


@router.get("", response_model=ResourceSnapshot)
async def get_resources() -> ResourceSnapshot:
    """Compose one GPU/CPU/RAM snapshot at call time.

    No internal caching — each call hits ``nvidia-smi`` + psutil
    fresh. Pod is loopback-only and traffic is bounded by the Mac
    client's poll cadence (15 s by design).

    Failure handling: ``default_health_snapshot()`` already returns
    ``None`` for fields whose source is unavailable (RP2 — match
    diagnostics endpoint policy). We surface that as the typed
    ``ResourceSnapshot`` (200) rather than 502 so the Mac client
    can render ``--`` for the missing field. A *catastrophic*
    failure (snapshot provider raises) maps to ``RESOURCES_UNAVAILABLE``
    (502).
    """
    try:
        raw = await default_health_snapshot()
    except Exception as exc:  # noqa: BLE001 — defensive
        raise APIError(
            ErrorCode.RESOURCES_UNAVAILABLE,
            status=502,
            detail=f"snapshot provider failed: {type(exc).__name__}: {exc}",
        ) from exc
    return ResourceSnapshot.model_validate(raw)


__all__ = ["router"]
