"""Control-plane heartbeat DTOs (POST /api/v1/control/heartbeat).

Background — see :mod:`ryotenkai_pod.runner.api.control` for the full
context. While the orchestrator process is alive, it sends a
heartbeat to ``POST /api/v1/control/heartbeat`` every 30 s. The
endpoint refreshes :class:`MacHeartbeat` with a longer TTL (default
120 s = 2× the recommended ping interval) so a single dropped ping
doesn't immediately stale the heartbeat.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ControlHeartbeatRequest(BaseModel):
    """Optional request body for ``POST /control/heartbeat``.

    Mac orchestrators that ping at the recommended 30 s interval can
    omit the body entirely — the runner falls back to the default
    explicit TTL (120 s).
    """

    ttl_seconds: float | None = Field(
        default=None,
        gt=0.0,
        description=(
            "How long the heartbeat remains 'fresh' from this beat. "
            "None ⇒ runner uses 120 s (Phase 11.E default, 2× the "
            "recommended client ping interval)."
        ),
    )


class ControlHeartbeatResponse(BaseModel):
    """Acknowledgement returned to the Mac orchestrator.

    Exposes the actual TTL applied + the current heartbeat alive
    status so the client can sanity-check the round-trip.
    """

    ok: bool
    ttl_seconds_applied: float


__all__ = [
    "ControlHeartbeatRequest",
    "ControlHeartbeatResponse",
]
