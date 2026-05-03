"""Phase 11.E — control-plane heartbeat endpoint.

Background
----------
Phase 11.B introduced :class:`MacHeartbeat` and wired implicit pings
into the WebSocket and REST GET handlers. Those work fine for
runner traffic, but :class:`ModelRetriever` does its heavy work
(SCP-streaming model adapters via ``download_directory``) over a
**separate SSH stream** that bypasses the runner's FastAPI entirely.
After ~60 s of SCP, the implicit heartbeat goes stale and
:class:`PodTerminator` would podStop the pod mid-download.

The fix is an **explicit "process is active" signal** from the Mac
orchestrator: while the orchestrator process is alive, it sends a
heartbeat to ``POST /api/v1/control/heartbeat`` every 30 s. The
endpoint refreshes :class:`MacHeartbeat` with a longer TTL (default
120 s = 2× the recommended ping interval) so a single dropped ping
doesn't immediately stale the heartbeat. Combined with retry logic
in :class:`PodTerminator` (Phase 11.E part 3), this gives the
orchestrator a robust safety net for any in-flight Mac-side work
including but not limited to ``ModelRetriever``.

Auth model
----------
The runner binds ``127.0.0.1:8080`` only — see
``docker/training/entrypoint.sh``. All traffic comes through the
SSH tunnel held open by the Mac orchestrator. We don't add a token
or shared secret on this endpoint for the same reason ``/internal/events``
doesn't have one: anyone who can reach 127.0.0.1:8080 inside the
pod is already trusted.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

from fastapi import APIRouter, Body, Depends
from pydantic import BaseModel, Field

from ryotenkai_pod.runner.api.deps import get_heartbeat

if TYPE_CHECKING:
    from ryotenkai_pod.runner.heartbeat import MacHeartbeat


__all__ = ["ControlHeartbeatRequest", "router"]


router = APIRouter(prefix="/control", tags=["control"])


class ControlHeartbeatRequest(BaseModel):
    """Optional request body for ``POST /control/heartbeat``.

    Mac orchestrators that ping at the recommended 30 s interval can
    omit the body entirely — the runner falls back to the default
    explicit TTL (:attr:`MacHeartbeat.EXPLICIT_HEARTBEAT_TTL_SECONDS`,
    120 s). Operators experimenting with shorter / longer ping
    cadences can override.
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


@router.post("/heartbeat", response_model=ControlHeartbeatResponse)
def post_heartbeat(
    heartbeat: "MacHeartbeat" = Depends(get_heartbeat),
    body: Annotated[ControlHeartbeatRequest | None, Body()] = None,
) -> ControlHeartbeatResponse:
    """Refresh the Mac heartbeat ledger with an explicit ping.

    Called every ~30 s by the Mac orchestrator's
    :class:`ControlPlaneHeartbeat` service for the duration of the
    orchestrator process lifetime. Idempotent — multiple pings
    within the same FastAPI loop iteration just overwrite the
    timestamp.

    Returns ``ok=True`` plus the TTL actually applied. The Mac
    client can use this for a sanity check; otherwise it just
    discards the response.
    """
    # Phase 14.E (V8) — module-level constant import (was re-importing
    # ``MacHeartbeat`` class for a one-off value read pre-14.E).
    from ryotenkai_pod.runner.heartbeat import EXPLICIT_HEARTBEAT_TTL_SECONDS
    explicit_default = EXPLICIT_HEARTBEAT_TTL_SECONDS

    ttl = (
        body.ttl_seconds
        if body is not None and body.ttl_seconds is not None
        else explicit_default
    )
    heartbeat.mark_active(ttl_seconds=ttl)
    return ControlHeartbeatResponse(ok=True, ttl_seconds_applied=ttl)
