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

# Phase 0 (transport-unification-v2): canonical DTO definitions live
# in ``ryotenkai_shared.contracts.runner_api.control``. This module
# re-exports them so existing import sites in this package keep
# working; PR-0b migrates them to the canonical location.
from ryotenkai_shared.contracts.runner_api.control import (
    ControlHeartbeatRequest,
    ControlHeartbeatResponse,
)

from ryotenkai_pod.runner.api.deps import get_heartbeat

if TYPE_CHECKING:
    from ryotenkai_pod.runner.heartbeat import MacHeartbeat


__all__ = ["ControlHeartbeatRequest", "ControlHeartbeatResponse", "router"]


router = APIRouter(prefix="/control", tags=["control"])


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
