"""REST endpoints for job lifecycle.

Phase 1 wire contract (subject to extension in Phase 2+ as the
supervisor and plugin unpacker land):

    POST   /api/v1/jobs                multipart submit, → 202
    GET    /api/v1/jobs/{job_id}       current snapshot
    POST   /api/v1/jobs/{job_id}/stop  request graceful stop, → 202

The multipart submit accepts two parts:
- ``job_spec`` (string, application/json) — :class:`JobSpec`
- ``plugins_payload`` (file, application/zip) — bundled community/
  reward plugins (:class:`PluginPacker` output, Phase 6). Phase 1
  reads + discards the body so wire compatibility is established
  early; the actual unpack lands when :class:`PluginUnpacker`
  arrives in Phase 6.

Failure modes:
- Malformed ``job_spec`` JSON → 422 (Pydantic validation).
- Submitting while a non-terminal job is in progress → 409 with
  the FSM's ``InvalidTransitionError`` rendered as JSON detail.
- ``stop`` called from a non-running state → 409.
- ``GET`` for a job_id that does not match the current FSM snapshot
  → 404.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import ValidationError

from src.runner.api.deps import get_bus, get_fsm, get_supervisor
from src.runner.api.schemas import (
    JobSnapshotResponse,
    JobSpec,
    JobStopAcceptedResponse,
    JobSubmittedResponse,
)
from src.runner.state import (
    InvalidTransitionError,
    JobState,
)
from src.runner.supervisor import SupervisorBusy

if TYPE_CHECKING:
    from src.runner.event_bus import EventBus
    from src.runner.state import JobLifecycleFSM
    from src.runner.supervisor import Supervisor

router = APIRouter(prefix="/jobs", tags=["jobs"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def _snapshot_to_response(
    fsm: "JobLifecycleFSM", bus: "EventBus",
) -> JobSnapshotResponse:
    """Render the FSM's current snapshot + the bus' next-offset cursor."""
    snap = fsm.current()
    if snap is None:
        # Caller is responsible for guarding — if we reach this it's
        # a server bug, not a client error.
        raise RuntimeError("snapshot requested with no active FSM state")
    return JobSnapshotResponse(
        job_id=snap.job_id,
        state=snap.state.value,
        sequence=snap.sequence,
        started_at=snap.started_at,
        updated_at=snap.updated_at,
        message=snap.message,
        # Cursor of the latest event the client could have observed.
        # Subtract 1 from ``next_offset`` since the bus' counter is
        # the slot of the *next* event to be assigned, not the last
        # one issued.
        last_event_offset=max(0, bus.next_offset - 1),
    )


def _get_active_or_404(fsm: "JobLifecycleFSM", job_id: str) -> None:
    """Raise 404 if the FSM is empty or holds a different job."""
    snap = fsm.current()
    if snap is None or snap.job_id != job_id:
        raise HTTPException(
            status_code=404,
            detail={"code": "job_not_found", "job_id": job_id},
        )


# ---------------------------------------------------------------------------
# POST /jobs — multipart submit
# ---------------------------------------------------------------------------


@router.post(
    "",
    status_code=202,
    response_model=JobSubmittedResponse,
    summary="Submit a new training job",
)
async def submit_job(
    job_spec: str = Form(
        ...,
        description="JSON-encoded :class:`JobSpec` payload.",
    ),
    plugins_payload: UploadFile = File(  # noqa: B008 — FastAPI dependency
        ...,
        description=(
            "ZIP archive bundling the community/ reward plugins the "
            "trainer needs. Phase 1 reads + discards the body."
        ),
    ),
    fsm: "JobLifecycleFSM" = Depends(get_fsm),
    bus: "EventBus" = Depends(get_bus),
    supervisor: "Supervisor" = Depends(get_supervisor),
) -> JobSubmittedResponse:
    # Validate the JSON part — extra="forbid" surfaces typos at 422
    # rather than silently dropping them.
    try:
        spec = JobSpec.model_validate(json.loads(job_spec))
    except (ValidationError, json.JSONDecodeError) as exc:
        raise HTTPException(
            status_code=422,
            detail={"code": "invalid_job_spec", "errors": str(exc)},
        ) from exc

    # Drain the upload — Phase 6's PluginUnpacker will replace this
    # with the real extract-into-/workspace/community/ flow. Reading
    # the body now (instead of ignoring it) keeps the upstream client
    # happy: ASGI doesn't always discard the unread body, depending on
    # the server. Discard explicitly.
    _ = await plugins_payload.read()
    await plugins_payload.close()

    # Atomic submit + spawn — the supervisor owns both the FSM
    # ``preparing → running`` transition and the subprocess launch.
    # On spawn failure the FSM is rolled forward to ``failed`` so a
    # future restart never sees a stuck ``preparing``.
    try:
        await supervisor.submit_and_spawn(
            spec.job_id, spec.command, env=spec.env or None,
        )
    except SupervisorBusy as exc:
        # Either a trainer is still running, or the FSM holds an
        # active non-terminal job.
        current = fsm.current()
        raise HTTPException(
            status_code=409,
            detail={
                "code": "job_in_progress",
                "current_state": current.state.value if current else "unknown",
                "message": str(exc),
            },
        ) from exc
    except (FileNotFoundError, OSError) as exc:
        # The command can't be exec()'d. ``submit_and_spawn`` already
        # rolled the FSM to ``failed`` and emitted ``spawn_failed``.
        raise HTTPException(
            status_code=422,
            detail={
                "code": "spawn_failed",
                "message": str(exc),
            },
        ) from exc

    snap = fsm.current()
    if snap is None:  # pragma: no cover — defensive; submit_and_spawn must have set it
        raise RuntimeError("FSM has no snapshot after successful submit_and_spawn")

    return JobSubmittedResponse(
        job_id=snap.job_id,
        sequence=snap.sequence,
        # The bus has at least two events at this point (job_submitted
        # at offset 0, trainer_spawned at offset 1). We point the
        # client at the first so it can replay both with since=0.
        offset=0,
    )


# ---------------------------------------------------------------------------
# GET /jobs/{job_id}
# ---------------------------------------------------------------------------


@router.get(
    "/{job_id}",
    response_model=JobSnapshotResponse,
    summary="Get current snapshot of a job",
)
def get_job(
    job_id: str,
    fsm: "JobLifecycleFSM" = Depends(get_fsm),
    bus: "EventBus" = Depends(get_bus),
) -> JobSnapshotResponse:
    _get_active_or_404(fsm, job_id)
    return _snapshot_to_response(fsm, bus)


# ---------------------------------------------------------------------------
# POST /jobs/{job_id}/stop
# ---------------------------------------------------------------------------


@router.post(
    "/{job_id}/stop",
    status_code=202,
    response_model=JobStopAcceptedResponse,
    summary="Request graceful stop of a job",
)
async def stop_job(
    job_id: str,
    fsm: "JobLifecycleFSM" = Depends(get_fsm),
    supervisor: "Supervisor" = Depends(get_supervisor),
) -> JobStopAcceptedResponse:
    _get_active_or_404(fsm, job_id)

    # Pre-check: only ``running`` accepts a stop. We could let the
    # supervisor guard this (it does — InvalidTransitionError lands
    # in the no-op branch), but surfacing 409 here is cleaner UX —
    # the client gets the actual current_state instead of a generic
    # 202 that secretly did nothing.
    snap = fsm.current()
    if snap is None or snap.state != JobState.RUNNING:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "stop_not_allowed",
                "current_state": snap.state.value if snap is not None else "unknown",
                "message": "stop only valid from running state",
            },
        )

    # ``request_stop`` is split: the FSM transition + SIGTERM happen
    # synchronously before it returns; the SIGKILL escalation runs as
    # a background task spawned inside the supervisor. So awaiting
    # the call costs ~one event-loop tick — the response is still
    # essentially 202-and-go.
    await supervisor.request_stop()

    snap = fsm.current()
    assert snap is not None  # _get_active_or_404 above guards this
    return JobStopAcceptedResponse(
        job_id=snap.job_id,
        state=snap.state.value,
        sequence=snap.sequence,
    )
