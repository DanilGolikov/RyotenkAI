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
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from pydantic import ValidationError

from ryotenkai_pod.runner.api.deps import (
    get_bus,
    get_fsm,
    get_plugin_unpacker,
    get_supervisor,
)
from ryotenkai_pod.runner.api.schemas import (
    JobSnapshotResponse,
    JobSpec,
    JobStopAcceptedResponse,
    JobSubmittedResponse,
)
from ryotenkai_pod.runner.plugin_unpacker import PluginUnpackError
from ryotenkai_pod.runner.state import (
    JobState,
)
from ryotenkai_pod.runner.supervisor import SupervisorBusy

if TYPE_CHECKING:
    from ryotenkai_pod.runner.event_bus import EventBus
    from ryotenkai_pod.runner.plugin_unpacker import PluginUnpacker
    from ryotenkai_pod.runner.state import JobLifecycleFSM
    from ryotenkai_pod.runner.supervisor import Supervisor

router = APIRouter(prefix="/jobs", tags=["jobs"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def _snapshot_to_response(
    fsm: JobLifecycleFSM, bus: EventBus,
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


def _get_active_or_404(fsm: JobLifecycleFSM, job_id: str) -> None:
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
    plugins_payload: UploadFile = File(
        ...,
        description=(
            "ZIP archive bundling the community/ reward plugins the "
            "trainer needs. Phase 1 reads + discards the body."
        ),
    ),
    fsm: JobLifecycleFSM = Depends(get_fsm),
    bus: EventBus = Depends(get_bus),
    supervisor: Supervisor = Depends(get_supervisor),
    plugin_unpacker: PluginUnpacker = Depends(get_plugin_unpacker),
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

    # Read + extract the plugins payload BEFORE spawning the trainer:
    # if the unpack fails we still hold ``preparing → failed`` ground
    # without a half-running subprocess, and the trainer never starts
    # against a missing community/. Empty bytes (no reward plugins,
    # e.g. SFT-only) is the no-op path.
    payload_bytes = await plugins_payload.read()
    await plugins_payload.close()
    try:
        unpack_result = plugin_unpacker.unpack(payload_bytes)
    except PluginUnpackError as exc:
        raise HTTPException(
            status_code=422,
            detail={"code": "plugin_unpack_failed", "message": str(exc)},
        ) from exc

    bus.publish(
        "plugins_unpacked",
        {
            "installed": list(unpack_result.installed),
            "skipped": list(unpack_result.skipped),
            "total_bytes": unpack_result.total_bytes,
        },
    )

    # Atomic submit + spawn — the supervisor owns both the FSM
    # ``preparing → running`` transition and the subprocess launch.
    # On spawn failure the FSM is rolled forward to ``failed`` so a
    # future restart never sees a stuck ``preparing``.
    try:
        # ``spec.workdir`` is the absolute pod-side directory to spawn
        # the trainer in — typically ``/workspace/runs/<run_id>``.
        # When None we fall through to inheriting uvicorn's cwd (legacy
        # behaviour pre-Phase 6.7); Mac clients always set this since
        # the inherited ``/root`` makes relative paths in the trainer's
        # config / argv misresolve.
        workdir = Path(spec.workdir) if spec.workdir else None
        await supervisor.submit_and_spawn(
            spec.job_id, spec.command,
            env=spec.env or None,
            workdir=workdir,
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
    request: Request,
    fsm: JobLifecycleFSM = Depends(get_fsm),
    bus: EventBus = Depends(get_bus),
) -> JobSnapshotResponse:
    _get_active_or_404(fsm, job_id)
    # Phase 14.E (V3) — heartbeat marking via centralized helper
    # (was inlined ``getattr(...) + if-not-None`` block). The
    # ModelRetriever polls this endpoint while it SCPs adapters
    # off the pod; each successful GET refreshes the heartbeat so
    # PodTerminator's natural-completion grace window stretches
    # to cover the whole download.
    from ryotenkai_pod.runner.api._activity import mark_heartbeat_if_present
    mark_heartbeat_if_present(request.app.state)
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
    fsm: JobLifecycleFSM = Depends(get_fsm),
    supervisor: Supervisor = Depends(get_supervisor),
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
