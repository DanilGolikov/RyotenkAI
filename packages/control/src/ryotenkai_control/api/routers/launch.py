from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.concurrency import run_in_threadpool

from ryotenkai_control.api.dependencies import resolve_run_dir
from ryotenkai_control.api.schemas.launch import (
    InterruptResponse,
    LaunchRequestSchema,
    LaunchResponse,
    RestartPointsResponse,
)
from ryotenkai_control.api.services import launch_service

router = APIRouter(prefix="/runs/{run_id:path}", tags=["launch"])


@router.get("/restart-points", response_model=RestartPointsResponse)
def restart_points(
    config_path: str | None = Query(default=None),
    run_dir: Path = Depends(resolve_run_dir),
) -> RestartPointsResponse:
    resolved_config = Path(config_path).expanduser().resolve() if config_path else None
    try:
        return launch_service.list_restart_points(run_dir, resolved_config)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/default-launch-mode")
def default_launch_mode(run_dir: Path = Depends(resolve_run_dir)) -> dict[str, str]:
    return {"mode": launch_service.default_launch_mode(run_dir)}


@router.post("/launch", response_model=LaunchResponse, status_code=202)
async def launch(
    body: LaunchRequestSchema,
    run_dir: Path = Depends(resolve_run_dir),
) -> LaunchResponse:
    try:
        return await run_in_threadpool(launch_service.launch, run_dir, body)
    except launch_service.LaunchAlreadyRunningError as exc:
        raise HTTPException(
            status_code=409,
            detail={"code": "run_already_running", "pid": exc.pid, "message": str(exc)},
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/interrupt", response_model=InterruptResponse)
def interrupt(run_dir: Path = Depends(resolve_run_dir)) -> InterruptResponse:
    return launch_service.interrupt(run_dir)


@router.post("/resume-pod", status_code=200)
async def resume_pod(
    run_dir: Path = Depends(resolve_run_dir),
) -> dict[str, str | bool]:
    """Phase 11.C-2 — wake a sleeping pod for the given run.

    Web UI's "Resume" button calls this BEFORE re-launching the
    pipeline. Idempotent — pod already RUNNING returns success
    without any GraphQL round-trip on the wake side.

    Response shape:

    * ``availability`` — final :class:`PodAvailability` enum value.
    * ``ok`` — true iff the pod is in ``RUNNING`` state after this
      call (either was already running, or wake succeeded).
    * ``message`` — human-readable detail for UI display.

    Status codes:

    * ``200`` — service ran cleanly. Inspect ``ok`` field for the
      verdict; failure modes (capacity exhausted, GONE pod) come
      back with ``200 ok=false`` so the UI can render a meaningful
      error rather than a generic 5xx.
    """
    response = await run_in_threadpool(
        launch_service.resume_pod_for_run, run_dir,
    )
    return response.to_dict()
