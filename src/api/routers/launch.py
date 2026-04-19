from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.concurrency import run_in_threadpool

from src.api.dependencies import resolve_run_dir
from src.api.schemas.launch import (
    InterruptResponse,
    LaunchRequestSchema,
    LaunchResponse,
    RestartPointsResponse,
)
from src.api.services import launch_service

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
