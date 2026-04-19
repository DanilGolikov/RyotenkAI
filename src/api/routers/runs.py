from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException

from src.api.dependencies import get_runs_dir, resolve_run_dir
from src.api.schemas.delete import DeleteResultSchema
from src.api.schemas.run import CreateRunRequest, RunDetail, RunSummary, RunsListResponse
from src.api.services import delete_service, run_service

router = APIRouter(prefix="/runs", tags=["runs"])


@router.get("", response_model=RunsListResponse)
def list_runs(runs_dir: Path = Depends(get_runs_dir)) -> RunsListResponse:
    return run_service.list_runs(runs_dir)


@router.post("", response_model=RunSummary, status_code=201)
def create_run(body: CreateRunRequest, runs_dir: Path = Depends(get_runs_dir)) -> RunSummary:
    return run_service.create_empty_run(runs_dir, body.run_id, body.subgroup)


@router.get("/{run_id:path}", response_model=RunDetail)
def get_run(
    run_dir: Path = Depends(resolve_run_dir),
    runs_dir: Path = Depends(get_runs_dir),
) -> RunDetail:
    return run_service.get_run_detail(run_dir, runs_dir)


@router.delete("/{run_id:path}", response_model=DeleteResultSchema)
def delete_run(
    run_dir: Path = Depends(resolve_run_dir),
    mode: str = "local_and_mlflow",
) -> DeleteResultSchema:
    try:
        return delete_service.delete_run(run_dir, mode)
    except delete_service.RunIsActiveError as exc:
        raise HTTPException(status_code=409, detail={"code": "run_active", "pid": exc.pid, "message": str(exc)}) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
