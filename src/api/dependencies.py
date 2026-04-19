from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from fastapi import Depends, HTTPException

from src.api.config import ApiSettings
from src.pipeline.state import PipelineStateStore


@lru_cache(maxsize=1)
def get_settings() -> ApiSettings:
    return ApiSettings()


def get_runs_dir(settings: ApiSettings = Depends(get_settings)) -> Path:
    runs_dir = settings.runs_dir_resolved
    if not runs_dir.exists():
        runs_dir.mkdir(parents=True, exist_ok=True)
    return runs_dir


def resolve_run_dir(run_id: str, runs_dir: Path = Depends(get_runs_dir)) -> Path:
    """Resolve a run_id (directory name, possibly nested under a subgroup) to an
    absolute Path, rejecting path traversal.
    """
    if not run_id or ".." in run_id.replace("\\", "/").split("/"):
        raise HTTPException(status_code=400, detail="invalid_run_id")
    run_dir = (runs_dir / run_id).resolve()
    # Ensure path stays inside runs_dir
    try:
        run_dir.relative_to(runs_dir)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="run_id_outside_runs_dir") from exc
    if not run_dir.exists() or not run_dir.is_dir():
        raise HTTPException(status_code=404, detail="run_not_found")
    return run_dir


def get_state_store(run_dir: Path = Depends(resolve_run_dir)) -> PipelineStateStore:
    return PipelineStateStore(run_dir)
