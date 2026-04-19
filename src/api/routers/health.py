from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends

from src.api.dependencies import get_runs_dir
from src.api.schemas.health import HealthStatus

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthStatus)
def health(runs_dir: Path = Depends(get_runs_dir)) -> HealthStatus:
    readable = runs_dir.exists() and runs_dir.is_dir()
    return HealthStatus(
        status="ok" if readable else "degraded",
        runs_dir=str(runs_dir),
        runs_dir_readable=readable,
    )
