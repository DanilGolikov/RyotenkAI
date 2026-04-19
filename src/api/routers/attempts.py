from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends

from src.api.dependencies import resolve_run_dir
from src.api.schemas.attempt import AttemptDetail, StagesResponse
from src.api.services import run_service

router = APIRouter(prefix="/runs/{run_id:path}/attempts", tags=["attempts"])


@router.get("/{attempt_no}", response_model=AttemptDetail)
def get_attempt(
    attempt_no: int,
    run_dir: Path = Depends(resolve_run_dir),
) -> AttemptDetail:
    return run_service.get_attempt_detail(run_dir, attempt_no)


@router.get("/{attempt_no}/stages", response_model=StagesResponse)
def get_stages(
    attempt_no: int,
    run_dir: Path = Depends(resolve_run_dir),
) -> StagesResponse:
    return run_service.get_attempt_stages(run_dir, attempt_no)
