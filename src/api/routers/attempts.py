from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import APIRouter, Depends, HTTPException, Request, Response

from src.api.dependencies import resolve_run_dir
from src.api.http_cache import apply_cache_headers, is_fresh
from src.api.schemas.attempt import AttemptDetail, StagesResponse
from src.api.services import run_service

if TYPE_CHECKING:
    from pathlib import Path

router = APIRouter(prefix="/runs/{run_id:path}/attempts", tags=["attempts"])


@router.get("/{attempt_no}", response_model=AttemptDetail)
def get_attempt(
    request: Request,
    response: Response,
    attempt_no: int,
    run_dir: Path = Depends(resolve_run_dir),
) -> AttemptDetail | Response:
    try:
        detail, snapshot = run_service.get_attempt_detail(run_dir, attempt_no)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if is_fresh(request, snapshot.mtime_ns):
        not_modified = Response(status_code=304)
        apply_cache_headers(not_modified, snapshot.mtime_ns)
        return not_modified
    apply_cache_headers(response, snapshot.mtime_ns)
    return detail


@router.get("/{attempt_no}/stages", response_model=StagesResponse)
def get_stages(
    request: Request,
    response: Response,
    attempt_no: int,
    run_dir: Path = Depends(resolve_run_dir),
) -> StagesResponse | Response:
    try:
        stages, snapshot = run_service.get_attempt_stages(run_dir, attempt_no)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if is_fresh(request, snapshot.mtime_ns):
        not_modified = Response(status_code=304)
        apply_cache_headers(not_modified, snapshot.mtime_ns)
        return not_modified
    apply_cache_headers(response, snapshot.mtime_ns)
    return stages
