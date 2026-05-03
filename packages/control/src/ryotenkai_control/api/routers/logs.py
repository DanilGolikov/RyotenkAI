from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query

from src.api.config import ApiSettings
from src.api.dependencies import get_settings, resolve_run_dir
from src.api.schemas.log import LogChunk, LogFileInfo
from src.api.services import log_service

router = APIRouter(prefix="/runs/{run_id:path}/attempts/{attempt_no}/logs", tags=["logs"])


@router.get("", response_model=LogChunk)
def get_log_chunk(
    attempt_no: int,
    file: str = Query(default="pipeline.log"),
    offset: int = Query(default=0, ge=0),
    max_bytes: int = Query(default=0, ge=0),
    run_dir: Path = Depends(resolve_run_dir),
    settings: ApiSettings = Depends(get_settings),
) -> LogChunk:
    effective_max = max_bytes or settings.max_log_chunk_bytes
    effective_max = min(effective_max, settings.max_log_chunk_bytes)
    try:
        return log_service.read_chunk(run_dir, attempt_no, file, offset=offset, max_bytes=effective_max)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/files", response_model=list[LogFileInfo])
def list_files(
    attempt_no: int,
    run_dir: Path = Depends(resolve_run_dir),
) -> list[LogFileInfo]:
    return log_service.list_log_files(run_dir, attempt_no)
