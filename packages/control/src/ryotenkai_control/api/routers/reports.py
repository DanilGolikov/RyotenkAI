from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException

from src.api.dependencies import resolve_run_dir
from src.api.schemas.report import ReportResponse
from src.api.services import report_service

router = APIRouter(prefix="/runs/{run_id:path}", tags=["reports"])


@router.get("/report", response_model=ReportResponse)
def get_report(
    regenerate: bool = False,
    run_dir: Path = Depends(resolve_run_dir),
) -> ReportResponse:
    try:
        return report_service.get_or_generate_report(run_dir, regenerate=regenerate)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 — MLflow errors bubble up as 503
        raise HTTPException(status_code=503, detail=f"report generation failed: {exc}") from exc
