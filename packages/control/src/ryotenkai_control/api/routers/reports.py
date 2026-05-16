from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends

from ryotenkai_control.api.dependencies import resolve_run_dir
from ryotenkai_control.api.schemas.report import ReportResponse
from ryotenkai_control.api.services import report_service
from ryotenkai_shared.errors import JobSpecInvalidError, ReportGenerationFailedError

router = APIRouter(prefix="/runs/{run_id:path}", tags=["reports"])


@router.get("/report", response_model=ReportResponse)
def get_report(
    regenerate: bool = False,
    run_dir: Path = Depends(resolve_run_dir),
) -> ReportResponse:
    try:
        return report_service.get_or_generate_report(run_dir, regenerate=regenerate)
    except ValueError as exc:
        raise JobSpecInvalidError(detail=str(exc), cause=exc) from exc
    except Exception as exc:  # noqa: BLE001 — MLflow errors bubble up as 503
        raise ReportGenerationFailedError(
            detail=f"report generation failed: {exc}",
            cause=exc,
        ) from exc
