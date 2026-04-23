from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.api.schemas.plugin import PluginKind, PluginListResponse
from src.api.services import plugin_service
from src.reports.plugins.defaults import DEFAULT_REPORT_SECTIONS

router = APIRouter(prefix="/plugins", tags=["plugins"])

_KINDS: tuple[PluginKind, ...] = ("reward", "validation", "evaluation", "reports")


class ReportDefaultsResponse(BaseModel):
    """The built-in report-section order used when the pipeline config
    doesn't declare ``reports.sections``. Surfaced so the frontend can
    pre-fill its reports section with a sensible starting list."""

    sections: list[str]


# NOTE: this route must be declared BEFORE the generic ``/{kind}`` below
# — FastAPI matches routes in declaration order, so swapping them would
# make "/plugins/reports/defaults" try to call ``list_plugins`` with
# ``kind='reports/defaults'``.
@router.get("/reports/defaults", response_model=ReportDefaultsResponse)
def get_report_defaults() -> ReportDefaultsResponse:
    return ReportDefaultsResponse(sections=list(DEFAULT_REPORT_SECTIONS))


@router.get("/{kind}", response_model=PluginListResponse)
def list_plugins(kind: PluginKind) -> PluginListResponse:
    if kind not in _KINDS:
        raise HTTPException(
            status_code=404,
            detail=f"unknown plugin kind {kind!r}; expected one of {list(_KINDS)}",
        )
    return plugin_service.list_plugins(kind)
