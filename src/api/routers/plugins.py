from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ValidationError

from src.api.schemas.plugin import (
    PluginKind,
    PluginListResponse,
    PreflightRequest,
    PreflightResponse,
)
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


@router.post("/preflight", response_model=PreflightResponse)
def preflight(request: PreflightRequest) -> PreflightResponse:
    """Check that every plugin in ``request.config`` has its non-optional
    ``[[required_env]]`` keys set in process env / project env.

    The Launch modal calls this before enabling the launch button so a
    user without the right credentials sees a "set up before launch"
    chip with a deep link to the right Settings tab — instead of a
    pipeline that runs for minutes and crashes mid-stage on the missing
    key.

    Returns ``ok=true`` plus an empty ``missing`` list when the launch
    is safe; ``ok=false`` with structured rows otherwise. A malformed
    config payload surfaces as a 422 with the full per-field error
    list (Pydantic does the work).
    """
    try:
        return plugin_service.preflight(request.config, request.project_env)
    except ValidationError as exc:
        # FastAPI's default ValidationError handler is for *request*
        # validation; here we're validating an arbitrary payload that
        # happens to be a config file. Surface the full error list so
        # the front-end can highlight the offending fields.
        raise HTTPException(status_code=422, detail=exc.errors()) from exc


@router.get("/{kind}", response_model=PluginListResponse)
def list_plugins(kind: PluginKind) -> PluginListResponse:
    if kind not in _KINDS:
        raise HTTPException(
            status_code=404,
            detail=f"unknown plugin kind {kind!r}; expected one of {list(_KINDS)}",
        )
    return plugin_service.list_plugins(kind)
