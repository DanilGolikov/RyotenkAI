from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import APIRouter, Depends, Request, Response

from ryotenkai_control.api.dependencies import get_runs_dir, resolve_run_dir
from ryotenkai_control.api.http_cache import apply_cache_headers, is_fresh
from ryotenkai_control.api.schemas.delete import DeleteResultSchema
from ryotenkai_control.api.schemas.run import CreateRunRequest, RunDetail, RunsListResponse, RunSummary
from ryotenkai_control.api.services import delete_service, run_service

if TYPE_CHECKING:
    from pathlib import Path

router = APIRouter(prefix="/runs", tags=["runs"])


@router.get("", response_model=RunsListResponse)
def list_runs(runs_dir: Path = Depends(get_runs_dir)) -> RunsListResponse:
    return run_service.list_runs(runs_dir)


@router.post("", response_model=RunSummary, status_code=201)
def create_run(body: CreateRunRequest, runs_dir: Path = Depends(get_runs_dir)) -> RunSummary:
    # Phase C: ``run_service.create_empty_run`` raises ``ValueError``
    # on duplicate / invalid run id. Convert to the typed
    # :class:`JobSpecInvalidError` (422, ``JOB_SPEC_INVALID``) so the
    # response carries a stable code via the shared problem+json
    # contract.
    from ryotenkai_shared.errors import JobSpecInvalidError

    try:
        return run_service.create_empty_run(runs_dir, body.run_id, body.subgroup)
    except ValueError as exc:
        raise JobSpecInvalidError(str(exc)) from exc


@router.get("/{run_id:path}", response_model=RunDetail)
def get_run(
    request: Request,
    response: Response,
    run_dir: Path = Depends(resolve_run_dir),
    runs_dir: Path = Depends(get_runs_dir),
) -> RunDetail | Response:
    detail, snapshot = run_service.get_run_detail(run_dir, runs_dir)
    if is_fresh(request, snapshot.mtime_ns):
        # 304 has no body per RFC 7232 §4.1; attach validators so the client
        # refreshes them for the next roundtrip.
        not_modified = Response(status_code=304)
        apply_cache_headers(not_modified, snapshot.mtime_ns)
        return not_modified
    apply_cache_headers(response, snapshot.mtime_ns)
    return detail


@router.delete("/{run_id:path}", response_model=DeleteResultSchema)
def delete_run(
    run_dir: Path = Depends(resolve_run_dir),
    mode: str = "local_and_mlflow",
) -> DeleteResultSchema:
    # Phase C: ``RunIsActiveError`` is a typed ``DomainError`` (409,
    # ``RUN_IS_ACTIVE``); it flows through the shared exception handler
    # without an ad-hoc adapter here. Bare ``ValueError`` becomes a
    # typed ``JobSpecInvalidError`` so it carries a stable error code.
    from ryotenkai_shared.errors import JobSpecInvalidError

    try:
        return delete_service.delete_run(run_dir, mode)
    except ValueError as exc:
        raise JobSpecInvalidError(str(exc), context={"pid_field": "mode"}) from exc
