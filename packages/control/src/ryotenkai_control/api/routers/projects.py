from __future__ import annotations

from fastapi import APIRouter, Depends

from ryotenkai_control.api.dependencies import get_project_registry
from ryotenkai_control.api.schemas.config_validate import ConfigValidationResult
from ryotenkai_control.api.schemas.project import (
    ConfigResponse,
    ConfigVersionDetail,
    ConfigVersionsResponse,
    CreateProjectRequest,
    ProjectDetail,
    ProjectEnvRequest,
    ProjectEnvResponse,
    ProjectRunEntry,
    ProjectRunsResponse,
    ProjectSummary,
    SaveConfigRequest,
    SaveConfigResponse,
    ToggleFavoriteRequest,
    ToggleFavoriteResponse,
    UpdateProjectDescriptionRequest,
)
from ryotenkai_control.api.services import project_service
from ryotenkai_control.api.services.project_service import ProjectServiceError
from ryotenkai_control.workspace.projects import ProjectRegistry
from ryotenkai_shared.errors import (
    ConfigInvalidError,
    ProjectNotFoundError,
)

router = APIRouter(prefix="/projects", tags=["projects"])


def _raise_400(exc: ProjectServiceError) -> None:
    raise ConfigInvalidError(detail=str(exc), cause=exc)


@router.get("", response_model=list[ProjectSummary])
def list_projects(registry: ProjectRegistry = Depends(get_project_registry)) -> list[ProjectSummary]:
    return project_service.list_summaries(registry)


@router.post("", response_model=ProjectSummary, status_code=201)
def create_project(
    body: CreateProjectRequest,
    registry: ProjectRegistry = Depends(get_project_registry),
) -> ProjectSummary:
    try:
        return project_service.create_project(
            registry,
            name=body.name,
            project_id=body.id,
            path=body.path,
            description=body.description,
        )
    except ProjectServiceError as exc:
        raise ConfigInvalidError(detail=str(exc), cause=exc) from exc


@router.get("/{project_id}", response_model=ProjectDetail)
def get_project(
    project_id: str,
    registry: ProjectRegistry = Depends(get_project_registry),
) -> ProjectDetail:
    try:
        return project_service.get_detail(registry, project_id)
    except ProjectServiceError as exc:
        raise ProjectNotFoundError(detail=str(exc), cause=exc) from exc


@router.put("/{project_id}/description", response_model=ProjectDetail)
def update_description(
    project_id: str,
    body: UpdateProjectDescriptionRequest,
    registry: ProjectRegistry = Depends(get_project_registry),
) -> ProjectDetail:
    try:
        return project_service.update_description(registry, project_id, body.description)
    except ProjectServiceError as exc:
        raise ProjectNotFoundError(detail=str(exc), cause=exc) from exc


@router.delete("/{project_id}", status_code=204)
def delete_project(
    project_id: str,
    delete_files: bool = True,
    registry: ProjectRegistry = Depends(get_project_registry),
) -> None:
    """Unregister a project. By default also removes the on-disk
    workspace — pass ``?delete_files=false`` to keep the directory."""
    try:
        removed = project_service.unregister(
            registry, project_id, delete_files=delete_files
        )
    except ProjectServiceError as exc:
        raise ConfigInvalidError(detail=str(exc), cause=exc) from exc
    if not removed:
        raise ProjectNotFoundError(
            detail=f"project {project_id!r} not registered",
            context={"project_id": project_id},
        )


@router.get("/{project_id}/config", response_model=ConfigResponse)
def get_config(
    project_id: str,
    registry: ProjectRegistry = Depends(get_project_registry),
) -> ConfigResponse:
    try:
        return project_service.get_config(registry, project_id)
    except ProjectServiceError as exc:
        raise ProjectNotFoundError(detail=str(exc), cause=exc) from exc


@router.put("/{project_id}/config", response_model=SaveConfigResponse)
def save_config(
    project_id: str,
    body: SaveConfigRequest,
    registry: ProjectRegistry = Depends(get_project_registry),
) -> SaveConfigResponse:
    try:
        snapshot = project_service.save_config(registry, project_id, body.yaml)
    except ProjectServiceError as exc:
        raise ProjectNotFoundError(detail=str(exc), cause=exc) from exc
    return SaveConfigResponse(ok=True, snapshot_filename=snapshot)


@router.get("/{project_id}/env", response_model=ProjectEnvResponse)
def get_project_env(
    project_id: str,
    registry: ProjectRegistry = Depends(get_project_registry),
) -> ProjectEnvResponse:
    try:
        return ProjectEnvResponse(env=project_service.read_env(registry, project_id))
    except ProjectServiceError as exc:
        raise ProjectNotFoundError(detail=str(exc), cause=exc) from exc


@router.put("/{project_id}/env", response_model=ProjectEnvResponse)
def save_project_env(
    project_id: str,
    body: ProjectEnvRequest,
    registry: ProjectRegistry = Depends(get_project_registry),
) -> ProjectEnvResponse:
    try:
        return ProjectEnvResponse(
            env=project_service.write_env(registry, project_id, body.env),
        )
    except ProjectServiceError as exc:
        raise ProjectNotFoundError(detail=str(exc), cause=exc) from exc


@router.post("/{project_id}/config/validate", response_model=ConfigValidationResult)
def validate_config(
    project_id: str,
    body: SaveConfigRequest,
    registry: ProjectRegistry = Depends(get_project_registry),
) -> ConfigValidationResult:
    try:
        return project_service.validate_yaml(registry, project_id, body.yaml)
    except ProjectServiceError as exc:
        raise ProjectNotFoundError(detail=str(exc), cause=exc) from exc


@router.get("/{project_id}/config/versions", response_model=ConfigVersionsResponse)
def list_versions(
    project_id: str,
    registry: ProjectRegistry = Depends(get_project_registry),
) -> ConfigVersionsResponse:
    try:
        return project_service.list_versions(registry, project_id)
    except ProjectServiceError as exc:
        raise ProjectNotFoundError(detail=str(exc), cause=exc) from exc


@router.get("/{project_id}/config/versions/{filename}", response_model=ConfigVersionDetail)
def read_version(
    project_id: str,
    filename: str,
    registry: ProjectRegistry = Depends(get_project_registry),
) -> ConfigVersionDetail:
    try:
        text = project_service.read_version(registry, project_id, filename)
    except ProjectServiceError as exc:
        raise ProjectNotFoundError(detail=str(exc), cause=exc) from exc
    return ConfigVersionDetail(filename=filename, yaml=text)


@router.post("/{project_id}/config/versions/{filename}/restore", response_model=SaveConfigResponse)
def restore_version(
    project_id: str,
    filename: str,
    registry: ProjectRegistry = Depends(get_project_registry),
) -> SaveConfigResponse:
    try:
        snapshot = project_service.restore_version(registry, project_id, filename)
    except ProjectServiceError as exc:
        raise ProjectNotFoundError(detail=str(exc), cause=exc) from exc
    return SaveConfigResponse(ok=True, snapshot_filename=snapshot)


@router.put(
    "/{project_id}/config/versions/{filename}/favorite",
    response_model=ToggleFavoriteResponse,
)
def toggle_favorite(
    project_id: str,
    filename: str,
    body: ToggleFavoriteRequest,
    registry: ProjectRegistry = Depends(get_project_registry),
) -> ToggleFavoriteResponse:
    try:
        favorites = project_service.set_favorite(
            registry, project_id, filename, favorite=body.favorite
        )
    except ProjectServiceError as exc:
        raise ProjectNotFoundError(detail=str(exc), cause=exc) from exc
    return ToggleFavoriteResponse(favorite_versions=favorites)


# ---------------------------------------------------------------------------
# Project runs listing
# ---------------------------------------------------------------------------


@router.get("/{project_id}/runs", response_model=ProjectRunsResponse)
def list_runs(
    project_id: str,
    status: str | None = None,
    limit: int | None = None,
    registry: ProjectRegistry = Depends(get_project_registry),
) -> ProjectRunsResponse:
    """List runs launched from this project, newest-first.

    Walks ``<project>/runs/`` directly — every sub-directory containing
    ``pipeline_state.json`` is a run. ``status=running`` / ``?limit=20``
    filter the result. Returns an empty list when no runs have been
    launched yet — that's the expected steady-state for a fresh
    project, not an error.
    """
    try:
        rows = project_service.list_project_runs(
            registry, project_id, status=status, limit=limit,
        )
    except ProjectServiceError as exc:
        raise ProjectNotFoundError(detail=str(exc), cause=exc) from exc

    # Defensive: malformed rows (e.g., corrupt pipeline_state.json that
    # the scanner managed to surface anyway) are dropped rather than
    # 5XX'ing the whole response.
    entries: list[ProjectRunEntry] = []
    for row in rows:
        try:
            entries.append(ProjectRunEntry(**row))
        except Exception:
            continue
    return ProjectRunsResponse(runs=entries)
