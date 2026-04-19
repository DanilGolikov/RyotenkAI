from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from src.api.dependencies import get_project_registry
from src.api.schemas.config_validate import ConfigValidationResult
from src.api.schemas.project import (
    ConfigResponse,
    ConfigVersionDetail,
    ConfigVersionsResponse,
    CreateProjectRequest,
    ProjectDetail,
    ProjectSummary,
    SaveConfigRequest,
    SaveConfigResponse,
)
from src.api.services import project_service
from src.api.services.project_service import ProjectServiceError
from src.pipeline.project import ProjectRegistry

router = APIRouter(prefix="/projects", tags=["projects"])


def _raise_400(exc: ProjectServiceError) -> None:
    raise HTTPException(status_code=400, detail=str(exc))


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
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/{project_id}", response_model=ProjectDetail)
def get_project(
    project_id: str,
    registry: ProjectRegistry = Depends(get_project_registry),
) -> ProjectDetail:
    try:
        return project_service.get_detail(registry, project_id)
    except ProjectServiceError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.delete("/{project_id}", status_code=204)
def delete_project(
    project_id: str,
    registry: ProjectRegistry = Depends(get_project_registry),
) -> None:
    """Unregister a project (does not delete files on disk)."""
    removed = project_service.unregister(registry, project_id)
    if not removed:
        raise HTTPException(status_code=404, detail=f"project {project_id!r} not registered")


@router.get("/{project_id}/config", response_model=ConfigResponse)
def get_config(
    project_id: str,
    registry: ProjectRegistry = Depends(get_project_registry),
) -> ConfigResponse:
    try:
        return project_service.get_config(registry, project_id)
    except ProjectServiceError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.put("/{project_id}/config", response_model=SaveConfigResponse)
def save_config(
    project_id: str,
    body: SaveConfigRequest,
    registry: ProjectRegistry = Depends(get_project_registry),
) -> SaveConfigResponse:
    try:
        snapshot = project_service.save_config(registry, project_id, body.yaml)
    except ProjectServiceError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return SaveConfigResponse(ok=True, snapshot_filename=snapshot)


@router.post("/{project_id}/config/validate", response_model=ConfigValidationResult)
def validate_config(
    project_id: str,
    body: SaveConfigRequest,
    registry: ProjectRegistry = Depends(get_project_registry),
) -> ConfigValidationResult:
    try:
        return project_service.validate_yaml(registry, project_id, body.yaml)
    except ProjectServiceError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/{project_id}/config/versions", response_model=ConfigVersionsResponse)
def list_versions(
    project_id: str,
    registry: ProjectRegistry = Depends(get_project_registry),
) -> ConfigVersionsResponse:
    try:
        return project_service.list_versions(registry, project_id)
    except ProjectServiceError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/{project_id}/config/versions/{filename}", response_model=ConfigVersionDetail)
def read_version(
    project_id: str,
    filename: str,
    registry: ProjectRegistry = Depends(get_project_registry),
) -> ConfigVersionDetail:
    try:
        text = project_service.read_version(registry, project_id, filename)
    except ProjectServiceError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
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
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return SaveConfigResponse(ok=True, snapshot_filename=snapshot)
