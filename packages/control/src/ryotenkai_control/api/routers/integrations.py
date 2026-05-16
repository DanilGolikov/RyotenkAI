from __future__ import annotations

from pathlib import Path

import yaml
from fastapi import APIRouter, Depends

from ryotenkai_control.api.dependencies import get_integration_registry, get_token_crypto
from ryotenkai_control.api.schemas.config_validate import ConfigValidationResult
from ryotenkai_control.api.schemas.integration import (
    ConnectionTestResult,
    CreateIntegrationRequest,
    IntegrationConfigResponse,
    IntegrationConfigVersionDetail,
    IntegrationConfigVersionsResponse,
    IntegrationDetail,
    IntegrationSaveConfigRequest,
    IntegrationSaveConfigResponse,
    IntegrationSummary,
    IntegrationTokenRequest,
    IntegrationTypesResponse,
)
from ryotenkai_control.api.services import integration_service
from ryotenkai_control.api.services.connection_test import test_integration
from ryotenkai_control.api.services.integration_service import IntegrationServiceError
from ryotenkai_shared.errors import (
    ConfigInvalidError,
    IntegrationNotFoundError,
    ProjectDirectoryMissingError,
)
from ryotenkai_shared.utils.crypto.token_crypto import TokenCrypto, read_token_file
from ryotenkai_control.workspace.integrations import IntegrationRegistry, IntegrationStore

router = APIRouter(prefix="/integrations", tags=["integrations"])


@router.get("/types", response_model=IntegrationTypesResponse)
def list_types() -> IntegrationTypesResponse:
    return integration_service.list_types()


@router.get("", response_model=list[IntegrationSummary])
def list_integrations(
    registry: IntegrationRegistry = Depends(get_integration_registry),
) -> list[IntegrationSummary]:
    return integration_service.list_summaries(registry)


@router.post("", response_model=IntegrationSummary, status_code=201)
def create_integration(
    body: CreateIntegrationRequest,
    registry: IntegrationRegistry = Depends(get_integration_registry),
) -> IntegrationSummary:
    try:
        return integration_service.create_integration(
            registry,
            name=body.name,
            type=body.type,
            integration_id=body.id,
            path=body.path,
            description=body.description,
        )
    except IntegrationServiceError as exc:
        raise ConfigInvalidError(detail=str(exc), cause=exc) from exc


@router.get("/{integration_id}", response_model=IntegrationDetail)
def get_integration(
    integration_id: str,
    registry: IntegrationRegistry = Depends(get_integration_registry),
) -> IntegrationDetail:
    try:
        return integration_service.get_detail(registry, integration_id)
    except IntegrationServiceError as exc:
        raise IntegrationNotFoundError(detail=str(exc), cause=exc) from exc


@router.delete("/{integration_id}", status_code=204)
def delete_integration(
    integration_id: str,
    registry: IntegrationRegistry = Depends(get_integration_registry),
) -> None:
    removed = integration_service.unregister(registry, integration_id)
    if not removed:
        raise IntegrationNotFoundError(
            detail=f"integration {integration_id!r} not registered",
            context={"integration_id": integration_id},
        )


@router.get("/{integration_id}/config", response_model=IntegrationConfigResponse)
def get_config(
    integration_id: str,
    registry: IntegrationRegistry = Depends(get_integration_registry),
) -> IntegrationConfigResponse:
    try:
        return integration_service.get_config(registry, integration_id)
    except IntegrationServiceError as exc:
        raise IntegrationNotFoundError(detail=str(exc), cause=exc) from exc


@router.put("/{integration_id}/config", response_model=IntegrationSaveConfigResponse)
def save_config(
    integration_id: str,
    body: IntegrationSaveConfigRequest,
    registry: IntegrationRegistry = Depends(get_integration_registry),
) -> IntegrationSaveConfigResponse:
    try:
        snapshot = integration_service.save_config(registry, integration_id, body.yaml)
    except IntegrationServiceError as exc:
        raise IntegrationNotFoundError(detail=str(exc), cause=exc) from exc
    return IntegrationSaveConfigResponse(ok=True, snapshot_filename=snapshot)


@router.post(
    "/{integration_id}/config/validate", response_model=ConfigValidationResult
)
def validate_config(
    integration_id: str,
    body: IntegrationSaveConfigRequest,
    registry: IntegrationRegistry = Depends(get_integration_registry),
) -> ConfigValidationResult:
    try:
        return integration_service.validate_yaml(registry, integration_id, body.yaml)
    except IntegrationServiceError as exc:
        raise IntegrationNotFoundError(detail=str(exc), cause=exc) from exc


@router.get(
    "/{integration_id}/config/versions", response_model=IntegrationConfigVersionsResponse
)
def list_versions(
    integration_id: str,
    registry: IntegrationRegistry = Depends(get_integration_registry),
) -> IntegrationConfigVersionsResponse:
    try:
        return integration_service.list_versions(registry, integration_id)
    except IntegrationServiceError as exc:
        raise IntegrationNotFoundError(detail=str(exc), cause=exc) from exc


@router.get(
    "/{integration_id}/config/versions/{filename}",
    response_model=IntegrationConfigVersionDetail,
)
def read_version(
    integration_id: str,
    filename: str,
    registry: IntegrationRegistry = Depends(get_integration_registry),
) -> IntegrationConfigVersionDetail:
    try:
        text = integration_service.read_version(registry, integration_id, filename)
    except IntegrationServiceError as exc:
        raise IntegrationNotFoundError(detail=str(exc), cause=exc) from exc
    return IntegrationConfigVersionDetail(filename=filename, yaml=text)


@router.post(
    "/{integration_id}/config/versions/{filename}/restore",
    response_model=IntegrationSaveConfigResponse,
)
def restore_version(
    integration_id: str,
    filename: str,
    registry: IntegrationRegistry = Depends(get_integration_registry),
) -> IntegrationSaveConfigResponse:
    try:
        snapshot = integration_service.restore_version(registry, integration_id, filename)
    except IntegrationServiceError as exc:
        raise IntegrationNotFoundError(detail=str(exc), cause=exc) from exc
    return IntegrationSaveConfigResponse(ok=True, snapshot_filename=snapshot)


# ---------- Tokens & test-connection --------------------------------------


@router.put("/{integration_id}/token", status_code=204)
def set_token(
    integration_id: str,
    body: IntegrationTokenRequest,
    registry: IntegrationRegistry = Depends(get_integration_registry),
    crypto: TokenCrypto = Depends(get_token_crypto),
) -> None:
    """Write-only — body is never echoed back. Responses contain no token."""
    try:
        integration_service.set_token(registry, integration_id, body.token, crypto)
    except IntegrationServiceError as exc:
        raise IntegrationNotFoundError(detail=str(exc), cause=exc) from exc


@router.delete("/{integration_id}/token", status_code=204)
def delete_token(
    integration_id: str,
    registry: IntegrationRegistry = Depends(get_integration_registry),
) -> None:
    try:
        integration_service.delete_token(registry, integration_id)
    except IntegrationServiceError as exc:
        raise IntegrationNotFoundError(detail=str(exc), cause=exc) from exc


@router.post("/{integration_id}/test-connection", response_model=ConnectionTestResult)
def test_connection(
    integration_id: str,
    registry: IntegrationRegistry = Depends(get_integration_registry),
    crypto: TokenCrypto = Depends(get_token_crypto),
) -> ConnectionTestResult:
    try:
        entry = registry.resolve(integration_id)
    except Exception as exc:
        raise IntegrationNotFoundError(detail=str(exc), cause=exc) from exc

    store = IntegrationStore(Path(entry.path))
    if not store.exists():
        raise ProjectDirectoryMissingError(
            detail=f"integration workspace missing: {store.root}",
            context={"integration_id": integration_id},
        )

    yaml_text = store.current_yaml_text()
    try:
        parsed = yaml.safe_load(yaml_text) if yaml_text.strip() else {}
    except yaml.YAMLError:
        parsed = {}
    if not isinstance(parsed, dict):
        parsed = {}

    token = read_token_file(store.token_path, crypto)
    return test_integration(entry.type, parsed, token)


__all__ = ["router"]
