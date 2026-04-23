from __future__ import annotations

from pathlib import Path

import yaml
from fastapi import APIRouter, Depends, HTTPException

from src.api.dependencies import get_provider_registry, get_token_crypto
from src.api.schemas.config_validate import ConfigValidationResult
from src.api.schemas.integration import ConnectionTestResult, IntegrationTokenRequest
from src.api.schemas.provider import (
    CreateProviderRequest,
    ProviderConfigResponse,
    ProviderConfigVersionDetail,
    ProviderConfigVersionsResponse,
    ProviderDetail,
    ProviderSaveConfigRequest,
    ProviderSaveConfigResponse,
    ProviderSummary,
    ProviderTypesResponse,
)
from src.api.services import provider_service
from src.api.services.connection_test import test_provider
from src.api.services.provider_service import ProviderServiceError
from src.api.services.token_crypto import TokenCrypto, read_token_file
from src.pipeline.settings.providers import ProviderRegistry, ProviderStore

router = APIRouter(prefix="/providers", tags=["providers"])


@router.get("/types", response_model=ProviderTypesResponse)
def list_types() -> ProviderTypesResponse:
    return provider_service.list_types()


@router.get("", response_model=list[ProviderSummary])
def list_providers(
    registry: ProviderRegistry = Depends(get_provider_registry),
) -> list[ProviderSummary]:
    return provider_service.list_summaries(registry)


@router.post("", response_model=ProviderSummary, status_code=201)
def create_provider(
    body: CreateProviderRequest,
    registry: ProviderRegistry = Depends(get_provider_registry),
) -> ProviderSummary:
    try:
        return provider_service.create_provider(
            registry,
            name=body.name,
            type=body.type,
            provider_id=body.id,
            path=body.path,
            description=body.description,
        )
    except ProviderServiceError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/{provider_id}", response_model=ProviderDetail)
def get_provider(
    provider_id: str,
    registry: ProviderRegistry = Depends(get_provider_registry),
) -> ProviderDetail:
    try:
        return provider_service.get_detail(registry, provider_id)
    except ProviderServiceError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.delete("/{provider_id}", status_code=204)
def delete_provider(
    provider_id: str,
    registry: ProviderRegistry = Depends(get_provider_registry),
) -> None:
    removed = provider_service.unregister(registry, provider_id)
    if not removed:
        raise HTTPException(status_code=404, detail=f"provider {provider_id!r} not registered")


@router.get("/{provider_id}/config", response_model=ProviderConfigResponse)
def get_config(
    provider_id: str,
    registry: ProviderRegistry = Depends(get_provider_registry),
) -> ProviderConfigResponse:
    try:
        return provider_service.get_config(registry, provider_id)
    except ProviderServiceError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.put("/{provider_id}/config", response_model=ProviderSaveConfigResponse)
def save_config(
    provider_id: str,
    body: ProviderSaveConfigRequest,
    registry: ProviderRegistry = Depends(get_provider_registry),
) -> ProviderSaveConfigResponse:
    try:
        snapshot = provider_service.save_config(registry, provider_id, body.yaml)
    except ProviderServiceError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return ProviderSaveConfigResponse(ok=True, snapshot_filename=snapshot)


@router.post("/{provider_id}/config/validate", response_model=ConfigValidationResult)
def validate_config(
    provider_id: str,
    body: ProviderSaveConfigRequest,
    registry: ProviderRegistry = Depends(get_provider_registry),
) -> ConfigValidationResult:
    try:
        return provider_service.validate_yaml(registry, provider_id, body.yaml)
    except ProviderServiceError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/{provider_id}/config/versions", response_model=ProviderConfigVersionsResponse)
def list_versions(
    provider_id: str,
    registry: ProviderRegistry = Depends(get_provider_registry),
) -> ProviderConfigVersionsResponse:
    try:
        return provider_service.list_versions(registry, provider_id)
    except ProviderServiceError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get(
    "/{provider_id}/config/versions/{filename}",
    response_model=ProviderConfigVersionDetail,
)
def read_version(
    provider_id: str,
    filename: str,
    registry: ProviderRegistry = Depends(get_provider_registry),
) -> ProviderConfigVersionDetail:
    try:
        text = provider_service.read_version(registry, provider_id, filename)
    except ProviderServiceError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return ProviderConfigVersionDetail(filename=filename, yaml=text)


@router.post(
    "/{provider_id}/config/versions/{filename}/restore",
    response_model=ProviderSaveConfigResponse,
)
def restore_version(
    provider_id: str,
    filename: str,
    registry: ProviderRegistry = Depends(get_provider_registry),
) -> ProviderSaveConfigResponse:
    try:
        snapshot = provider_service.restore_version(registry, provider_id, filename)
    except ProviderServiceError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return ProviderSaveConfigResponse(ok=True, snapshot_filename=snapshot)


# ---------- Tokens & test-connection --------------------------------------


@router.put("/{provider_id}/token", status_code=204)
def set_token(
    provider_id: str,
    body: IntegrationTokenRequest,
    registry: ProviderRegistry = Depends(get_provider_registry),
    crypto: TokenCrypto = Depends(get_token_crypto),
) -> None:
    """Write-only token upload (e.g. RUNPOD_API_KEY). Never echoed back."""
    try:
        provider_service.set_token(registry, provider_id, body.token, crypto)
    except ProviderServiceError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.delete("/{provider_id}/token", status_code=204)
def delete_token(
    provider_id: str,
    registry: ProviderRegistry = Depends(get_provider_registry),
) -> None:
    try:
        provider_service.delete_token(registry, provider_id)
    except ProviderServiceError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/{provider_id}/test-connection", response_model=ConnectionTestResult)
def test_connection(
    provider_id: str,
    registry: ProviderRegistry = Depends(get_provider_registry),
    crypto: TokenCrypto = Depends(get_token_crypto),
) -> ConnectionTestResult:
    try:
        entry = registry.resolve(provider_id)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    store = ProviderStore(Path(entry.path))
    if not store.exists():
        raise HTTPException(
            status_code=404, detail=f"provider workspace missing: {store.root}"
        )

    yaml_text = store.current_yaml_text()
    try:
        parsed = yaml.safe_load(yaml_text) if yaml_text.strip() else {}
    except yaml.YAMLError:
        parsed = {}
    if not isinstance(parsed, dict):
        parsed = {}

    token = read_token_file(store.root / "token.enc", crypto)
    return test_provider(entry.type, parsed, token)
