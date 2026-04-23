"""Business logic for the integration workspace endpoints."""

from __future__ import annotations

import re
from pathlib import Path

import yaml

from src.api.schemas.config_validate import ConfigCheck, ConfigValidationResult
from src.api.schemas.integration import (
    IntegrationConfigResponse,
    IntegrationConfigVersion,
    IntegrationConfigVersionsResponse,
    IntegrationDetail,
    IntegrationSummary,
    IntegrationTypeInfo,
    IntegrationTypesResponse,
)
from src.api.services.token_crypto import (
    TokenCrypto,
    delete_token_file,
    write_token_file,
)
from src.config.integrations.registry import INTEGRATION_TYPES
from src.pipeline.settings.integrations import (
    IntegrationMetadata,
    IntegrationRegistry,
    IntegrationRegistryEntry,
    IntegrationStore,
)
from src.pipeline.settings.integrations.registry import (
    IntegrationRegistryError,
    validate_integration_id,
)
from src.pipeline.settings.integrations.store import IntegrationStoreError

_SLUG_STRIP_RE = re.compile(r"[^a-z0-9]+")


class IntegrationServiceError(RuntimeError):
    """Surface-level errors mapped to HTTP 4xx by the router."""


def slugify(value: str) -> str:
    slug = _SLUG_STRIP_RE.sub("-", value.strip().lower()).strip("-")
    return slug[:48] or "integration"


# ---------- Types --------------------------------------------------------


def list_types() -> IntegrationTypesResponse:
    entries = sorted(INTEGRATION_TYPES.values(), key=lambda t: t.label.lower())
    return IntegrationTypesResponse(
        types=[
            IntegrationTypeInfo(
                id=t.id,
                label=t.label,
                requires_token=t.requires_token,
                json_schema=t.schema.model_json_schema(),
            )
            for t in entries
        ]
    )


# ---------- Summary / Detail ---------------------------------------------


def _load(
    registry: IntegrationRegistry, integration_id: str
) -> tuple[IntegrationRegistryEntry, IntegrationStore, IntegrationMetadata]:
    try:
        entry = registry.resolve(integration_id)
    except IntegrationRegistryError as exc:
        raise IntegrationServiceError(str(exc)) from exc

    store = IntegrationStore(Path(entry.path))
    if not store.exists():
        raise IntegrationServiceError(
            f"integration directory missing on disk: {store.root}. "
            "The registry points to a folder that no longer exists.",
        )
    return entry, store, store.load()


def list_summaries(registry: IntegrationRegistry) -> list[IntegrationSummary]:
    summaries: list[IntegrationSummary] = []
    for entry in registry.list():
        store = IntegrationStore(Path(entry.path))
        description = ""
        if store.exists():
            try:
                description = store.load().description
            except (OSError, ValueError):
                description = ""
        summaries.append(
            IntegrationSummary(
                id=entry.id,
                name=entry.name,
                type=entry.type,
                path=entry.path,
                created_at=entry.created_at,
                description=description,
                has_token=store.has_token() if store.exists() else False,
            )
        )
    return summaries


def get_detail(registry: IntegrationRegistry, integration_id: str) -> IntegrationDetail:
    entry, store, metadata = _load(registry, integration_id)
    return IntegrationDetail(
        id=metadata.id,
        name=metadata.name,
        type=metadata.type,
        path=entry.path,
        description=metadata.description,
        created_at=metadata.created_at,
        updated_at=metadata.updated_at,
        current_config_yaml=store.current_yaml_text(),
        has_token=store.has_token(),
    )


# ---------- Create / Delete ----------------------------------------------


def create_integration(
    registry: IntegrationRegistry,
    *,
    name: str,
    type: str,
    integration_id: str | None = None,
    path: str | None = None,
    description: str = "",
) -> IntegrationSummary:
    if type not in INTEGRATION_TYPES:
        raise IntegrationServiceError(
            f"unknown integration type {type!r}. Known types: {sorted(INTEGRATION_TYPES.keys())}"
        )

    resolved_id = integration_id or slugify(name)
    try:
        validate_integration_id(resolved_id)
    except IntegrationRegistryError as exc:
        raise IntegrationServiceError(str(exc)) from exc

    target = (
        Path(path).expanduser().resolve()
        if path
        else registry.default_integration_path(resolved_id)
    )
    store = IntegrationStore(target)
    if store.exists():
        raise IntegrationServiceError(f"integration already exists at {target}")

    try:
        store.create(id=resolved_id, name=name, type=type, description=description)
        entry = registry.register(
            integration_id=resolved_id, name=name, type=type, path=target
        )
    except (IntegrationStoreError, IntegrationRegistryError) as exc:
        raise IntegrationServiceError(str(exc)) from exc

    return IntegrationSummary(
        id=entry.id,
        name=entry.name,
        type=entry.type,
        path=entry.path,
        created_at=entry.created_at,
        description=description,
        has_token=False,
    )


def unregister(registry: IntegrationRegistry, integration_id: str) -> bool:
    return registry.unregister(integration_id)


# ---------- Config --------------------------------------------------------


def _parse_yaml_safely(text: str) -> dict | None:
    if not text.strip():
        return None
    try:
        parsed = yaml.safe_load(text)
    except yaml.YAMLError:
        return None
    return parsed if isinstance(parsed, dict) else None


def get_config(
    registry: IntegrationRegistry, integration_id: str
) -> IntegrationConfigResponse:
    _, store, _ = _load(registry, integration_id)
    yaml_text = store.current_yaml_text()
    return IntegrationConfigResponse(
        yaml=yaml_text, parsed_json=_parse_yaml_safely(yaml_text)
    )


def save_config(
    registry: IntegrationRegistry, integration_id: str, yaml_text: str
) -> str | None:
    _, store, _ = _load(registry, integration_id)
    return store.save_config(yaml_text)


def validate_yaml(
    registry: IntegrationRegistry, integration_id: str, yaml_text: str
) -> ConfigValidationResult:
    _, _, metadata = _load(registry, integration_id)
    integration_type = INTEGRATION_TYPES.get(metadata.type)
    checks: list[ConfigCheck] = []

    parsed = _parse_yaml_safely(yaml_text)
    if parsed is None and yaml_text.strip():
        checks.append(
            ConfigCheck(
                label="YAML parses as mapping",
                status="fail",
                detail="YAML is not a mapping",
            )
        )
        return ConfigValidationResult(
            ok=False,
            config_path=f"integration:{integration_id}",
            checks=checks,
        )
    checks.append(ConfigCheck(label="YAML parses", status="ok"))

    if integration_type is None:
        checks.append(
            ConfigCheck(
                label=f"Unknown integration type {metadata.type!r}",
                status="fail",
                detail="No schema registered",
            )
        )
        return ConfigValidationResult(
            ok=False,
            config_path=f"integration:{integration_id}",
            checks=checks,
        )

    try:
        integration_type.schema(**(parsed or {}))
        checks.append(
            ConfigCheck(label=f"{integration_type.schema_name} schema", status="ok")
        )
    except Exception as exc:
        checks.append(
            ConfigCheck(
                label=f"{integration_type.schema_name} schema",
                status="fail",
                detail=str(exc),
            )
        )

    ok = not any(c.status == "fail" for c in checks)
    return ConfigValidationResult(
        ok=ok, config_path=f"integration:{integration_id}", checks=checks
    )


def list_versions(
    registry: IntegrationRegistry, integration_id: str
) -> IntegrationConfigVersionsResponse:
    _, store, _ = _load(registry, integration_id)
    versions = [
        IntegrationConfigVersion(
            filename=v.filename, created_at=v.created_at, size_bytes=v.size_bytes
        )
        for v in store.list_versions()
    ]
    return IntegrationConfigVersionsResponse(versions=versions)


def read_version(
    registry: IntegrationRegistry, integration_id: str, filename: str
) -> str:
    _, store, _ = _load(registry, integration_id)
    try:
        return store.read_version(filename)
    except IntegrationStoreError as exc:
        raise IntegrationServiceError(str(exc)) from exc


def restore_version(
    registry: IntegrationRegistry, integration_id: str, filename: str
) -> str | None:
    _, store, _ = _load(registry, integration_id)
    try:
        return store.restore_version(filename)
    except IntegrationStoreError as exc:
        raise IntegrationServiceError(str(exc)) from exc


# ---------- Tokens --------------------------------------------------------


def set_token(
    registry: IntegrationRegistry,
    integration_id: str,
    plaintext: str,
    crypto: TokenCrypto,
) -> None:
    _, store, _ = _load(registry, integration_id)
    write_token_file(store.token_path, plaintext, crypto)


def delete_token(registry: IntegrationRegistry, integration_id: str) -> bool:
    _, store, _ = _load(registry, integration_id)
    return delete_token_file(store.token_path)


__all__ = [
    "IntegrationServiceError",
    "create_integration",
    "delete_token",
    "get_config",
    "get_detail",
    "list_summaries",
    "list_types",
    "list_versions",
    "read_version",
    "restore_version",
    "save_config",
    "set_token",
    "slugify",
    "unregister",
    "validate_yaml",
]
