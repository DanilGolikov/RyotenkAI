"""Business logic for the provider workspace endpoints."""

from __future__ import annotations

import re
from pathlib import Path

import yaml

from src.api.schemas.config_validate import ConfigCheck, ConfigValidationResult
from src.api.schemas.provider import (
    ProviderConfigResponse,
    ProviderConfigVersion,
    ProviderConfigVersionsResponse,
    ProviderDetail,
    ProviderSummary,
    ProviderTypeInfo,
    ProviderTypesResponse,
)
from src.config.providers.registry import PROVIDER_TYPES
from src.pipeline.settings.providers import (
    ProviderMetadata,
    ProviderRegistry,
    ProviderRegistryEntry,
    ProviderStore,
)
from src.pipeline.settings.providers.registry import (
    ProviderRegistryError,
    validate_provider_id,
)
from src.pipeline.settings.providers.store import ProviderStoreError

_SLUG_STRIP_RE = re.compile(r"[^a-z0-9]+")


class ProviderServiceError(RuntimeError):
    """Surface-level errors mapped to HTTP 4xx by the router."""


def slugify(value: str) -> str:
    slug = _SLUG_STRIP_RE.sub("-", value.strip().lower()).strip("-")
    return slug[:48] or "provider"


# ---------- Types ---------------------------------------------------------


def list_types() -> ProviderTypesResponse:
    entries = sorted(PROVIDER_TYPES.values(), key=lambda t: t.label.lower())
    return ProviderTypesResponse(
        types=[
            ProviderTypeInfo(id=t.id, label=t.label, json_schema=t.schema.model_json_schema())
            for t in entries
        ]
    )


# ---------- Summary / Detail ---------------------------------------------


def _load_provider(
    registry: ProviderRegistry, provider_id: str
) -> tuple[ProviderRegistryEntry, ProviderStore, ProviderMetadata]:
    try:
        entry = registry.resolve(provider_id)
    except ProviderRegistryError as exc:
        raise ProviderServiceError(str(exc)) from exc

    store = ProviderStore(Path(entry.path))
    if not store.exists():
        raise ProviderServiceError(
            f"provider directory missing on disk: {store.root}. "
            "The registry points to a folder that no longer exists.",
        )
    return entry, store, store.load()


def list_summaries(registry: ProviderRegistry) -> list[ProviderSummary]:
    summaries: list[ProviderSummary] = []
    for entry in registry.list():
        store = ProviderStore(Path(entry.path))
        description = ""
        if store.exists():
            try:
                description = store.load().description
            except (OSError, ValueError):
                description = ""
        summaries.append(
            ProviderSummary(
                id=entry.id,
                name=entry.name,
                type=entry.type,
                path=entry.path,
                created_at=entry.created_at,
                description=description,
            )
        )
    return summaries


def get_detail(registry: ProviderRegistry, provider_id: str) -> ProviderDetail:
    entry, store, metadata = _load_provider(registry, provider_id)
    return ProviderDetail(
        id=metadata.id,
        name=metadata.name,
        type=metadata.type,
        path=entry.path,
        description=metadata.description,
        created_at=metadata.created_at,
        updated_at=metadata.updated_at,
        current_config_yaml=store.current_yaml_text(),
    )


# ---------- Create / Delete ----------------------------------------------


def create_provider(
    registry: ProviderRegistry,
    *,
    name: str,
    type: str,
    provider_id: str | None = None,
    path: str | None = None,
    description: str = "",
) -> ProviderSummary:
    if type not in PROVIDER_TYPES:
        raise ProviderServiceError(
            f"unknown provider type {type!r}. Known types: {sorted(PROVIDER_TYPES.keys())}"
        )

    resolved_id = provider_id or slugify(name)
    try:
        validate_provider_id(resolved_id)
    except ProviderRegistryError as exc:
        raise ProviderServiceError(str(exc)) from exc

    target = Path(path).expanduser().resolve() if path else registry.default_provider_path(resolved_id)
    store = ProviderStore(target)
    if store.exists():
        raise ProviderServiceError(f"provider already exists at {target}")

    try:
        store.create(id=resolved_id, name=name, type=type, description=description)
        entry = registry.register(provider_id=resolved_id, name=name, type=type, path=target)
    except (ProviderStoreError, ProviderRegistryError) as exc:
        raise ProviderServiceError(str(exc)) from exc

    return ProviderSummary(
        id=entry.id,
        name=entry.name,
        type=entry.type,
        path=entry.path,
        created_at=entry.created_at,
        description=description,
    )


def unregister(registry: ProviderRegistry, provider_id: str) -> bool:
    return registry.unregister(provider_id)


# ---------- Config --------------------------------------------------------


def _parse_yaml_safely(text: str) -> dict | None:
    if not text.strip():
        return None
    try:
        parsed = yaml.safe_load(text)
    except yaml.YAMLError:
        return None
    return parsed if isinstance(parsed, dict) else None


def get_config(registry: ProviderRegistry, provider_id: str) -> ProviderConfigResponse:
    _, store, _ = _load_provider(registry, provider_id)
    yaml_text = store.current_yaml_text()
    return ProviderConfigResponse(yaml=yaml_text, parsed_json=_parse_yaml_safely(yaml_text))


def save_config(registry: ProviderRegistry, provider_id: str, yaml_text: str) -> str | None:
    _, store, _ = _load_provider(registry, provider_id)
    return store.save_config(yaml_text)


def validate_yaml(
    registry: ProviderRegistry, provider_id: str, yaml_text: str
) -> ConfigValidationResult:
    _, _, metadata = _load_provider(registry, provider_id)
    provider_type = PROVIDER_TYPES.get(metadata.type)
    checks: list[ConfigCheck] = []

    parsed = _parse_yaml_safely(yaml_text)
    if parsed is None and yaml_text.strip():
        checks.append(ConfigCheck(label="YAML parses as mapping", status="fail", detail="YAML is not a mapping"))
        return ConfigValidationResult(ok=False, config_path=f"provider:{provider_id}", checks=checks)
    checks.append(ConfigCheck(label="YAML parses", status="ok"))

    if provider_type is None:
        checks.append(
            ConfigCheck(
                label=f"Unknown provider type {metadata.type!r}",
                status="fail",
                detail="No schema registered",
            )
        )
        return ConfigValidationResult(ok=False, config_path=f"provider:{provider_id}", checks=checks)

    try:
        provider_type.schema(**(parsed or {}))
        checks.append(ConfigCheck(label=f"{provider_type.schema_name} schema", status="ok"))
    except Exception as exc:  # noqa: BLE001 — surface full details to UI
        checks.append(
            ConfigCheck(
                label=f"{provider_type.schema_name} schema",
                status="fail",
                detail=str(exc),
            )
        )

    ok = not any(c.status == "fail" for c in checks)
    return ConfigValidationResult(ok=ok, config_path=f"provider:{provider_id}", checks=checks)


def list_versions(registry: ProviderRegistry, provider_id: str) -> ProviderConfigVersionsResponse:
    _, store, _ = _load_provider(registry, provider_id)
    versions = [
        ProviderConfigVersion(filename=v.filename, created_at=v.created_at, size_bytes=v.size_bytes)
        for v in store.list_versions()
    ]
    return ProviderConfigVersionsResponse(versions=versions)


def read_version(registry: ProviderRegistry, provider_id: str, filename: str) -> str:
    _, store, _ = _load_provider(registry, provider_id)
    try:
        return store.read_version(filename)
    except ProviderStoreError as exc:
        raise ProviderServiceError(str(exc)) from exc


def restore_version(registry: ProviderRegistry, provider_id: str, filename: str) -> str | None:
    _, store, _ = _load_provider(registry, provider_id)
    try:
        return store.restore_version(filename)
    except ProviderStoreError as exc:
        raise ProviderServiceError(str(exc)) from exc


__all__ = [
    "ProviderServiceError",
    "create_provider",
    "get_config",
    "get_detail",
    "list_summaries",
    "list_types",
    "list_versions",
    "read_version",
    "restore_version",
    "save_config",
    "slugify",
    "unregister",
    "validate_yaml",
]
