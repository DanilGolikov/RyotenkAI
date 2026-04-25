"""Business logic for the project workspace endpoints."""

from __future__ import annotations

import re
import tempfile
from pathlib import Path

import yaml

from src.api.schemas.config_validate import ConfigValidationResult
from src.api.schemas.project import (
    ConfigResponse,
    ConfigVersion,
    ConfigVersionsResponse,
    ProjectDetail,
    ProjectSummary,
    StalePluginEntry,
)
from src.api.services import config_service
from src.pipeline.project import (
    ProjectMetadata,
    ProjectRegistry,
    ProjectRegistryEntry,
    ProjectStore,
)
from src.pipeline.project.registry import ProjectRegistryError, validate_project_id
from src.pipeline.project.store import ProjectStoreError

_SLUG_STRIP_RE = re.compile(r"[^a-z0-9]+")


class ProjectServiceError(RuntimeError):
    """Surface-level errors mapped to HTTP 4xx by the router."""


def slugify(value: str) -> str:
    slug = _SLUG_STRIP_RE.sub("-", value.strip().lower()).strip("-")
    return slug[:48] or "project"


def _load_project(registry: ProjectRegistry, project_id: str) -> tuple[ProjectRegistryEntry, ProjectStore, ProjectMetadata]:
    try:
        entry = registry.resolve(project_id)
    except ProjectRegistryError as exc:
        raise ProjectServiceError(str(exc)) from exc

    store = ProjectStore(Path(entry.path))
    if not store.exists():
        raise ProjectServiceError(
            f"project directory missing on disk: {store.root}. "
            "The registry points to a folder that no longer exists.",
        )
    return entry, store, store.load()


# ---------- Summary / Detail ---------------------------------------------


def list_summaries(registry: ProjectRegistry) -> list[ProjectSummary]:
    summaries: list[ProjectSummary] = []
    for entry in registry.list():
        store = ProjectStore(Path(entry.path))
        description = ""
        if store.exists():
            try:
                description = store.load().description
            except (OSError, ValueError):
                description = ""
        summaries.append(
            ProjectSummary(
                id=entry.id,
                name=entry.name,
                path=entry.path,
                created_at=entry.created_at,
                description=description,
            )
        )
    return summaries


def get_detail(registry: ProjectRegistry, project_id: str) -> ProjectDetail:
    entry, store, metadata = _load_project(registry, project_id)
    return ProjectDetail(
        id=metadata.id,
        name=metadata.name,
        path=entry.path,
        description=metadata.description,
        created_at=metadata.created_at,
        updated_at=metadata.updated_at,
        current_config_yaml=store.current_yaml_text(),
    )


# ---------- Create / Delete ----------------------------------------------


def create_project(
    registry: ProjectRegistry,
    *,
    name: str,
    project_id: str | None = None,
    path: str | None = None,
    description: str = "",
) -> ProjectSummary:
    resolved_id = project_id or slugify(name)
    try:
        validate_project_id(resolved_id)
    except ProjectRegistryError as exc:
        raise ProjectServiceError(str(exc)) from exc

    target = Path(path).expanduser().resolve() if path else registry.default_project_path(resolved_id)
    store = ProjectStore(target)
    if store.exists():
        raise ProjectServiceError(f"project already exists at {target}")

    try:
        store.create(id=resolved_id, name=name, description=description)
        entry = registry.register(project_id=resolved_id, name=name, path=target)
    except (ProjectStoreError, ProjectRegistryError) as exc:
        raise ProjectServiceError(str(exc)) from exc

    return ProjectSummary(
        id=entry.id,
        name=entry.name,
        path=entry.path,
        created_at=entry.created_at,
        description=description,
    )


def update_description(
    registry: ProjectRegistry,
    project_id: str,
    description: str,
) -> ProjectDetail:
    """Patch only the description (plus updated_at) on disk. Name, id, and
    path are fixed identity fields and stay put."""
    entry, store, _ = _load_project(registry, project_id)
    try:
        metadata = store.update_description(description)
    except ProjectStoreError as exc:
        raise ProjectServiceError(str(exc)) from exc
    return ProjectDetail(
        id=metadata.id,
        name=metadata.name,
        path=entry.path,
        description=metadata.description,
        created_at=metadata.created_at,
        updated_at=metadata.updated_at,
        current_config_yaml=store.current_yaml_text(),
    )


def unregister(
    registry: ProjectRegistry,
    project_id: str,
    *,
    delete_files: bool = False,
) -> bool:
    """Remove the project from the registry. When ``delete_files`` is
    true the on-disk workspace is also rm -rf'd so the UI's "delete"
    actually deletes. Registry removal is attempted first so a crash in
    the filesystem step doesn't leave an orphan registry entry.
    """
    try:
        entry = registry.resolve(project_id)
    except ProjectRegistryError:
        entry = None

    removed = registry.unregister(project_id)

    if delete_files and entry is not None:
        import shutil

        path = Path(entry.path).expanduser().resolve()
        if path.exists() and path.is_dir():
            # Guard: refuse to delete obviously wrong paths (root, home,
            # anything outside a reasonable depth).
            if len(path.parts) < 3:
                raise ProjectServiceError(
                    f"refusing to delete suspicious path: {path}",
                )
            shutil.rmtree(path, ignore_errors=False)

    return removed


# ---------- Config --------------------------------------------------------


def _parse_yaml_safely(text: str) -> dict | None:
    if not text.strip():
        return None
    try:
        parsed = yaml.safe_load(text)
    except yaml.YAMLError:
        return None
    return parsed if isinstance(parsed, dict) else None


def get_config(registry: ProjectRegistry, project_id: str) -> ConfigResponse:
    _, store, _ = _load_project(registry, project_id)
    yaml_text = store.current_yaml_text()
    parsed = _parse_yaml_safely(yaml_text)
    return ConfigResponse(
        yaml=yaml_text,
        parsed_json=parsed,
        stale_plugins=_collect_stale_plugins(parsed),
    )


def _collect_stale_plugins(parsed: dict | None) -> list[StalePluginEntry]:
    """Walk the saved config for plugin references that no longer match
    a registered community plugin.

    Empty list when ``parsed`` is None (config didn't parse) or when
    :class:`PipelineConfig` validation fails — config-level errors
    surface through the dedicated validate endpoint, not here. We
    swallow the validation error so a get_config call never fails on
    config issues alone (the user can still open the editor and fix).
    """
    if parsed is None:
        return []

    # Lazy imports keep the project_service module light at import time
    # — community catalog pulls torch/datasets via plugin loaders, and
    # the get_config endpoint is hot enough that we only want that hit
    # when there's actually parsed YAML to walk.
    from pydantic import ValidationError

    from src.community.stale_plugins import find_stale_plugins
    from src.utils.config import PipelineConfig

    try:
        config = PipelineConfig.model_validate(parsed)
    except ValidationError:
        return []

    return [
        StalePluginEntry(
            plugin_kind=ref.plugin_kind,
            plugin_name=ref.plugin_name,
            instance_id=ref.instance_id,
            location=ref.location,
        )
        for ref in find_stale_plugins(config)
    ]


def save_config(registry: ProjectRegistry, project_id: str, yaml_text: str) -> str | None:
    _, store, _ = _load_project(registry, project_id)
    return store.save_config(yaml_text)


def read_env(registry: ProjectRegistry, project_id: str) -> dict[str, str]:
    _, store, _ = _load_project(registry, project_id)
    try:
        return store.read_env()
    except ProjectStoreError as exc:
        raise ProjectServiceError(str(exc)) from exc


def write_env(
    registry: ProjectRegistry, project_id: str, env: dict[str, str]
) -> dict[str, str]:
    _, store, _ = _load_project(registry, project_id)
    try:
        store.write_env(env)
        return store.read_env()
    except ProjectStoreError as exc:
        raise ProjectServiceError(str(exc)) from exc


def validate_yaml(registry: ProjectRegistry, project_id: str, yaml_text: str) -> ConfigValidationResult:
    _load_project(registry, project_id)  # ensure the project exists
    with tempfile.NamedTemporaryFile(
        "w", suffix=".yaml", delete=False, encoding="utf-8"
    ) as tmp:
        tmp.write(yaml_text)
        tmp_path = Path(tmp.name)
    try:
        return config_service.validate_config(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)


def list_versions(registry: ProjectRegistry, project_id: str) -> ConfigVersionsResponse:
    _, store, metadata = _load_project(registry, project_id)
    favorites = set(metadata.favorite_versions)
    versions = [
        ConfigVersion(
            filename=v.filename,
            created_at=v.created_at,
            size_bytes=v.size_bytes,
            is_favorite=v.filename in favorites,
        )
        for v in store.list_versions()
    ]
    # Favorites first, newest-first within each group.
    versions.sort(key=lambda v: (not v.is_favorite, -len(v.filename), v.filename), reverse=False)
    versions.sort(key=lambda v: v.filename, reverse=True)
    versions.sort(key=lambda v: not v.is_favorite)
    return ConfigVersionsResponse(versions=versions)


def set_favorite(
    registry: ProjectRegistry,
    project_id: str,
    filename: str,
    *,
    favorite: bool,
) -> list[str]:
    _, store, _ = _load_project(registry, project_id)
    try:
        return store.toggle_favorite_version(filename, pinned=favorite)
    except Exception as exc:
        raise ProjectServiceError(str(exc)) from exc


def read_version(registry: ProjectRegistry, project_id: str, filename: str) -> str:
    _, store, _ = _load_project(registry, project_id)
    try:
        return store.read_version(filename)
    except ProjectStoreError as exc:
        raise ProjectServiceError(str(exc)) from exc


def restore_version(registry: ProjectRegistry, project_id: str, filename: str) -> str | None:
    _, store, _ = _load_project(registry, project_id)
    try:
        return store.restore_version(filename)
    except ProjectStoreError as exc:
        raise ProjectServiceError(str(exc)) from exc


# ---------- Runs directory ------------------------------------------------


def get_runs_dir(registry: ProjectRegistry, project_id: str) -> Path:
    _, store, _ = _load_project(registry, project_id)
    store.runs_dir.mkdir(parents=True, exist_ok=True)
    return store.runs_dir


__all__ = [
    "ProjectServiceError",
    "create_project",
    "get_config",
    "get_detail",
    "get_runs_dir",
    "list_summaries",
    "list_versions",
    "read_version",
    "update_description",
    "restore_version",
    "save_config",
    "set_favorite",
    "slugify",
    "unregister",
    "validate_yaml",
]
