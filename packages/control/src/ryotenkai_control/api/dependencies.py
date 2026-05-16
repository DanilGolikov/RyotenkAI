from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
from fastapi import Depends

from ryotenkai_control.api.config import ApiSettings
from ryotenkai_control.workspace.projects import ProjectRegistry, ProjectStore
from ryotenkai_control.workspace.projects.registry import ProjectRegistryError
from ryotenkai_control.workspace.integrations import IntegrationRegistry
from ryotenkai_control.workspace.providers import ProviderRegistry
from ryotenkai_control.pipeline.state import PipelineStateStore
from ryotenkai_shared.errors import (
    AttemptInvalidError,
    ConfigInvalidError,
    DatasetNotFoundError,
    ProjectDirectoryMissingError,
    ProjectNotFoundError,
    RunNotFoundError,
)

if TYPE_CHECKING:
    from ryotenkai_shared.utils.crypto.token_crypto import TokenCrypto
    from ryotenkai_shared.config.datasets.schema import DatasetConfig


@lru_cache(maxsize=1)
def get_settings() -> ApiSettings:
    return ApiSettings()


def get_runs_dir(settings: ApiSettings = Depends(get_settings)) -> Path:
    runs_dir = settings.runs_dir_resolved
    if not runs_dir.exists():
        runs_dir.mkdir(parents=True, exist_ok=True)
    return runs_dir


def resolve_run_dir(run_id: str, runs_dir: Path = Depends(get_runs_dir)) -> Path:
    """Resolve a run_id (directory name, possibly nested under a subgroup) to an
    absolute Path, rejecting path traversal.

    Defence-in-depth: both sides of the containment check are fully resolved
    (symlinks followed) so a symlink farm under ``runs_dir`` can't let a
    crafted ``run_id`` escape the configured root. ``get_runs_dir`` already
    returns a resolved path, but callers that bypass the dep (tests, future
    middleware) must still be safe.
    """
    if not run_id or ".." in run_id.replace("\\", "/").split("/"):
        raise AttemptInvalidError(
            detail="invalid_run_id",
            context={"run_id": run_id},
        )
    runs_root = runs_dir.resolve()
    run_dir = (runs_root / run_id).resolve()
    if not run_dir.is_relative_to(runs_root):
        raise AttemptInvalidError(
            detail="run_id_outside_runs_dir",
            context={"run_id": run_id},
        )
    if not run_dir.exists() or not run_dir.is_dir():
        raise RunNotFoundError(
            detail=f"run not found: {run_id}",
            context={"run_id": run_id},
        )
    return run_dir


def get_state_store(run_dir: Path = Depends(resolve_run_dir)) -> PipelineStateStore:
    return PipelineStateStore(run_dir)


def get_project_registry(settings: ApiSettings = Depends(get_settings)) -> ProjectRegistry:
    return ProjectRegistry(settings.projects_root_resolved)


def get_provider_registry(settings: ApiSettings = Depends(get_settings)) -> ProviderRegistry:
    """Reusable provider registry. Shares the same workspace root as projects."""
    return ProviderRegistry(settings.projects_root_resolved)


def get_integration_registry(
    settings: ApiSettings = Depends(get_settings),
) -> IntegrationRegistry:
    """Reusable integration registry. Same workspace root as projects/providers."""
    return IntegrationRegistry(settings.projects_root_resolved)


@dataclass(frozen=True)
class DatasetRequestContext:
    """Resolved context for a dataset-scoped HTTP request.

    Holds the parsed pipeline config (dict — *not* a typed Pydantic
    instance, because that would require the full config to be valid;
    preview/validate endpoints must work even when other unrelated
    parts of the config are broken), the dataset key, and the parsed
    :class:`DatasetConfig` for that key.
    """

    project_id: str
    project_root: Path
    dataset_key: str
    dataset_config: "DatasetConfig"
    parsed_pipeline_config: dict[str, Any]


def resolve_dataset_key(
    project_id: str,
    dataset_key: str,
    registry: ProjectRegistry = Depends(get_project_registry),
) -> DatasetRequestContext:
    """Look up ``datasets.<dataset_key>`` in the project's current YAML
    config and return a resolved request context.

    Raises:
        404 — project not found / dataset key not present
        400 — invalid project_id or dataset_key (path traversal, empty)
        422 — dataset block can't be parsed into ``DatasetConfig``
    """
    if not project_id or "/" in project_id or "\\" in project_id or ".." in project_id:
        raise AttemptInvalidError(
            detail="invalid_project_id",
            context={"project_id": project_id},
        )
    if not dataset_key or "/" in dataset_key or "\\" in dataset_key or ".." in dataset_key:
        raise AttemptInvalidError(
            detail="invalid_dataset_key",
            context={"dataset_key": dataset_key},
        )

    try:
        entry = registry.resolve(project_id)
    except ProjectRegistryError as exc:
        raise ProjectNotFoundError(
            detail=f"project_not_found: {exc}",
            context={"project_id": project_id},
            cause=exc,
        ) from exc

    project_root = Path(entry.path)
    store = ProjectStore(project_root)
    if not store.exists():
        raise ProjectDirectoryMissingError(
            detail="project_directory_missing",
            context={"project_id": project_id},
        )

    yaml_text = store.current_yaml_text()
    try:
        parsed = yaml.safe_load(yaml_text) or {}
    except yaml.YAMLError as exc:
        raise ConfigInvalidError(
            detail=f"config_yaml_parse_error: {exc}",
            cause=exc,
        ) from exc

    if not isinstance(parsed, dict):
        raise ConfigInvalidError(detail="config_yaml_not_object")

    datasets_block = parsed.get("datasets") or {}
    if not isinstance(datasets_block, dict) or dataset_key not in datasets_block:
        raise DatasetNotFoundError(
            detail=f"dataset_not_found: {dataset_key}",
            context={"dataset_key": dataset_key},
        )

    raw = datasets_block[dataset_key]
    if not isinstance(raw, dict):
        raise ConfigInvalidError(
            detail=f"dataset_entry_not_object: {dataset_key}",
            context={"dataset_key": dataset_key},
        )

    # Local import — config schema imports validators which transitively
    # touch many subsystems; keeping the import lazy avoids slowing down
    # API boot for endpoints that don't need it.
    from ryotenkai_shared.config.datasets.schema import DatasetConfig

    try:
        dataset_config = DatasetConfig.model_validate(raw)
    except Exception as exc:  # ValidationError or any pydantic-raised
        raise ConfigInvalidError(
            detail=f"dataset_config_invalid: {exc}",
            cause=exc,
        ) from exc

    return DatasetRequestContext(
        project_id=project_id,
        project_root=project_root.resolve(),
        dataset_key=dataset_key,
        dataset_config=dataset_config,
        parsed_pipeline_config=parsed,
    )


@lru_cache(maxsize=1)
def get_token_crypto() -> TokenCrypto:
    """AES-GCM wrapper tied to the master key.

    Cached — one instance per process. The master key is auto-generated
    on first call when absent (see ``token_crypto.load_or_create_master_key``).
    """
    from ryotenkai_shared.utils.crypto.token_crypto import TokenCrypto as _TokenCrypto

    return _TokenCrypto()
