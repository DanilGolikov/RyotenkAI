from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import Depends, HTTPException

from src.api.config import ApiSettings
from src.pipeline.project import ProjectRegistry
from src.pipeline.settings.integrations import IntegrationRegistry
from src.pipeline.settings.providers import ProviderRegistry
from src.pipeline.state import PipelineStateStore

if TYPE_CHECKING:
    from src.api.services.token_crypto import TokenCrypto


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
        raise HTTPException(status_code=400, detail="invalid_run_id")
    runs_root = runs_dir.resolve()
    run_dir = (runs_root / run_id).resolve()
    if not run_dir.is_relative_to(runs_root):
        raise HTTPException(status_code=400, detail="run_id_outside_runs_dir")
    if not run_dir.exists() or not run_dir.is_dir():
        raise HTTPException(status_code=404, detail="run_not_found")
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


@lru_cache(maxsize=1)
def get_token_crypto() -> TokenCrypto:
    """AES-GCM wrapper tied to the master key.

    Cached — one instance per process. The master key is auto-generated
    on first call when absent (see ``token_crypto.load_or_create_master_key``).
    """
    from src.api.services.token_crypto import TokenCrypto as _TokenCrypto

    return _TokenCrypto()
