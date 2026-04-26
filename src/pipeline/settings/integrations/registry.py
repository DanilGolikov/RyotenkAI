"""Global integration registry — index at ``~/.ryotenkai/integrations.json``."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from src.pipeline._fs import utc_now_iso
from src.pipeline._workspace_registry import (
    REGISTRY_SCHEMA_VERSION,
    WorkspaceRegistry,
    WorkspaceRegistryError,
    default_workspace_root,
)
from src.pipeline.settings.integrations.models import IntegrationRegistryEntry

if TYPE_CHECKING:
    from pathlib import Path


class IntegrationRegistryError(WorkspaceRegistryError):
    """Raised for recoverable integration-registry issues."""


def validate_integration_id(integration_id: str) -> None:
    IntegrationRegistry.validate_id(integration_id)


class IntegrationRegistry(WorkspaceRegistry[IntegrationRegistryEntry]):
    """File-backed index of reusable integrations."""

    payload_key: ClassVar[str] = "integrations"
    resource_subdir: ClassVar[str] = "integrations"
    registry_filename: ClassVar[str] = "integrations.json"
    error_class: ClassVar[type[IntegrationRegistryError]] = IntegrationRegistryError

    def _decode_entry(self, raw: dict[str, Any]) -> IntegrationRegistryEntry:
        return IntegrationRegistryEntry(
            id=str(raw["id"]),
            name=str(raw.get("name", raw["id"])),
            type=str(raw.get("type", "")),
            path=str(raw["path"]),
            created_at=str(raw.get("created_at", "")),
        )

    # ---------- Integration-specific accessors -----------------------------

    @property
    def integrations_dir(self) -> Path:
        """Alias for :attr:`resources_dir` kept for legacy call sites."""
        return self.resources_dir

    def default_integration_path(self, integration_id: str) -> Path:
        return self.default_resource_path(integration_id)

    # ---------- Public API -------------------------------------------------

    def register(
        self,
        *,
        integration_id: str,
        name: str,
        type: str,  # noqa: A002 — public API
        path: Path,
    ) -> IntegrationRegistryEntry:
        self.validate_id(integration_id)
        from pathlib import Path as _Path

        entry = IntegrationRegistryEntry(
            id=integration_id,
            name=name,
            type=type,
            path=str(_Path(path).expanduser().resolve()),
            created_at=utc_now_iso(),
        )
        return self._append_entry(entry)


def default_root() -> Path:
    """Backward-compat alias for ``default_workspace_root``."""
    return default_workspace_root()


__all__ = [
    "REGISTRY_SCHEMA_VERSION",
    "IntegrationRegistry",
    "IntegrationRegistryError",
    "default_root",
    "validate_integration_id",
]
