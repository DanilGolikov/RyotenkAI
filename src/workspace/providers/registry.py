"""Global provider registry — lightweight index at ``~/.ryotenkai/providers.json``.

Thin subclass of :class:`WorkspaceRegistry`; all I/O + validation lives
in the base. Only provider-specific bits are declared here:

* the entry dataclass shape (carries ``type`` alongside id/name/path),
* the JSON layout names (``providers.json``, ``providers``, etc.),
* the ``register()`` signature with the provider's required kwargs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from src.pipeline._workspace_registry import (
    REGISTRY_SCHEMA_VERSION,
    WorkspaceRegistry,
    WorkspaceRegistryError,
    default_workspace_root,
)
from src.pipeline._fs import utc_now_iso
from src.workspace.providers.models import ProviderRegistryEntry

if TYPE_CHECKING:
    from pathlib import Path


class ProviderRegistryError(WorkspaceRegistryError):
    """Raised for recoverable provider-registry issues."""


def validate_provider_id(provider_id: str) -> None:
    ProviderRegistry.validate_id(provider_id)


class ProviderRegistry(WorkspaceRegistry[ProviderRegistryEntry]):
    """File-backed index of reusable providers."""

    payload_key: ClassVar[str] = "providers"
    resource_subdir: ClassVar[str] = "providers"
    registry_filename: ClassVar[str] = "providers.json"
    error_class: ClassVar[type[ProviderRegistryError]] = ProviderRegistryError

    def _decode_entry(self, raw: dict[str, Any]) -> ProviderRegistryEntry:
        return ProviderRegistryEntry(
            id=str(raw["id"]),
            name=str(raw.get("name", raw["id"])),
            type=str(raw.get("type", "")),
            path=str(raw["path"]),
            created_at=str(raw.get("created_at", "")),
        )

    # ---------- Provider-specific accessors --------------------------------

    @property
    def providers_dir(self) -> Path:
        """Alias for :attr:`resources_dir` kept for legacy call sites."""
        return self.resources_dir

    def default_provider_path(self, provider_id: str) -> Path:
        return self.default_resource_path(provider_id)

    # ---------- Public API -------------------------------------------------

    def register(
        self,
        *,
        provider_id: str,
        name: str,
        type: str,  # noqa: A002 — public API field
        path: Path,
    ) -> ProviderRegistryEntry:
        self.validate_id(provider_id)
        from pathlib import Path as _Path  # local import to avoid TYPE_CHECKING dance

        entry = ProviderRegistryEntry(
            id=provider_id,
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
    "ProviderRegistry",
    "ProviderRegistryError",
    "default_root",
    "validate_provider_id",
]
