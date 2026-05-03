"""On-disk layout for a single provider workspace.

Layout::

    <provider_root>/
      provider.json
      current.yaml
      history/
        2026-04-19T10-30-45Z.yaml

Thin :class:`WorkspaceStore` subclass — only metadata decoding +
``create()`` are provider-specific; everything else (save/restore/list
versions, atomic writes, snapshot semantics) lives in the base.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from src.utils.atomic_fs import atomic_write_json, atomic_write_text, utc_now_iso
from src.workspace._registry_base import WorkspaceStore, WorkspaceStoreError
from src.workspace.providers.models import (
    ProviderConfigVersion,
    ProviderMetadata,
)

if TYPE_CHECKING:
    from pathlib import Path

PROVIDER_SCHEMA_VERSION = 1


class ProviderStoreError(WorkspaceStoreError):
    """Raised for recoverable provider-store issues."""


class ProviderStore(WorkspaceStore[ProviderMetadata, ProviderConfigVersion]):
    """File-backed store for a single reusable provider configuration."""

    metadata_filename: ClassVar[str] = "provider.json"
    schema_version: ClassVar[int] = PROVIDER_SCHEMA_VERSION
    error_class: ClassVar[type[ProviderStoreError]] = ProviderStoreError

    # ---------- Hooks ------------------------------------------------------

    def _decode_metadata(self, raw: dict[str, Any]) -> ProviderMetadata:
        return ProviderMetadata(
            schema_version=int(raw.get("schema_version", PROVIDER_SCHEMA_VERSION)),
            id=str(raw["id"]),
            name=str(raw.get("name", raw["id"])),
            type=str(raw.get("type", "")),
            description=str(raw.get("description", "")),
            created_at=str(raw.get("created_at", "")),
            updated_at=str(raw.get("updated_at", "")),
        )

    def _make_config_version(
        self, *, filename: str, created_at: str, size_bytes: int, path: Path
    ) -> ProviderConfigVersion:
        return ProviderConfigVersion(
            filename=filename,
            created_at=created_at,
            size_bytes=size_bytes,
            path=path,
        )

    # ---------- Lifecycle --------------------------------------------------

    def create(
        self,
        *,
        id: str,  # noqa: A002 — public API
        name: str,
        type: str,  # noqa: A002 — public API
        description: str = "",
    ) -> ProviderMetadata:
        if self.exists():
            raise self.error_class(f"provider already exists at {self.root}")

        self.root.mkdir(parents=True, exist_ok=True)
        self.history_dir.mkdir(parents=True, exist_ok=True)

        now = utc_now_iso()
        metadata = ProviderMetadata(
            schema_version=PROVIDER_SCHEMA_VERSION,
            id=id,
            name=name,
            type=type,
            description=description,
            created_at=now,
            updated_at=now,
        )
        atomic_write_json(self.metadata_path, metadata.to_dict())

        if not self.current_config_path.exists():
            atomic_write_text(self.current_config_path, "")
        return metadata


__all__ = [
    "PROVIDER_SCHEMA_VERSION",
    "ProviderStore",
    "ProviderStoreError",
]
