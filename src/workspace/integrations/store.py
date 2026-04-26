"""On-disk layout for a single integration workspace.

Layout::

    <integration_root>/
      integration.json
      current.yaml
      token.enc            # AES-GCM encrypted; managed by api.services.token_crypto
      history/
        2026-04-22T10-30-45Z.yaml

Subclass of :class:`WorkspaceStore` that adds two integration-specific
extras on top of the shared lifecycle:

* ``token_path`` / :meth:`has_token` — encrypted token blob lives next
  to the metadata; never appears in ``current.yaml``.
* ``create()`` carries the integration's ``type`` field through to
  metadata.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from src.pipeline._fs import atomic_write_json, atomic_write_text, utc_now_iso
from src.pipeline._workspace_registry import WorkspaceStore, WorkspaceStoreError
from src.workspace.integrations.models import (
    IntegrationConfigVersion,
    IntegrationMetadata,
)

if TYPE_CHECKING:
    from pathlib import Path

INTEGRATION_SCHEMA_VERSION = 1
TOKEN_FILENAME = "token.enc"


class IntegrationStoreError(WorkspaceStoreError):
    """Raised for recoverable integration-store issues."""


class IntegrationStore(WorkspaceStore[IntegrationMetadata, IntegrationConfigVersion]):
    """File-backed store for a single reusable integration configuration."""

    metadata_filename: ClassVar[str] = "integration.json"
    schema_version: ClassVar[int] = INTEGRATION_SCHEMA_VERSION
    error_class: ClassVar[type[IntegrationStoreError]] = IntegrationStoreError

    # ---------- Hooks ------------------------------------------------------

    def _decode_metadata(self, raw: dict[str, Any]) -> IntegrationMetadata:
        return IntegrationMetadata(
            schema_version=int(raw.get("schema_version", INTEGRATION_SCHEMA_VERSION)),
            id=str(raw["id"]),
            name=str(raw.get("name", raw["id"])),
            type=str(raw.get("type", "")),
            description=str(raw.get("description", "")),
            created_at=str(raw.get("created_at", "")),
            updated_at=str(raw.get("updated_at", "")),
        )

    def _make_config_version(
        self, *, filename: str, created_at: str, size_bytes: int, path: Path
    ) -> IntegrationConfigVersion:
        return IntegrationConfigVersion(
            filename=filename,
            created_at=created_at,
            size_bytes=size_bytes,
            path=path,
        )

    # ---------- Token storage (integration-specific) -----------------------

    @property
    def token_path(self) -> Path:
        return self.root / TOKEN_FILENAME

    def has_token(self) -> bool:
        return self.token_path.is_file()

    # ---------- Lifecycle --------------------------------------------------

    def create(
        self,
        *,
        id: str,  # noqa: A002 — public API
        name: str,
        type: str,  # noqa: A002 — public API
        description: str = "",
    ) -> IntegrationMetadata:
        if self.exists():
            raise self.error_class(f"integration already exists at {self.root}")

        self.root.mkdir(parents=True, exist_ok=True)
        self.history_dir.mkdir(parents=True, exist_ok=True)

        now = utc_now_iso()
        metadata = IntegrationMetadata(
            schema_version=INTEGRATION_SCHEMA_VERSION,
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
    "INTEGRATION_SCHEMA_VERSION",
    "TOKEN_FILENAME",
    "IntegrationStore",
    "IntegrationStoreError",
]
