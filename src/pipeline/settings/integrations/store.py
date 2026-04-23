"""On-disk layout for a single integration workspace.

Layout::

    <integration_root>/
      integration.json
      current.yaml
      token.enc            # AES-GCM encrypted; managed by api.services.token_crypto
      history/
        2026-04-22T10-30-45Z.yaml

Same atomic-write + snapshot-per-save contract as ``ProviderStore``.
Token never appears in ``current.yaml`` — it lives exclusively in
``token.enc`` and is read/written via separate endpoints.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from src.pipeline._fs import (
    atomic_write_json,
    atomic_write_text,
    created_at_from_filename,
    unique_snapshot_path,
    utc_now_iso,
)
from src.pipeline.settings.integrations.models import (
    IntegrationConfigVersion,
    IntegrationMetadata,
)

INTEGRATION_SCHEMA_VERSION = 1
TOKEN_FILENAME = "token.enc"


class IntegrationStoreError(RuntimeError):
    """Raised for recoverable integration-store issues."""


class IntegrationStore:
    """File-backed store for a single reusable integration configuration."""

    def __init__(self, root: Path):
        self.root = Path(root).expanduser().resolve()

    @property
    def metadata_path(self) -> Path:
        return self.root / "integration.json"

    @property
    def current_config_path(self) -> Path:
        return self.root / "current.yaml"

    @property
    def history_dir(self) -> Path:
        return self.root / "history"

    @property
    def token_path(self) -> Path:
        return self.root / TOKEN_FILENAME

    def exists(self) -> bool:
        return self.metadata_path.is_file()

    def has_token(self) -> bool:
        return self.token_path.is_file()

    def create(
        self,
        *,
        id: str,
        name: str,
        type: str,
        description: str = "",
    ) -> IntegrationMetadata:
        if self.exists():
            raise IntegrationStoreError(f"integration already exists at {self.root}")

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

    def load(self) -> IntegrationMetadata:
        if not self.exists():
            raise IntegrationStoreError(f"integration not found at {self.root}")
        raw = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        return IntegrationMetadata(
            schema_version=int(raw.get("schema_version", INTEGRATION_SCHEMA_VERSION)),
            id=str(raw["id"]),
            name=str(raw.get("name", raw["id"])),
            type=str(raw.get("type", "")),
            description=str(raw.get("description", "")),
            created_at=str(raw.get("created_at", "")),
            updated_at=str(raw.get("updated_at", "")),
        )

    def touch(self) -> None:
        if not self.exists():
            return
        metadata = self.load()
        metadata.updated_at = utc_now_iso()
        atomic_write_json(self.metadata_path, metadata.to_dict())

    def current_yaml_text(self) -> str:
        if self.current_config_path.is_file():
            return self.current_config_path.read_text(encoding="utf-8")
        return ""

    def save_config(self, yaml_text: str) -> str | None:
        self.history_dir.mkdir(parents=True, exist_ok=True)
        snapshot_name: str | None = None
        if self.current_config_path.is_file():
            previous = self.current_config_path.read_text(encoding="utf-8")
            if previous:
                snapshot_path = unique_snapshot_path(self.history_dir)
                atomic_write_text(snapshot_path, previous)
                snapshot_name = snapshot_path.name
        atomic_write_text(self.current_config_path, yaml_text)
        self.touch()
        return snapshot_name

    def list_versions(self) -> list[IntegrationConfigVersion]:
        if not self.history_dir.is_dir():
            return []
        entries: list[IntegrationConfigVersion] = []
        for path in sorted(self.history_dir.glob("*.yaml")):
            stat = path.stat()
            entries.append(
                IntegrationConfigVersion(
                    filename=path.name,
                    created_at=created_at_from_filename(path.name),
                    size_bytes=stat.st_size,
                    path=path,
                )
            )
        entries.sort(key=lambda v: v.filename, reverse=True)
        return entries

    def read_version(self, filename: str) -> str:
        if "/" in filename or ".." in filename:
            raise IntegrationStoreError(f"invalid version filename: {filename!r}")
        path = self.history_dir / filename
        if not path.is_file():
            raise IntegrationStoreError(f"version not found: {filename}")
        return path.read_text(encoding="utf-8")

    def restore_version(self, filename: str) -> str | None:
        content = self.read_version(filename)
        return self.save_config(content)

    def remove(self) -> None:
        if self.root.is_dir():
            shutil.rmtree(self.root)


__all__ = [
    "INTEGRATION_SCHEMA_VERSION",
    "TOKEN_FILENAME",
    "IntegrationStore",
    "IntegrationStoreError",
]
