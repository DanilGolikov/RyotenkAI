"""Global integration registry — index at ``~/.ryotenkai/integrations.json``."""

from __future__ import annotations

import json
import re
from pathlib import Path

from src.pipeline._fs import atomic_write_json, utc_now_iso
from src.pipeline.settings.integrations.models import IntegrationRegistryEntry

REGISTRY_SCHEMA_VERSION = 1
_INTEGRATION_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_\-]{0,63}$")


class IntegrationRegistryError(RuntimeError):
    """Raised for recoverable registry issues."""


def default_root() -> Path:
    return Path.home() / ".ryotenkai"


def validate_integration_id(integration_id: str) -> None:
    if not _INTEGRATION_ID_RE.match(integration_id):
        raise IntegrationRegistryError(
            f"invalid integration id {integration_id!r}: must match {_INTEGRATION_ID_RE.pattern}"
        )


class IntegrationRegistry:
    """File-backed index of reusable integrations."""

    def __init__(self, root: Path | None = None):
        self.root = Path(root).expanduser().resolve() if root else default_root()

    @property
    def registry_path(self) -> Path:
        return self.root / "integrations.json"

    @property
    def integrations_dir(self) -> Path:
        return self.root / "integrations"

    def _load_raw(self) -> dict:
        if not self.registry_path.is_file():
            return {"schema_version": REGISTRY_SCHEMA_VERSION, "integrations": []}
        try:
            return json.loads(self.registry_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise IntegrationRegistryError(
                f"failed to read {self.registry_path}: {exc}"
            ) from exc

    def _save_raw(self, payload: dict) -> None:
        atomic_write_json(self.registry_path, payload)

    def list(self) -> list[IntegrationRegistryEntry]:
        raw = self._load_raw()
        return [
            IntegrationRegistryEntry(
                id=str(item["id"]),
                name=str(item.get("name", item["id"])),
                type=str(item.get("type", "")),
                path=str(item["path"]),
                created_at=str(item.get("created_at", "")),
            )
            for item in raw.get("integrations", [])
        ]

    def resolve(self, integration_id: str) -> IntegrationRegistryEntry:
        for entry in self.list():
            if entry.id == integration_id:
                return entry
        raise IntegrationRegistryError(
            f"integration {integration_id!r} not found in registry"
        )

    def register(
        self,
        *,
        integration_id: str,
        name: str,
        type: str,
        path: Path,
    ) -> IntegrationRegistryEntry:
        validate_integration_id(integration_id)
        payload = self._load_raw()
        existing_ids = {str(item["id"]) for item in payload.get("integrations", [])}
        if integration_id in existing_ids:
            raise IntegrationRegistryError(
                f"integration id {integration_id!r} already registered"
            )
        entry = IntegrationRegistryEntry(
            id=integration_id,
            name=name,
            type=type,
            path=str(Path(path).expanduser().resolve()),
            created_at=utc_now_iso(),
        )
        payload.setdefault("integrations", []).append(entry.to_dict())
        payload["schema_version"] = REGISTRY_SCHEMA_VERSION
        self._save_raw(payload)
        return entry

    def unregister(self, integration_id: str) -> bool:
        payload = self._load_raw()
        items = payload.get("integrations", [])
        remaining = [item for item in items if str(item["id"]) != integration_id]
        if len(remaining) == len(items):
            return False
        payload["integrations"] = remaining
        self._save_raw(payload)
        return True

    def default_integration_path(self, integration_id: str) -> Path:
        return self.integrations_dir / integration_id


__all__ = [
    "REGISTRY_SCHEMA_VERSION",
    "IntegrationRegistry",
    "IntegrationRegistryError",
    "default_root",
    "validate_integration_id",
]
