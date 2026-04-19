"""Global provider registry — lightweight index at ``~/.ryotenkai/providers.json``.

Mirrors the ProjectRegistry pattern.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from src.pipeline._fs import atomic_write_json, utc_now_iso
from src.pipeline.settings.providers.models import ProviderRegistryEntry

REGISTRY_SCHEMA_VERSION = 1
_PROVIDER_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_\-]{0,63}$")


class ProviderRegistryError(RuntimeError):
    """Raised for recoverable registry issues."""


def default_root() -> Path:
    return Path.home() / ".ryotenkai"


def validate_provider_id(provider_id: str) -> None:
    if not _PROVIDER_ID_RE.match(provider_id):
        raise ProviderRegistryError(
            f"invalid provider id {provider_id!r}: must match {_PROVIDER_ID_RE.pattern}"
        )


class ProviderRegistry:
    """File-backed index of reusable providers."""

    def __init__(self, root: Path | None = None):
        self.root = Path(root).expanduser().resolve() if root else default_root()

    @property
    def registry_path(self) -> Path:
        return self.root / "providers.json"

    @property
    def providers_dir(self) -> Path:
        return self.root / "providers"

    # ---------- IO ---------------------------------------------------------

    def _load_raw(self) -> dict:
        if not self.registry_path.is_file():
            return {"schema_version": REGISTRY_SCHEMA_VERSION, "providers": []}
        try:
            return json.loads(self.registry_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ProviderRegistryError(f"failed to read {self.registry_path}: {exc}") from exc

    def _save_raw(self, payload: dict) -> None:
        atomic_write_json(self.registry_path, payload)

    # ---------- Public API -------------------------------------------------

    def list(self) -> list[ProviderRegistryEntry]:
        raw = self._load_raw()
        return [
            ProviderRegistryEntry(
                id=str(item["id"]),
                name=str(item.get("name", item["id"])),
                type=str(item.get("type", "")),
                path=str(item["path"]),
                created_at=str(item.get("created_at", "")),
            )
            for item in raw.get("providers", [])
        ]

    def resolve(self, provider_id: str) -> ProviderRegistryEntry:
        for entry in self.list():
            if entry.id == provider_id:
                return entry
        raise ProviderRegistryError(f"provider {provider_id!r} not found in registry")

    def register(
        self,
        *,
        provider_id: str,
        name: str,
        type: str,
        path: Path,
    ) -> ProviderRegistryEntry:
        validate_provider_id(provider_id)
        payload = self._load_raw()
        existing_ids = {str(item["id"]) for item in payload.get("providers", [])}
        if provider_id in existing_ids:
            raise ProviderRegistryError(f"provider id {provider_id!r} already registered")

        entry = ProviderRegistryEntry(
            id=provider_id,
            name=name,
            type=type,
            path=str(Path(path).expanduser().resolve()),
            created_at=utc_now_iso(),
        )
        payload.setdefault("providers", []).append(entry.to_dict())
        payload["schema_version"] = REGISTRY_SCHEMA_VERSION
        self._save_raw(payload)
        return entry

    def unregister(self, provider_id: str) -> bool:
        payload = self._load_raw()
        providers = payload.get("providers", [])
        remaining = [item for item in providers if str(item["id"]) != provider_id]
        if len(remaining) == len(providers):
            return False
        payload["providers"] = remaining
        self._save_raw(payload)
        return True

    def default_provider_path(self, provider_id: str) -> Path:
        return self.providers_dir / provider_id


__all__ = [
    "REGISTRY_SCHEMA_VERSION",
    "ProviderRegistry",
    "ProviderRegistryError",
    "default_root",
    "validate_provider_id",
]
