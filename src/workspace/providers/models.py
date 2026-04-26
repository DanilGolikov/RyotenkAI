"""Dataclasses for the provider workspace."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class ProviderMetadata:
    """On-disk metadata for a single provider (``<provider_dir>/provider.json``)."""

    schema_version: int
    id: str
    name: str
    type: str
    description: str
    created_at: str
    updated_at: str

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "description": self.description,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


@dataclass(slots=True)
class ProviderRegistryEntry:
    """Index entry stored in ``~/.ryotenkai/providers.json``."""

    id: str
    name: str
    type: str
    path: str
    created_at: str

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "path": self.path,
            "created_at": self.created_at,
        }


@dataclass(slots=True)
class ProviderConfigVersion:
    """A snapshot of the provider's config at a point in time."""

    filename: str
    created_at: str
    size_bytes: int
    path: Path = field(repr=False)


__all__ = [
    "ProviderConfigVersion",
    "ProviderMetadata",
    "ProviderRegistryEntry",
]
