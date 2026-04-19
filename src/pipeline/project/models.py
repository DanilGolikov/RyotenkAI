"""Dataclasses for project workspaces."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class ProjectMetadata:
    """On-disk metadata for a single project (``<project_dir>/project.json``)."""

    schema_version: int
    id: str
    name: str
    description: str
    created_at: str
    updated_at: str

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


@dataclass(slots=True)
class ProjectRegistryEntry:
    """Index entry stored in ``~/.ryotenkai/projects.json``."""

    id: str
    name: str
    path: str
    created_at: str

    def to_dict(self) -> dict:
        return {"id": self.id, "name": self.name, "path": self.path, "created_at": self.created_at}


@dataclass(slots=True)
class ProjectConfigVersion:
    """A snapshot of the project's pipeline config at a point in time."""

    filename: str
    created_at: str
    size_bytes: int
    path: Path = field(repr=False)


__all__ = [
    "ProjectConfigVersion",
    "ProjectMetadata",
    "ProjectRegistryEntry",
]
