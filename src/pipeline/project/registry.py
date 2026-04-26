"""Global project registry — lightweight index of known projects on disk.

Stored at ``<root>/projects.json`` (default root: ``~/.ryotenkai/``).
Each entry points to a project directory; actual project state lives inside
that directory.

Thin :class:`WorkspaceRegistry` subclass — same shape as Provider /
Integration registries, just without the ``type`` field on entries.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from src.pipeline._fs import utc_now_iso
from src.pipeline._workspace_registry import (
    REGISTRY_SCHEMA_VERSION,
    WorkspaceRegistry,
    WorkspaceRegistryError,
    default_workspace_root,
)
from src.pipeline.project.models import ProjectRegistryEntry

if TYPE_CHECKING:
    from pathlib import Path


class ProjectRegistryError(WorkspaceRegistryError):
    """Raised for recoverable project-registry issues."""


def validate_project_id(project_id: str) -> None:
    ProjectRegistry.validate_id(project_id)


class ProjectRegistry(WorkspaceRegistry[ProjectRegistryEntry]):
    """File-backed index of projects.

    The registry intentionally carries minimal metadata (id, name, path,
    created_at). Authoritative project data lives in each project's
    ``project.json``; the registry is just a lookup table.
    """

    payload_key: ClassVar[str] = "projects"
    resource_subdir: ClassVar[str] = "projects"
    registry_filename: ClassVar[str] = "projects.json"
    error_class: ClassVar[type[ProjectRegistryError]] = ProjectRegistryError

    def _decode_entry(self, raw: dict[str, Any]) -> ProjectRegistryEntry:
        return ProjectRegistryEntry(
            id=str(raw["id"]),
            name=str(raw.get("name", raw["id"])),
            path=str(raw["path"]),
            created_at=str(raw.get("created_at", "")),
        )

    # ---------- Project-specific accessors ---------------------------------

    @property
    def projects_dir(self) -> Path:
        """Alias for :attr:`resources_dir` kept for legacy call sites."""
        return self.resources_dir

    def default_project_path(self, project_id: str) -> Path:
        return self.default_resource_path(project_id)

    # ---------- Public API -------------------------------------------------

    def register(
        self,
        *,
        project_id: str,
        name: str,
        path: Path,
    ) -> ProjectRegistryEntry:
        self.validate_id(project_id)
        from pathlib import Path as _Path

        entry = ProjectRegistryEntry(
            id=project_id,
            name=name,
            path=str(_Path(path).expanduser().resolve()),
            created_at=utc_now_iso(),
        )
        return self._append_entry(entry)


def default_root() -> Path:
    """Backward-compat alias for ``default_workspace_root``."""
    return default_workspace_root()


__all__ = [
    "REGISTRY_SCHEMA_VERSION",
    "ProjectRegistry",
    "ProjectRegistryError",
    "default_root",
    "validate_project_id",
]
