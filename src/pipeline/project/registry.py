"""Global project registry — lightweight index of known projects on disk.

Stored at ``<root>/projects.json`` (default root: ``~/.ryotenkai/``).
Each entry points to a project directory; actual project state lives inside
that directory.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from src.pipeline.project.models import ProjectRegistryEntry

REGISTRY_SCHEMA_VERSION = 1
_PROJECT_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_\-]{0,63}$")


class ProjectRegistryError(RuntimeError):
    """Raised for recoverable registry issues."""


def default_root() -> Path:
    return Path.home() / ".ryotenkai"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", delete=False, dir=path.parent, encoding="utf-8", newline="\n"
    ) as tmp:
        json.dump(payload, tmp, ensure_ascii=False, indent=2, sort_keys=True)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_name = tmp.name
    Path(tmp_name).replace(path)


def validate_project_id(project_id: str) -> None:
    if not _PROJECT_ID_RE.match(project_id):
        raise ProjectRegistryError(
            f"invalid project id {project_id!r}: must match {_PROJECT_ID_RE.pattern}"
        )


class ProjectRegistry:
    """File-backed index of projects.

    The registry intentionally carries minimal metadata (id, name, path,
    created_at). Authoritative project data lives in each project's
    ``project.json``; the registry is just a lookup table.
    """

    def __init__(self, root: Path | None = None):
        self.root = Path(root).expanduser().resolve() if root else default_root()

    @property
    def registry_path(self) -> Path:
        return self.root / "projects.json"

    @property
    def projects_dir(self) -> Path:
        return self.root / "projects"

    # ---------- IO ---------------------------------------------------------

    def _load_raw(self) -> dict:
        if not self.registry_path.is_file():
            return {"schema_version": REGISTRY_SCHEMA_VERSION, "projects": []}
        try:
            return json.loads(self.registry_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ProjectRegistryError(f"failed to read {self.registry_path}: {exc}") from exc

    def _save_raw(self, payload: dict) -> None:
        _atomic_write_json(self.registry_path, payload)

    # ---------- Public API -------------------------------------------------

    def list(self) -> list[ProjectRegistryEntry]:
        raw = self._load_raw()
        return [
            ProjectRegistryEntry(
                id=str(item["id"]),
                name=str(item.get("name", item["id"])),
                path=str(item["path"]),
                created_at=str(item.get("created_at", "")),
            )
            for item in raw.get("projects", [])
        ]

    def resolve(self, project_id: str) -> ProjectRegistryEntry:
        for entry in self.list():
            if entry.id == project_id:
                return entry
        raise ProjectRegistryError(f"project {project_id!r} not found in registry")

    def register(
        self,
        *,
        project_id: str,
        name: str,
        path: Path,
    ) -> ProjectRegistryEntry:
        validate_project_id(project_id)
        payload = self._load_raw()
        existing_ids = {str(item["id"]) for item in payload.get("projects", [])}
        if project_id in existing_ids:
            raise ProjectRegistryError(f"project id {project_id!r} already registered")

        entry = ProjectRegistryEntry(
            id=project_id,
            name=name,
            path=str(Path(path).expanduser().resolve()),
            created_at=_utc_now_iso(),
        )
        payload.setdefault("projects", []).append(entry.to_dict())
        payload["schema_version"] = REGISTRY_SCHEMA_VERSION
        self._save_raw(payload)
        return entry

    def unregister(self, project_id: str) -> bool:
        payload = self._load_raw()
        projects = payload.get("projects", [])
        remaining = [item for item in projects if str(item["id"]) != project_id]
        if len(remaining) == len(projects):
            return False
        payload["projects"] = remaining
        self._save_raw(payload)
        return True

    def default_project_path(self, project_id: str) -> Path:
        return self.projects_dir / project_id


__all__ = [
    "REGISTRY_SCHEMA_VERSION",
    "ProjectRegistry",
    "ProjectRegistryError",
    "default_root",
    "validate_project_id",
]
