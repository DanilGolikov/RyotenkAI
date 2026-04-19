"""On-disk layout for a single project.

Layout::

    <project_root>/
      project.json
      configs/
        current.yaml
        history/
          2026-04-19T10-30-45Z.yaml
      runs/

Writes are atomic (temp-file replace) and each ``save_config`` call snapshots
the previous ``current.yaml`` into ``configs/history/`` before overwriting.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from src.pipeline.project.models import ProjectConfigVersion, ProjectMetadata

PROJECT_SCHEMA_VERSION = 1


class ProjectStoreError(RuntimeError):
    """Raised for recoverable project-store issues."""


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _snapshot_filename(ts: str | None = None) -> str:
    # colons are awkward on macOS + Windows filesystems, so use dashes.
    return (ts or _utc_now_iso()).replace(":", "-") + ".yaml"


def _created_at_from_filename(filename: str) -> str:
    """Recover an ISO-8601 timestamp from ``<YYYY-MM-DDTHH-MM-SSZ[-N]>.yaml``."""
    stem = Path(filename).stem
    # Trim optional "-N" collision suffix (e.g. "...Z-2").
    if "Z-" in stem:
        stem = stem.split("Z-")[0] + "Z"
    # Reverse the dash→colon substitution done at write time.
    if "T" in stem:
        date_part, _, time_part = stem.partition("T")
        time_iso = time_part.replace("-", ":", 2)
        return f"{date_part}T{time_iso}"
    return stem


def _unique_snapshot_path(history_dir: Path) -> Path:
    """Return a snapshot path that does not yet exist in ``history_dir``.

    Resolves same-second collisions by appending ``-N`` before the extension
    (e.g. ``2026-04-19T10-45-50Z-1.yaml``).
    """
    base = _snapshot_filename()
    candidate = history_dir / base
    if not candidate.exists():
        return candidate
    stem = candidate.stem
    for n in range(1, 10_000):
        alt = history_dir / f"{stem}-{n}.yaml"
        if not alt.exists():
            return alt
    raise RuntimeError("exhausted snapshot filename attempts")


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", delete=False, dir=path.parent, encoding="utf-8", newline="\n"
    ) as tmp:
        tmp.write(text)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_name = tmp.name
    Path(tmp_name).replace(path)


def _atomic_write_json(path: Path, payload: dict) -> None:
    _atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


class ProjectStore:
    """File-backed store for a single project workspace."""

    def __init__(self, root: Path):
        self.root = Path(root).expanduser().resolve()

    # ---------- Paths ------------------------------------------------------

    @property
    def metadata_path(self) -> Path:
        return self.root / "project.json"

    @property
    def configs_dir(self) -> Path:
        return self.root / "configs"

    @property
    def current_config_path(self) -> Path:
        return self.configs_dir / "current.yaml"

    @property
    def history_dir(self) -> Path:
        return self.configs_dir / "history"

    @property
    def runs_dir(self) -> Path:
        return self.root / "runs"

    # ---------- Lifecycle --------------------------------------------------

    def exists(self) -> bool:
        return self.metadata_path.is_file()

    def create(self, *, id: str, name: str, description: str = "") -> ProjectMetadata:
        if self.exists():
            raise ProjectStoreError(f"project already exists at {self.root}")

        self.root.mkdir(parents=True, exist_ok=True)
        self.configs_dir.mkdir(parents=True, exist_ok=True)
        self.history_dir.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(parents=True, exist_ok=True)

        now = _utc_now_iso()
        metadata = ProjectMetadata(
            schema_version=PROJECT_SCHEMA_VERSION,
            id=id,
            name=name,
            description=description,
            created_at=now,
            updated_at=now,
        )
        _atomic_write_json(self.metadata_path, metadata.to_dict())

        # Seed empty config so the first GET has something to hand back.
        if not self.current_config_path.exists():
            _atomic_write_text(self.current_config_path, "")
        return metadata

    def load(self) -> ProjectMetadata:
        if not self.exists():
            raise ProjectStoreError(f"project not found at {self.root}")
        raw = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        return ProjectMetadata(
            schema_version=int(raw.get("schema_version", PROJECT_SCHEMA_VERSION)),
            id=str(raw["id"]),
            name=str(raw.get("name", raw["id"])),
            description=str(raw.get("description", "")),
            created_at=str(raw.get("created_at", "")),
            updated_at=str(raw.get("updated_at", "")),
        )

    def touch(self) -> None:
        """Bump ``updated_at`` in metadata."""
        if not self.exists():
            return
        metadata = self.load()
        metadata.updated_at = _utc_now_iso()
        _atomic_write_json(self.metadata_path, metadata.to_dict())

    # ---------- Config ------------------------------------------------------

    def current_yaml_text(self) -> str:
        if self.current_config_path.is_file():
            return self.current_config_path.read_text(encoding="utf-8")
        return ""

    def save_config(self, yaml_text: str) -> str | None:
        """Write ``yaml_text`` to ``configs/current.yaml`` atomically.

        If a previous ``current.yaml`` existed, its contents are first copied
        into ``configs/history/<iso>.yaml``. Returns the filename of the
        snapshot (or ``None`` if there was no prior config).
        """
        self.configs_dir.mkdir(parents=True, exist_ok=True)
        self.history_dir.mkdir(parents=True, exist_ok=True)

        snapshot_name: str | None = None
        if self.current_config_path.is_file():
            previous = self.current_config_path.read_text(encoding="utf-8")
            if previous:
                snapshot_path = _unique_snapshot_path(self.history_dir)
                _atomic_write_text(snapshot_path, previous)
                snapshot_name = snapshot_path.name

        _atomic_write_text(self.current_config_path, yaml_text)
        self.touch()
        return snapshot_name

    def list_versions(self) -> list[ProjectConfigVersion]:
        if not self.history_dir.is_dir():
            return []
        entries: list[ProjectConfigVersion] = []
        for path in sorted(self.history_dir.glob("*.yaml")):
            stat = path.stat()
            entries.append(
                ProjectConfigVersion(
                    filename=path.name,
                    created_at=_created_at_from_filename(path.name),
                    size_bytes=stat.st_size,
                    path=path,
                )
            )
        # Newest first
        entries.sort(key=lambda v: v.filename, reverse=True)
        return entries

    def read_version(self, filename: str) -> str:
        if "/" in filename or ".." in filename:
            raise ProjectStoreError(f"invalid version filename: {filename!r}")
        path = self.history_dir / filename
        if not path.is_file():
            raise ProjectStoreError(f"version not found: {filename}")
        return path.read_text(encoding="utf-8")

    def restore_version(self, filename: str) -> str | None:
        """Copy a history snapshot into ``current.yaml``.

        First snapshots the current config, then overwrites. Returns the
        filename of the new snapshot (or ``None`` if current was empty).
        """
        content = self.read_version(filename)
        return self.save_config(content)

    # ---------- Cleanup -----------------------------------------------------

    def remove(self) -> None:
        """Delete the entire project directory. Use with caution."""
        if self.root.is_dir():
            shutil.rmtree(self.root)


__all__ = ["PROJECT_SCHEMA_VERSION", "ProjectStore", "ProjectStoreError"]
