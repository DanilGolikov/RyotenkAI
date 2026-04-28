"""On-disk layout for a single project.

Layout::

    <project_root>/
      project.json
      configs/
        current.yaml
        history/
          2026-04-19T10-30-45Z.yaml
      env.json                 # optional env-var overrides
      runs/
        index.json             # append-only ledger of launches (Step 6)

Project layout is richer than Provider/Integration: configs live in a
``configs/`` sub-directory, projects own ``runs/`` + ``env.json``, and
:meth:`save_config` seeds a ``v1`` snapshot on the very first save so
the Versions tab is never empty after a user's first Save click.

The shared :class:`WorkspaceStore` base handles atomic writes,
metadata I/O, version listing and restore. ProjectStore overrides
``current_config_path`` / ``history_dir`` to point at the ``configs/``
sub-directory and adds env-vars + favorite-versions + runs-index APIs.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, ClassVar

from src.utils.atomic_fs import (
    atomic_write_json,
    atomic_write_text,
    unique_snapshot_path,
    utc_now_iso,
)
from src.workspace._registry_base import WorkspaceStore, WorkspaceStoreError
from src.workspace.projects.models import ProjectConfigVersion, ProjectMetadata

if TYPE_CHECKING:
    from pathlib import Path

PROJECT_SCHEMA_VERSION = 1
RUNS_INDEX_SCHEMA_VERSION = 1
RUNS_INDEX_FILENAME = "index.json"


class ProjectStoreError(WorkspaceStoreError):
    """Raised for recoverable project-store issues."""


class ProjectStore(WorkspaceStore[ProjectMetadata, ProjectConfigVersion]):
    """File-backed store for a single project workspace."""

    metadata_filename: ClassVar[str] = "project.json"
    schema_version: ClassVar[int] = PROJECT_SCHEMA_VERSION
    error_class: ClassVar[type[ProjectStoreError]] = ProjectStoreError

    # ---------- Layout overrides ------------------------------------------

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

    @property
    def env_path(self) -> Path:
        return self.root / "env.json"

    # ---------- Hooks ------------------------------------------------------

    def _decode_metadata(self, raw: dict[str, Any]) -> ProjectMetadata:
        favorites = raw.get("favorite_versions") or []
        return ProjectMetadata(
            schema_version=int(raw.get("schema_version", PROJECT_SCHEMA_VERSION)),
            id=str(raw["id"]),
            name=str(raw.get("name", raw["id"])),
            description=str(raw.get("description", "")),
            created_at=str(raw.get("created_at", "")),
            updated_at=str(raw.get("updated_at", "")),
            favorite_versions=[str(v) for v in favorites if isinstance(v, str)],
        )

    def _make_config_version(
        self, *, filename: str, created_at: str, size_bytes: int, path: Path
    ) -> ProjectConfigVersion:
        return ProjectConfigVersion(
            filename=filename,
            created_at=created_at,
            size_bytes=size_bytes,
            path=path,
        )

    # ---------- Env vars (project-specific) -------------------------------

    def read_env(self) -> dict[str, str]:
        """Return the project's env-var overrides, or {} when unset."""
        if not self.env_path.is_file():
            return {}
        try:
            with self.env_path.open("r", encoding="utf-8") as fh:
                raw = json.load(fh)
        except Exception as exc:
            raise self.error_class(f"failed to read {self.env_path}: {exc}") from exc
        if not isinstance(raw, dict):
            raise self.error_class(
                f"{self.env_path} must be a JSON object, got {type(raw).__name__}",
            )
        return {str(k): str(v) for k, v in raw.items()}

    def write_env(self, env: dict[str, str]) -> None:
        atomic_write_json(self.env_path, {k: v for k, v in env.items() if v != ""})

    # ---------- Lifecycle --------------------------------------------------

    def create(
        self,
        *,
        id: str,
        name: str,
        description: str = "",
    ) -> ProjectMetadata:
        if self.exists():
            raise self.error_class(f"project already exists at {self.root}")

        self.root.mkdir(parents=True, exist_ok=True)
        self.configs_dir.mkdir(parents=True, exist_ok=True)
        self.history_dir.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(parents=True, exist_ok=True)

        now = utc_now_iso()
        metadata = ProjectMetadata(
            schema_version=PROJECT_SCHEMA_VERSION,
            id=id,
            name=name,
            description=description,
            created_at=now,
            updated_at=now,
        )
        atomic_write_json(self.metadata_path, metadata.to_dict())

        # Seed empty config so the first GET has something to hand back.
        if not self.current_config_path.exists():
            atomic_write_text(self.current_config_path, "")
        return metadata

    def update_description(self, description: str) -> ProjectMetadata:
        """Patch ``description`` + ``updated_at`` in metadata, leaving every
        other identity field (id, name, created_at, favorites) untouched.
        Returns the fresh metadata so callers can echo it back."""
        if not self.exists():
            raise self.error_class(f"project not found at {self.root}")
        metadata = self.load()
        metadata.description = description
        metadata.updated_at = utc_now_iso()
        atomic_write_json(self.metadata_path, metadata.to_dict())
        return metadata

    # ---------- Config -----------------------------------------------------

    def save_config(self, yaml_text: str) -> str | None:
        """Write ``yaml_text`` to ``configs/current.yaml`` atomically.

        If a previous ``current.yaml`` existed, its contents are first copied
        into ``configs/history/<iso>.yaml``. On the FIRST save (no prior
        content, no history yet) we snapshot the new content instead, so
        the user sees a v1 in the versions list immediately.
        """
        self.configs_dir.mkdir(parents=True, exist_ok=True)
        self.history_dir.mkdir(parents=True, exist_ok=True)

        snapshot_name: str | None = None
        previous: str | None = None
        if self.current_config_path.is_file():
            previous = self.current_config_path.read_text(encoding="utf-8")

        if previous:
            snapshot_path = unique_snapshot_path(self.history_dir)
            atomic_write_text(snapshot_path, previous)
            snapshot_name = snapshot_path.name
        elif not any(self.history_dir.iterdir()):
            # First save on an empty project — seed v1 from the incoming
            # content so the Versions tab isn't empty after the user's
            # first Save click.
            snapshot_path = unique_snapshot_path(self.history_dir)
            atomic_write_text(snapshot_path, yaml_text)
            snapshot_name = snapshot_path.name

        atomic_write_text(self.current_config_path, yaml_text)
        self.touch()
        return snapshot_name

    # ---------- Favorites --------------------------------------------------

    def toggle_favorite_version(self, filename: str, *, pinned: bool) -> list[str]:
        """Add / remove ``filename`` from the project's favorite versions.

        Raises ``ProjectStoreError`` if the snapshot does not exist.
        Returns the updated list.
        """
        if not self.exists():
            raise self.error_class(f"project not found at {self.root}")
        if "/" in filename or ".." in filename:
            raise self.error_class(f"invalid version filename: {filename!r}")
        if not (self.history_dir / filename).is_file():
            raise self.error_class(f"version not found: {filename}")
        metadata = self.load()
        favs = list(metadata.favorite_versions)
        if pinned:
            if filename not in favs:
                favs.append(filename)
        else:
            favs = [f for f in favs if f != filename]
        metadata.favorite_versions = favs
        metadata.updated_at = utc_now_iso()
        atomic_write_json(self.metadata_path, metadata.to_dict())
        return favs

    # ---------- Runs index (Step 6) ----------------------------------------

    @property
    def runs_index_path(self) -> Path:
        return self.runs_dir / RUNS_INDEX_FILENAME

    def list_runs(
        self,
        *,
        status: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return the project's launched-run entries, newest first.

        ``runs/index.json`` is an append-only ledger written by
        :meth:`register_run`. Entries are filtered by ``status`` if given
        (e.g. ``"running"`` / ``"completed"`` / ``"failed"`` — values
        match what ``register_run`` was called with) and capped by
        ``limit`` after the sort.

        Returns ``[]`` when the index doesn't exist yet (project has
        never been launched). Malformed entries are dropped, never
        raised — surface stays usable even when one entry is corrupt.
        """
        payload = self._read_runs_index()
        entries = list(payload.get("runs", []))

        # Filter on status before sort + limit so ``limit`` is the
        # number of *matching* entries the caller wants, not "look at
        # last N then filter".
        if status is not None:
            entries = [e for e in entries if e.get("status") == status]
        # Newest first by ``started_at`` ISO string (sorts lexico-
        # graphically when both are UTC ISO with leading zeros, which
        # ``utc_now_iso`` guarantees).
        entries.sort(key=lambda e: str(e.get("started_at", "")), reverse=True)
        if limit is not None and limit >= 0:
            entries = entries[:limit]
        return entries

    def register_run(
        self,
        *,
        run_id: str,
        started_at: str | None = None,
        status: str = "running",
        mlflow_run_id: str | None = None,
        config_version_hash: str | None = None,
        actor: str | None = None,
        run_directory: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Append a run entry to ``runs/index.json``.

        Append-only by design (R16 from the plan): the index is a
        ledger, not a state store. Status updates after the fact go
        through :meth:`update_run_status` which writes a new revision
        of the same ``run_id`` entry rather than mutating the original
        in place — keeping the audit trail intact.

        Atomic write semantics — the file is read, mutated in memory,
        then written through ``atomic_write_json`` (tmp + rename). On
        the rare interleaved write (two processes register at the
        exact same instant) the loser's entry can be lost; callers
        for whom that matters should add ``fcntl.flock`` outside this
        method. For the typical "one process per run launch" pattern
        the atomic rename is sufficient.

        Returns the entry as written (with all fields populated).
        """
        entry: dict[str, Any] = {
            "run_id": run_id,
            "started_at": started_at or utc_now_iso(),
            "status": status,
        }
        if mlflow_run_id is not None:
            entry["mlflow_run_id"] = mlflow_run_id
        if config_version_hash is not None:
            entry["config_version_hash"] = config_version_hash
        if actor is not None:
            entry["actor"] = actor
        if run_directory is not None:
            entry["run_directory"] = run_directory
        if extra:
            # Caller-supplied extras flow through unchanged but never
            # override the canonical keys above (which the UI relies on).
            for k, v in extra.items():
                entry.setdefault(k, v)

        payload = self._read_runs_index()
        runs = list(payload.get("runs", []))
        runs.append(entry)
        payload["schema_version"] = RUNS_INDEX_SCHEMA_VERSION
        payload["runs"] = runs

        self.runs_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(self.runs_index_path, payload)
        return entry

    def update_run_status(
        self,
        run_id: str,
        *,
        status: str,
        finished_at: str | None = None,
    ) -> dict[str, Any] | None:
        """Mark a previously-registered run with a terminal status.

        Updates the *most recent* entry whose ``run_id`` matches —
        callers append once at launch time then update once at
        completion. Returns the updated entry, or ``None`` when no
        matching entry exists (silent no-op for resilience: a run
        finishing whose register-call was lost shouldn't crash the
        finisher).
        """
        payload = self._read_runs_index()
        runs = list(payload.get("runs", []))
        for entry in reversed(runs):
            if entry.get("run_id") != run_id:
                continue
            entry["status"] = status
            if finished_at is not None:
                entry["finished_at"] = finished_at
            payload["runs"] = runs
            payload["schema_version"] = RUNS_INDEX_SCHEMA_VERSION
            self.runs_dir.mkdir(parents=True, exist_ok=True)
            atomic_write_json(self.runs_index_path, payload)
            return entry
        return None

    def _read_runs_index(self) -> dict[str, Any]:
        """Read ``runs/index.json``; return ``{}`` when missing.

        Defensive: schema-version mismatch and malformed JSON both
        return ``{}`` rather than raising — the index is a UX nicety,
        not authoritative state. Authoritative run state still lives
        under ``~/.ryotenkai/runs/<run_id>/pipeline_state.json``.
        """
        if not self.runs_index_path.is_file():
            return {}
        try:
            with self.runs_index_path.open("r", encoding="utf-8") as fh:
                raw = json.load(fh)
        except (OSError, json.JSONDecodeError):
            return {}
        if not isinstance(raw, dict):
            return {}
        return raw


__all__ = [
    "PROJECT_SCHEMA_VERSION",
    "RUNS_INDEX_FILENAME",
    "RUNS_INDEX_SCHEMA_VERSION",
    "ProjectStore",
    "ProjectStoreError",
]
