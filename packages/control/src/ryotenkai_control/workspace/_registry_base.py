"""Generic base classes for workspace-scoped registries + stores.

Three near-identical pairs of (registry, store) live in the workspace
umbrella: ``projects/``, ``providers/``, ``integrations/``. Each pair
manages:

* a JSON index file at ``<root>/<resource>.json`` listing entries by id;
* a per-entry directory at ``<root>/<resource>/<id>/`` holding metadata
  (``<resource>.json``) + ``current.yaml`` + ``history/<iso>.yaml``
  snapshots.

Without this module the I/O wrapping was ~400 LOC of triplicated
boilerplate. Two generic bases capture the shape:

* :class:`WorkspaceRegistry` — id index, validate / list / resolve /
  register / unregister.
* :class:`WorkspaceStore` — per-entry directory: metadata I/O,
  current-yaml read/write with snapshot-on-save, version listing,
  restore.

Subclasses declare what's specific (entry dataclass, payload key,
resource name, schema version, metadata layout) via classvars + a
small set of hooks. Provider and Integration store/registry become
thin wrappers; Project keeps its own extras (env vars, favorite
versions, ``configs/`` sublayout, first-save seed) on top of the
shared base.
"""

from __future__ import annotations

import json
import re
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Generic, Protocol, TypeVar

from src.utils.atomic_fs import (
    atomic_write_json,
    atomic_write_text,
    created_at_from_filename,
    unique_snapshot_path,
    utc_now_iso,
)

if TYPE_CHECKING:
    from collections.abc import Iterable


REGISTRY_SCHEMA_VERSION = 1


def default_workspace_root() -> Path:
    """User-level workspace root (``~/.ryotenkai/``) shared by all registries."""
    return Path.home() / ".ryotenkai"


# ---------------------------------------------------------------------------
# Errors — concrete subclasses keep the historical names so callers that
# catch ``ProjectRegistryError`` / ``ProviderStoreError`` etc. still work.
# ---------------------------------------------------------------------------


class WorkspaceRegistryError(RuntimeError):
    """Base class for workspace-registry failures (recoverable)."""


class WorkspaceStoreError(RuntimeError):
    """Base class for workspace-store failures (recoverable)."""


# ---------------------------------------------------------------------------
# Registry generic
# ---------------------------------------------------------------------------


class _RegistryEntry(Protocol):
    """Structural protocol the entry dataclass must satisfy."""

    id: str

    def to_dict(self) -> dict[str, Any]: ...


EntryT = TypeVar("EntryT", bound=_RegistryEntry)


class WorkspaceRegistry(ABC, Generic[EntryT]):
    """File-backed JSON index of workspace resources keyed by id.

    Subclasses must declare:

    * ``payload_key`` — list-key in the JSON payload (``"projects"`` /
      ``"providers"`` / ``"integrations"``).
    * ``resource_subdir`` — folder name under ``root`` where individual
      entries live (also ``"projects"`` / ``"providers"`` / etc.).
    * ``registry_filename`` — JSON index filename (``"projects.json"`` /
      ``"providers.json"`` / ``"integrations.json"``).
    * ``id_pattern`` — regex string for ``validate_id``.
    * ``error_class`` — subclass-specific error type.
    * :meth:`_decode_entry` — turn one raw dict into an ``EntryT``.
    """

    payload_key: ClassVar[str]
    resource_subdir: ClassVar[str]
    registry_filename: ClassVar[str]
    id_pattern: ClassVar[str] = r"^[a-z0-9][a-z0-9_\-]{0,63}$"
    error_class: ClassVar[type[WorkspaceRegistryError]] = WorkspaceRegistryError

    def __init__(self, root: Path | None = None) -> None:
        self.root = (
            Path(root).expanduser().resolve() if root else default_workspace_root()
        )

    # ---------- Abstract hooks --------------------------------------------

    @abstractmethod
    def _decode_entry(self, raw: dict[str, Any]) -> EntryT:
        """Build one ``EntryT`` from the JSON-decoded entry dict."""

    # ---------- Path properties -------------------------------------------

    @property
    def registry_path(self) -> Path:
        return self.root / self.registry_filename

    @property
    def resources_dir(self) -> Path:
        return self.root / self.resource_subdir

    # ---------- Validation -------------------------------------------------

    @classmethod
    def validate_id(cls, resource_id: str) -> None:
        if not re.match(cls.id_pattern, resource_id):
            raise cls.error_class(
                f"invalid {cls.resource_subdir.rstrip('s')} id {resource_id!r}: "
                f"must match {cls.id_pattern}"
            )

    # ---------- I/O --------------------------------------------------------

    def _load_raw(self) -> dict[str, Any]:
        if not self.registry_path.is_file():
            return {"schema_version": REGISTRY_SCHEMA_VERSION, self.payload_key: []}
        try:
            return json.loads(self.registry_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise self.error_class(
                f"failed to read {self.registry_path}: {exc}"
            ) from exc

    def _save_raw(self, payload: dict[str, Any]) -> None:
        atomic_write_json(self.registry_path, payload)

    def _items(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        items = payload.get(self.payload_key)
        return list(items) if isinstance(items, list) else []

    # ---------- Public API -------------------------------------------------

    def list(self) -> list[EntryT]:
        return [self._decode_entry(item) for item in self._items(self._load_raw())]

    def resolve(self, resource_id: str) -> EntryT:
        for entry in self.list():
            if entry.id == resource_id:
                return entry
        raise self.error_class(
            f"{self.resource_subdir.rstrip('s')} {resource_id!r} not found in registry"
        )

    def unregister(self, resource_id: str) -> bool:
        payload = self._load_raw()
        items = self._items(payload)
        remaining = [item for item in items if str(item.get("id")) != resource_id]
        if len(remaining) == len(items):
            return False
        payload[self.payload_key] = remaining
        self._save_raw(payload)
        return True

    def default_resource_path(self, resource_id: str) -> Path:
        """Default on-disk directory for a resource id."""
        return self.resources_dir / resource_id

    # Lower-level helper used by subclass register() implementations that
    # carry resource-specific create kwargs.
    def _append_entry(self, entry: EntryT) -> EntryT:
        payload = self._load_raw()
        items = self._items(payload)
        existing_ids = {str(item.get("id")) for item in items}
        if entry.id in existing_ids:
            raise self.error_class(
                f"{self.resource_subdir.rstrip('s')} id {entry.id!r} already registered"
            )
        items.append(entry.to_dict())
        payload[self.payload_key] = items
        payload["schema_version"] = REGISTRY_SCHEMA_VERSION
        self._save_raw(payload)
        return entry


# ---------------------------------------------------------------------------
# Store generic
# ---------------------------------------------------------------------------


class _MetadataLike(Protocol):
    """Structural protocol the metadata dataclass must satisfy."""

    updated_at: str

    def to_dict(self) -> dict[str, Any]: ...


MetadataT = TypeVar("MetadataT", bound=_MetadataLike)
ConfigVersionT = TypeVar("ConfigVersionT")


class WorkspaceStore(ABC, Generic[MetadataT, ConfigVersionT]):
    """File-backed store for a single workspace resource directory.

    Layout managed::

        <root>/
          <metadata.json>          # subclass-specific filename
          current.yaml             # current configuration
          history/
            <iso>.yaml             # historical snapshots

    Subclasses declare:

    * ``metadata_filename`` — JSON metadata filename (``"project.json"`` /
      ``"provider.json"`` / ``"integration.json"``).
    * ``schema_version`` — int.
    * ``error_class`` — subclass-specific error type.
    * :meth:`_decode_metadata` / :meth:`_make_config_version` —
      dataclass constructors.
    * Optionally override ``current_config_path`` / ``history_dir`` /
      ``configs_dir`` to support nested layouts (Project does this).

    The base intentionally does NOT include resource-specific extras
    (token storage for integrations, env-var management for projects,
    favorites). Those stay in subclasses as additional methods —
    composition over leaky base abstractions.
    """

    metadata_filename: ClassVar[str]
    schema_version: ClassVar[int]
    error_class: ClassVar[type[WorkspaceStoreError]] = WorkspaceStoreError

    def __init__(self, root: Path) -> None:
        self.root = Path(root).expanduser().resolve()

    # ---------- Abstract hooks --------------------------------------------

    @abstractmethod
    def _decode_metadata(self, raw: dict[str, Any]) -> MetadataT:
        """Build the metadata dataclass from a JSON dict."""

    @abstractmethod
    def _make_config_version(
        self, *, filename: str, created_at: str, size_bytes: int, path: Path
    ) -> ConfigVersionT:
        """Build one ConfigVersion dataclass entry."""

    # ---------- Path properties (overridable) -----------------------------

    @property
    def metadata_path(self) -> Path:
        return self.root / self.metadata_filename

    @property
    def current_config_path(self) -> Path:
        return self.root / "current.yaml"

    @property
    def history_dir(self) -> Path:
        return self.root / "history"

    # ---------- Lifecycle --------------------------------------------------

    def exists(self) -> bool:
        return self.metadata_path.is_file()

    def load(self) -> MetadataT:
        if not self.exists():
            raise self.error_class(f"{self.metadata_filename} not found at {self.root}")
        raw = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        return self._decode_metadata(raw)

    def touch(self) -> None:
        if not self.exists():
            return
        metadata = self.load()
        metadata.updated_at = utc_now_iso()
        atomic_write_json(self.metadata_path, metadata.to_dict())

    def remove(self) -> None:
        if self.root.is_dir():
            shutil.rmtree(self.root)

    # ---------- Config -----------------------------------------------------

    def current_yaml_text(self) -> str:
        if self.current_config_path.is_file():
            return self.current_config_path.read_text(encoding="utf-8")
        return ""

    def save_config(self, yaml_text: str) -> str | None:
        """Write yaml_text atomically; snapshot the previous content first.

        Returns the snapshot filename or ``None`` if there was nothing
        to snapshot. Subclasses can override (e.g. ProjectStore seeds
        v1 on the very first save) but should call ``touch()`` at the
        end to bump ``updated_at``.
        """
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

    def list_versions(self) -> list[ConfigVersionT]:
        if not self.history_dir.is_dir():
            return []
        entries: list[ConfigVersionT] = []
        for path in sorted(self.history_dir.glob("*.yaml")):
            stat = path.stat()
            entries.append(
                self._make_config_version(
                    filename=path.name,
                    created_at=created_at_from_filename(path.name),
                    size_bytes=stat.st_size,
                    path=path,
                )
            )
        # Newest first.
        entries.sort(key=lambda v: v.filename, reverse=True)  # type: ignore[attr-defined]
        return entries

    def read_version(self, filename: str) -> str:
        if "/" in filename or ".." in filename:
            raise self.error_class(f"invalid version filename: {filename!r}")
        path = self.history_dir / filename
        if not path.is_file():
            raise self.error_class(f"version not found: {filename}")
        return path.read_text(encoding="utf-8")

    def restore_version(self, filename: str) -> str | None:
        return self.save_config(self.read_version(filename))


__all__ = [
    "REGISTRY_SCHEMA_VERSION",
    "ConfigVersionT",
    "EntryT",
    "MetadataT",
    "WorkspaceRegistry",
    "WorkspaceRegistryError",
    "WorkspaceStore",
    "WorkspaceStoreError",
    "default_workspace_root",
]
