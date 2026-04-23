"""Lazy, in-memory catalogue of everything under ``community/``.

The catalogue is the single entry point for the rest of the codebase:

- ``catalog.ensure_loaded()`` — idempotent load + transparent re-load
  when the ``community/`` tree changed on disk (mtime fingerprint).
- ``catalog.plugins(kind)`` — ``list[LoadedPlugin]`` for one plugin kind.
- ``catalog.get(kind, id)`` — single ``LoadedPlugin`` or raises ``KeyError``.
- ``catalog.presets()`` — ``list[LoadedPreset]``.
- ``catalog.reload()`` — forced reload (tests, ``/community/reload`` endpoint).

The first call populates downstream registries (validation / evaluation /
reward / reports) through their ``register_from_community`` hooks.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import TYPE_CHECKING

from src.community.constants import ALL_PLUGIN_KINDS, COMMUNITY_ROOT
from src.community.loader import (
    LoadedPlugin,
    LoadedPreset,
    load_all_plugins,
    load_presets,
)
from src.utils.logger import logger

if TYPE_CHECKING:
    from src.community.manifest import PluginKind


def _append(
    entries: list[tuple[str, float]],
    path: Path,
    root: Path,
) -> None:
    """Record ``(relpath, mtime)`` into the fingerprint, swallowing races
    where the file disappears between iterdir and stat."""
    try:
        entries.append((str(path.relative_to(root)), path.stat().st_mtime))
    except OSError:
        return


class CommunityCatalog:
    def __init__(self, root: Path = COMMUNITY_ROOT) -> None:
        self._root = root
        self._plugins: dict[str, list[LoadedPlugin]] = {}
        self._presets: list[LoadedPreset] = []
        self._loaded = False
        self._lock = threading.Lock()
        self._fingerprint: tuple[tuple[str, float], ...] = ()

    @property
    def root(self) -> Path:
        return self._root

    def ensure_loaded(self) -> None:
        """Load (or reload) the catalogue if the on-disk tree changed.

        Cheap path: already loaded AND fingerprint matches → just return.
        Stale path: acquire lock, recheck under lock, then ``_load_locked``.
        The fingerprint is a stable-sorted tuple of ``(relpath, mtime)``
        over every ``manifest.toml`` / ``preset.yaml`` / ``*.zip`` under
        ``community/`` — catches additions, deletions, and edits.
        """
        if self._loaded and self._fingerprint == self._compute_fingerprint():
            return
        with self._lock:
            fresh = self._compute_fingerprint()
            if self._loaded and fresh == self._fingerprint:
                return
            self._reset_state()
            self._load_locked()
            self._fingerprint = fresh

    def reload(self) -> None:
        """Force a reload regardless of fingerprint (tests, manual refresh)."""
        with self._lock:
            self._reset_state()
            self._load_locked()
            self._fingerprint = self._compute_fingerprint()

    def _reset_state(self) -> None:
        self._plugins.clear()
        self._presets.clear()
        self._loaded = False

    def _compute_fingerprint(self) -> tuple[tuple[str, float], ...]:
        """Stable snapshot of files whose state would change the load.

        Walks one level deep into each recognised ``community/<kind>/``
        directory and stats at most 3 files per plugin folder — skipping
        hidden trees like ``community/.cache/`` and ``__pycache__``.
        Cheap enough (~hundreds of µs) to call on every request.
        """
        if not self._root.is_dir():
            return ()
        entries: list[tuple[str, float]] = []
        for kind_dir_name in (*ALL_PLUGIN_KINDS, "presets"):
            kind_dir = self._root / kind_dir_name
            if not kind_dir.is_dir():
                continue
            for child in kind_dir.iterdir():
                name = child.name
                if name.startswith(".") or name == "__pycache__":
                    continue
                # Zip archives are self-fingerprinted by their own mtime.
                if child.is_file() and name.endswith(".zip"):
                    _append(entries, child, self._root)
                    continue
                if not child.is_dir():
                    continue
                for filename in ("manifest.toml", "preset.yaml"):
                    candidate = child / filename
                    if candidate.is_file():
                        _append(entries, candidate, self._root)
        return tuple(sorted(entries))

    def _load_locked(self) -> None:
        logger.info("[COMMUNITY_CATALOG] loading from %s", self._root)
        self._plugins = load_all_plugins(root=self._root)
        self._presets = load_presets(root=self._root)
        self._populate_registries()
        self._loaded = True
        counts = ", ".join(
            f"{kind}={len(items)}" for kind, items in self._plugins.items()
        )
        logger.info(
            "[COMMUNITY_CATALOG] loaded: %s, presets=%d",
            counts,
            len(self._presets),
        )

    def _populate_registries(self) -> None:
        """Push loaded plugins into the legacy per-kind registries.

        Imports are done lazily inside the method to keep ``src/community/``
        free of hard dependencies on unrelated subsystems.
        """
        from src.data.validation.registry import ValidationPluginRegistry
        from src.evaluation.plugins.registry import EvaluatorPluginRegistry
        from src.reports.plugins.registry import ReportPluginRegistry
        from src.training.reward_plugins.registry import RewardPluginRegistry

        ValidationPluginRegistry.clear()
        EvaluatorPluginRegistry.clear()
        RewardPluginRegistry.clear()
        ReportPluginRegistry.clear()

        for loaded in self._plugins.get("validation", []):
            ValidationPluginRegistry.register_from_community(loaded)
        for loaded in self._plugins.get("evaluation", []):
            EvaluatorPluginRegistry.register_from_community(loaded)
        for loaded in self._plugins.get("reward", []):
            RewardPluginRegistry.register_from_community(loaded)
        for loaded in self._plugins.get("reports", []):
            ReportPluginRegistry.register_from_community(loaded)

    # ----- public accessors -------------------------------------------------

    def plugins(self, kind: PluginKind) -> list[LoadedPlugin]:
        self.ensure_loaded()
        return list(self._plugins.get(kind, []))

    def get(self, kind: PluginKind, plugin_id: str) -> LoadedPlugin:
        for loaded in self.plugins(kind):
            if loaded.manifest.plugin.id == plugin_id:
                return loaded
        raise KeyError(f"plugin {plugin_id!r} of kind={kind!r} is not in community catalog")

    def presets(self) -> list[LoadedPreset]:
        self.ensure_loaded()
        return list(self._presets)

    def list_kinds(self) -> tuple[str, ...]:
        return ALL_PLUGIN_KINDS


catalog = CommunityCatalog()


__all__ = ["CommunityCatalog", "catalog"]
