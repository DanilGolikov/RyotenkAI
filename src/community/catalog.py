"""Lazy, in-memory catalogue of everything under ``community/``.

The catalogue is the single entry point for the rest of the codebase:

- ``catalog.ensure_loaded()`` — idempotent, loads every plugin and preset.
- ``catalog.plugins(kind)`` — ``list[LoadedPlugin]`` for one plugin kind.
- ``catalog.get(kind, id)`` — single ``LoadedPlugin`` or raises ``KeyError``.
- ``catalog.presets()`` — ``list[LoadedPreset]``.
- ``catalog.reload()`` — forced reload (tests, hot-reload scenarios).

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


class CommunityCatalog:
    def __init__(self, root: Path = COMMUNITY_ROOT) -> None:
        self._root = root
        self._plugins: dict[str, list[LoadedPlugin]] = {}
        self._presets: list[LoadedPreset] = []
        self._loaded = False
        self._lock = threading.Lock()

    @property
    def root(self) -> Path:
        return self._root

    def ensure_loaded(self) -> None:
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            self._load_locked()

    def reload(self) -> None:
        with self._lock:
            self._plugins.clear()
            self._presets.clear()
            self._loaded = False
            self._load_locked()

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
