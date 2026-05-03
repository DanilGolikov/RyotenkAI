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
from src.community.libs import (
    LibLoadFailure,
    LoadedLib,
    libs_fingerprint_entries,
    libs_root_for,
    load_libs,
)
from src.community.loader import (
    LoadedPlugin,
    LoadedPreset,
    LoadFailure,
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
        self._libs: list[LoadedLib] = []
        #: Per-kind list of plugins that failed to load on the last
        #: refresh. Surfaced via :meth:`failures` and through the
        #: ``GET /plugins/{kind}`` API so the UI can render an error
        #: banner with traceback context. ``"presets"`` key holds preset
        #: failures; ``"all"`` is reserved by the API endpoint for a
        #: combined view.
        self._failures: dict[str, list[LoadFailure]] = {}
        #: Lib failures kept on a separate axis from plugin failures —
        #: lib loading runs first and a broken lib will probably
        #: cascade into plugin failures (missing import), so users
        #: should see them grouped, not interleaved.
        self._lib_failures: list[LibLoadFailure] = []
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
        self._libs.clear()
        self._failures.clear()
        self._lib_failures.clear()
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
        # community/libs/<lib>/{__init__.py,*.py} — surface that decides
        # which libs are preloaded into community_libs.* and what their
        # top-level modules are. Deep edits inside libs aren't tracked
        # (same rule as src/) — restart the backend after those.
        entries.extend(libs_fingerprint_entries(libs_root_for(self._root)))
        return tuple(sorted(entries))

    def _load_locked(self) -> None:
        logger.info("[COMMUNITY_CATALOG] loading from %s", self._root)
        # Load shared libs FIRST: plugin modules import from
        # ``community_libs.<name>`` at module-load time, so the
        # namespace must exist before _import_plugin_class executes.
        # ``load_libs`` parses each lib's manifest, registers the
        # subpackage in sys.modules, and returns successes + failures.
        lib_result = load_libs(libs_root=libs_root_for(self._root))
        self._libs = list(lib_result.libs)
        self._lib_failures = list(lib_result.failures)
        if self._libs:
            logger.info(
                "[COMMUNITY_CATALOG] loaded libs: %s",
                ", ".join(
                    f"{lib.manifest.lib.id}@{lib.manifest.lib.version}"
                    for lib in self._libs
                ),
            )
        if self._lib_failures:
            logger.warning(
                "[COMMUNITY_CATALOG] %d lib(s) failed to load",
                len(self._lib_failures),
            )
        libs_by_id = {lib.manifest.lib.id: lib.manifest for lib in self._libs}
        all_results = load_all_plugins(root=self._root, libs_by_id=libs_by_id)
        # Split the LoadResult shape into the catalog's separate stores —
        # registries only see successes, the UI gets failures alongside.
        self._plugins = {kind: result.plugins for kind, result in all_results.items()}
        self._failures = {
            kind: list(result.failures)
            for kind, result in all_results.items()
            if result.failures
        }

        preset_result = load_presets(root=self._root)
        self._presets = list(preset_result.presets)
        if preset_result.failures:
            self._failures["presets"] = list(preset_result.failures)

        self._populate_registries()
        self._loaded = True
        counts = ", ".join(
            f"{kind}={len(items)}" for kind, items in self._plugins.items()
        )
        total_failures = sum(len(v) for v in self._failures.values())
        if total_failures:
            logger.warning(
                "[COMMUNITY_CATALOG] loaded: %s, presets=%d (failures=%d)",
                counts,
                len(self._presets),
                total_failures,
            )
        else:
            logger.info(
                "[COMMUNITY_CATALOG] loaded: %s, presets=%d",
                counts,
                len(self._presets),
            )

    def _populate_registries(self) -> None:
        """Push loaded plugins into the per-kind registries.

        Imports are done lazily inside the method to keep ``src/community/``
        free of hard dependencies on unrelated subsystems. Each kind module
        exports a singleton registry instance — the catalog drives the
        clear/register lifecycle through that instance, never directly
        through the class.
        """
        from src.data.validation.registry import validation_registry
        from src.evaluation.plugins.registry import evaluator_registry
        from src.reports.plugins.registry import report_registry
        from src.training.reward_plugins.registry import reward_registry

        validation_registry.clear()
        evaluator_registry.clear()
        reward_registry.clear()
        report_registry.clear()

        for loaded in self._plugins.get("validation", []):
            validation_registry.register_from_community(loaded)
        for loaded in self._plugins.get("evaluation", []):
            evaluator_registry.register_from_community(loaded)
        for loaded in self._plugins.get("reward", []):
            reward_registry.register_from_community(loaded)
        for loaded in self._plugins.get("reports", []):
            report_registry.register_from_community(loaded)

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

    def libs(self) -> list[LoadedLib]:
        """Return every successfully loaded :class:`LoadedLib`.

        Used by the loader's plugin lib-version cross-check (each
        plugin's ``[[lib_requirements]]`` is matched against the
        installed lib manifests) and by the future ``GET /libs`` API
        endpoint for the catalogue UI.
        """
        self.ensure_loaded()
        return list(self._libs)

    def get_lib(self, lib_id: str) -> LoadedLib:
        """Return one :class:`LoadedLib` by id, or raise ``KeyError``.

        Symmetric with :meth:`get` for plugins. Used by the loader's
        version-check at plugin load time.
        """
        for lib in self.libs():
            if lib.manifest.lib.id == lib_id:
                return lib
        raise KeyError(f"lib {lib_id!r} is not in community catalog")

    def lib_failures(self) -> list[LibLoadFailure]:
        """Return lib load failures from the most recent refresh."""
        self.ensure_loaded()
        return list(self._lib_failures)

    def failures(self, kind: str | None = None) -> list[LoadFailure]:
        """Return load failures from the most recent refresh.

        ``kind=None`` flattens across all kinds (plugins + presets) so
        an admin endpoint can surface everything at once. ``kind="…"``
        narrows to one kind — used by ``GET /plugins/{kind}`` to attach
        only the relevant failures to each catalog response.
        """
        self.ensure_loaded()
        if kind is None:
            return [f for entries in self._failures.values() for f in entries]
        return list(self._failures.get(kind, []))

    def list_kinds(self) -> tuple[str, ...]:
        return ALL_PLUGIN_KINDS


catalog = CommunityCatalog()


__all__ = ["CommunityCatalog", "catalog"]
