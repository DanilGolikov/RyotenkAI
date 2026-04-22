"""Loader for community plugins and presets.

Scans ``community/<kind>/`` (for plugins) and ``community/presets/`` (for
presets), returning structured ``LoadedPlugin`` / ``LoadedPreset`` records.
Each entry is backed by either a folder or a ZIP archive; archives are
transparently extracted to the cache and loaded from there.
"""

from __future__ import annotations

import importlib.util
import sys
import tomllib
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

from src.community.archive import ensure_extracted, resolve_extraction_root
from src.community.constants import (
    ALL_PLUGIN_KINDS,
    COMMUNITY_ROOT,
    MANIFEST_FILENAME,
    PLUGIN_KIND_DIRS,
    PRESET_DIR_NAME,
)
from src.community.manifest import PluginKind, PluginManifest, PresetManifest
from src.utils.logger import logger


@dataclass(frozen=True, slots=True)
class LoadedPlugin:
    manifest: PluginManifest
    plugin_cls: type
    source_path: Path


@dataclass(frozen=True, slots=True)
class LoadedPreset:
    manifest: PresetManifest
    yaml_text: str
    source_path: Path


def _iter_entries(kind_dir: Path) -> Iterator[tuple[str, Path]]:
    """Yield loadable entries under ``kind_dir``.

    Precedence when the same stem exists both as a folder and as a ``.zip``:
    **the folder wins**. This is the dev-time default — the folder is the
    source of truth while a plugin is being written; the archive is a
    distributable snapshot produced by ``ryotenkai community pack`` that
    should only be used when the source folder is absent. A warning is
    logged so the author notices a stale ``.zip`` sitting next to live
    sources.
    """
    if not kind_dir.exists():
        return

    folders: dict[str, Path] = {}
    zips: dict[str, Path] = {}
    for entry in sorted(kind_dir.iterdir()):
        if entry.name.startswith(".") or entry.name == "__pycache__":
            continue
        if entry.is_dir():
            folders[entry.name] = entry
        elif entry.is_file() and entry.suffix == ".zip":
            zips[entry.stem] = entry

    # Folders first (source form takes precedence).
    for name, path in folders.items():
        if name in zips:
            logger.warning(
                "[COMMUNITY_LOADER] %s shadows %s — folder wins; "
                "delete the folder to use the archive, or delete the archive if it is stale",
                path,
                zips[name],
            )
        yield name, path

    # Zips whose stem is NOT also a folder — these are archive-only plugins.
    for stem, path in zips.items():
        if stem in folders:
            continue
        yield path.name, path


def _resolve_source_root(entry: Path) -> Path:
    if entry.is_dir():
        return entry
    if entry.suffix == ".zip":
        extracted = ensure_extracted(entry)
        return resolve_extraction_root(extracted)
    raise ValueError(f"unsupported community entry: {entry}")


def _read_manifest_text(source_root: Path) -> str:
    manifest_path = source_root / MANIFEST_FILENAME
    if not manifest_path.exists():
        raise FileNotFoundError(f"{MANIFEST_FILENAME} not found in {source_root}")
    return manifest_path.read_text(encoding="utf-8")


def _parse_toml(text: str) -> dict:
    return tomllib.loads(text)


def _import_plugin_class(source_root: Path, module_name: str, class_name: str) -> type:
    """Import ``class_name`` from ``<source_root>/<module_name>.py`` (or package).

    Uses ``importlib.util.spec_from_file_location`` so that ``community/``
    does not need to be on ``sys.path`` and plugins with identically named
    modules can coexist.
    """
    candidates = [
        source_root / f"{module_name}.py",
        source_root / module_name / "__init__.py",
    ]
    spec_source = next((path for path in candidates if path.exists()), None)
    if spec_source is None:
        raise FileNotFoundError(
            f"entry point module {module_name!r} not found under {source_root}; "
            f"expected one of: {[str(p.relative_to(source_root)) for p in candidates]}"
        )

    unique_suffix = uuid.uuid4().hex[:8]
    unique_name = f"_community_{source_root.name}_{module_name}_{unique_suffix}"
    spec = importlib.util.spec_from_file_location(
        unique_name,
        spec_source,
        submodule_search_locations=[str(source_root / module_name)]
        if spec_source.name == "__init__.py"
        else None,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot create import spec for {spec_source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[unique_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(unique_name, None)
        raise
    if not hasattr(module, class_name):
        raise AttributeError(
            f"module {spec_source} does not define class {class_name!r}"
        )
    return getattr(module, class_name)


def _attach_community_metadata(
    plugin_cls: type, manifest: PluginManifest, source_path: Path
) -> None:
    """Mirror manifest fields onto the plugin class so runtime code can read them."""
    plugin_cls.name = manifest.plugin.id  # type: ignore[attr-defined]
    plugin_cls.priority = manifest.plugin.priority  # type: ignore[attr-defined]
    plugin_cls.version = manifest.plugin.version  # type: ignore[attr-defined]
    plugin_cls._required_secrets = tuple(manifest.secrets.required)  # type: ignore[attr-defined]
    plugin_cls._community_manifest = manifest  # type: ignore[attr-defined]
    plugin_cls._community_source_path = source_path  # type: ignore[attr-defined]


def load_plugins(
    kind: PluginKind, *, root: Path = COMMUNITY_ROOT
) -> list[LoadedPlugin]:
    """Load every plugin of the given kind under ``root/<kind>/``."""
    kind_dir = root / PLUGIN_KIND_DIRS[kind]
    loaded: list[LoadedPlugin] = []
    seen_ids: set[str] = set()

    for entry_name, entry_path in _iter_entries(kind_dir):
        try:
            source_root = _resolve_source_root(entry_path)
            manifest_dict = _parse_toml(_read_manifest_text(source_root))
            manifest = PluginManifest.model_validate(manifest_dict)
        except Exception as exc:
            logger.error(
                "[COMMUNITY_LOADER] kind=%s entry=%s failed to load manifest: %s",
                kind,
                entry_name,
                exc,
            )
            continue

        if manifest.plugin.kind != kind:
            logger.error(
                "[COMMUNITY_LOADER] kind mismatch: %s manifest declares kind=%s",
                entry_name,
                manifest.plugin.kind,
            )
            continue

        if manifest.plugin.id in seen_ids:
            raise ValueError(
                f"duplicate plugin id {manifest.plugin.id!r} in kind={kind}"
            )
        seen_ids.add(manifest.plugin.id)

        try:
            plugin_cls = _import_plugin_class(
                source_root,
                manifest.plugin.entry_point.module,
                manifest.plugin.entry_point.class_name,
            )
        except Exception as exc:
            logger.error(
                "[COMMUNITY_LOADER] kind=%s id=%s failed to import entry point: %s",
                kind,
                manifest.plugin.id,
                exc,
            )
            continue

        _attach_community_metadata(plugin_cls, manifest, source_root)
        loaded.append(
            LoadedPlugin(
                manifest=manifest, plugin_cls=plugin_cls, source_path=source_root
            )
        )
        logger.debug(
            "[COMMUNITY_LOADER] kind=%s id=%s loaded from %s",
            kind,
            manifest.plugin.id,
            source_root,
        )

    return loaded


def load_presets(*, root: Path = COMMUNITY_ROOT) -> list[LoadedPreset]:
    """Load every preset under ``root/presets/``."""
    presets_dir = root / PRESET_DIR_NAME
    loaded: list[LoadedPreset] = []
    seen_ids: set[str] = set()

    for entry_name, entry_path in _iter_entries(presets_dir):
        try:
            source_root = _resolve_source_root(entry_path)
            manifest_dict = _parse_toml(_read_manifest_text(source_root))
            manifest = PresetManifest.model_validate(manifest_dict)
        except Exception as exc:
            logger.error(
                "[COMMUNITY_LOADER] preset=%s failed to load manifest: %s",
                entry_name,
                exc,
            )
            continue

        if manifest.preset.id in seen_ids:
            raise ValueError(f"duplicate preset id {manifest.preset.id!r}")
        seen_ids.add(manifest.preset.id)

        yaml_path = source_root / manifest.preset.entry_point.file
        if not yaml_path.exists():
            logger.error(
                "[COMMUNITY_LOADER] preset=%s: YAML file not found at %s",
                manifest.preset.id,
                yaml_path,
            )
            continue

        loaded.append(
            LoadedPreset(
                manifest=manifest,
                yaml_text=yaml_path.read_text(encoding="utf-8"),
                source_path=yaml_path,
            )
        )
        logger.debug(
            "[COMMUNITY_LOADER] preset=%s loaded from %s",
            manifest.preset.id,
            yaml_path,
        )

    return loaded


def load_all_plugins(*, root: Path = COMMUNITY_ROOT) -> dict[str, list[LoadedPlugin]]:
    return {kind: load_plugins(kind, root=root) for kind in ALL_PLUGIN_KINDS}


__all__ = [
    "LoadedPlugin",
    "LoadedPreset",
    "load_all_plugins",
    "load_plugins",
    "load_presets",
]
