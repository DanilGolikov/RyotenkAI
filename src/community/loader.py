"""Loader for community plugins and presets.

Scans ``community/<kind>/`` (for plugins) and ``community/presets/`` (for
presets), returning structured :class:`LoadedPlugin` / :class:`LoadedPreset`
records alongside :class:`LoadFailure` entries for entries the loader
couldn't import.

Two modes (controlled by the ``strict`` flag and the ``COMMUNITY_STRICT``
env var):

- **loose** (production default): swallow per-entry exceptions, log
  them, and surface them as ``LoadFailure`` rows so the API/UI can
  show the full picture without bricking the whole catalog.
- **strict** (test / dev): re-raise the original exception so the test
  run fails loudly. Triggered when the env var is set to a truthy value
  or the caller passes ``strict=True``.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import tomllib
import traceback
import uuid
from dataclasses import dataclass, field
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


#: Truthy values for the ``COMMUNITY_STRICT`` env var. Anything else
#: (unset, ``"0"``, ``""``, ``"false"``, etc.) keeps the default loose mode.
_STRICT_TRUE = frozenset({"1", "true", "yes", "on"})


def _resolve_strict(explicit: bool | None) -> bool:
    """Combine an explicit caller flag with the env-var override.

    ``strict=True`` from the caller always wins. ``strict=None`` (the
    default for top-level entry points) defers to ``COMMUNITY_STRICT``.
    """
    if explicit is True:
        return True
    if explicit is False:
        return False
    return os.environ.get("COMMUNITY_STRICT", "").strip().lower() in _STRICT_TRUE


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


@dataclass(frozen=True, slots=True)
class LoadFailure:
    """One entry the loader couldn't bring online.

    Surfaced through :class:`CommunityCatalog._failures` and
    ``GET /plugins/{kind}`` so the UI can render an error banner with
    enough context for the plugin author to debug — and so a
    ``COMMUNITY_STRICT=1 pytest`` run fails loudly on the same input.
    """

    #: ``"validation" | "evaluation" | "reward" | "reports" | "presets"`` —
    #: which on-disk subtree the entry came from. ``"presets"`` is reserved
    #: for the preset loader.
    kind: str
    #: Folder/zip name as seen on disk (``community/<kind>/<entry_name>``).
    entry_name: str
    #: Manifest id when available; falls back to :data:`None` when the
    #: manifest itself didn't parse.
    plugin_id: str | None
    #: Stable machine-readable failure category.
    #: ``"manifest_parse" | "kind_mismatch" | "duplicate_id" | "import_error" | "missing_yaml"``.
    error_type: str
    #: Human-readable summary suitable for the UI banner.
    message: str
    #: Full traceback for the developer drilldown. Empty when the failure
    #: is a synthetic check (e.g. ``kind_mismatch``) without a real exc.
    traceback: str = ""


@dataclass(frozen=True, slots=True)
class LoadResult:
    """Success/failure pair returned by :func:`load_plugins`.

    Two fields keep the wire-shape obvious to callers — the catalog
    stores ``failures`` separately from ``plugins`` so the UI can treat
    them independently. The result is iterable / indexable / sized so
    code that only cares about the happy path can keep treating it as
    ``list[LoadedPlugin]``.
    """

    plugins: list[LoadedPlugin] = field(default_factory=list)
    failures: list[LoadFailure] = field(default_factory=list)

    def __iter__(self):
        return iter(self.plugins)

    def __len__(self) -> int:
        return len(self.plugins)

    def __getitem__(self, index):
        return self.plugins[index]


@dataclass(frozen=True, slots=True)
class PresetLoadResult:
    presets: list[LoadedPreset] = field(default_factory=list)
    failures: list[LoadFailure] = field(default_factory=list)

    def __iter__(self):
        return iter(self.presets)

    def __len__(self) -> int:
        return len(self.presets)

    def __getitem__(self, index):
        return self.presets[index]


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


def _crosscheck_required_env(
    plugin_cls: type, manifest: PluginManifest
) -> None:
    """If the class declares ``REQUIRED_ENV``, verify it matches the manifest.

    Compares Python's tuple to the manifest's ``[[required_env]]`` list
    element-wise on (name, optional, secret, managed_by). Description
    text is intentionally **not** compared — authors often massage it
    in the TOML for end users without re-deriving from a Python-side
    one-liner.

    Empty Python-side tuple (the default for plugins that don't import
    :class:`BasePlugin`) skips the check; the manifest stays
    authoritative. Mismatch raises :class:`ValueError` with a precise
    per-field diff so the author can decide which side to fix.
    """
    declared = getattr(plugin_cls, "REQUIRED_ENV", ())
    if not declared:
        return

    # Normalise the Python side to plain dicts so we can compare against
    # the Pydantic model_dump() of the manifest entries on equal terms.
    py_entries: list[dict[str, object]] = []
    for spec in declared:
        if hasattr(spec, "model_dump"):
            py_entries.append(spec.model_dump())
        elif isinstance(spec, dict):
            py_entries.append(dict(spec))
        else:
            raise TypeError(
                f"plugin {plugin_cls.__name__}.REQUIRED_ENV entries must be "
                f"RequiredEnvSpec instances or matching dicts; got "
                f"{type(spec).__name__}"
            )

    manifest_entries = [spec.model_dump() for spec in manifest.required_env]

    py_by_name = {e["name"]: e for e in py_entries}
    toml_by_name = {e["name"]: e for e in manifest_entries}

    diffs: list[str] = []

    only_in_py = sorted(set(py_by_name) - set(toml_by_name))
    only_in_toml = sorted(set(toml_by_name) - set(py_by_name))
    if only_in_py:
        diffs.append(
            f"declared in REQUIRED_ENV but missing from manifest: {only_in_py}"
        )
    if only_in_toml:
        diffs.append(
            f"declared in manifest but missing from REQUIRED_ENV: {only_in_toml}"
        )

    for name in sorted(set(py_by_name) & set(toml_by_name)):
        py = py_by_name[name]
        toml = toml_by_name[name]
        per_key: list[str] = []
        for key in ("optional", "secret", "managed_by"):
            if py.get(key) != toml.get(key):
                per_key.append(
                    f"{key}: code={py.get(key)!r} vs toml={toml.get(key)!r}"
                )
        if per_key:
            diffs.append(f"{name}: " + "; ".join(per_key))

    if diffs:
        raise ValueError(
            f"REQUIRED_ENV ↔ manifest cross-check failed for plugin "
            f"{manifest.plugin.id!r}:\n  - "
            + "\n  - ".join(diffs)
            + "\nRun `ryotenkai community sync-envs <plugin>` to align "
            "the manifest with REQUIRED_ENV."
        )


def _attach_community_metadata(
    plugin_cls: type, manifest: PluginManifest, source_path: Path
) -> None:
    """Mirror manifest fields onto the plugin class so runtime code can read them.

    ``_required_secrets`` is derived from ``required_env`` entries that
    are both ``secret=true`` and ``optional=false`` — the registry uses
    that ClassVar at instantiate time to fetch values via the per-kind
    :class:`PluginSecretsResolver`. Optional or non-secret envs are
    handled by the plugin itself (or B2's ``_env`` helper).

    Also runs the REQUIRED_ENV cross-check (A7) — see
    :func:`_crosscheck_required_env` — *before* mirroring metadata, so
    a manifest/code mismatch fails the load fast with a precise diff.
    """
    _crosscheck_required_env(plugin_cls, manifest)

    plugin_cls.name = manifest.plugin.id  # type: ignore[attr-defined]
    plugin_cls.version = manifest.plugin.version  # type: ignore[attr-defined]
    plugin_cls._required_secrets = manifest.required_secret_names()  # type: ignore[attr-defined]
    plugin_cls._community_manifest = manifest  # type: ignore[attr-defined]
    plugin_cls._community_source_path = source_path  # type: ignore[attr-defined]


def load_plugins(
    kind: PluginKind,
    *,
    root: Path = COMMUNITY_ROOT,
    strict: bool | None = None,
) -> LoadResult:
    """Load every plugin of the given kind under ``root/<kind>/``.

    Returns a :class:`LoadResult` carrying both successes and per-entry
    failures. Iterating over the result still yields ``LoadedPlugin`` —
    callers that only care about the happy path can keep the
    ``for p in load_plugins(kind, root=…):`` shape unchanged.

    ``strict`` controls fail-fast behaviour: ``True`` re-raises the
    first exception, ``False`` swallows everything into ``failures``,
    ``None`` (default) consults the ``COMMUNITY_STRICT`` env var.
    """
    is_strict = _resolve_strict(strict)
    kind_dir = root / PLUGIN_KIND_DIRS[kind]
    plugins: list[LoadedPlugin] = []
    failures: list[LoadFailure] = []
    seen_ids: set[str] = set()

    for entry_name, entry_path in _iter_entries(kind_dir):
        try:
            source_root = _resolve_source_root(entry_path)
            manifest_dict = _parse_toml(_read_manifest_text(source_root))
            manifest = PluginManifest.model_validate(manifest_dict)
        except Exception as exc:
            if is_strict:
                raise
            logger.error(
                "[COMMUNITY_LOADER] kind=%s entry=%s failed to load manifest: %s",
                kind,
                entry_name,
                exc,
            )
            failures.append(LoadFailure(
                kind=kind,
                entry_name=entry_name,
                plugin_id=None,
                error_type="manifest_parse",
                message=str(exc),
                traceback=traceback.format_exc(),
            ))
            continue

        if manifest.plugin.kind != kind:
            msg = (
                f"manifest declares kind={manifest.plugin.kind!r} "
                f"but lives under community/{kind}/"
            )
            if is_strict:
                raise ValueError(f"[{entry_name}] {msg}")
            logger.error(
                "[COMMUNITY_LOADER] kind mismatch: %s manifest declares kind=%s",
                entry_name,
                manifest.plugin.kind,
            )
            failures.append(LoadFailure(
                kind=kind,
                entry_name=entry_name,
                plugin_id=manifest.plugin.id,
                error_type="kind_mismatch",
                message=msg,
            ))
            continue

        if manifest.plugin.id in seen_ids:
            # Duplicate ids are *always* fatal — they're a programming
            # mistake, not a transient runtime issue. Strict mode here
            # would be redundant.
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
            if is_strict:
                raise
            logger.error(
                "[COMMUNITY_LOADER] kind=%s id=%s failed to import entry point: %s",
                kind,
                manifest.plugin.id,
                exc,
            )
            failures.append(LoadFailure(
                kind=kind,
                entry_name=entry_name,
                plugin_id=manifest.plugin.id,
                error_type="import_error",
                message=str(exc),
                traceback=traceback.format_exc(),
            ))
            continue

        # Metadata attachment runs the REQUIRED_ENV cross-check (A7)
        # which can raise on Python ↔ TOML drift. Wrap in the same
        # strict/loose-mode contract as the import step so a single
        # plugin's contract violation doesn't take the whole catalog
        # down in production.
        try:
            _attach_community_metadata(plugin_cls, manifest, source_root)
        except Exception as exc:
            if is_strict:
                raise
            logger.error(
                "[COMMUNITY_LOADER] kind=%s id=%s metadata attach failed: %s",
                kind,
                manifest.plugin.id,
                exc,
            )
            failures.append(LoadFailure(
                kind=kind,
                entry_name=entry_name,
                plugin_id=manifest.plugin.id,
                error_type="metadata_error",
                message=str(exc),
                traceback=traceback.format_exc(),
            ))
            continue

        plugins.append(
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

    return LoadResult(plugins=plugins, failures=failures)


def load_presets(
    *, root: Path = COMMUNITY_ROOT, strict: bool | None = None
) -> PresetLoadResult:
    """Load every preset under ``root/presets/``.

    Mirrors :func:`load_plugins` — returns a result object with both
    successes and per-entry failures. Strict-mode handling is shared.
    """
    is_strict = _resolve_strict(strict)
    presets_dir = root / PRESET_DIR_NAME
    presets: list[LoadedPreset] = []
    failures: list[LoadFailure] = []
    seen_ids: set[str] = set()

    for entry_name, entry_path in _iter_entries(presets_dir):
        try:
            source_root = _resolve_source_root(entry_path)
            manifest_dict = _parse_toml(_read_manifest_text(source_root))
            manifest = PresetManifest.model_validate(manifest_dict)
        except Exception as exc:
            if is_strict:
                raise
            logger.error(
                "[COMMUNITY_LOADER] preset=%s failed to load manifest: %s",
                entry_name,
                exc,
            )
            failures.append(LoadFailure(
                kind="presets",
                entry_name=entry_name,
                plugin_id=None,
                error_type="manifest_parse",
                message=str(exc),
                traceback=traceback.format_exc(),
            ))
            continue

        if manifest.preset.id in seen_ids:
            raise ValueError(f"duplicate preset id {manifest.preset.id!r}")
        seen_ids.add(manifest.preset.id)

        yaml_path = source_root / manifest.preset.entry_point.file
        if not yaml_path.exists():
            msg = f"YAML file not found at {yaml_path}"
            if is_strict:
                raise FileNotFoundError(msg)
            logger.error(
                "[COMMUNITY_LOADER] preset=%s: %s",
                manifest.preset.id,
                msg,
            )
            failures.append(LoadFailure(
                kind="presets",
                entry_name=entry_name,
                plugin_id=manifest.preset.id,
                error_type="missing_yaml",
                message=msg,
            ))
            continue

        presets.append(
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

    return PresetLoadResult(presets=presets, failures=failures)


def load_all_plugins(
    *, root: Path = COMMUNITY_ROOT, strict: bool | None = None
) -> dict[str, LoadResult]:
    return {kind: load_plugins(kind, root=root, strict=strict) for kind in ALL_PLUGIN_KINDS}


__all__ = [
    "LoadFailure",
    "LoadResult",
    "LoadedPlugin",
    "LoadedPreset",
    "PresetLoadResult",
    "load_all_plugins",
    "load_plugins",
    "load_presets",
]
