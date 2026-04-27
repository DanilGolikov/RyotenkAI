"""Discovery, loading and preloading of ``community/libs/<name>/``.

Background — *why this exists*
------------------------------
``community/`` is intentionally not a Python package, and the loader
imports plugin classes via :func:`importlib.util.spec_from_file_location`
under one-shot unique names. That keeps plugins isolated from each
other but it also means a community plugin **cannot** ``from
community.foo import …`` to share helpers with another plugin.

To keep ``src/`` free of domain code and still let plugin authors
share modules across kinds, this module materialises
``community/libs/<name>/`` as a real Python package whose import
name is :data:`~src.community.constants.LIBS_NAMESPACE` (default
``community_libs``). Each direct subdirectory becomes
``community_libs.<id>``; subpackages are resolved lazily by the
standard import machinery once preload is done.

What's loaded
-------------
A lib lives at ``community/libs/<name>/`` (folder) **or**
``community/libs/<name>.zip`` (archive — same precedence semantics
as plugin packs: folder wins on collision). Either form must contain
a ``manifest.toml`` carrying a ``[lib]`` block with at least ``id``
and ``version`` (PEP 440). The manifest is the canonical source of
truth for the lib's identity and is required — a folder without
``manifest.toml`` is a load failure (no legacy passthrough).

Plugins import as if ``community_libs`` were a normal pip-installed
package::

    from community_libs.helixql.compiler import get_compiler
    from community_libs.helixql.extract import extract_query_text

The catalog calls :func:`preload_community_libs` once before loading
plugins, and again on every reload. The function is idempotent (same
``libs_root`` → no-op) and self-cleaning (different ``libs_root`` →
drop old subpackages from ``sys.modules``).
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path

from src.community.constants import (
    LIBS_DIR_NAME,
    LIBS_NAMESPACE,
    MANIFEST_FILENAME,
)
from src.community.manifest import LibManifest
from src.utils.logger import logger


@dataclass(frozen=True, slots=True)
class LoadedLib:
    """One successfully loaded community lib.

    Carried by :class:`~src.community.catalog.CommunityCatalog` so
    callers (the ``GET /libs`` API endpoint, the plugin loader's
    version-check) can look up libs by id.
    """

    manifest: LibManifest
    source_path: Path  #: Folder on disk hosting the lib's Python files.


@dataclass(frozen=True, slots=True)
class LibLoadFailure:
    """One lib entry the loader couldn't bring online."""

    entry_name: str
    lib_id: str | None
    error_type: str
    message: str
    traceback: str = ""


@dataclass(frozen=True, slots=True)
class LibLoadResult:
    libs: list[LoadedLib]
    failures: list[LibLoadFailure]


def _list_lib_entries(libs_root: Path) -> list[tuple[str, Path]]:
    """Return ``(entry_name, entry_path)`` for every lib candidate.

    Mirrors :func:`src.community.loader._iter_entries` — folders
    AND ``.zip`` archives both qualify, with the folder winning on
    name collision (a warning is logged so a stale archive next to
    live sources doesn't get used silently).
    """
    if not libs_root.is_dir():
        return []
    folders: dict[str, Path] = {}
    zips: dict[str, Path] = {}
    for entry in sorted(libs_root.iterdir()):
        if entry.name.startswith(".") or entry.name == "__pycache__":
            continue
        if entry.is_dir():
            folders[entry.name] = entry
        elif entry.is_file() and entry.suffix == ".zip":
            zips[entry.stem] = entry

    out: list[tuple[str, Path]] = []
    for name, path in folders.items():
        if name in zips:
            logger.warning(
                "[COMMUNITY_LIBS] %s shadows %s — folder wins; delete the "
                "folder to use the archive, or delete the archive if it is stale",
                path,
                zips[name],
            )
        out.append((name, path))
    for stem, path in zips.items():
        if stem in folders:
            continue
        out.append((path.name, path))
    return out


def _purge_namespace_from_sys_modules() -> None:
    """Drop ``community_libs`` and every ``community_libs.*`` from sys.modules."""
    prefix = f"{LIBS_NAMESPACE}."
    stale = [name for name in sys.modules if name == LIBS_NAMESPACE or name.startswith(prefix)]
    for name in stale:
        del sys.modules[name]


def _register_namespace(libs_root: Path) -> None:
    """Materialise :data:`LIBS_NAMESPACE` as a namespace-style package.

    Idempotent on the same ``libs_root``. Replaces the namespace when
    the root changes (purges any cached subpackages first so
    ``community_libs.helixql`` is freshly resolved against the new
    extraction roots).
    """
    libs_path_str = str(libs_root)
    existing = sys.modules.get(LIBS_NAMESPACE)
    if existing is not None and list(getattr(existing, "__path__", ())) == [libs_path_str]:
        return

    _purge_namespace_from_sys_modules()
    spec = importlib.machinery.ModuleSpec(
        name=LIBS_NAMESPACE,
        loader=None,
        is_package=True,
    )
    spec.submodule_search_locations = [libs_path_str]
    module = importlib.util.module_from_spec(spec)
    module.__path__ = [libs_path_str]  # type: ignore[attr-defined]
    sys.modules[LIBS_NAMESPACE] = module


def _register_subpackage(lib_id: str, source_path: Path) -> None:
    """Pre-populate ``community_libs.<lib_id>`` pointing at ``source_path``.

    Necessary when the lib is distributed as a zip archive — the
    extracted directory lives under ``community/.cache/<hash>/...``
    and is **not** a child of ``community/libs/``, so the namespace
    package's ``__path__`` (the ``libs_root``) wouldn't find it.

    For libs distributed as folders directly under ``community/libs/``,
    this still helps: it makes the package importable without
    consulting the file finder, and lets us key the cache on the
    canonical lib id (which matches the manifest, not necessarily the
    zip's stem).
    """
    full_name = f"{LIBS_NAMESPACE}.{lib_id}"
    if full_name in sys.modules:
        # Already registered — leave the cached module alone so plugin
        # imports captured at load time stay valid.
        return
    init = source_path / "__init__.py"
    if not init.is_file():
        raise ValueError(
            f"lib {lib_id!r} at {source_path} has no __init__.py — every "
            f"lib must be a regular Python package."
        )
    spec = importlib.util.spec_from_file_location(
        full_name,
        init,
        submodule_search_locations=[str(source_path)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot create import spec for {init}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(full_name, None)
        raise


def load_libs(*, libs_root: Path) -> LibLoadResult:
    """Resolve every lib under ``libs_root`` (folder or zip).

    Reads each entry's ``manifest.toml``, validates against
    :class:`LibManifest`, and returns a :class:`LibLoadResult`.
    Failures (missing manifest, bad version, name collision, etc.)
    are captured per entry; the catalog presents them through the
    same ``failures`` surface as plugin failures so the UI can
    render an actionable error banner.

    Side effects: each successfully resolved lib is pre-registered in
    :mod:`sys.modules` as ``community_libs.<id>``. The top-level
    namespace package is registered exactly once (or replaced when
    ``libs_root`` changes).
    """
    # Late imports — these helpers live in loader.py and we deliberately
    # avoid circular import at module load.
    from src.community.archive import ensure_extracted, resolve_extraction_root

    libs: list[LoadedLib] = []
    failures: list[LibLoadFailure] = []
    seen_ids: set[str] = set()

    if not libs_root.is_dir():
        # Nothing to load. Don't tear down any pre-existing namespace —
        # that's reserved for the explicit replacement path below.
        return LibLoadResult(libs=[], failures=[])

    entries = _list_lib_entries(libs_root)
    if not entries:
        # An empty libs/ directory: don't register the top-level
        # namespace at all. Pollution of ``sys.modules`` with an
        # empty ``community_libs`` would be visible to ``import
        # community_libs`` in unrelated code.
        return LibLoadResult(libs=[], failures=[])

    # Qualifying entries exist — register the top-level namespace
    # BEFORE walking subpackages so ``community_libs.<id>`` resolves
    # via the canonical search path before we explicitly preload
    # each subpackage.
    _register_namespace(libs_root)

    for entry_name, entry_path in entries:
        try:
            if entry_path.is_dir():
                source_root = entry_path
            elif entry_path.suffix == ".zip":
                source_root = resolve_extraction_root(ensure_extracted(entry_path))
            else:
                raise ValueError(f"unsupported lib entry: {entry_path}")

            manifest_path = source_root / MANIFEST_FILENAME
            if not manifest_path.is_file():
                raise FileNotFoundError(
                    f"{MANIFEST_FILENAME} not found in {source_root} "
                    f"(every lib must declare its [lib] block; minimum: "
                    f"id, version)"
                )
            import tomllib

            manifest_dict = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
            manifest = LibManifest.model_validate(manifest_dict)

            # The manifest's id must match the directory / zip stem to
            # keep import path predictable: ``community_libs.<id>`` is
            # the only entry the namespace knows about.
            stem = entry_name[:-4] if entry_name.endswith(".zip") else entry_name
            if manifest.lib.id != stem:
                raise ValueError(
                    f"lib at {entry_path} declares id={manifest.lib.id!r} "
                    f"but its directory/zip stem is {stem!r}. The two must "
                    f"match — rename the folder or fix the manifest."
                )

            if manifest.lib.id in seen_ids:
                # Should be impossible (folder/zip dedup happens earlier),
                # but be explicit just in case.
                raise ValueError(
                    f"duplicate lib id {manifest.lib.id!r} under {libs_root}"
                )
            seen_ids.add(manifest.lib.id)

            _register_subpackage(manifest.lib.id, source_root)
            libs.append(LoadedLib(manifest=manifest, source_path=source_root))
            logger.debug(
                "[COMMUNITY_LIBS] loaded id=%s version=%s from %s",
                manifest.lib.id,
                manifest.lib.version,
                source_root,
            )
        except Exception as exc:
            logger.error(
                "[COMMUNITY_LIBS] entry=%s failed to load: %s",
                entry_name,
                exc,
            )
            failures.append(
                LibLoadFailure(
                    entry_name=entry_name,
                    lib_id=None,
                    error_type=type(exc).__name__,
                    message=str(exc),
                    traceback=traceback.format_exc(),
                )
            )

    return LibLoadResult(libs=libs, failures=failures)


def preload_community_libs(libs_root: Path) -> tuple[str, ...]:
    """Convenience wrapper used by the catalog and CLI.

    Calls :func:`load_libs` and returns the tuple of successfully
    loaded lib ids (sorted) for the catalog's log line. Failures are
    discarded here — :class:`~src.community.catalog.CommunityCatalog`
    surfaces them through its own ``failures()`` accessor by calling
    :func:`load_libs` directly.
    """
    if not libs_root.is_dir():
        # No libs/ directory? Don't tear down any pre-existing namespace
        # set up by the production catalog (relevant in tests with tmp
        # roots that have no libs/).
        return ()
    result = load_libs(libs_root=libs_root)
    return tuple(sorted(lib.manifest.lib.id for lib in result.libs))


def libs_fingerprint_entries(libs_root: Path) -> list[tuple[str, float]]:
    """Files whose mtime invalidates the libs preload (catalog fingerprint).

    Tracks each lib's ``manifest.toml`` (drives identity / version) +
    every direct ``.py`` file at the lib root (drives the public
    surface that subpackages resolve through). Zip archives are
    fingerprinted by their own mtime; deeper edits inside a lib's
    subdirectories require a backend restart, same rule as ``src/``.
    """
    if not libs_root.is_dir():
        return []
    entries: list[tuple[str, float]] = []
    for entry_name, entry_path in _list_lib_entries(libs_root):
        try:
            if entry_path.is_file():
                # Zip — fingerprint the archive itself.
                entries.append(
                    (str(entry_path.relative_to(libs_root.parent)), entry_path.stat().st_mtime)
                )
                continue
            manifest = entry_path / MANIFEST_FILENAME
            if manifest.is_file():
                entries.append(
                    (str(manifest.relative_to(libs_root.parent)), manifest.stat().st_mtime)
                )
            for child in sorted(entry_path.iterdir()):
                if not child.is_file() or child.suffix != ".py":
                    continue
                entries.append(
                    (str(child.relative_to(libs_root.parent)), child.stat().st_mtime)
                )
        except OSError:
            pass
    return entries


def libs_root_for(community_root: Path) -> Path:
    """Return the libs directory for a given ``community/`` root."""
    return community_root / LIBS_DIR_NAME


__all__ = [
    "LibLoadFailure",
    "LibLoadResult",
    "LoadedLib",
    "libs_fingerprint_entries",
    "libs_root_for",
    "load_libs",
    "preload_community_libs",
]
