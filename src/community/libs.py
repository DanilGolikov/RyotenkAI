"""Preload :data:`~src.community.constants.LIBS_NAMESPACE` from ``community/libs/``.

Background — *why this exists*
------------------------------
``community/`` is intentionally not a Python package, and the loader
imports plugin classes via :func:`importlib.util.spec_from_file_location`
under one-shot unique names. That keeps plugins isolated from each
other but it also means a community plugin **cannot** ``from
community.foo import …`` to share helpers with another plugin.

Pure ``src/utils/domains/<domain>.py`` modules used to fill that gap —
but doing so violates the layering rule ("``src/`` is the platform,
``community/`` is the domain"). To keep ``src/`` free of domain code
and still let plugin authors share modules across kinds, this module
materialises ``community/libs/`` as a real Python package whose import
name is :data:`~src.community.constants.LIBS_NAMESPACE` (default
``community_libs``). Each direct subdirectory with an ``__init__.py``
becomes ``community_libs.<name>``; subpackages are resolved lazily by
the standard import machinery.

Plugins import as if ``community_libs`` were a normal pip-installed
package::

    from community_libs.helixql.compiler import get_compiler
    from community_libs.helixql.extract import extract_query_text

The catalog calls :func:`preload_community_libs` once before loading
plugins, and again on every reload. The function is idempotent (same
``libs_root`` → no-op) and self-cleaning (different ``libs_root`` → drop
old subpackages from ``sys.modules`` so tests with temporary roots get
clean state).
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from src.community.constants import LIBS_DIR_NAME, LIBS_NAMESPACE
from src.utils.logger import logger

if TYPE_CHECKING:
    pass


def _list_lib_subpackages(libs_root: Path) -> list[Path]:
    """Return direct subdirs of ``libs_root`` that look like Python packages.

    A directory qualifies when it contains an ``__init__.py`` and its
    name doesn't start with ``.`` or equal ``__pycache__``. The order
    is ``sorted`` for determinism (used in fingerprinting and logs).
    """
    if not libs_root.is_dir():
        return []
    out: list[Path] = []
    for entry in sorted(libs_root.iterdir()):
        if not entry.is_dir():
            continue
        if entry.name.startswith(".") or entry.name == "__pycache__":
            continue
        if not (entry / "__init__.py").is_file():
            continue
        out.append(entry)
    return out


def _purge_namespace_from_sys_modules() -> None:
    """Drop ``community_libs`` and every ``community_libs.*`` from sys.modules."""
    prefix = f"{LIBS_NAMESPACE}."
    stale = [name for name in sys.modules if name == LIBS_NAMESPACE or name.startswith(prefix)]
    for name in stale:
        del sys.modules[name]


def preload_community_libs(libs_root: Path) -> tuple[str, ...]:
    """Make ``community/libs/`` importable as :data:`LIBS_NAMESPACE`.

    Steps performed in order:

    1. If ``libs_root`` is missing or empty (no qualifying subpackages),
       purge any stale ``community_libs.*`` entries from
       :mod:`sys.modules` and return an empty tuple. Catalog stays
       functional — libs are an opt-in feature.
    2. If ``community_libs`` is already in ``sys.modules`` and its
       ``__path__`` matches ``libs_root``, return early — idempotent.
    3. Otherwise: purge the old namespace (if any), construct a fresh
       :class:`importlib.machinery.ModuleSpec` with ``loader=None`` and
       ``is_package=True``, build a module from it whose ``__path__``
       points at ``libs_root``, and register it in
       :mod:`sys.modules`. Subpackages (``community_libs.helixql``,
       …) load lazily on first import via the default file finder
       reading ``__path__``.

    Returns the tuple of subpackage names registered (sorted), useful
    for tests and for the catalog log line.
    """
    subpackages = _list_lib_subpackages(libs_root)
    libs_path_str = str(libs_root)
    existing = sys.modules.get(LIBS_NAMESPACE)
    existing_path = list(getattr(existing, "__path__", ())) if existing is not None else []

    if not subpackages:
        # Nothing to preload. Two cases:
        # 1. The existing namespace points at *this* root — keep it (no-op).
        # 2. The existing namespace points elsewhere (or doesn't exist) —
        #    leave it alone. We don't tear down a namespace that another
        #    catalog (typically the real one in tests using a tmp root)
        #    set up, because doing so would invalidate plugin imports
        #    captured at module load time. The "tmp1 with libs → tmp2
        #    without libs" leak is a contrived scenario we explicitly
        #    accept to keep the production catalog's namespace stable.
        return ()

    if existing is not None and existing_path == [libs_path_str]:
        # Same root: keep the existing namespace and any cached
        # subpackages — they're still valid.
        return tuple(p.name for p in subpackages)

    # Different root or never registered: clean slate.
    _purge_namespace_from_sys_modules()

    spec = importlib.machinery.ModuleSpec(
        name=LIBS_NAMESPACE,
        loader=None,
        is_package=True,
    )
    # ``submodule_search_locations`` is what makes ``from community_libs.x
    # import …`` find ``<libs_root>/x/__init__.py``.
    spec.submodule_search_locations = [libs_path_str]
    module = importlib.util.module_from_spec(spec)
    # Mirror onto __path__ — some import paths consult the attribute
    # directly rather than going through the spec.
    module.__path__ = [libs_path_str]  # type: ignore[attr-defined]
    sys.modules[LIBS_NAMESPACE] = module

    names = tuple(p.name for p in subpackages)
    logger.debug(
        "[COMMUNITY_LIBS] preloaded %s.{%s} from %s",
        LIBS_NAMESPACE,
        ",".join(names),
        libs_root,
    )
    return names


def libs_fingerprint_entries(libs_root: Path) -> list[tuple[str, float]]:
    """Files whose mtime invalidates the libs preload (for catalog fingerprint).

    Walks ``libs_root/<lib>/__init__.py`` plus every direct ``.py`` file
    one level deeper (covers ``compiler.py`` / ``extract.py`` / etc.).
    Deep changes inside lib subpackages aren't fingerprinted — adding
    or editing those still requires a backend restart, like any
    framework code change. We track only the surface that affects
    *which* libs exist and what their public exports are.
    """
    if not libs_root.is_dir():
        return []
    entries: list[tuple[str, float]] = []
    for lib_dir in _list_lib_subpackages(libs_root):
        # __init__.py — re-exports are the most-broken-on-rename surface.
        init = lib_dir / "__init__.py"
        try:
            entries.append((str(init.relative_to(libs_root.parent)), init.stat().st_mtime))
        except OSError:
            pass
        # Direct top-level submodules (compiler.py, extract.py, ...).
        for child in sorted(lib_dir.iterdir()):
            if not child.is_file() or child.suffix != ".py" or child.name == "__init__.py":
                continue
            try:
                entries.append((str(child.relative_to(libs_root.parent)), child.stat().st_mtime))
            except OSError:
                pass
    return entries


def libs_root_for(community_root: Path) -> Path:
    """Return the libs directory for a given ``community/`` root."""
    return community_root / LIBS_DIR_NAME


__all__ = [
    "libs_fingerprint_entries",
    "libs_root_for",
    "preload_community_libs",
]
