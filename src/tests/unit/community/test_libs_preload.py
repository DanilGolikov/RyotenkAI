"""
Tests for ``src.community.libs.preload_community_libs``.

The preload step is what makes ``community/libs/<name>/`` importable
as ``community_libs.<name>``. It runs once per catalog load (before
plugins) and is fingerprinted alongside plugin manifests, so any
behaviour change here is plugin-author-visible. The test matrix
covers the contract that downstream callers depend on:

- **Happy path** — a libs/ tree with one package gets registered as
  ``community_libs.<name>`` in :mod:`sys.modules`, and submodules
  resolve by ordinary ``from community_libs.<name>... import …``.
- **Idempotence** — calling preload twice with the same root is a
  no-op (same module object is preserved, no spurious churn).
- **Reload-with-different-root** — pointing the namespace at a new
  on-disk root drops the old subpackages from ``sys.modules`` so
  the new content actually wins.
- **Missing libs/** — preload silently no-ops; catalog still works.
- **Empty libs/** (root exists, no qualifying packages) — same as
  above, no namespace registered.
- **Non-package directories** — folders without ``__init__.py`` /
  hidden / ``__pycache__`` are skipped.
- **Catalog-driven** — ``catalog.ensure_loaded()`` triggers the
  preload before the first plugin is imported.
- **Fingerprint** — :func:`libs_fingerprint_entries` covers
  ``__init__.py`` + every direct ``*.py`` of every lib, in a
  deterministic order.

We instantiate :class:`CommunityCatalog` against a tmp tree for each
test — not the real catalog — so global ``sys.modules`` state never
ends up dirty after the suite. The shared autouse fixture below
takes a snapshot of ``community_libs*`` keys at start and restores
them at teardown to keep the surrounding catalogue tests
deterministic.
"""

from __future__ import annotations

import importlib
import sys
import textwrap
from pathlib import Path

import pytest

from src.community.catalog import CommunityCatalog
from src.community.constants import LIBS_NAMESPACE
from src.community.libs import (
    libs_fingerprint_entries,
    libs_root_for,
    preload_community_libs,
)


# ---------------------------------------------------------------------------
# Shared scaffolding
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _restore_libs_namespace():
    """Snapshot/restore ``community_libs*`` keys around each test, and
    invalidate the global catalog so the next caller does a real load.

    Without this:

    1. ``sys.modules[community_libs*]`` leaks across tests — modifying
       it in one test pollutes the next.
    2. Tests that create a fresh ``CommunityCatalog(root=tmp_path)``
       and call ``ensure_loaded()`` mutate the **global** plugin
       registries (validation/evaluation/reward/reports) since those
       are module-level singletons. A subsequent test that calls
       ``catalog.ensure_loaded()`` on the global catalog short-circuits
       (``_loaded=True``, fingerprint unchanged) and never re-populates
       — so the next test sees empty registries.

    Defence:

    - At setup, take a snapshot of every ``community_libs*`` key.
    - At teardown, drop everything currently under that namespace
      and restore the snapshot.
    - At teardown, also poke the global ``catalog._loaded = False``
      so the next ``ensure_loaded`` call does the real work and
      re-populates registries — cheap, lazy, and only charged once
      to the next test that actually needs the catalog.
    """
    prefix = f"{LIBS_NAMESPACE}."
    snapshot = {
        name: sys.modules[name]
        for name in list(sys.modules)
        if name == LIBS_NAMESPACE or name.startswith(prefix)
    }
    yield
    # Drop everything the test added under our namespace.
    for name in list(sys.modules):
        if name == LIBS_NAMESPACE or name.startswith(prefix):
            del sys.modules[name]
    # Restore the original modules.
    for name, module in snapshot.items():
        sys.modules[name] = module
    # Force the global catalog to do a real load on its next access —
    # tests that built tmp catalogs may have cleared the global plugin
    # registries via ``_populate_registries``.
    from src.community.catalog import catalog as _global_catalog

    _global_catalog._loaded = False


def _make_libs_root(parent: Path, libs: dict[str, dict[str, str]]) -> Path:
    """Materialise a libs/ tree from ``{lib_name: {filename: source}}``.

    Each ``lib_name`` becomes a subdirectory; its ``files`` dict is
    written verbatim. Callers that want the package to actually be a
    package must include ``"__init__.py"`` in ``files``. ``parent``
    may be a not-yet-existing directory — we create it (and the
    ``libs`` child) recursively.
    """
    root = parent / "libs"
    root.mkdir(parents=True, exist_ok=True)
    for name, files in libs.items():
        lib_dir = root / name
        lib_dir.mkdir(parents=True, exist_ok=True)
        for fname, source in files.items():
            (lib_dir / fname).write_text(textwrap.dedent(source))
    return root


# ---------------------------------------------------------------------------
# Positive paths
# ---------------------------------------------------------------------------


class TestPreloadHappyPath:
    def test_registers_top_level_namespace(self, tmp_path: Path) -> None:
        root = _make_libs_root(
            tmp_path,
            {"alpha": {"__init__.py": "value = 42\n"}},
        )
        names = preload_community_libs(root)
        assert names == ("alpha",)
        assert LIBS_NAMESPACE in sys.modules
        # __path__ is what makes ``import community_libs.alpha`` find the dir.
        assert list(sys.modules[LIBS_NAMESPACE].__path__) == [str(root)]

    def test_subpackage_is_importable(self, tmp_path: Path) -> None:
        root = _make_libs_root(
            tmp_path,
            {
                "alpha": {
                    "__init__.py": "from community_libs.alpha.inner import secret\n",
                    "inner.py": "secret = 'opensesame'\n",
                },
            },
        )
        preload_community_libs(root)
        module = importlib.import_module(f"{LIBS_NAMESPACE}.alpha")
        assert module.secret == "opensesame"

    def test_multiple_libs_register(self, tmp_path: Path) -> None:
        root = _make_libs_root(
            tmp_path,
            {
                "alpha": {"__init__.py": ""},
                "beta": {"__init__.py": ""},
                "gamma": {"__init__.py": ""},
            },
        )
        names = preload_community_libs(root)
        # Sorted for determinism.
        assert names == ("alpha", "beta", "gamma")

    def test_top_level_module_resolves(self, tmp_path: Path) -> None:
        # ``from community_libs.alpha.compiler import X`` only works if
        # the namespace's __path__ feeds the file finder for nested files.
        root = _make_libs_root(
            tmp_path,
            {
                "alpha": {
                    "__init__.py": "",
                    "compiler.py": "VALUE = 'x'\n",
                },
            },
        )
        preload_community_libs(root)
        compiler = importlib.import_module(f"{LIBS_NAMESPACE}.alpha.compiler")
        assert compiler.VALUE == "x"


# ---------------------------------------------------------------------------
# Idempotence + replacement semantics
# ---------------------------------------------------------------------------


class TestPreloadIdempotence:
    def test_same_root_twice_is_noop(self, tmp_path: Path) -> None:
        root = _make_libs_root(
            tmp_path,
            {"alpha": {"__init__.py": "value = 1\n"}},
        )
        preload_community_libs(root)
        first_module = sys.modules[LIBS_NAMESPACE]
        preload_community_libs(root)
        # Object identity preserved — no needless re-creation.
        assert sys.modules[LIBS_NAMESPACE] is first_module

    def test_subpackage_cache_survives_no_op_preload(self, tmp_path: Path) -> None:
        root = _make_libs_root(
            tmp_path,
            {
                "alpha": {
                    "__init__.py": "value = 1\n",
                },
            },
        )
        preload_community_libs(root)
        sub = importlib.import_module(f"{LIBS_NAMESPACE}.alpha")
        preload_community_libs(root)
        # Same root → cached subpackage is kept (importing a second
        # time returns the same object). This matters because plugins
        # may have already captured references via ``from … import …``.
        assert importlib.import_module(f"{LIBS_NAMESPACE}.alpha") is sub

    def test_different_root_replaces_namespace(self, tmp_path: Path) -> None:
        first_root = _make_libs_root(
            tmp_path / "v1",
            {"alpha": {"__init__.py": "value = 1\n"}},
        )
        preload_community_libs(first_root)
        first_alpha = importlib.import_module(f"{LIBS_NAMESPACE}.alpha")
        assert first_alpha.value == 1

        second_root = _make_libs_root(
            tmp_path / "v2",
            {"alpha": {"__init__.py": "value = 2\n"}},
        )
        preload_community_libs(second_root)

        # Old subpackage was purged from sys.modules — re-importing
        # picks up v2's content.
        assert f"{LIBS_NAMESPACE}.alpha" not in sys.modules
        second_alpha = importlib.import_module(f"{LIBS_NAMESPACE}.alpha")
        assert second_alpha.value == 2


# ---------------------------------------------------------------------------
# Edge cases — missing / empty / weird shapes
# ---------------------------------------------------------------------------


class TestPreloadEdgeCases:
    def test_missing_root_returns_empty(self, tmp_path: Path) -> None:
        # The autouse fixture clears community_libs at start, so we
        # know the namespace is absent here. The empty preload must
        # not try to register anything new.
        sys.modules.pop(LIBS_NAMESPACE, None)
        names = preload_community_libs(tmp_path / "does-not-exist")
        assert names == ()
        assert LIBS_NAMESPACE not in sys.modules

    def test_empty_root_returns_empty(self, tmp_path: Path) -> None:
        sys.modules.pop(LIBS_NAMESPACE, None)
        root = tmp_path / "libs"
        root.mkdir()
        names = preload_community_libs(root)
        assert names == ()
        assert LIBS_NAMESPACE not in sys.modules

    def test_directory_without_init_is_skipped(self, tmp_path: Path) -> None:
        root = tmp_path / "libs"
        (root / "alpha").mkdir(parents=True)  # no __init__.py
        names = preload_community_libs(root)
        assert names == ()

    def test_hidden_directories_skipped(self, tmp_path: Path) -> None:
        root = _make_libs_root(
            tmp_path,
            {
                "alpha": {"__init__.py": ""},
                ".hidden": {"__init__.py": ""},
            },
        )
        names = preload_community_libs(root)
        assert ".hidden" not in names
        assert "alpha" in names

    def test_pycache_skipped(self, tmp_path: Path) -> None:
        root = _make_libs_root(
            tmp_path,
            {
                "alpha": {"__init__.py": ""},
                "__pycache__": {"__init__.py": ""},
            },
        )
        names = preload_community_libs(root)
        assert "__pycache__" not in names

    def test_files_at_libs_root_skipped(self, tmp_path: Path) -> None:
        root = _make_libs_root(tmp_path, {"alpha": {"__init__.py": ""}})
        (root / "stray.py").write_text("# not a package\n")
        names = preload_community_libs(root)
        assert names == ("alpha",)

    def test_empty_second_call_leaves_existing_namespace_alone(self, tmp_path: Path) -> None:
        """A second call with no libs/ does NOT tear down the prior namespace.

        Production has exactly one ``community_libs`` namespace (set up by
        the global catalog at startup); tests routinely create extra
        ``CommunityCatalog`` instances rooted at tmp paths with no libs/
        of their own. Tearing down the namespace in those cases would
        invalidate the real plugins' captured imports, which is far worse
        than tolerating the unlikely "tmp catalog with libs → other tmp
        catalog without libs" leak.
        """
        first_root = _make_libs_root(
            tmp_path / "v1",
            {"alpha": {"__init__.py": ""}},
        )
        preload_community_libs(first_root)
        assert LIBS_NAMESPACE in sys.modules

        # Second call: empty/missing root → leave existing namespace alone.
        preload_community_libs(tmp_path / "ghost")
        assert LIBS_NAMESPACE in sys.modules
        # Existing path is preserved — no swap.
        assert list(sys.modules[LIBS_NAMESPACE].__path__) == [str(first_root)]


# ---------------------------------------------------------------------------
# Catalog integration
# ---------------------------------------------------------------------------


class TestCatalogPreloads:
    def test_ensure_loaded_triggers_preload(self, tmp_community_root: Path) -> None:
        # Drop a minimal lib in the temp community/ tree.
        libs_dir = tmp_community_root / "libs" / "demo"
        libs_dir.mkdir(parents=True)
        (libs_dir / "__init__.py").write_text("answer = 42\n")

        catalog = CommunityCatalog(root=tmp_community_root)
        catalog.ensure_loaded()

        assert LIBS_NAMESPACE in sys.modules
        demo = importlib.import_module(f"{LIBS_NAMESPACE}.demo")
        assert demo.answer == 42

    def test_reload_picks_up_new_lib(self, tmp_community_root: Path) -> None:
        libs_root = tmp_community_root / "libs"
        libs_root.mkdir()
        (libs_root / "first").mkdir()
        (libs_root / "first" / "__init__.py").write_text("")

        catalog = CommunityCatalog(root=tmp_community_root)
        catalog.ensure_loaded()
        # First call only sees `first`.
        assert sys.modules[LIBS_NAMESPACE].__path__ == [str(libs_root)]
        first = importlib.import_module(f"{LIBS_NAMESPACE}.first")
        assert first is not None

        # Add a second lib and reload.
        (libs_root / "second").mkdir()
        (libs_root / "second" / "__init__.py").write_text("value = 'two'\n")
        catalog.reload()

        second = importlib.import_module(f"{LIBS_NAMESPACE}.second")
        assert second.value == "two"

    def test_no_libs_dir_does_not_break_catalog(self, tmp_community_root: Path) -> None:
        # tmp_community_root exists but has no libs/ subdirectory.
        # The catalog must load without raising. We DON'T assert that
        # ``community_libs`` is absent from sys.modules — preload
        # intentionally leaves any pre-existing namespace alone (set
        # up by the global catalog) to keep production imports stable
        # under tmp-catalog tests.
        sys.modules.pop(LIBS_NAMESPACE, None)
        catalog = CommunityCatalog(root=tmp_community_root)
        catalog.ensure_loaded()  # must not raise
        # When we explicitly start with a clean slate (above pop) and
        # there's no libs/ to preload, no namespace is created.
        assert LIBS_NAMESPACE not in sys.modules


# ---------------------------------------------------------------------------
# Fingerprint coverage
# ---------------------------------------------------------------------------


class TestLibsFingerprint:
    def test_returns_empty_for_missing_root(self, tmp_path: Path) -> None:
        assert libs_fingerprint_entries(tmp_path / "ghost") == []

    def test_includes_init_and_top_level_modules(self, tmp_path: Path) -> None:
        root = _make_libs_root(
            tmp_path,
            {
                "alpha": {
                    "__init__.py": "",
                    "compiler.py": "",
                    "extract.py": "",
                },
            },
        )
        entries = libs_fingerprint_entries(root)
        relpaths = {p for p, _ in entries}
        assert "libs/alpha/__init__.py" in relpaths
        assert "libs/alpha/compiler.py" in relpaths
        assert "libs/alpha/extract.py" in relpaths

    def test_skips_subdirectories(self, tmp_path: Path) -> None:
        root = _make_libs_root(
            tmp_path,
            {"alpha": {"__init__.py": ""}},
        )
        # Nested subdir with .py files should NOT appear in fingerprint.
        (root / "alpha" / "tests").mkdir()
        (root / "alpha" / "tests" / "__init__.py").write_text("")
        (root / "alpha" / "tests" / "test_x.py").write_text("")

        relpaths = {p for p, _ in libs_fingerprint_entries(root)}
        assert "libs/alpha/tests/__init__.py" not in relpaths
        assert "libs/alpha/tests/test_x.py" not in relpaths

    def test_libs_root_for_helper(self, tmp_path: Path) -> None:
        assert libs_root_for(tmp_path) == tmp_path / "libs"


# ---------------------------------------------------------------------------
# Plugin import chain — end-to-end integration with the real helixql lib
# ---------------------------------------------------------------------------


class TestRealHelixqlLibImports:
    """Regression check: every HelixQL plugin imports through the lib.

    If we accidentally drop a plugin's import path or move a symbol
    out of the public ``community_libs.helixql`` re-exports, this test
    fires before the per-plugin tests do.
    """

    def test_real_catalog_preloads_helixql(self) -> None:
        from src.community.catalog import catalog as real_catalog

        # Force reload — our autouse ``_restore_libs_namespace`` fixture
        # cleared sys.modules but left the catalog's ``_loaded=True``,
        # so plain ``ensure_loaded`` would short-circuit and skip the
        # preload that re-registers ``community_libs``.
        real_catalog.reload()
        assert f"{LIBS_NAMESPACE}.helixql" in sys.modules

    def test_public_symbols_re_exported(self) -> None:
        from src.community.catalog import catalog as real_catalog

        real_catalog.reload()
        helixql = importlib.import_module(f"{LIBS_NAMESPACE}.helixql")
        for symbol in (
            "CompileResult",
            "HelixCompiler",
            "extract_query_text",
            "extract_schema_block",
            "extract_schema_and_query",
            "get_compiler",
            "hard_eval_errors",
            "normalize_query_text",
            "semantic_match_details",
        ):
            assert hasattr(helixql, symbol), f"missing public symbol {symbol!r}"

    @pytest.mark.parametrize(
        "module_path",
        [
            "community.validation.helixql_sapo_prompt_contract.plugin",
            "community.validation.helixql_preference_semantics.plugin",
            "community.validation.helixql_gold_syntax_backend.plugin",
            "community.evaluation.helixql_semantic_match.plugin",
            "community.evaluation.helixql_generated_syntax_backend.plugin",
            "community.reward.helixql_compiler_semantic.plugin",
        ],
    )
    def test_plugin_modules_load_through_catalog(self, module_path: str) -> None:
        """Every HelixQL plugin must register cleanly via the real catalog.

        We don't import the plugin modules directly (the loader uses
        unique private names); instead we trust the catalog log: if
        all 6 are listed under their kind, everything that imports
        ``community_libs.helixql`` got resolved correctly.
        """
        from src.community.catalog import catalog as real_catalog

        real_catalog.reload()
        plugin_id = module_path.split(".")[-2]
        kind = module_path.split(".")[1]
        ids = [p.manifest.plugin.id for p in real_catalog.plugins(kind)]
        assert plugin_id in ids, f"{plugin_id} missing from {kind} plugins: {ids}"
