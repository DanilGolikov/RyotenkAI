"""
Tests for the community libs loader (``src/community/libs.py``).

The libs loader is the single chokepoint between ``community/libs/<name>/``
on disk and the ``community_libs.<id>`` namespace plugins import from.
This test suite pins down five behaviours the rest of the platform
relies on:

1. **Preload mechanics** — top-level namespace + per-lib subpackage
   registration in :mod:`sys.modules`; idempotence on the same root;
   namespace replacement when ``libs_root`` changes; clean state on
   missing/empty roots.
2. **Manifest contract** — every lib needs a ``manifest.toml`` with a
   valid ``[lib]`` block (``id`` matching the folder/zip stem,
   PEP 440 ``version``). Failures are surfaced as
   :class:`LibLoadFailure` records, not silent skips.
3. **Zip distribution** — a lib distributed as a ``.zip`` archive
   loads via :mod:`src.community.archive` (cache-extracted root)
   exactly like a folder lib; folder wins on collision.
4. **Catalog integration** — ``catalog.libs()`` /
   ``catalog.get_lib(id)`` return the loaded manifests; failures
   surface through ``catalog.lib_failures()``.
5. **Plugin isolation** — content under ``community/libs/`` is
   never picked up by the plugin loader, even if it carries a
   ``manifest.toml`` shaped like a plugin manifest.
"""

from __future__ import annotations

import importlib
import sys
import textwrap
import zipfile
from pathlib import Path

import pytest

from src.community.catalog import CommunityCatalog
from src.community.constants import LIBS_NAMESPACE
from src.community.libs import (
    LibLoadFailure,
    LibLoadResult,
    LoadedLib,
    libs_fingerprint_entries,
    libs_root_for,
    load_libs,
    preload_community_libs,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _restore_libs_namespace():
    """Snapshot/restore ``community_libs*`` keys around each test.

    Without this the production ``community_libs.helixql`` from earlier
    tests would leak into tests that swap the namespace to a tmp tree
    (and vice versa).
    """
    prefix = f"{LIBS_NAMESPACE}."
    snapshot = {
        name: sys.modules[name]
        for name in list(sys.modules)
        if name == LIBS_NAMESPACE or name.startswith(prefix)
    }
    yield
    for name in list(sys.modules):
        if name == LIBS_NAMESPACE or name.startswith(prefix):
            del sys.modules[name]
    for name, module in snapshot.items():
        sys.modules[name] = module


def _lib_manifest(lib_id: str, version: str = "1.0.0") -> str:
    return textwrap.dedent(
        f"""
        schema_version = 1
        [lib]
        id = "{lib_id}"
        version = "{version}"
        """
    ).strip() + "\n"


def _make_lib_folder(libs_root: Path, lib_id: str, *, version: str = "1.0.0",
                    extra_files: dict[str, str] | None = None) -> Path:
    libs_root.mkdir(parents=True, exist_ok=True)
    lib_dir = libs_root / lib_id
    lib_dir.mkdir()
    (lib_dir / "__init__.py").write_text(f"VERSION = {version!r}\n")
    (lib_dir / "manifest.toml").write_text(_lib_manifest(lib_id, version))
    for fname, content in (extra_files or {}).items():
        (lib_dir / fname).write_text(content)
    return lib_dir


def _make_lib_zip(libs_root: Path, lib_id: str, *, version: str = "1.0.0") -> Path:
    """Build a libs/<id>.zip archive directly (without going through pack)."""
    libs_root.mkdir(parents=True, exist_ok=True)
    zip_path = libs_root / f"{lib_id}.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr(f"{lib_id}/__init__.py", f"VERSION = {version!r}\n")
        zf.writestr(f"{lib_id}/manifest.toml", _lib_manifest(lib_id, version))
    return zip_path


# ---------------------------------------------------------------------------
# 1. Preload mechanics
# ---------------------------------------------------------------------------


class TestPreloadMechanics:
    def test_registers_top_level_namespace_when_libs_present(self, tmp_path: Path) -> None:
        libs_root = tmp_path / "libs"
        _make_lib_folder(libs_root, "alpha")
        load_libs(libs_root=libs_root)
        assert LIBS_NAMESPACE in sys.modules
        assert list(sys.modules[LIBS_NAMESPACE].__path__) == [str(libs_root)]

    def test_subpackage_importable_after_preload(self, tmp_path: Path) -> None:
        libs_root = tmp_path / "libs"
        _make_lib_folder(libs_root, "alpha", extra_files={"util.py": "answer = 42\n"})
        load_libs(libs_root=libs_root)
        util = importlib.import_module(f"{LIBS_NAMESPACE}.alpha.util")
        assert util.answer == 42

    def test_multiple_libs_register(self, tmp_path: Path) -> None:
        libs_root = tmp_path / "libs"
        for name in ("alpha", "beta", "gamma"):
            _make_lib_folder(libs_root, name)
        names = preload_community_libs(libs_root)
        assert names == ("alpha", "beta", "gamma")

    def test_idempotent_on_same_root(self, tmp_path: Path) -> None:
        libs_root = tmp_path / "libs"
        _make_lib_folder(libs_root, "alpha")
        load_libs(libs_root=libs_root)
        first_module = sys.modules[LIBS_NAMESPACE]
        load_libs(libs_root=libs_root)
        assert sys.modules[LIBS_NAMESPACE] is first_module

    def test_replaces_namespace_on_root_change(self, tmp_path: Path) -> None:
        v1 = tmp_path / "v1" / "libs"
        v2 = tmp_path / "v2" / "libs"
        _make_lib_folder(v1, "alpha", version="1.0.0")
        _make_lib_folder(v2, "alpha", version="2.0.0")
        load_libs(libs_root=v1)
        a1 = importlib.import_module(f"{LIBS_NAMESPACE}.alpha")
        assert a1.VERSION == "1.0.0"
        load_libs(libs_root=v2)
        # Subpackage cache should be repopulated against v2.
        a2 = importlib.import_module(f"{LIBS_NAMESPACE}.alpha")
        assert a2.VERSION == "2.0.0"


# ---------------------------------------------------------------------------
# 2. Manifest contract
# ---------------------------------------------------------------------------


class TestManifestContract:
    def test_missing_manifest_is_a_load_failure(self, tmp_path: Path) -> None:
        libs_root = tmp_path / "libs"
        libs_root.mkdir(parents=True)
        broken = libs_root / "alpha"
        broken.mkdir()
        (broken / "__init__.py").write_text("")
        # No manifest.toml.
        result = load_libs(libs_root=libs_root)
        assert result.libs == []
        assert len(result.failures) == 1
        assert "manifest.toml" in result.failures[0].message

    def test_id_must_match_folder_name(self, tmp_path: Path) -> None:
        libs_root = tmp_path / "libs"
        _make_lib_folder(libs_root, "alpha")
        # Rewrite manifest with a mismatched id.
        (libs_root / "alpha" / "manifest.toml").write_text(
            _lib_manifest("beta")
        )
        result = load_libs(libs_root=libs_root)
        assert result.libs == []
        assert any("alpha" in f.message and "beta" in f.message for f in result.failures)

    def test_invalid_pep440_version_rejected(self, tmp_path: Path) -> None:
        libs_root = tmp_path / "libs"
        _make_lib_folder(libs_root, "alpha")
        (libs_root / "alpha" / "manifest.toml").write_text(
            'schema_version = 1\n[lib]\nid = "alpha"\nversion = "not-a-version"\n'
        )
        result = load_libs(libs_root=libs_root)
        assert result.libs == []
        assert any("PEP 440" in f.message or "not a valid" in f.message for f in result.failures)

    def test_missing_init_py_rejected(self, tmp_path: Path) -> None:
        # Folder + manifest, but no __init__.py — every lib must be a
        # real Python package.
        libs_root = tmp_path / "libs"
        libs_root.mkdir(parents=True)
        broken = libs_root / "alpha"
        broken.mkdir()
        (broken / "manifest.toml").write_text(_lib_manifest("alpha"))
        result = load_libs(libs_root=libs_root)
        assert result.libs == []
        assert any("__init__.py" in f.message for f in result.failures)

    def test_unknown_schema_version_rejected(self, tmp_path: Path) -> None:
        libs_root = tmp_path / "libs"
        _make_lib_folder(libs_root, "alpha")
        (libs_root / "alpha" / "manifest.toml").write_text(
            'schema_version = 99\n[lib]\nid = "alpha"\nversion = "1.0.0"\n'
        )
        result = load_libs(libs_root=libs_root)
        assert result.libs == []
        assert any("schema_version" in f.message for f in result.failures)


# ---------------------------------------------------------------------------
# 3. Zip distribution
# ---------------------------------------------------------------------------


class TestZipDistribution:
    def test_zip_lib_loads(self, tmp_path: Path) -> None:
        libs_root = tmp_path / "libs"
        _make_lib_zip(libs_root, "alpha", version="2.0.0")
        result = load_libs(libs_root=libs_root)
        assert len(result.libs) == 1
        loaded = result.libs[0]
        assert loaded.manifest.lib.id == "alpha"
        assert loaded.manifest.lib.version == "2.0.0"
        # Subpackage import resolves to the extracted root.
        mod = importlib.import_module(f"{LIBS_NAMESPACE}.alpha")
        assert mod.VERSION == "2.0.0"
        # source_path points at the cache-extracted folder, not the zip.
        assert "/.cache/" in str(loaded.source_path)

    def test_folder_wins_over_zip(self, tmp_path: Path) -> None:
        libs_root = tmp_path / "libs"
        _make_lib_folder(libs_root, "alpha", version="1.0.0")
        _make_lib_zip(libs_root, "alpha", version="9.9.9")
        result = load_libs(libs_root=libs_root)
        assert len(result.libs) == 1
        # Folder version wins.
        assert result.libs[0].manifest.lib.version == "1.0.0"

    def test_zip_only_no_folder(self, tmp_path: Path) -> None:
        libs_root = tmp_path / "libs"
        _make_lib_zip(libs_root, "alpha", version="3.0.0")
        result = load_libs(libs_root=libs_root)
        assert len(result.libs) == 1
        assert result.libs[0].manifest.lib.version == "3.0.0"


# ---------------------------------------------------------------------------
# 4. Catalog integration
# ---------------------------------------------------------------------------


class TestCatalogIntegration:
    def test_catalog_libs_accessor(self, tmp_community_root: Path) -> None:
        _make_lib_folder(tmp_community_root / "libs", "demo", version="0.1.0")
        catalog = CommunityCatalog(root=tmp_community_root)
        catalog.ensure_loaded()
        libs = catalog.libs()
        assert len(libs) == 1
        assert libs[0].manifest.lib.id == "demo"

    def test_catalog_get_lib_raises_for_missing(self, tmp_community_root: Path) -> None:
        catalog = CommunityCatalog(root=tmp_community_root)
        catalog.ensure_loaded()
        with pytest.raises(KeyError, match="ghost"):
            catalog.get_lib("ghost")

    def test_catalog_surfaces_lib_failures(self, tmp_community_root: Path) -> None:
        # Build a broken lib (missing __init__.py) so the catalog
        # captures it as a failure.
        libs_root = tmp_community_root / "libs"
        libs_root.mkdir(parents=True)
        bad = libs_root / "bad"
        bad.mkdir()
        (bad / "manifest.toml").write_text(_lib_manifest("bad"))
        catalog = CommunityCatalog(root=tmp_community_root)
        catalog.ensure_loaded()
        failures = catalog.lib_failures()
        assert len(failures) == 1
        assert "bad" in failures[0].entry_name

    def test_real_catalog_carries_helixql(self) -> None:
        from src.community.catalog import catalog as real_catalog

        real_catalog.reload()
        ids = sorted(lib.manifest.lib.id for lib in real_catalog.libs())
        assert "helixql" in ids


# ---------------------------------------------------------------------------
# 5. Plugin loader isolation
# ---------------------------------------------------------------------------


class TestLibsInvisibleToPluginLoader:
    def test_lib_with_plugin_shaped_manifest_ignored(
        self, tmp_community_root: Path
    ) -> None:
        # Drop something inside libs/ that LOOKS like a plugin (has
        # [plugin] section). The plugin loader must NOT walk into
        # libs/.
        from src.community.loader import load_all_plugins

        bogus = tmp_community_root / "libs" / "shape_shifter"
        bogus.mkdir(parents=True)
        (bogus / "__init__.py").write_text("")
        (bogus / "manifest.toml").write_text(
            'schema_version = 5\n[plugin]\nid = "bogus"\n'
            'kind = "validation"\nversion = "1.0.0"\n'
            '[plugin.entry_point]\nmodule = "x"\nclass = "Y"\n'
        )
        result = load_all_plugins(root=tmp_community_root)
        for kind, kind_result in result.items():
            ids = [p.manifest.plugin.id for p in kind_result.plugins]
            assert "bogus" not in ids


# ---------------------------------------------------------------------------
# 6. Fingerprint coverage
# ---------------------------------------------------------------------------


class TestLibsFingerprint:
    def test_returns_empty_for_missing_root(self, tmp_path: Path) -> None:
        assert libs_fingerprint_entries(tmp_path / "ghost") == []

    def test_includes_manifest_and_top_level_modules(self, tmp_path: Path) -> None:
        libs_root = tmp_path / "libs"
        _make_lib_folder(
            libs_root,
            "alpha",
            extra_files={"compiler.py": "", "extract.py": ""},
        )
        relpaths = {p for p, _ in libs_fingerprint_entries(libs_root)}
        assert "libs/alpha/manifest.toml" in relpaths
        assert "libs/alpha/compiler.py" in relpaths
        assert "libs/alpha/extract.py" in relpaths
        assert "libs/alpha/__init__.py" in relpaths

    def test_fingerprint_picks_up_zip_archives(self, tmp_path: Path) -> None:
        libs_root = tmp_path / "libs"
        _make_lib_zip(libs_root, "alpha")
        relpaths = {p for p, _ in libs_fingerprint_entries(libs_root)}
        assert "libs/alpha.zip" in relpaths

    def test_libs_root_for_helper(self, tmp_path: Path) -> None:
        assert libs_root_for(tmp_path) == tmp_path / "libs"


# ---------------------------------------------------------------------------
# 7. Empty / missing root edge cases
# ---------------------------------------------------------------------------


class TestEmptyAndMissingRoots:
    def test_missing_root_returns_empty(self, tmp_path: Path) -> None:
        sys.modules.pop(LIBS_NAMESPACE, None)
        result = load_libs(libs_root=tmp_path / "ghost")
        assert result == LibLoadResult(libs=[], failures=[])
        assert LIBS_NAMESPACE not in sys.modules

    def test_empty_root_returns_empty(self, tmp_path: Path) -> None:
        sys.modules.pop(LIBS_NAMESPACE, None)
        libs_root = tmp_path / "libs"
        libs_root.mkdir()
        result = load_libs(libs_root=libs_root)
        assert result == LibLoadResult(libs=[], failures=[])
        assert LIBS_NAMESPACE not in sys.modules

    def test_hidden_directories_skipped(self, tmp_path: Path) -> None:
        libs_root = tmp_path / "libs"
        _make_lib_folder(libs_root, "alpha")
        hidden = libs_root / ".secret"
        hidden.mkdir()
        (hidden / "__init__.py").write_text("")
        (hidden / "manifest.toml").write_text(_lib_manifest(".secret"))
        result = load_libs(libs_root=libs_root)
        ids = [lib.manifest.lib.id for lib in result.libs]
        assert ".secret" not in ids
        assert "alpha" in ids
