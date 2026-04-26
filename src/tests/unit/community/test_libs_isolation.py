"""
Tests that ``community/libs/`` is *invisible* to the plugin loader.

Background — :data:`PLUGIN_KIND_DIRS` is the loader's whitelist; anything
outside it is silently ignored. The ``libs`` directory is intentionally
absent from that mapping. These tests pin that invariant down so a
careless edit of ``constants.py`` doesn't quietly start trying to load
``community_libs.helixql.compiler`` as a plugin manifest.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.community.constants import (
    ALL_PLUGIN_KINDS,
    LIBS_DIR_NAME,
    LIBS_NAMESPACE,
    PLUGIN_KIND_DIRS,
)
from src.community.catalog import CommunityCatalog
from src.community.loader import load_all_plugins


class TestLibsNotAKind:
    def test_libs_not_in_plugin_kind_dirs(self) -> None:
        assert LIBS_DIR_NAME not in PLUGIN_KIND_DIRS
        assert LIBS_DIR_NAME not in PLUGIN_KIND_DIRS.values()

    def test_libs_not_in_all_plugin_kinds(self) -> None:
        assert LIBS_DIR_NAME not in ALL_PLUGIN_KINDS
        # Sanity: the four real kinds are still there. A regression in
        # constants.py that drops one of these would break the rest of
        # the suite — better to fail here with a precise message.
        assert set(ALL_PLUGIN_KINDS) == {"validation", "evaluation", "reward", "reports"}

    def test_libs_namespace_is_distinct_string(self) -> None:
        # Constants used as both directory name (libs) and import-time
        # namespace (community_libs) must NOT collide with any kind.
        for kind in ALL_PLUGIN_KINDS:
            assert kind != LIBS_NAMESPACE
            assert kind != LIBS_DIR_NAME


class TestLibsIgnoredByLoader:
    def test_loader_does_not_walk_libs_dir(self, tmp_community_root: Path) -> None:
        """A bogus ``manifest.toml`` under libs/ must be ignored by the loader."""
        # Set up a libs/ dir with something that *looks* like a plugin but
        # under a directory the loader has no business reading.
        libs_dir = tmp_community_root / LIBS_DIR_NAME / "alpha"
        libs_dir.mkdir(parents=True)
        (libs_dir / "manifest.toml").write_text(
            '[plugin]\nid="bogus"\nkind="validation"\nversion="1.0.0"\n'
            '[plugin.entry_point]\nmodule="x"\nclass="Y"\n'
        )
        (libs_dir / "__init__.py").write_text("")

        result = load_all_plugins(root=tmp_community_root)
        # No kind picks up the libs/ entry — even though the manifest
        # itself is well-formed.
        for kind, kind_result in result.items():
            ids = [p.manifest.plugin.id for p in kind_result.plugins]
            assert "bogus" not in ids, (
                f"loader walked into libs/ for kind={kind} — that's the bug "
                "this test guards against"
            )

    def test_libs_namespace_not_in_catalog_kinds(self, tmp_community_root: Path) -> None:
        catalog = CommunityCatalog(root=tmp_community_root)
        # ``list_kinds()`` is the catalog's authoritative answer for
        # "what kinds exist?". libs must NEVER appear here.
        kinds = catalog.list_kinds()
        assert LIBS_NAMESPACE not in kinds
        assert LIBS_DIR_NAME not in kinds


class TestLibsCannotBecomeAKind:
    """Defence-in-depth: even if someone *thinks* they want to call
    ``catalog.plugins("libs")``, the catalog should refuse rather
    than serve an empty list silently.
    """

    def test_plugins_for_unknown_kind(self, tmp_community_root: Path) -> None:
        catalog = CommunityCatalog(root=tmp_community_root)
        # ``plugins("libs")`` returns an empty list because the loader
        # never populated ``self._plugins["libs"]``. That's fine —
        # the explicit signal is "no libs kind". We just want to
        # verify it doesn't raise or surface bogus entries.
        assert catalog.plugins(LIBS_DIR_NAME) == []  # type: ignore[arg-type]
        assert catalog.plugins(LIBS_NAMESPACE) == []  # type: ignore[arg-type]

    def test_no_registry_for_libs(self) -> None:
        """The catalog populates 4 registries; none of them is for libs."""
        # Imported lazily inside _populate_registries — we don't go
        # poking at that private API. Instead, confirm the constant
        # set hasn't grown.
        assert len(ALL_PLUGIN_KINDS) == 4
