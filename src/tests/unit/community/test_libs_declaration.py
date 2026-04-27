"""
Tests for the ``[plugin].libs`` manifest field + ``REQUIRED_LIBS`` ClassVar.

Coverage matrix (mirrors the platform-wide 7-category taxonomy used in
``test_phase_complete_coverage.py``):

1. **Positive** — plugin with a valid lib loads cleanly; ``ui_manifest()``
   surfaces the field; the registry returns the plugin.
2. **Negative** — declared lib is absent on disk; cross-check mismatch;
   bad name format; duplicates.
3. **Boundary** — empty list is the default and skips both checks; long
   list with multiple libs.
4. **Invariant** — manifest field name format (snake_case
   identifier); Python-side declaration must be a tuple of strings;
   set-based comparison ignores order.
5. **Dependency errors** — libs declared in TOML but
   ``community/libs/<name>/`` doesn't exist or has no ``__init__.py``.
6. **Regression** — plugins without a ``libs`` field still load
   identically to before this feature; ``ui_manifest()`` always emits
   the field (even when empty).
7. **Logic-specific** — ``REQUIRED_LIBS`` empty tuple skips the
   cross-check; declaring on Python side without TOML side is caught
   with a clear diff; sync-libs round-trips set normalisation.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from src.community.libs import preload_community_libs
from src.community.loader import load_plugins
from src.community.manifest import PluginManifest, PluginSpec


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_community_with_libs(tmp_community_root: Path) -> Path:
    """Augment ``tmp_community_root`` with a ``libs/<name>/`` package.

    Tests that drop a plugin declaring ``libs = ["alpha"]`` need
    ``alpha`` to be a real package on disk for the loader to accept
    them. Single fixture saves ~5 lines per test.
    """
    libs_dir = tmp_community_root / "libs" / "alpha"
    libs_dir.mkdir(parents=True)
    (libs_dir / "__init__.py").write_text("value = 1\n")
    return tmp_community_root


def _write_plugin(
    plugin_dir: Path,
    *,
    plugin_id: str,
    kind: str,
    libs_block: str = "",
    class_body: str = "    pass",
    class_name: str = "TestPlugin",
) -> Path:
    plugin_dir.mkdir(parents=True, exist_ok=True)
    libs_line = f"\nlibs = {libs_block}" if libs_block else ""
    (plugin_dir / "manifest.toml").write_text(
        textwrap.dedent(
            f"""
            [plugin]
            id = "{plugin_id}"
            kind = "{kind}"
            version = "1.0.0"{libs_line}

            [plugin.entry_point]
            module = "plugin"
            class = "{class_name}"
            """
        ).strip()
        + "\n"
    )
    (plugin_dir / "plugin.py").write_text(
        f"class {class_name}:\n{class_body}\n"
    )
    return plugin_dir


# ---------------------------------------------------------------------------
# 1. Positive
# ---------------------------------------------------------------------------


class TestLibsDeclarationPositive:
    def test_plugin_with_existing_lib_loads(
        self, tmp_community_with_libs: Path
    ) -> None:
        _write_plugin(
            tmp_community_with_libs / "validation" / "uses_alpha",
            plugin_id="uses_alpha",
            kind="validation",
            libs_block='["alpha"]',
        )
        result = load_plugins("validation", root=tmp_community_with_libs, strict=True)
        ids = [p.manifest.plugin.id for p in result.plugins]
        assert "uses_alpha" in ids
        loaded = next(p for p in result.plugins if p.manifest.plugin.id == "uses_alpha")
        assert loaded.manifest.plugin.libs == ["alpha"]

    def test_ui_manifest_surfaces_libs_field(self) -> None:
        # Default empty list must still appear in ui_manifest so the
        # web UI can rely on the key being present.
        manifest = PluginManifest(
            plugin=PluginSpec(
                id="x",
                kind="validation",
                entry_point={"module": "plugin", "class": "X"},
            )
        )
        assert manifest.ui_manifest()["libs"] == []

    def test_ui_manifest_passes_through_declared_libs(self) -> None:
        manifest = PluginManifest(
            plugin=PluginSpec(
                id="x",
                kind="validation",
                libs=["alpha", "beta"],
                entry_point={"module": "plugin", "class": "X"},
            )
        )
        assert manifest.ui_manifest()["libs"] == ["alpha", "beta"]


# ---------------------------------------------------------------------------
# 2. Negative
# ---------------------------------------------------------------------------


class TestLibsDeclarationNegative:
    def test_missing_lib_on_disk_fails_load(
        self, tmp_community_root: Path
    ) -> None:
        _write_plugin(
            tmp_community_root / "validation" / "needs_ghost",
            plugin_id="needs_ghost",
            kind="validation",
            libs_block='["ghost"]',
        )
        # Strict mode: the loader raises with a helpful message.
        with pytest.raises(ValueError, match="ghost"):
            load_plugins("validation", root=tmp_community_root, strict=True)

    def test_missing_lib_collected_as_failure_in_loose_mode(
        self, tmp_community_root: Path
    ) -> None:
        _write_plugin(
            tmp_community_root / "validation" / "needs_ghost",
            plugin_id="needs_ghost",
            kind="validation",
            libs_block='["ghost"]',
        )
        result = load_plugins("validation", root=tmp_community_root, strict=False)
        assert any("ghost" in f.message for f in result.failures)
        assert "needs_ghost" not in [p.manifest.plugin.id for p in result.plugins]

    def test_mismatched_required_libs_raises_with_diff(
        self, tmp_community_with_libs: Path
    ) -> None:
        _write_plugin(
            tmp_community_with_libs / "validation" / "drift",
            plugin_id="drift",
            kind="validation",
            libs_block='["alpha"]',
            class_body='    REQUIRED_LIBS = ("alpha", "beta")',
        )
        with pytest.raises(ValueError, match="REQUIRED_LIBS.*beta") as excinfo:
            load_plugins("validation", root=tmp_community_with_libs, strict=True)
        # Diff names BOTH the side missing the entry and the offender.
        msg = str(excinfo.value)
        assert "REQUIRED_LIBS but missing from manifest" in msg

    def test_invalid_name_format_rejected_at_manifest_parse(self) -> None:
        with pytest.raises(ValueError, match="snake_case"):
            PluginSpec(
                id="x",
                kind="validation",
                libs=["bad-name"],
                entry_point={"module": "plugin", "class": "X"},
            )

    def test_uppercase_name_rejected(self) -> None:
        with pytest.raises(ValueError, match="snake_case"):
            PluginSpec(
                id="x",
                kind="validation",
                libs=["BadName"],
                entry_point={"module": "plugin", "class": "X"},
            )

    def test_duplicate_libs_rejected(self) -> None:
        with pytest.raises(ValueError, match="duplicate"):
            PluginSpec(
                id="x",
                kind="validation",
                libs=["alpha", "alpha"],
                entry_point={"module": "plugin", "class": "X"},
            )


# ---------------------------------------------------------------------------
# 3. Boundary
# ---------------------------------------------------------------------------


class TestLibsDeclarationBoundary:
    def test_empty_libs_is_the_default(self) -> None:
        manifest = PluginManifest(
            plugin=PluginSpec(
                id="x",
                kind="validation",
                entry_point={"module": "plugin", "class": "X"},
            )
        )
        assert manifest.plugin.libs == []

    def test_empty_libs_skips_cross_check(self, tmp_community_root: Path) -> None:
        # Plugin has no libs in TOML and no REQUIRED_LIBS on the class →
        # no cross-check, no presence check, plain load.
        _write_plugin(
            tmp_community_root / "validation" / "no_libs",
            plugin_id="no_libs",
            kind="validation",
        )
        result = load_plugins("validation", root=tmp_community_root, strict=True)
        ids = [p.manifest.plugin.id for p in result.plugins]
        assert "no_libs" in ids

    def test_multiple_libs_all_validated(self, tmp_community_root: Path) -> None:
        for name in ("alpha", "beta", "gamma"):
            (tmp_community_root / "libs" / name).mkdir(parents=True)
            (tmp_community_root / "libs" / name / "__init__.py").write_text("")
        _write_plugin(
            tmp_community_root / "validation" / "multi",
            plugin_id="multi",
            kind="validation",
            libs_block='["alpha", "beta", "gamma"]',
        )
        result = load_plugins("validation", root=tmp_community_root, strict=True)
        loaded = next(p for p in result.plugins if p.manifest.plugin.id == "multi")
        assert sorted(loaded.manifest.plugin.libs) == ["alpha", "beta", "gamma"]


# ---------------------------------------------------------------------------
# 4. Invariant
# ---------------------------------------------------------------------------


class TestLibsDeclarationInvariants:
    def test_required_libs_must_be_tuple_of_strings(
        self, tmp_community_with_libs: Path
    ) -> None:
        _write_plugin(
            tmp_community_with_libs / "validation" / "weird",
            plugin_id="weird",
            kind="validation",
            libs_block='["alpha"]',
            class_body='    REQUIRED_LIBS = ["alpha"]',  # list, not tuple
        )
        with pytest.raises(TypeError, match="REQUIRED_LIBS must be a tuple"):
            load_plugins("validation", root=tmp_community_with_libs, strict=True)

    def test_required_libs_set_based_comparison_ignores_order(
        self, tmp_community_root: Path
    ) -> None:
        for name in ("alpha", "beta"):
            (tmp_community_root / "libs" / name).mkdir(parents=True)
            (tmp_community_root / "libs" / name / "__init__.py").write_text("")
        # TOML order ["beta", "alpha"]; Python tuple ("alpha", "beta").
        # Cross-check must accept this — order does NOT matter.
        _write_plugin(
            tmp_community_root / "validation" / "ordered",
            plugin_id="ordered",
            kind="validation",
            libs_block='["beta", "alpha"]',
            class_body='    REQUIRED_LIBS = ("alpha", "beta")',
        )
        result = load_plugins("validation", root=tmp_community_root, strict=True)
        ids = [p.manifest.plugin.id for p in result.plugins]
        assert "ordered" in ids


# ---------------------------------------------------------------------------
# 5. Dependency errors (lib path resolution)
# ---------------------------------------------------------------------------


class TestLibsDependencyResolution:
    def test_lib_dir_without_init_py_treated_as_missing(
        self, tmp_community_root: Path
    ) -> None:
        # Folder exists but no __init__.py — not a real package.
        (tmp_community_root / "libs" / "alpha").mkdir(parents=True)
        # NO __init__.py file.
        _write_plugin(
            tmp_community_root / "validation" / "uses_broken",
            plugin_id="uses_broken",
            kind="validation",
            libs_block='["alpha"]',
        )
        with pytest.raises(ValueError, match="alpha"):
            load_plugins("validation", root=tmp_community_root, strict=True)

    def test_libs_root_outside_community_root_does_not_match(
        self, tmp_community_root: Path, tmp_path: Path
    ) -> None:
        # A libs/ in some other tmp dir doesn't satisfy the plugin —
        # presence check is rooted at the catalog's own community_root.
        (tmp_path / "libs" / "alpha").mkdir(parents=True)
        (tmp_path / "libs" / "alpha" / "__init__.py").write_text("")
        _write_plugin(
            tmp_community_root / "validation" / "scoped",
            plugin_id="scoped",
            kind="validation",
            libs_block='["alpha"]',
        )
        with pytest.raises(ValueError, match="alpha"):
            load_plugins("validation", root=tmp_community_root, strict=True)


# ---------------------------------------------------------------------------
# 6. Regression — pre-libs plugins keep working
# ---------------------------------------------------------------------------


class TestLibsRegression:
    def test_legacy_plugin_without_libs_field_loads_unchanged(
        self, tmp_community_root: Path
    ) -> None:
        # No `libs = …` in TOML; no REQUIRED_LIBS on the class. This
        # is the shape of every plugin before this feature landed.
        _write_plugin(
            tmp_community_root / "validation" / "legacy",
            plugin_id="legacy",
            kind="validation",
        )
        result = load_plugins("validation", root=tmp_community_root, strict=True)
        ids = [p.manifest.plugin.id for p in result.plugins]
        assert "legacy" in ids
        legacy = next(p for p in result.plugins if p.manifest.plugin.id == "legacy")
        assert legacy.manifest.plugin.libs == []

    def test_real_helixql_plugins_still_load(self) -> None:
        """The 6 HelixQL plugins were updated to declare libs=['helixql'];
        the real catalog must still load them (and only them) via the
        production preload path.
        """
        from src.community.catalog import catalog as real_catalog

        real_catalog.reload()
        helixql_plugins: list[str] = []
        for kind in ("validation", "evaluation", "reward"):
            for p in real_catalog.plugins(kind):
                if "helixql" in p.manifest.plugin.libs:
                    helixql_plugins.append(p.manifest.plugin.id)
        assert sorted(helixql_plugins) == [
            "helixql_compiler_semantic",
            "helixql_generated_syntax_backend",
            "helixql_gold_syntax_backend",
            "helixql_preference_semantics",
            "helixql_sapo_prompt_contract",
            "helixql_semantic_match",
        ]


# ---------------------------------------------------------------------------
# 7. Logic-specific (REQUIRED_LIBS contract details)
# ---------------------------------------------------------------------------


class TestLibsLogicSpecific:
    def test_empty_required_libs_skips_check(
        self, tmp_community_with_libs: Path
    ) -> None:
        # TOML declares libs=["alpha"], class declares REQUIRED_LIBS=()
        # → cross-check is skipped (the empty tuple opts out). Manifest
        # is the source of truth in that case.
        _write_plugin(
            tmp_community_with_libs / "validation" / "py_silent",
            plugin_id="py_silent",
            kind="validation",
            libs_block='["alpha"]',
            class_body='    REQUIRED_LIBS = ()',
        )
        result = load_plugins("validation", root=tmp_community_with_libs, strict=True)
        ids = [p.manifest.plugin.id for p in result.plugins]
        assert "py_silent" in ids

    def test_required_libs_extra_in_python_only(
        self, tmp_community_with_libs: Path
    ) -> None:
        # Python declares ("alpha", "extra") but TOML has only ["alpha"]
        # → cross-check fails. Order of error message names matters
        # for the user — we report ``extra`` as missing from manifest.
        (tmp_community_with_libs / "libs" / "extra").mkdir(parents=True)
        (tmp_community_with_libs / "libs" / "extra" / "__init__.py").write_text("")
        _write_plugin(
            tmp_community_with_libs / "validation" / "py_extra",
            plugin_id="py_extra",
            kind="validation",
            libs_block='["alpha"]',
            class_body='    REQUIRED_LIBS = ("alpha", "extra")',
        )
        with pytest.raises(ValueError, match="extra") as excinfo:
            load_plugins("validation", root=tmp_community_with_libs, strict=True)
        assert "REQUIRED_LIBS but missing from manifest" in str(excinfo.value)

    def test_required_libs_extra_in_toml_only(
        self, tmp_community_with_libs: Path
    ) -> None:
        (tmp_community_with_libs / "libs" / "extra").mkdir(parents=True)
        (tmp_community_with_libs / "libs" / "extra" / "__init__.py").write_text("")
        _write_plugin(
            tmp_community_with_libs / "validation" / "toml_extra",
            plugin_id="toml_extra",
            kind="validation",
            libs_block='["alpha", "extra"]',
            class_body='    REQUIRED_LIBS = ("alpha",)',
        )
        with pytest.raises(ValueError, match="extra") as excinfo:
            load_plugins("validation", root=tmp_community_with_libs, strict=True)
        assert "manifest but missing from REQUIRED_LIBS" in str(excinfo.value)

    def test_sync_libs_normalises_to_sorted_unique(
        self, tmp_community_with_libs: Path
    ) -> None:
        from src.community.sync import sync_plugin_libs

        (tmp_community_with_libs / "libs" / "beta").mkdir(parents=True)
        (tmp_community_with_libs / "libs" / "beta" / "__init__.py").write_text("")
        # Python declares unsorted with a duplicate import-style alias
        # — sync should write sorted unique names regardless.
        _write_plugin(
            tmp_community_with_libs / "validation" / "to_sync",
            plugin_id="to_sync",
            kind="validation",
            libs_block='["beta", "alpha"]',
            class_body='    REQUIRED_LIBS = ("beta", "alpha", "alpha")',
        )
        # Preload so the plugin module imports cleanly.
        preload_community_libs(tmp_community_with_libs / "libs")
        result = sync_plugin_libs(
            tmp_community_with_libs / "validation" / "to_sync"
        )
        # Final manifest text contains alphabetically sorted libs.
        assert 'libs = ["alpha", "beta"]' in result.new_text
