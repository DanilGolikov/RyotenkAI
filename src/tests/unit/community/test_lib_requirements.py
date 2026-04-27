"""
Tests for the v5 ``[[lib_requirements]]`` plugin contract.

Parallel to :mod:`test_libs_preload` (which covers the *libs* side
of the picture); this suite covers the *plugin* side: how plugins
declare their lib dependencies, what the loader does with them, and
how ``ryotenkai community sync`` keeps the manifest in sync with the
class-level ``REQUIRED_LIBS`` ClassVar.

Coverage matrix mirrors the platform-wide 7-category taxonomy used
by ``test_phase_complete_coverage.py``:

1. **Schema** — :class:`LibRequirement` validation; ``PluginManifest.lib_requirements``
   accepts arrays of tables; v4-style ``[plugin].libs = [...]`` is
   rejected with a precise migration hint; ``ui_manifest()`` surfaces
   the field.
2. **REQUIRED_LIBS shapes** — bare names, ``(name, version)`` tuples,
   :class:`LibRequirement` instances all normalise correctly.
3. **Cross-check** — Python ↔ TOML drift is caught with set-based
   diffing on names + byte comparison on versions.
4. **Version satisfaction** — at plugin load, the catalog's loaded
   libs are matched against each requirement; mismatches block the
   plugin.
5. **Sync** — extending an existing plugin's ``REQUIRED_LIBS`` is
   reflected in ``[[lib_requirements]]`` after sync, with
   author-typed version specifiers preserved when code is silent.
6. **Author** — ``[plugin].author`` round-trips through
   ``ui_manifest()`` and is preserved by sync.
7. **Regression** — the 6 real HelixQL plugins still load through
   the production catalog with the new shape.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from src.community.libs import preload_community_libs
from src.community.loader import (
    _normalise_required_libs,
    load_plugins,
)
from src.community.manifest import (
    EntryPoint,
    LATEST_SCHEMA_VERSION,
    LibManifest,
    LibRequirement,
    LibSpec,
    PluginManifest,
    PluginSpec,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def with_alpha_lib(tmp_community_root: Path) -> Path:
    """Drop a minimal ``community/libs/alpha/`` (1.0.0) into the tmp tree.

    Plugins under the same ``tmp_community_root`` can declare
    ``[[lib_requirements]] name = "alpha"`` and the loader's
    presence/version check will pass.
    """
    libs_root = tmp_community_root / "libs" / "alpha"
    libs_root.mkdir(parents=True)
    (libs_root / "__init__.py").write_text("")
    (libs_root / "manifest.toml").write_text(
        textwrap.dedent(
            """
            schema_version = 1
            [lib]
            id = "alpha"
            version = "1.0.0"
            """
        ).strip() + "\n"
    )
    return tmp_community_root


def _write_plugin(
    plugin_dir: Path,
    *,
    plugin_id: str,
    kind: str,
    lib_requirements_block: str = "",
    class_body: str = "    pass",
    class_name: str = "TestPlugin",
    author: str = "",
) -> Path:
    plugin_dir.mkdir(parents=True, exist_ok=True)
    author_line = f'\nauthor = "{author}"' if author else ""
    (plugin_dir / "manifest.toml").write_text(
        textwrap.dedent(
            f"""
            schema_version = {LATEST_SCHEMA_VERSION}
            [plugin]
            id = "{plugin_id}"
            kind = "{kind}"
            version = "1.0.0"{author_line}

            [plugin.entry_point]
            module = "plugin"
            class = "{class_name}"
            """
        ).strip()
        + "\n"
        + (lib_requirements_block.strip() + "\n" if lib_requirements_block else "")
    )
    (plugin_dir / "plugin.py").write_text(f"class {class_name}:\n{class_body}\n")
    return plugin_dir


def _libs_by_id(community_root: Path) -> dict[str, LibManifest]:
    """Mirror what the catalog passes to ``load_plugins``."""
    from src.community.libs import load_libs, libs_root_for

    result = load_libs(libs_root=libs_root_for(community_root))
    return {lib.manifest.lib.id: lib.manifest for lib in result.libs}


# ---------------------------------------------------------------------------
# 1. Schema
# ---------------------------------------------------------------------------


class TestSchema:
    def test_lib_requirement_accepts_empty_version(self) -> None:
        req = LibRequirement(name="alpha")
        assert req.version == ""

    def test_lib_requirement_accepts_pep440_specifier(self) -> None:
        req = LibRequirement(name="alpha", version=">=1.0.0,<2.0.0")
        assert req.version == ">=1.0.0,<2.0.0"

    def test_invalid_specifier_rejected(self) -> None:
        with pytest.raises(ValueError, match="PEP 440"):
            LibRequirement(name="alpha", version="not-a-specifier")

    def test_invalid_lib_name_rejected(self) -> None:
        with pytest.raises(ValueError, match="snake_case"):
            LibRequirement(name="Bad-Name", version="")

    def test_v4_libs_field_raises_migration_hint(self) -> None:
        # A leftover v4 manifest with [plugin].libs = [...] should
        # surface a migration error, not Pydantic's vague "extra fields".
        with pytest.raises(ValueError, match="schema v5"):
            PluginSpec(
                id="x",
                kind="validation",
                libs=["alpha"],
                entry_point=EntryPoint(module="plugin", **{"class": "X"}),
            )

    def test_lib_requirements_unique_by_name(self) -> None:
        with pytest.raises(ValueError, match="duplicate"):
            PluginManifest(
                plugin=PluginSpec(
                    id="x",
                    kind="validation",
                    entry_point=EntryPoint(module="plugin", **{"class": "X"}),
                ),
                lib_requirements=[
                    LibRequirement(name="alpha", version=">=1.0"),
                    LibRequirement(name="alpha", version="<2.0"),
                ],
            )

    def test_ui_manifest_surfaces_lib_requirements(self) -> None:
        m = PluginManifest(
            plugin=PluginSpec(
                id="x",
                kind="validation",
                entry_point=EntryPoint(module="plugin", **{"class": "X"}),
            ),
            lib_requirements=[
                LibRequirement(name="alpha", version=">=1.0"),
                LibRequirement(name="beta"),
            ],
        )
        ui = m.ui_manifest()
        assert ui["lib_requirements"] == [
            {"name": "alpha", "version": ">=1.0"},
            {"name": "beta", "version": ""},
        ]


class TestLibManifestSchema:
    def test_minimal_lib_manifest(self) -> None:
        m = LibManifest(lib=LibSpec(id="alpha", version="1.0.0"))
        assert m.schema_version == 1
        assert m.lib.id == "alpha"
        assert m.lib.version == "1.0.0"
        assert m.lib.author == ""

    def test_lib_id_must_be_snake_case(self) -> None:
        with pytest.raises(ValueError, match="snake_case"):
            LibSpec(id="Bad-Name", version="1.0.0")

    def test_invalid_pep440_version(self) -> None:
        with pytest.raises(ValueError, match="PEP 440"):
            LibSpec(id="alpha", version="not-a-version")

    def test_unknown_schema_version_rejected(self) -> None:
        with pytest.raises(ValueError, match="schema_version"):
            LibManifest(schema_version=99, lib=LibSpec(id="alpha", version="1.0.0"))

    def test_ui_manifest(self) -> None:
        m = LibManifest(
            lib=LibSpec(
                id="alpha",
                version="1.0.0",
                description="x",
                author="A <a@b.com>",
            )
        )
        assert m.ui_manifest() == {
            "schema_version": 1,
            "id": "alpha",
            "version": "1.0.0",
            "description": "x",
            "author": "A <a@b.com>",
        }


# ---------------------------------------------------------------------------
# 2. REQUIRED_LIBS normalisation
# ---------------------------------------------------------------------------


class TestRequiredLibsShapes:
    def test_string_only(self) -> None:
        out = _normalise_required_libs(("alpha",), plugin_cls_name="X")
        assert out == [LibRequirement(name="alpha", version="")]

    def test_tuple_with_version(self) -> None:
        out = _normalise_required_libs((("alpha", ">=1.0"),), plugin_cls_name="X")
        assert out == [LibRequirement(name="alpha", version=">=1.0")]

    def test_lib_requirement_instance(self) -> None:
        req = LibRequirement(name="alpha", version=">=1.0")
        out = _normalise_required_libs((req,), plugin_cls_name="X")
        assert out == [req]

    def test_mixed_shapes(self) -> None:
        out = _normalise_required_libs(
            (
                "alpha",
                ("beta", "<2.0"),
                LibRequirement(name="gamma", version="==1.0"),
            ),
            plugin_cls_name="X",
        )
        assert {r.name for r in out} == {"alpha", "beta", "gamma"}

    def test_non_tuple_outer_rejected(self) -> None:
        with pytest.raises(TypeError, match="must be a tuple"):
            _normalise_required_libs(["alpha"], plugin_cls_name="X")

    def test_unknown_inner_type_rejected(self) -> None:
        with pytest.raises(TypeError):
            _normalise_required_libs((123,), plugin_cls_name="X")


# ---------------------------------------------------------------------------
# 3. Cross-check Python ↔ TOML drift
# ---------------------------------------------------------------------------


class TestCrossCheck:
    def test_matching_no_op(self, with_alpha_lib: Path) -> None:
        community_root = with_alpha_lib
        _write_plugin(
            community_root / "validation" / "ok",
            plugin_id="ok",
            kind="validation",
            lib_requirements_block='\n[[lib_requirements]]\nname = "alpha"\nversion = ">=1.0"',
            class_body='    REQUIRED_LIBS = (("alpha", ">=1.0"),)',
        )
        result = load_plugins(
            "validation",
            root=community_root,
            strict=True,
            libs_by_id=_libs_by_id(community_root),
        )
        ids = [p.manifest.plugin.id for p in result.plugins]
        assert "ok" in ids

    def test_python_only_lib_diff(self, with_alpha_lib: Path) -> None:
        community_root = with_alpha_lib
        _write_plugin(
            community_root / "validation" / "drift",
            plugin_id="drift",
            kind="validation",
            class_body='    REQUIRED_LIBS = ("alpha",)',
        )
        with pytest.raises(ValueError, match="REQUIRED_LIBS but missing from manifest"):
            load_plugins(
                "validation",
                root=community_root,
                strict=True,
                libs_by_id=_libs_by_id(community_root),
            )

    def test_toml_only_lib_diff(self, with_alpha_lib: Path) -> None:
        # TOML carries an entry that ``REQUIRED_LIBS`` doesn't mention.
        # Cross-check should fire because ``REQUIRED_LIBS`` is non-empty
        # (the empty case is covered by ``test_empty_required_libs_skips_check``).
        community_root = with_alpha_lib
        # Need a second lib so the diff has both sides — the loader's
        # presence check fires before the cross-check otherwise.
        beta = community_root / "libs" / "beta"
        beta.mkdir(parents=True)
        (beta / "__init__.py").write_text("")
        (beta / "manifest.toml").write_text(
            'schema_version = 1\n[lib]\nid = "beta"\nversion = "1.0.0"\n'
        )
        _write_plugin(
            community_root / "validation" / "drift",
            plugin_id="drift",
            kind="validation",
            lib_requirements_block=(
                '\n[[lib_requirements]]\nname = "alpha"\n'
                '\n[[lib_requirements]]\nname = "beta"'
            ),
            class_body='    REQUIRED_LIBS = ("alpha",)',
        )
        with pytest.raises(
            ValueError, match="manifest but missing from REQUIRED_LIBS.*beta"
        ):
            load_plugins(
                "validation",
                root=community_root,
                strict=True,
                libs_by_id=_libs_by_id(community_root),
            )

    def test_version_byte_mismatch_caught(self, with_alpha_lib: Path) -> None:
        community_root = with_alpha_lib
        _write_plugin(
            community_root / "validation" / "drift",
            plugin_id="drift",
            kind="validation",
            lib_requirements_block='\n[[lib_requirements]]\nname = "alpha"\nversion = ">=1.0"',
            class_body='    REQUIRED_LIBS = (("alpha", ">=2.0"),)',
        )
        with pytest.raises(ValueError, match=r"alpha: version code='>=2\.0' vs toml='>=1\.0'"):
            load_plugins(
                "validation",
                root=community_root,
                strict=True,
                libs_by_id=_libs_by_id(community_root),
            )

    def test_empty_required_libs_skips_check(self, with_alpha_lib: Path) -> None:
        # TOML declares the requirement; Python has no REQUIRED_LIBS —
        # cross-check is skipped (manifest stays authoritative).
        community_root = with_alpha_lib
        _write_plugin(
            community_root / "validation" / "py_silent",
            plugin_id="py_silent",
            kind="validation",
            lib_requirements_block='\n[[lib_requirements]]\nname = "alpha"',
        )
        result = load_plugins(
            "validation",
            root=community_root,
            strict=True,
            libs_by_id=_libs_by_id(community_root),
        )
        ids = [p.manifest.plugin.id for p in result.plugins]
        assert "py_silent" in ids


# ---------------------------------------------------------------------------
# 4. Version satisfaction at plugin load
# ---------------------------------------------------------------------------


class TestVersionSatisfaction:
    def test_constraint_satisfied(self, with_alpha_lib: Path) -> None:
        community_root = with_alpha_lib
        _write_plugin(
            community_root / "validation" / "needs_alpha",
            plugin_id="needs_alpha",
            kind="validation",
            lib_requirements_block='\n[[lib_requirements]]\nname = "alpha"\nversion = ">=1.0,<2.0"',
        )
        result = load_plugins(
            "validation",
            root=community_root,
            strict=True,
            libs_by_id=_libs_by_id(community_root),
        )
        ids = [p.manifest.plugin.id for p in result.plugins]
        assert "needs_alpha" in ids

    def test_constraint_unsatisfied(self, with_alpha_lib: Path) -> None:
        community_root = with_alpha_lib
        _write_plugin(
            community_root / "validation" / "needs_alpha2",
            plugin_id="needs_alpha2",
            kind="validation",
            lib_requirements_block='\n[[lib_requirements]]\nname = "alpha"\nversion = ">=2.0"',
        )
        with pytest.raises(ValueError, match="lib 'alpha' is at version 1.0.0"):
            load_plugins(
                "validation",
                root=community_root,
                strict=True,
                libs_by_id=_libs_by_id(community_root),
            )

    def test_missing_lib_blocks_load(self, tmp_community_root: Path) -> None:
        # Plugin requires 'ghost' but no community/libs/ghost/ exists.
        _write_plugin(
            tmp_community_root / "validation" / "needs_ghost",
            plugin_id="needs_ghost",
            kind="validation",
            lib_requirements_block='\n[[lib_requirements]]\nname = "ghost"',
        )
        with pytest.raises(ValueError, match="requires lib 'ghost'"):
            load_plugins(
                "validation",
                root=tmp_community_root,
                strict=True,
                libs_by_id=_libs_by_id(tmp_community_root),
            )

    def test_no_constraint_just_presence(self, with_alpha_lib: Path) -> None:
        # Empty version field → loader only checks presence, not version.
        community_root = with_alpha_lib
        _write_plugin(
            community_root / "validation" / "any_alpha",
            plugin_id="any_alpha",
            kind="validation",
            lib_requirements_block='\n[[lib_requirements]]\nname = "alpha"',
        )
        result = load_plugins(
            "validation",
            root=community_root,
            strict=True,
            libs_by_id=_libs_by_id(community_root),
        )
        ids = [p.manifest.plugin.id for p in result.plugins]
        assert "any_alpha" in ids


# ---------------------------------------------------------------------------
# 5. Sync — REQUIRED_LIBS → manifest
# ---------------------------------------------------------------------------


class TestSyncIntegration:
    def test_sync_writes_lib_requirements_from_class(
        self, with_alpha_lib: Path
    ) -> None:
        from src.community.sync import sync_plugin_manifest

        community_root = with_alpha_lib
        plugin_dir = community_root / "validation" / "to_sync"
        # Plugin has REQUIRED_LIBS in code but no lib_requirements in TOML.
        plugin_dir.mkdir(parents=True)
        (plugin_dir / "manifest.toml").write_text(
            textwrap.dedent(
                f"""
                schema_version = {LATEST_SCHEMA_VERSION}
                [plugin]
                id = "to_sync"
                kind = "validation"
                version = "1.0.0"

                [plugin.entry_point]
                module = "plugin"
                class = "ToSyncPlugin"
                """
            ).strip()
            + "\n"
        )
        (plugin_dir / "plugin.py").write_text(
            "from src.data.validation.base import ValidationPlugin\n"
            "class ToSyncPlugin(ValidationPlugin):\n"
            '    REQUIRED_LIBS = (("alpha", ">=1.0"),)\n'
            "    def validate(self, dataset):\n        ...\n"
        )
        # Preload so the plugin module imports cleanly during sync.
        preload_community_libs(community_root / "libs")
        result = sync_plugin_manifest(plugin_dir, bump="patch")
        assert "lib_requirements" in result.new_text
        assert 'name = "alpha"' in result.new_text
        assert 'version = ">=1.0"' in result.new_text

    def test_sync_preserves_author(self, with_alpha_lib: Path) -> None:
        from src.community.sync import sync_plugin_manifest

        community_root = with_alpha_lib
        plugin_dir = community_root / "validation" / "with_author"
        plugin_dir.mkdir(parents=True)
        (plugin_dir / "manifest.toml").write_text(
            textwrap.dedent(
                f"""
                schema_version = {LATEST_SCHEMA_VERSION}
                [plugin]
                id = "with_author"
                kind = "validation"
                version = "1.0.0"
                author = "Alice <alice@example.com>"

                [plugin.entry_point]
                module = "plugin"
                class = "AuthoredPlugin"
                """
            ).strip()
            + "\n"
        )
        (plugin_dir / "plugin.py").write_text(
            "from src.data.validation.base import ValidationPlugin\n"
            "class AuthoredPlugin(ValidationPlugin):\n"
            "    def validate(self, dataset):\n        ...\n"
        )
        preload_community_libs(community_root / "libs")
        result = sync_plugin_manifest(plugin_dir, bump="patch")
        assert 'author = "Alice <alice@example.com>"' in result.new_text


# ---------------------------------------------------------------------------
# 6. Author field
# ---------------------------------------------------------------------------


class TestAuthor:
    def test_author_default_empty(self) -> None:
        m = PluginManifest(
            plugin=PluginSpec(
                id="x",
                kind="validation",
                entry_point=EntryPoint(module="plugin", **{"class": "X"}),
            )
        )
        assert m.plugin.author == ""

    def test_author_round_trips_through_ui_manifest(self) -> None:
        m = PluginManifest(
            plugin=PluginSpec(
                id="x",
                kind="validation",
                author="Bob <bob@example.com>",
                entry_point=EntryPoint(module="plugin", **{"class": "X"}),
            )
        )
        assert m.ui_manifest()["author"] == "Bob <bob@example.com>"

    def test_lib_author_round_trips(self) -> None:
        m = LibManifest(
            lib=LibSpec(id="alpha", version="1.0.0", author="Carol")
        )
        assert m.ui_manifest()["author"] == "Carol"


# ---------------------------------------------------------------------------
# 7. Regression — real HelixQL plugins still load
# ---------------------------------------------------------------------------


class TestRealHelixqlPluginsRegression:
    def test_six_helixql_plugins_load_with_lib_requirement(self) -> None:
        from src.community.catalog import catalog as real_catalog

        real_catalog.reload()
        helixql_plugins: list[str] = []
        for kind in ("validation", "evaluation", "reward"):
            for p in real_catalog.plugins(kind):
                if any(req.name == "helixql" for req in p.manifest.lib_requirements):
                    helixql_plugins.append(p.manifest.plugin.id)
        assert sorted(helixql_plugins) == [
            "helixql_compiler_semantic",
            "helixql_generated_syntax_backend",
            "helixql_gold_syntax_backend",
            "helixql_preference_semantics",
            "helixql_sapo_prompt_contract",
            "helixql_semantic_match",
        ]

    def test_real_helixql_lib_at_version_one_zero(self) -> None:
        from src.community.catalog import catalog as real_catalog

        real_catalog.reload()
        helixql = real_catalog.get_lib("helixql")
        assert helixql.manifest.lib.version == "1.0.0"
        assert helixql.manifest.lib.author  # non-empty
