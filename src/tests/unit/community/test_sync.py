"""Unit tests for :mod:`src.community.sync`."""

from __future__ import annotations

import textwrap
import tomllib
from pathlib import Path

import pytest

from src.community.sync import sync_plugin_manifest, sync_preset_manifest

_VALIDATION_PLUGIN_SRC = textwrap.dedent('''
    from src.data.validation.base import ValidationPlugin

    class MyValidator(ValidationPlugin):
        """Checks the thing."""
        def validate(self, dataset):
            threshold = self._threshold("threshold", 100)
            sample_size = self._param("sample_size", 5000)
            return None
''')


def _write_plugin(
    tmp_path: Path,
    *,
    src: str = _VALIDATION_PLUGIN_SRC,
    manifest: str,
) -> Path:
    plugin_dir = tmp_path / "my_plugin"
    plugin_dir.mkdir()
    (plugin_dir / "plugin.py").write_text(src)
    (plugin_dir / "manifest.toml").write_text(manifest)
    return plugin_dir


# ---------------------------------------------------------------------------
# plugin sync
# ---------------------------------------------------------------------------


def test_version_bump_patch(tmp_path: Path) -> None:
    manifest = textwrap.dedent('''
        [plugin]
        id = "my_plugin"
        kind = "validation"
        name = "MyValidator"
        version = "1.2.3"
        description = "Checks the thing."

        [plugin.entry_point]
        module = "plugin"
        class = "MyValidator"
    ''')
    plugin_dir = _write_plugin(tmp_path, manifest=manifest)
    result = sync_plugin_manifest(plugin_dir, bump="patch")
    assert tomllib.loads(result.new_text)["plugin"]["version"] == "1.2.4"


def test_version_bump_minor_resets_patch(tmp_path: Path) -> None:
    manifest = textwrap.dedent('''
        [plugin]
        id = "my_plugin"
        kind = "validation"
        name = "MyValidator"
        version = "1.2.5"
        description = "."

        [plugin.entry_point]
        module = "plugin"
        class = "MyValidator"
    ''')
    plugin_dir = _write_plugin(tmp_path, manifest=manifest)
    result = sync_plugin_manifest(plugin_dir, bump="minor")
    assert tomllib.loads(result.new_text)["plugin"]["version"] == "1.3.0"


def test_user_override_of_schema_entry_preserved(tmp_path: Path) -> None:
    """If user added ``min``/``max`` to params_schema[key], sync keeps them."""
    manifest = textwrap.dedent('''
        [plugin]
        id = "my_plugin"
        kind = "validation"
        name = "MyValidator"
        version = "1.0.0"
        description = "."

        [plugin.entry_point]
        module = "plugin"
        class = "MyValidator"

        [params_schema.sample_size]
        type = "integer"
        default = 5000
        min = 1
        max = 100000
        description = "Rows to scan"
    ''')
    plugin_dir = _write_plugin(tmp_path, manifest=manifest)
    result = sync_plugin_manifest(plugin_dir, bump="patch")
    merged = tomllib.loads(result.new_text)
    entry = merged["params_schema"]["sample_size"]
    assert entry["min"] == 1
    assert entry["max"] == 100000
    assert entry["description"] == "Rows to scan"
    assert entry["type"] == "integer"
    assert entry["default"] == 5000


def test_removed_schema_key_is_dropped(tmp_path: Path) -> None:
    """Sync removes schema entries for keys the code no longer references."""
    manifest = textwrap.dedent('''
        [plugin]
        id = "my_plugin"
        kind = "validation"
        name = "MyValidator"
        version = "1.0.0"
        description = "."

        [plugin.entry_point]
        module = "plugin"
        class = "MyValidator"

        [params_schema.sample_size]
        type = "integer"
        default = 5000

        [params_schema.stale_key]
        type = "integer"
        default = 42
    ''')
    plugin_dir = _write_plugin(tmp_path, manifest=manifest)
    result = sync_plugin_manifest(plugin_dir, bump="patch")
    merged = tomllib.loads(result.new_text)
    assert "sample_size" in merged["params_schema"]
    assert "stale_key" not in merged["params_schema"]


def test_new_schema_key_is_added(tmp_path: Path) -> None:
    """Sync adds newly-seen params to the schema."""
    manifest = textwrap.dedent('''
        [plugin]
        id = "my_plugin"
        kind = "validation"
        name = "MyValidator"
        version = "1.0.0"
        description = "."

        [plugin.entry_point]
        module = "plugin"
        class = "MyValidator"

        [params_schema.sample_size]
        type = "integer"
        default = 5000
    ''')
    # A new ``_param("new_param", ...)`` call appears inside the class body.
    src = textwrap.dedent('''
        from src.data.validation.base import ValidationPlugin

        class MyValidator(ValidationPlugin):
            """Checks the thing."""
            def validate(self, dataset):
                threshold = self._threshold("threshold", 100)
                sample_size = self._param("sample_size", 5000)
                new_param = self._param("new_param", "hello")
                return None
    ''')
    plugin_dir = _write_plugin(tmp_path, src=src, manifest=manifest)
    result = sync_plugin_manifest(plugin_dir, bump="patch")
    merged = tomllib.loads(result.new_text)
    assert merged["params_schema"]["new_param"]["type"] == "string"
    assert merged["params_schema"]["new_param"]["default"] == "hello"


def test_suggested_param_orphan_removed(tmp_path: Path) -> None:
    manifest = textwrap.dedent('''
        [plugin]
        id = "my_plugin"
        kind = "validation"
        name = "MyValidator"
        version = "1.0.0"
        description = "."

        [plugin.entry_point]
        module = "plugin"
        class = "MyValidator"

        [params_schema.sample_size]
        type = "integer"
        default = 5000

        [suggested_params]
        sample_size = 5000
        orphan_key = 123
    ''')
    plugin_dir = _write_plugin(tmp_path, manifest=manifest)
    result = sync_plugin_manifest(plugin_dir, bump="patch")
    merged = tomllib.loads(result.new_text)
    assert "sample_size" in merged["suggested_params"]
    assert "orphan_key" not in merged["suggested_params"]


def test_required_env_is_merged_not_overwritten(tmp_path: Path) -> None:
    """Sync supplements ``[[required_env]]`` with inferred keys; existing
    entries (with hand-edited descriptions / non-default flags) are kept
    verbatim, and entries no longer inferred are preserved too."""
    src = textwrap.dedent('''
        from src.evaluation.plugins.base import EvaluatorPlugin

        class MyPlugin(EvaluatorPlugin):
            """."""
            _secrets: dict
            def evaluate(self, samples):
                k = self._secrets["EVAL_NEW_KEY"]
                return None
            def get_recommendations(self, r): return []
            @classmethod
            def get_description(cls): return ""
    ''')
    manifest = textwrap.dedent('''
        [plugin]
        id = "my_plugin"
        kind = "evaluation"
        name = "MyPlugin"
        version = "1.0.0"
        description = "."

        [plugin.entry_point]
        module = "plugin"
        class = "MyPlugin"

        [[required_env]]
        name = "EVAL_OPTIONAL_USER_ONLY"
        description = "Hand-edited helper text"
        optional = true
        secret = true
        managed_by = ""
    ''')
    plugin_dir = tmp_path / "my_plugin"
    plugin_dir.mkdir()
    (plugin_dir / "plugin.py").write_text(src)
    (plugin_dir / "manifest.toml").write_text(manifest)

    result = sync_plugin_manifest(plugin_dir, bump="patch")
    merged = tomllib.loads(result.new_text)
    names = [entry["name"] for entry in merged["required_env"]]
    # Inferred key first, existing-only entry kept after.
    assert names == ["EVAL_NEW_KEY", "EVAL_OPTIONAL_USER_ONLY"]
    # Existing entry's hand-edited fields survive intact.
    user_entry = next(e for e in merged["required_env"] if e["name"] == "EVAL_OPTIONAL_USER_ONLY")
    assert user_entry["description"] == "Hand-edited helper text"
    assert user_entry["optional"] is True


def test_sync_is_idempotent(tmp_path: Path) -> None:
    """Running sync twice with --bump patch should always bump, but content
    stays otherwise identical (no other drift)."""
    from src.community.scaffold import scaffold_plugin_manifest

    plugin_dir = tmp_path / "my_plugin"
    plugin_dir.mkdir()
    (plugin_dir / "plugin.py").write_text(_VALIDATION_PLUGIN_SRC)
    (plugin_dir / "manifest.toml").write_text(scaffold_plugin_manifest(plugin_dir))

    r1 = sync_plugin_manifest(plugin_dir, bump="patch")
    (plugin_dir / "manifest.toml").write_text(r1.new_text)
    r2 = sync_plugin_manifest(plugin_dir, bump="patch")
    # Only `version` line should change between r1 and r2
    r1_lines = [line for line in r1.new_text.splitlines() if "version" not in line]
    r2_lines = [line for line in r2.new_text.splitlines() if "version" not in line]
    assert r1_lines == r2_lines


def test_sync_result_changed_flag(tmp_path: Path) -> None:
    """If code is already in sync, sync with bump=patch still reports changed
    (version moved)."""
    from src.community.scaffold import scaffold_plugin_manifest

    plugin_dir = tmp_path / "my_plugin"
    plugin_dir.mkdir()
    (plugin_dir / "plugin.py").write_text(_VALIDATION_PLUGIN_SRC)
    (plugin_dir / "manifest.toml").write_text(scaffold_plugin_manifest(plugin_dir))

    result = sync_plugin_manifest(plugin_dir, bump="patch")
    assert result.changed is True
    assert "version" in result.diff


# ---------------------------------------------------------------------------
# preset sync
# ---------------------------------------------------------------------------


def test_preset_sync_bumps_version(tmp_path: Path) -> None:
    preset_dir = tmp_path / "starter"
    preset_dir.mkdir()
    (preset_dir / "preset.yaml").write_text("model: {}\n")
    (preset_dir / "manifest.toml").write_text(
        textwrap.dedent('''
            [preset]
            id = "starter"
            name = "Starter"
            description = "hi"
            size_tier = "small"
            version = "0.4.2"

            [preset.entry_point]
            file = "preset.yaml"
        ''')
    )
    result = sync_preset_manifest(preset_dir, bump="patch")
    merged = tomllib.loads(result.new_text)
    assert merged["preset"]["version"] == "0.4.3"


def test_preset_sync_preserves_user_fields(tmp_path: Path) -> None:
    preset_dir = tmp_path / "starter"
    preset_dir.mkdir()
    (preset_dir / "preset.yaml").write_text("x: 1\n")
    (preset_dir / "manifest.toml").write_text(
        textwrap.dedent('''
            [preset]
            id = "starter"
            name = "Custom Display"
            description = "a written blurb"
            size_tier = "large"
            version = "1.0.0"

            [preset.entry_point]
            file = "preset.yaml"
        ''')
    )
    result = sync_preset_manifest(preset_dir, bump="patch")
    merged = tomllib.loads(result.new_text)
    assert merged["preset"]["name"] == "Custom Display"
    assert merged["preset"]["description"] == "a written blurb"
    assert merged["preset"]["size_tier"] == "large"


def test_preset_sync_refreshes_entry_point_file(tmp_path: Path) -> None:
    """If the user renames the YAML, sync picks up the new filename."""
    preset_dir = tmp_path / "starter"
    preset_dir.mkdir()
    (preset_dir / "alt.yaml").write_text("x: 1\n")
    (preset_dir / "manifest.toml").write_text(
        textwrap.dedent('''
            [preset]
            id = "starter"
            name = "Starter"
            description = "."
            size_tier = "small"
            version = "0.1.0"

            [preset.entry_point]
            file = "preset.yaml"
        ''')
    )
    result = sync_preset_manifest(preset_dir, bump="patch")
    merged = tomllib.loads(result.new_text)
    assert merged["preset"]["entry_point"]["file"] == "alt.yaml"


def test_sync_missing_manifest_raises(tmp_path: Path) -> None:
    plugin_dir = tmp_path / "x"
    plugin_dir.mkdir()
    (plugin_dir / "plugin.py").write_text(_VALIDATION_PLUGIN_SRC)
    with pytest.raises(FileNotFoundError, match=r"no manifest\.toml"):
        sync_plugin_manifest(plugin_dir, bump="patch")
