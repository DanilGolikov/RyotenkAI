"""Unit tests for :mod:`src.community.scaffold`."""

from __future__ import annotations

import textwrap
import tomllib
from pathlib import Path

import pytest

from src.community.manifest import PluginManifest, PresetManifest
from src.community.scaffold import scaffold_plugin_manifest, scaffold_preset_manifest


def _write_plugin_py(plugin_dir: Path, source: str) -> None:
    plugin_dir.mkdir()
    (plugin_dir / "plugin.py").write_text(source)


def test_scaffold_plugin_is_fully_valid(tmp_path: Path) -> None:
    src = textwrap.dedent('''
        from src.data.validation.base import ValidationPlugin

        class MyValidator(ValidationPlugin):
            """Checks the thing."""
            def validate(self, dataset):
                threshold = self._threshold("threshold", 100)
                sample_size = self._param("sample_size", 5000)
                return None
    ''')
    plugin_dir = tmp_path / "my_plugin"
    _write_plugin_py(plugin_dir, src)

    text = scaffold_plugin_manifest(plugin_dir)
    parsed = tomllib.loads(text)

    # Must pass pydantic validation
    manifest = PluginManifest.model_validate(parsed)
    assert manifest.plugin.id == "my_plugin"
    assert manifest.plugin.kind == "validation"
    assert manifest.plugin.entry_point.class_name == "MyValidator"
    assert manifest.plugin.description == "Checks the thing."
    assert manifest.plugin.version == "0.1.0"
    assert manifest.params_schema["sample_size"].default == 5000
    assert manifest.thresholds_schema["threshold"].default == 100
    assert manifest.suggested_params == {"sample_size": 5000}
    assert manifest.suggested_thresholds == {"threshold": 100}


def test_scaffold_plugin_emits_todo_markers(tmp_path: Path) -> None:
    src = textwrap.dedent('''
        from src.data.validation.base import ValidationPlugin
        class X(ValidationPlugin):
            """doc."""
            def validate(self, d): return None
    ''')
    plugin_dir = tmp_path / "x"
    _write_plugin_py(plugin_dir, src)
    text = scaffold_plugin_manifest(plugin_dir)
    # TODO markers for the fields authors are expected to fill
    assert "# TODO: fill in" in text
    assert "category" in text


def test_scaffold_plugin_with_secrets(tmp_path: Path) -> None:
    """``self._secrets["KEY"]`` accesses become ``[[required_env]]`` blocks
    with the safe defaults (``secret=true, optional=false``)."""
    src = textwrap.dedent('''
        from src.evaluation.plugins.base import EvaluatorPlugin

        class MyPlugin(EvaluatorPlugin):
            """Judge."""
            _secrets: dict
            def evaluate(self, samples):
                api_key = self._secrets["EVAL_API_KEY"]
                return None
            def get_recommendations(self, r): return []
            @classmethod
            def get_description(cls): return ""
    ''')
    plugin_dir = tmp_path / "my_plugin"
    _write_plugin_py(plugin_dir, src)

    text = scaffold_plugin_manifest(plugin_dir)
    parsed = tomllib.loads(text)
    manifest = PluginManifest.model_validate(parsed)
    names = [entry.name for entry in manifest.required_env]
    assert names == ["EVAL_API_KEY"]
    entry = manifest.required_env[0]
    assert entry.secret is True
    assert entry.optional is False
    # Loader-derived runtime tuple matches.
    assert manifest.required_secret_names() == ("EVAL_API_KEY",)


def test_scaffold_preset_basic(tmp_path: Path) -> None:
    preset_dir = tmp_path / "starter"
    preset_dir.mkdir()
    (preset_dir / "preset.yaml").write_text("model:\n  name: demo\n")

    text = scaffold_preset_manifest(preset_dir)
    parsed = tomllib.loads(text)
    manifest = PresetManifest.model_validate(parsed)
    assert manifest.preset.id == "starter"
    assert manifest.preset.entry_point.file == "preset.yaml"
    assert manifest.preset.version == "0.1.0"
    assert "# TODO: fill in" in text    # description/size_tier are marked TODO


def test_scaffold_preset_alt_yaml_name(tmp_path: Path) -> None:
    preset_dir = tmp_path / "custom"
    preset_dir.mkdir()
    (preset_dir / "my_config.yaml").write_text("model: {}\n")

    text = scaffold_preset_manifest(preset_dir)
    manifest = PresetManifest.model_validate(tomllib.loads(text))
    assert manifest.preset.entry_point.file == "my_config.yaml"


def test_scaffold_preset_missing_yaml_raises(tmp_path: Path) -> None:
    preset_dir = tmp_path / "bad"
    preset_dir.mkdir()
    with pytest.raises(FileNotFoundError, match=r"no \*\.yaml"):
        scaffold_preset_manifest(preset_dir)
