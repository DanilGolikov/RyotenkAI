"""Unit tests for :mod:`src.community.inference`."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from src.community.inference import (
    bump_version,
    find_entry_module,
    infer_plugin,
)


def _write_plugin(
    tmp_path: Path,
    *,
    source: str,
    package: bool = False,
    extra_sibling: str | None = None,
) -> Path:
    plugin_dir = tmp_path / "my_plugin"
    plugin_dir.mkdir()
    if package:
        pkg = plugin_dir / "plugin"
        pkg.mkdir()
        (pkg / "__init__.py").write_text(source)
        if extra_sibling is not None:
            (pkg / "sibling.py").write_text(extra_sibling)
    else:
        (plugin_dir / "plugin.py").write_text(source)
    return plugin_dir


def test_find_entry_module_prefers_plugin_py(tmp_path: Path) -> None:
    plugin_dir = _write_plugin(tmp_path, source="class X: pass\n")
    assert find_entry_module(plugin_dir).name == "plugin.py"


def test_find_entry_module_falls_back_to_package(tmp_path: Path) -> None:
    plugin_dir = _write_plugin(tmp_path, source="class X: pass\n", package=True)
    assert find_entry_module(plugin_dir).name == "__init__.py"


def test_find_entry_module_missing_raises(tmp_path: Path) -> None:
    plugin_dir = tmp_path / "empty"
    plugin_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        find_entry_module(plugin_dir)


def test_infer_validation_plugin_basic(tmp_path: Path) -> None:
    src = textwrap.dedent('''
        from src.data.validation.base import ValidationPlugin

        class MyValidator(ValidationPlugin):
            """Checks that foo equals bar."""

            def validate(self, dataset):
                threshold = self._threshold("threshold", 100)
                sample_size = self._param("sample_size", 10_000)
                return None
    ''')
    plugin_dir = _write_plugin(tmp_path, source=src)
    inferred = infer_plugin(plugin_dir)

    assert inferred.kind == "validation"
    assert inferred.entry_class == "MyValidator"
    assert inferred.description == "Checks that foo equals bar."
    assert inferred.params["sample_size"].type == "integer"
    assert inferred.params["sample_size"].default == 10_000
    assert inferred.thresholds["threshold"].type == "integer"
    assert inferred.thresholds["threshold"].default == 100


def test_infer_evaluation_params_via_params_get(tmp_path: Path) -> None:
    """``self.params.get("key", default)`` is inferred alongside ``_param``."""
    src = textwrap.dedent('''
        from src.evaluation.plugins.base import EvaluatorPlugin

        class MyJudgePlugin(EvaluatorPlugin):
            """LLM judge plugin."""

            def evaluate(self, samples):
                temperature = float(self.params.get("temperature", 0.5))
                return None

            def get_recommendations(self, result):
                return []

            @classmethod
            def get_description(cls):
                return "Judge"
    ''')
    plugin_dir = _write_plugin(tmp_path, source=src)
    inferred = infer_plugin(plugin_dir)
    assert inferred.kind == "evaluation"
    assert inferred.params["temperature"].type == "number"
    assert inferred.params["temperature"].default == 0.5


def test_infer_reward_plugin(tmp_path: Path) -> None:
    src = textwrap.dedent('''
        from src.training.reward_plugins.base import RewardPlugin

        class MyReward(RewardPlugin):
            """Cheap lexical reward."""

            def build_trainer_kwargs(self, **kw):
                return {}
    ''')
    plugin_dir = _write_plugin(tmp_path, source=src)
    inferred = infer_plugin(plugin_dir)
    assert inferred.kind == "reward"
    assert inferred.entry_class == "MyReward"


def test_infer_report_plugin_duck_typed(tmp_path: Path) -> None:
    """Report plugins aren't derived from a base; detected via plugin_id/order/render."""
    src = textwrap.dedent('''
        class MyBlockPlugin:
            """Demo report block."""
            plugin_id = "my_block"
            order = 75
            title = "My Block"

            def render(self, ctx):
                return None
    ''')
    plugin_dir = _write_plugin(tmp_path, source=src)
    inferred = infer_plugin(plugin_dir)
    assert inferred.kind == "reports"
    assert inferred.entry_class == "MyBlockPlugin"


def test_infer_required_secrets(tmp_path: Path) -> None:
    src = textwrap.dedent('''
        from src.evaluation.plugins.base import EvaluatorPlugin

        class MyJudgePlugin(EvaluatorPlugin):
            """Uses a secret."""
            _secrets: dict

            def evaluate(self, samples):
                api_key = self._secrets["EVAL_CEREBRAS_API_KEY"]
                backup_key = self._secrets["EVAL_BACKUP"]
                return None

            def get_recommendations(self, result):
                return []

            @classmethod
            def get_description(cls):
                return ""
    ''')
    plugin_dir = _write_plugin(tmp_path, source=src)
    inferred = infer_plugin(plugin_dir)
    assert inferred.required_secrets == ("EVAL_CEREBRAS_API_KEY", "EVAL_BACKUP")


def test_infer_module_level_constant_resolves(tmp_path: Path) -> None:
    """Constants like ``DEFAULT_MODEL = "x"`` are followed when used as default."""
    src = textwrap.dedent('''
        from src.evaluation.plugins.base import EvaluatorPlugin

        DEFAULT_MODEL = "llama3.1-8b"

        class MyJudgePlugin(EvaluatorPlugin):
            """Uses a module-level default."""
            def evaluate(self, samples):
                model = self.params.get("model", DEFAULT_MODEL)
                return None

            def get_recommendations(self, result):
                return []

            @classmethod
            def get_description(cls):
                return ""
    ''')
    plugin_dir = _write_plugin(tmp_path, source=src)
    inferred = infer_plugin(plugin_dir)
    assert inferred.params["model"].type == "string"
    assert inferred.params["model"].default == "llama3.1-8b"


def test_infer_package_plugin_scans_siblings(tmp_path: Path) -> None:
    """Entry class can live in a sibling file alongside ``__init__.py``."""
    init_src = "from .sibling import MyValidator\n__all__ = ['MyValidator']\n"
    sibling_src = textwrap.dedent('''
        from src.data.validation.base import ValidationPlugin

        class MyValidator(ValidationPlugin):
            """In a sibling module."""
            def validate(self, dataset):
                return None
    ''')
    plugin_dir = _write_plugin(
        tmp_path, source=init_src, package=True, extra_sibling=sibling_src
    )
    inferred = infer_plugin(plugin_dir)
    assert inferred.entry_class == "MyValidator"
    assert inferred.kind == "validation"


def test_infer_no_entry_class_raises(tmp_path: Path) -> None:
    plugin_dir = _write_plugin(tmp_path, source="x = 1\n")
    with pytest.raises(ValueError, match="no plugin entry class found"):
        infer_plugin(plugin_dir)


def test_infer_dynamic_key_is_skipped(tmp_path: Path) -> None:
    """Non-literal keys log a warning but do not fail the scan."""
    src = textwrap.dedent('''
        from src.data.validation.base import ValidationPlugin

        class MyValidator(ValidationPlugin):
            """Has a dynamic param key."""
            def validate(self, dataset):
                key = "threshold"
                x = self._param(key, 42)   # skipped (key is not a literal)
                y = self._param("known", 1)
                return None
    ''')
    plugin_dir = _write_plugin(tmp_path, source=src)
    inferred = infer_plugin(plugin_dir)
    assert "known" in inferred.params
    assert "threshold" not in inferred.params


# ---------------------------------------------------------------------------
# bump_version
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "version,bump,expected",
    [
        ("1.2.3", "patch", "1.2.4"),
        ("1.2.3", "minor", "1.3.0"),
        ("1.2.3", "major", "2.0.0"),
        ("0.1.0", "patch", "0.1.1"),
        ("0.0.1", "major", "1.0.0"),
    ],
)
def test_bump_version(version: str, bump: str, expected: str) -> None:
    assert bump_version(version, bump) == expected  # type: ignore[arg-type]


def test_bump_version_invalid_semver() -> None:
    with pytest.raises(ValueError, match="version must be semver"):
        bump_version("1.2", "patch")


def test_bump_version_invalid_bump() -> None:
    with pytest.raises(ValueError, match="bump must be"):
        bump_version("1.2.3", "superpatch")  # type: ignore[arg-type]
