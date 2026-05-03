"""Tests for the REQUIRED_ENV ↔ manifest cross-check (A7).

Covers two halves of the contract:

- Loader-time check (:func:`_crosscheck_required_env`): when a plugin
  class declares a non-empty :attr:`BasePlugin.REQUIRED_ENV` tuple, the
  loader must reject any drift against the manifest's
  ``[[required_env]]`` block.
- Sync helper (:func:`sync_plugin_envs`): rewrites the manifest's
  ``[[required_env]]`` from the class's ClassVar so the cross-check
  stays green without manual TOML edits.
"""

from __future__ import annotations

import textwrap
import tomllib
from pathlib import Path

import pytest

from src.community.loader import load_plugins
from src.community.sync import sync_plugin_envs


def _write_eval_plugin_with_required_env(
    root: Path,
    plugin_id: str,
    *,
    py_required_env: str,
    toml_required_env: str,
    class_name: str = "TestPlugin",
) -> Path:
    """Drop an evaluation plugin where Python and TOML each declare the
    REQUIRED_ENV / [[required_env]] block independently — used to
    fabricate matching and mismatching scenarios.

    The plugin source is assembled with raw string concatenation rather
    than ``textwrap.dedent`` because ``py_required_env`` itself spans
    multiple lines with its own indentation, which would confuse
    dedent's common-prefix calculation.
    """
    plugin_dir = root / "evaluation" / plugin_id
    plugin_dir.mkdir(parents=True)
    manifest_lines = [
        "[plugin]",
        f'id = "{plugin_id}"',
        'kind = "evaluation"',
        'version = "1.0.0"',
        "",
        "[plugin.entry_point]",
        'module = "plugin"',
        f'class = "{class_name}"',
    ]
    manifest = "\n".join(manifest_lines)
    if toml_required_env.strip():
        manifest += "\n\n" + toml_required_env.strip() + "\n"
    else:
        manifest += "\n"
    (plugin_dir / "manifest.toml").write_text(manifest)
    plugin_source = (
        "from src.evaluation.plugins.base import EvalResult, EvaluatorPlugin\n"
        "from src.community.manifest import RequiredEnvSpec\n"
        "\n"
        "\n"
        f"class {class_name}(EvaluatorPlugin):\n"
        f"    REQUIRED_ENV = {py_required_env}\n"
        "\n"
        "    def evaluate(self, samples):\n"
        f'        return EvalResult(plugin_name="{plugin_id}", passed=True)\n'
        "\n"
        "    def get_recommendations(self, result):\n"
        "        return []\n"
    )
    (plugin_dir / "plugin.py").write_text(plugin_source)
    return plugin_dir


def test_matching_required_env_loads_clean(tmp_community_root: Path) -> None:
    py = '(\n    RequiredEnvSpec(name="EVAL_KEY", description="x", optional=False, secret=True, managed_by=""),\n)'
    toml = textwrap.dedent("""
        [[required_env]]
        name = "EVAL_KEY"
        description = "x"
        optional = false
        secret = true
        managed_by = ""
    """)
    _write_eval_plugin_with_required_env(
        tmp_community_root, "tiny", py_required_env=py, toml_required_env=toml,
    )
    loaded = load_plugins("evaluation", root=tmp_community_root, strict=True)
    assert len(loaded) == 1
    assert loaded[0].plugin_cls.name == "tiny"


def test_missing_in_toml_diff_message(tmp_community_root: Path) -> None:
    """REQUIRED_ENV declares an entry not present in the TOML."""
    py = '(\n    RequiredEnvSpec(name="EVAL_NEW_KEY", optional=False, secret=True, managed_by=""),\n)'
    _write_eval_plugin_with_required_env(
        tmp_community_root, "tiny", py_required_env=py, toml_required_env="",
    )
    with pytest.raises(ValueError, match=r"missing from manifest:.*EVAL_NEW_KEY"):
        load_plugins("evaluation", root=tmp_community_root, strict=True)


def test_missing_in_python_diff_message(tmp_community_root: Path) -> None:
    """TOML declares an entry that REQUIRED_ENV doesn't know about."""
    py = '(\n    RequiredEnvSpec(name="EVAL_KEY", optional=False, secret=True, managed_by=""),\n)'
    toml = textwrap.dedent("""
        [[required_env]]
        name = "EVAL_KEY"
        optional = false
        secret = true
        managed_by = ""

        [[required_env]]
        name = "EVAL_EXTRA"
        optional = false
        secret = true
        managed_by = ""
    """)
    _write_eval_plugin_with_required_env(
        tmp_community_root, "tiny", py_required_env=py, toml_required_env=toml,
    )
    with pytest.raises(ValueError, match=r"missing from REQUIRED_ENV:.*EVAL_EXTRA"):
        load_plugins("evaluation", root=tmp_community_root, strict=True)


def test_flag_drift_diff_message(tmp_community_root: Path) -> None:
    """Same name, different flags → per-key diff in the error."""
    py = '(\n    RequiredEnvSpec(name="EVAL_KEY", optional=False, secret=True, managed_by=""),\n)'
    toml = textwrap.dedent("""
        [[required_env]]
        name = "EVAL_KEY"
        optional = true
        secret = true
        managed_by = ""
    """)
    _write_eval_plugin_with_required_env(
        tmp_community_root, "tiny", py_required_env=py, toml_required_env=toml,
    )
    with pytest.raises(ValueError, match=r"EVAL_KEY:.*optional: code=False vs toml=True"):
        load_plugins("evaluation", root=tmp_community_root, strict=True)


def test_empty_required_env_skips_check(tmp_community_root: Path) -> None:
    """When ``REQUIRED_ENV = ()`` the manifest is the only source of truth.

    Plenty of plugins declare envs only in TOML — the cross-check is
    opt-in, not a hard requirement.
    """
    toml = textwrap.dedent("""
        [[required_env]]
        name = "EVAL_TOML_ONLY"
        optional = false
        secret = true
        managed_by = ""
    """)
    # No REQUIRED_ENV in the Python class — the helper skips entirely.
    _write_eval_plugin_with_required_env(
        tmp_community_root, "tiny", py_required_env="()", toml_required_env=toml,
    )
    loaded = load_plugins("evaluation", root=tmp_community_root, strict=True)
    assert len(loaded) == 1


def test_sync_envs_writes_required_env_block(tmp_community_root: Path) -> None:
    """``sync_plugin_envs`` rewrites the manifest from the class's tuple."""
    py = (
        '(\n'
        '    RequiredEnvSpec(name="EVAL_KEY_A", description="first", '
        'optional=False, secret=True, managed_by=""),\n'
        '    RequiredEnvSpec(name="EVAL_KEY_B", description="second", '
        'optional=True, secret=True, managed_by="integrations"),\n'
        ')'
    )
    plugin_dir = _write_eval_plugin_with_required_env(
        tmp_community_root, "tiny",
        py_required_env=py,
        toml_required_env="",  # start with no [[required_env]] in TOML
    )
    result = sync_plugin_envs(plugin_dir)
    assert result.changed is True
    written = tomllib.loads(result.new_text)
    by_name = {e["name"]: e for e in written["required_env"]}
    assert by_name["EVAL_KEY_A"]["description"] == "first"
    assert by_name["EVAL_KEY_B"]["managed_by"] == "integrations"
    assert by_name["EVAL_KEY_B"]["optional"] is True


def test_sync_envs_clears_block_when_required_env_is_empty(tmp_community_root: Path) -> None:
    """When ``REQUIRED_ENV = ()`` the helper drops the manifest block."""
    toml = textwrap.dedent("""
        [[required_env]]
        name = "EVAL_LEFT_OVER"
        optional = false
        secret = true
        managed_by = ""
    """)
    plugin_dir = _write_eval_plugin_with_required_env(
        tmp_community_root, "tiny", py_required_env="()", toml_required_env=toml,
    )
    result = sync_plugin_envs(plugin_dir)
    assert result.changed is True
    written = tomllib.loads(result.new_text)
    assert "required_env" not in written


def test_crosscheck_failure_becomes_load_failure_in_loose_mode(
    tmp_community_root: Path,
) -> None:
    """REQUIRED_ENV ↔ TOML drift should not take the whole catalog down.

    In loose mode (production default) the contract violation is captured
    as a structured ``LoadFailure`` row alongside the rest of the loaded
    plugins — so one author's mistake doesn't break every other plugin
    in the catalog. Strict mode still re-raises (covered above).
    """
    py = (
        '(\n'
        '    RequiredEnvSpec(name="EVAL_KEY", optional=False, secret=True, managed_by=""),\n'
        ')'
    )
    # TOML disagrees — optional flag flipped.
    toml = textwrap.dedent("""
        [[required_env]]
        name = "EVAL_KEY"
        optional = true
        secret = true
        managed_by = ""
    """)
    _write_eval_plugin_with_required_env(
        tmp_community_root, "drifty",
        py_required_env=py,
        toml_required_env=toml,
    )

    result = load_plugins("evaluation", root=tmp_community_root, strict=False)
    assert list(result) == []
    assert len(result.failures) == 1
    f = result.failures[0]
    assert f.error_type == "metadata_error"
    assert f.plugin_id == "drifty"
    assert "REQUIRED_ENV ↔ manifest cross-check failed" in f.message


def test_sync_envs_idempotent(tmp_community_root: Path) -> None:
    """Running sync-envs twice produces no further diff."""
    py = (
        '(\n'
        '    RequiredEnvSpec(name="EVAL_KEY", optional=False, secret=True, managed_by=""),\n'
        ')'
    )
    toml = textwrap.dedent("""
        [[required_env]]
        name = "EVAL_KEY"
        description = ""
        optional = false
        secret = true
        managed_by = ""
    """)
    plugin_dir = _write_eval_plugin_with_required_env(
        tmp_community_root, "tiny",
        py_required_env=py,
        toml_required_env=toml,
    )
    first = sync_plugin_envs(plugin_dir)
    if first.changed:
        (plugin_dir / "manifest.toml").write_text(first.new_text)
    second = sync_plugin_envs(plugin_dir)
    assert second.changed is False
