"""End-to-end CLI tests for ``python -m src.main community …``."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
from typer.testing import CliRunner

from src.cli.community import community_app

_PLUGIN_SRC = textwrap.dedent('''
    from src.data.validation.base import ValidationPlugin

    class MyValidator(ValidationPlugin):
        """Checks the thing."""
        def validate(self, dataset):
            threshold = self._threshold("threshold", 100)
            return None
''')


def _make_plugin_dir(tmp_path: Path) -> Path:
    plugin_dir = tmp_path / "validation" / "my_plugin"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.py").write_text(_PLUGIN_SRC)
    return plugin_dir


def _make_preset_dir(tmp_path: Path) -> Path:
    preset_dir = tmp_path / "presets" / "starter"
    preset_dir.mkdir(parents=True)
    (preset_dir / "preset.yaml").write_text("model: {}\n")
    return preset_dir


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


# ---------------------------------------------------------------------------
# scaffold
# ---------------------------------------------------------------------------


def test_scaffold_creates_plugin_manifest(
    runner: CliRunner, tmp_path: Path
) -> None:
    plugin_dir = _make_plugin_dir(tmp_path)
    result = runner.invoke(community_app, ["scaffold", str(plugin_dir)])
    assert result.exit_code == 0, result.output
    target = plugin_dir / "manifest.toml"
    assert target.exists()
    assert 'id = "my_plugin"' in target.read_text()


def test_scaffold_refuses_overwrite_without_force(
    runner: CliRunner, tmp_path: Path
) -> None:
    plugin_dir = _make_plugin_dir(tmp_path)
    (plugin_dir / "manifest.toml").write_text('[plugin]\nid="x"\n')
    result = runner.invoke(community_app, ["scaffold", str(plugin_dir)])
    assert result.exit_code == 1
    assert "already exists" in result.output or "already exists" in (result.stderr or "")


def test_scaffold_force_overwrites(runner: CliRunner, tmp_path: Path) -> None:
    plugin_dir = _make_plugin_dir(tmp_path)
    (plugin_dir / "manifest.toml").write_text('[plugin]\nid="old"\n')
    result = runner.invoke(community_app, ["scaffold", str(plugin_dir), "--force"])
    assert result.exit_code == 0, result.output
    assert 'id = "my_plugin"' in (plugin_dir / "manifest.toml").read_text()


def test_scaffold_auto_detects_preset(runner: CliRunner, tmp_path: Path) -> None:
    """Parent folder named ``presets`` → preset mode."""
    preset_dir = _make_preset_dir(tmp_path)
    result = runner.invoke(community_app, ["scaffold", str(preset_dir)])
    assert result.exit_code == 0, result.output
    text = (preset_dir / "manifest.toml").read_text()
    assert "[preset]" in text
    assert 'id = "starter"' in text


def test_scaffold_kind_override(runner: CliRunner, tmp_path: Path) -> None:
    """``--kind preset`` forces preset mode regardless of parent dir."""
    preset_dir = tmp_path / "weird_parent" / "my_preset"
    preset_dir.mkdir(parents=True)
    (preset_dir / "preset.yaml").write_text("x: 1\n")
    result = runner.invoke(
        community_app, ["scaffold", str(preset_dir), "--kind", "preset"]
    )
    assert result.exit_code == 0, result.output
    assert "[preset]" in (preset_dir / "manifest.toml").read_text()


# ---------------------------------------------------------------------------
# sync
# ---------------------------------------------------------------------------


def test_sync_dry_run_does_not_modify(runner: CliRunner, tmp_path: Path) -> None:
    plugin_dir = _make_plugin_dir(tmp_path)
    # scaffold first so there's something to sync
    runner.invoke(community_app, ["scaffold", str(plugin_dir)])
    before = (plugin_dir / "manifest.toml").read_text()

    result = runner.invoke(
        community_app, ["sync", str(plugin_dir), "--dry-run", "--bump", "minor"]
    )
    assert result.exit_code == 0, result.output
    assert "[dry-run]" in result.output
    assert (plugin_dir / "manifest.toml").read_text() == before


def test_sync_writes_when_not_dry_run(runner: CliRunner, tmp_path: Path) -> None:
    plugin_dir = _make_plugin_dir(tmp_path)
    runner.invoke(community_app, ["scaffold", str(plugin_dir)])
    result = runner.invoke(community_app, ["sync", str(plugin_dir), "--bump", "patch"])
    assert result.exit_code == 0, result.output
    new_text = (plugin_dir / "manifest.toml").read_text()
    # scaffold leaves version = "0.1.0", patch bump → "0.1.1"
    assert 'version = "0.1.1"' in new_text


def test_sync_rejects_bad_bump(runner: CliRunner, tmp_path: Path) -> None:
    plugin_dir = _make_plugin_dir(tmp_path)
    runner.invoke(community_app, ["scaffold", str(plugin_dir)])
    result = runner.invoke(
        community_app, ["sync", str(plugin_dir), "--bump", "superpatch"]
    )
    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# pack
# ---------------------------------------------------------------------------


def test_pack_creates_archive(runner: CliRunner, tmp_path: Path) -> None:
    import zipfile

    plugin_dir = _make_plugin_dir(tmp_path)
    runner.invoke(community_app, ["scaffold", str(plugin_dir)])

    result = runner.invoke(community_app, ["pack", str(plugin_dir)])
    assert result.exit_code == 0, result.output
    archive = plugin_dir.parent / "my_plugin.zip"
    assert archive.is_file()
    assert zipfile.is_zipfile(archive)


def test_pack_dry_run_lists_files(runner: CliRunner, tmp_path: Path) -> None:
    plugin_dir = _make_plugin_dir(tmp_path)
    runner.invoke(community_app, ["scaffold", str(plugin_dir)])

    result = runner.invoke(community_app, ["pack", str(plugin_dir), "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "would write" in result.output
    assert "my_plugin/plugin.py" in result.output
    assert not (plugin_dir.parent / "my_plugin.zip").exists()


def test_pack_refuses_overwrite_without_force(runner: CliRunner, tmp_path: Path) -> None:
    plugin_dir = _make_plugin_dir(tmp_path)
    runner.invoke(community_app, ["scaffold", str(plugin_dir)])
    (plugin_dir.parent / "my_plugin.zip").write_bytes(b"stale")

    result = runner.invoke(community_app, ["pack", str(plugin_dir)])
    assert result.exit_code == 1
    combined = result.output + (result.stderr or "")
    assert "already exists" in combined


def test_pack_force_overwrites(runner: CliRunner, tmp_path: Path) -> None:
    import zipfile

    plugin_dir = _make_plugin_dir(tmp_path)
    runner.invoke(community_app, ["scaffold", str(plugin_dir)])
    (plugin_dir.parent / "my_plugin.zip").write_bytes(b"stale")

    result = runner.invoke(community_app, ["pack", str(plugin_dir), "--force"])
    assert result.exit_code == 0, result.output
    assert zipfile.is_zipfile(plugin_dir.parent / "my_plugin.zip")


def test_pack_rejects_missing_manifest(runner: CliRunner, tmp_path: Path) -> None:
    plugin_dir = tmp_path / "validation" / "bare"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.py").write_text("x = 1\n")

    result = runner.invoke(community_app, ["pack", str(plugin_dir)])
    assert result.exit_code == 1
    combined = result.output + (result.stderr or "")
    assert "manifest.toml" in combined


# ---------------------------------------------------------------------------
# help alias
# ---------------------------------------------------------------------------


def test_help_subcommand_is_alias_for_dash_dash_help(runner: CliRunner) -> None:
    """`ryotenkai community help` should print the group help (not error)."""
    result = runner.invoke(community_app, ["help"])
    assert result.exit_code == 0, result.output
    # The group help contains the sub-command names.
    assert "scaffold" in result.output
    assert "sync" in result.output
    assert "pack" in result.output


# ---------------------------------------------------------------------------
# Batch mode — kind folders
# ---------------------------------------------------------------------------


def _make_community_tree(tmp_path: Path) -> Path:
    """Build a miniature community/ tree with two validators and one preset."""
    community = tmp_path / "community"
    (community / "validation" / "p1").mkdir(parents=True)
    (community / "validation" / "p1" / "plugin.py").write_text(_PLUGIN_SRC)
    (community / "validation" / "p2").mkdir(parents=True)
    (community / "validation" / "p2" / "plugin.py").write_text(_PLUGIN_SRC)
    (community / "presets" / "starter").mkdir(parents=True)
    (community / "presets" / "starter" / "preset.yaml").write_text("model: {}\n")
    return community


def test_scaffold_batch_on_kind_dir(runner: CliRunner, tmp_path: Path) -> None:
    """`scaffold community/validation` should generate a manifest for each plugin."""
    community = _make_community_tree(tmp_path)
    result = runner.invoke(community_app, ["scaffold", str(community / "validation")])
    assert result.exit_code == 0, result.output
    assert (community / "validation" / "p1" / "manifest.toml").exists()
    assert (community / "validation" / "p2" / "manifest.toml").exists()
    # Summary line mentions the count.
    assert "2 written" in result.output


def test_scaffold_batch_skips_existing_without_force(
    runner: CliRunner, tmp_path: Path
) -> None:
    community = _make_community_tree(tmp_path)
    # Pre-populate one manifest manually.
    (community / "validation" / "p1" / "manifest.toml").write_text('[plugin]\nid="p1"\n')
    result = runner.invoke(community_app, ["scaffold", str(community / "validation")])
    assert result.exit_code == 0, result.output
    assert (community / "validation" / "p2" / "manifest.toml").exists()
    # p1 was left alone.
    assert 'id="p1"' in (community / "validation" / "p1" / "manifest.toml").read_text()
    assert "1 written" in result.output
    assert "1 skipped" in result.output


def test_sync_batch_on_kind_dir(runner: CliRunner, tmp_path: Path) -> None:
    community = _make_community_tree(tmp_path)
    runner.invoke(community_app, ["scaffold", str(community / "validation")])

    result = runner.invoke(
        community_app, ["sync", str(community / "validation"), "--dry-run"]
    )
    assert result.exit_code == 0, result.output
    # Both plugins listed with the dry-run marker.
    assert "p1" in result.output
    assert "p2" in result.output
    assert "dry-run" in result.output.lower()
    # Dry-run must not modify files.
    for leaf in ("p1", "p2"):
        assert 'version = "0.1.0"' in (
            community / "validation" / leaf / "manifest.toml"
        ).read_text()


def test_sync_batch_flags_missing_manifests(
    runner: CliRunner, tmp_path: Path
) -> None:
    community = _make_community_tree(tmp_path)
    # Scaffold just p1 — p2 is left without a manifest.
    runner.invoke(
        community_app, ["scaffold", str(community / "validation" / "p1")]
    )

    result = runner.invoke(
        community_app, ["sync", str(community / "validation"), "--dry-run"]
    )
    assert result.exit_code == 0, result.output
    # p2 should be flagged, not crash.
    assert "no manifest.toml" in result.output
    assert "1 missing manifest" in result.output


def test_sync_batch_on_community_root(runner: CliRunner, tmp_path: Path) -> None:
    community = _make_community_tree(tmp_path)
    runner.invoke(community_app, ["scaffold", str(community)])

    result = runner.invoke(
        community_app, ["sync", str(community), "--dry-run"]
    )
    assert result.exit_code == 0, result.output
    # All three targets (2 plugins + 1 preset) visited.
    assert "3 folders" in result.output or "3 changed" in result.output


# ---------------------------------------------------------------------------
# Error messages
# ---------------------------------------------------------------------------


def test_sync_on_plugin_folder_without_manifest_suggests_scaffold(
    runner: CliRunner, tmp_path: Path
) -> None:
    plugin_dir = _make_plugin_dir(tmp_path)
    # No manifest.toml created.
    result = runner.invoke(community_app, ["sync", str(plugin_dir)])
    assert result.exit_code == 1
    combined = result.output + (result.stderr or "")
    assert "no manifest.toml" in combined
    # The hint references the scaffold command, not a raw stack trace.
    assert "scaffold" in combined


def test_sync_on_unknown_dir_shows_usage_hint(
    runner: CliRunner, tmp_path: Path
) -> None:
    bogus = tmp_path / "bogus"
    bogus.mkdir()
    result = runner.invoke(community_app, ["sync", str(bogus)])
    assert result.exit_code == 1
    combined = result.output + (result.stderr or "")
    assert "cannot tell" in combined or "Expected one of" in combined


# ---------------------------------------------------------------------------
# sync-envs (PR8 / A7) — propagates plugin class's REQUIRED_ENV into TOML.
# ---------------------------------------------------------------------------


def test_sync_envs_writes_required_env_block(
    runner: CliRunner, tmp_path: Path
) -> None:
    plugin_dir = tmp_path / "evaluation" / "tiny"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "manifest.toml").write_text(
        textwrap.dedent("""
            [plugin]
            id = "tiny"
            kind = "evaluation"
            version = "1.0.0"

            [plugin.entry_point]
            module = "plugin"
            class = "TinyPlugin"
        """).strip() + "\n"
    )
    plugin_source = (
        "from src.evaluation.plugins.base import EvalResult, EvaluatorPlugin\n"
        "from src.community.manifest import RequiredEnvSpec\n"
        "\n"
        "\n"
        "class TinyPlugin(EvaluatorPlugin):\n"
        '    REQUIRED_ENV = (\n'
        '        RequiredEnvSpec(name="EVAL_KEY", description="my key",\n'
        '                        optional=False, secret=True, managed_by=""),\n'
        '    )\n'
        "\n"
        "    def evaluate(self, samples):\n"
        '        return EvalResult(plugin_name="tiny", passed=True)\n'
        "\n"
        "    def get_recommendations(self, result):\n"
        "        return []\n"
    )
    (plugin_dir / "plugin.py").write_text(plugin_source)

    result = runner.invoke(community_app, ["sync-envs", str(plugin_dir)])
    assert result.exit_code == 0, result.output + (result.stderr or "")
    written = (plugin_dir / "manifest.toml").read_text()
    assert "[[required_env]]" in written
    assert 'name = "EVAL_KEY"' in written
    assert 'description = "my key"' in written


def test_sync_envs_dry_run_does_not_write(runner: CliRunner, tmp_path: Path) -> None:
    plugin_dir = tmp_path / "evaluation" / "tiny"
    plugin_dir.mkdir(parents=True)
    initial_manifest = (
        textwrap.dedent("""
            [plugin]
            id = "tiny"
            kind = "evaluation"
            version = "1.0.0"

            [plugin.entry_point]
            module = "plugin"
            class = "TinyPlugin"
        """).strip()
        + "\n"
    )
    (plugin_dir / "manifest.toml").write_text(initial_manifest)
    plugin_source = (
        "from src.evaluation.plugins.base import EvalResult, EvaluatorPlugin\n"
        "from src.community.manifest import RequiredEnvSpec\n"
        "\n"
        "\n"
        "class TinyPlugin(EvaluatorPlugin):\n"
        '    REQUIRED_ENV = (\n'
        '        RequiredEnvSpec(name="EVAL_KEY", optional=False, secret=True, managed_by=""),\n'
        '    )\n'
        "\n"
        "    def evaluate(self, samples):\n"
        '        return EvalResult(plugin_name="tiny", passed=True)\n'
        "\n"
        "    def get_recommendations(self, result):\n"
        "        return []\n"
    )
    (plugin_dir / "plugin.py").write_text(plugin_source)

    result = runner.invoke(community_app, ["sync-envs", str(plugin_dir), "--dry-run"])
    assert result.exit_code == 0, result.output + (result.stderr or "")
    assert (plugin_dir / "manifest.toml").read_text() == initial_manifest


def test_sync_envs_rejects_kind_dir(runner: CliRunner, tmp_path: Path) -> None:
    """sync-envs only operates on a single plugin folder — kind/root errors out."""
    kind_dir = tmp_path / "evaluation"
    kind_dir.mkdir()
    result = runner.invoke(community_app, ["sync-envs", str(kind_dir)])
    assert result.exit_code == 1
    combined = result.output + (result.stderr or "")
    assert "single plugin folder" in combined or "kind_dir" in combined
