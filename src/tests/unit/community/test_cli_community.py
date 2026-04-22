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
    assert "[dry-run]" in result.output
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
