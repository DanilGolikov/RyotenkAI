"""End-to-end CLI tests for ``ryotenkai plugin scaffold …`` (PR12 / B4).

Each scaffold output must be loadable via the community catalog
without further edits — that's the contract that makes the CLI
useful. We assert it for every kind by running the loader against
the freshly-scaffolded folder.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest
from typer.testing import CliRunner

from src.cli.plugin_scaffold import plugin_app
from src.community.loader import load_plugins
from src.community.manifest import LATEST_SCHEMA_VERSION


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture()
def fake_root(tmp_path: Path) -> Path:
    """A community root with the standard kind subdirs ready to receive
    scaffolded plugins."""
    root = tmp_path / "community"
    for sub in ("validation", "evaluation", "reward", "reports", "presets"):
        (root / sub).mkdir(parents=True)
    return root


# ---------------------------------------------------------------------------
# Output structure
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kind",
    ["validation", "evaluation", "reward", "reports"],
)
def test_scaffold_creates_expected_files(
    runner: CliRunner, fake_root: Path, kind: str
) -> None:
    result = runner.invoke(
        plugin_app,
        ["scaffold", kind, "hello_world", "--root", str(fake_root)],
    )
    assert result.exit_code == 0, result.output + (result.stderr or "")

    plugin_dir = fake_root / kind / "hello_world"
    assert (plugin_dir / "manifest.toml").is_file()
    assert (plugin_dir / "plugin.py").is_file()
    assert (plugin_dir / "README.md").is_file()
    assert (plugin_dir / "tests" / "test_plugin.py").is_file()
    assert (plugin_dir / "tests" / "__init__.py").is_file()


def test_scaffold_manifest_declares_latest_schema_version(
    runner: CliRunner, fake_root: Path
) -> None:
    runner.invoke(
        plugin_app,
        ["scaffold", "validation", "hello_world", "--root", str(fake_root)],
    )
    manifest = tomllib.loads(
        (fake_root / "validation" / "hello_world" / "manifest.toml").read_text()
    )
    assert manifest["schema_version"] == LATEST_SCHEMA_VERSION


# ---------------------------------------------------------------------------
# Loader compatibility — the scaffolded folder must be importable as-is
# for validation / evaluation / reports. Reward needs ``supported_strategies``
# filled in (intentional TODO that the loader rejects until set), so we
# assert THAT separately.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kind",
    ["validation", "evaluation", "reports"],
)
def test_scaffolded_plugin_loads_through_catalog(
    runner: CliRunner, fake_root: Path, kind: str
) -> None:
    runner.invoke(
        plugin_app,
        ["scaffold", kind, "hello_world", "--root", str(fake_root)],
    )
    result = load_plugins(kind, root=fake_root, strict=True)
    assert len(result.plugins) == 1
    loaded = result.plugins[0]
    assert loaded.manifest.plugin.id == "hello_world"
    assert loaded.plugin_cls.__name__ == "HelloWorldPlugin"


def test_reward_scaffold_emits_supported_strategies_todo(
    runner: CliRunner, fake_root: Path
) -> None:
    """Reward plugins need ``supported_strategies`` filled in. Scaffold
    emits an empty list with a TODO comment so the manifest validator
    rejects the plugin until the author sets it — matching the same
    nudge ``community scaffold`` already gives."""
    runner.invoke(
        plugin_app,
        ["scaffold", "reward", "fancy_reward", "--root", str(fake_root)],
    )
    manifest_text = (
        fake_root / "reward" / "fancy_reward" / "manifest.toml"
    ).read_text()
    assert "supported_strategies = []" in manifest_text
    assert "TODO" in manifest_text
    # Loader should reject — empty list trips the validator.
    with pytest.raises(Exception, match="supported_strategies"):
        load_plugins("reward", root=fake_root, strict=True)


# ---------------------------------------------------------------------------
# Validation / safety
# ---------------------------------------------------------------------------


def test_scaffold_rejects_invalid_plugin_id(
    runner: CliRunner, fake_root: Path
) -> None:
    result = runner.invoke(
        plugin_app,
        ["scaffold", "validation", "Bad-Name", "--root", str(fake_root)],
    )
    assert result.exit_code == 1
    combined = result.output + (result.stderr or "")
    assert "snake_case" in combined or "must match" in combined


def test_scaffold_refuses_overwrite_without_force(
    runner: CliRunner, fake_root: Path
) -> None:
    runner.invoke(
        plugin_app,
        ["scaffold", "validation", "hello_world", "--root", str(fake_root)],
    )
    result = runner.invoke(
        plugin_app,
        ["scaffold", "validation", "hello_world", "--root", str(fake_root)],
    )
    assert result.exit_code == 1
    combined = result.output + (result.stderr or "")
    assert "already exists" in combined
    assert "--force" in combined


def test_scaffold_force_overwrites(runner: CliRunner, fake_root: Path) -> None:
    runner.invoke(
        plugin_app,
        ["scaffold", "validation", "hello_world", "--root", str(fake_root)],
    )
    # Tamper with manifest so we can detect overwrite.
    manifest_path = fake_root / "validation" / "hello_world" / "manifest.toml"
    manifest_path.write_text("# tampered\n")
    result = runner.invoke(
        plugin_app,
        [
            "scaffold",
            "validation",
            "hello_world",
            "--root",
            str(fake_root),
            "--force",
        ],
    )
    assert result.exit_code == 0
    assert "# tampered" not in manifest_path.read_text()


# ---------------------------------------------------------------------------
# Class-name derivation
# ---------------------------------------------------------------------------


def test_class_name_capitalises_underscores(
    runner: CliRunner, fake_root: Path
) -> None:
    runner.invoke(
        plugin_app,
        ["scaffold", "validation", "two_word_thing", "--root", str(fake_root)],
    )
    plugin_text = (
        fake_root / "validation" / "two_word_thing" / "plugin.py"
    ).read_text()
    assert "class TwoWordThingPlugin(ValidationPlugin):" in plugin_text
