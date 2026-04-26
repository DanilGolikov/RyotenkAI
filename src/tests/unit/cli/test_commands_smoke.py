"""Targeted unit tests for the new noun-verb commands.

Cover the cheap, deterministic paths through each commands/<noun>.py
module: error branches, output-format switching, project-context
resolution. Heavy paths (orchestrator, MLflow, network) live behind
mocks where the test exercises them at all — most stay covered by the
existing pipeline-level tests.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from src.cli.app import app


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


# ---------------------------------------------------------------------------
# version
# ---------------------------------------------------------------------------


def test_version_text(runner: CliRunner) -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "ryotenkai " in result.stdout


def test_version_json(runner: CliRunner) -> None:
    result = runner.invoke(app, ["-o", "json", "version"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert "ryotenkai" in payload
    assert "python" in payload
    assert "platform" in payload


def test_version_yaml(runner: CliRunner) -> None:
    result = runner.invoke(app, ["-o", "yaml", "version"])
    assert result.exit_code == 0
    assert "ryotenkai:" in result.stdout
    assert "python:" in result.stdout


# ---------------------------------------------------------------------------
# runs ls — empty + populated dirs
# ---------------------------------------------------------------------------


def test_runs_ls_empty_text(runner: CliRunner, tmp_path: Path) -> None:
    result = runner.invoke(app, ["runs", "ls", str(tmp_path)])
    assert result.exit_code == 0
    assert "No runs found" in result.stdout


def test_runs_ls_empty_json(runner: CliRunner, tmp_path: Path) -> None:
    result = runner.invoke(app, ["-o", "json", "runs", "ls", str(tmp_path)])
    assert result.exit_code == 0
    assert json.loads(result.stdout) == []


# ---------------------------------------------------------------------------
# runs inspect — error path on missing run dir
# ---------------------------------------------------------------------------


def test_runs_inspect_missing_run_dir(runner: CliRunner, tmp_path: Path) -> None:
    result = runner.invoke(app, ["runs", "inspect", str(tmp_path / "nope")])
    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# config validate — missing file
# ---------------------------------------------------------------------------


def test_config_validate_missing_file(runner: CliRunner, tmp_path: Path) -> None:
    result = runner.invoke(
        app, ["config", "validate", "--config", str(tmp_path / "x.yaml")],
    )
    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# dataset validate — NR-04: explicit-plugins gate
# ---------------------------------------------------------------------------


def test_dataset_validate_no_validation_plugins(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the config has no validation plugins, exit 2 with a hint."""
    from src.cli.commands import dataset as dataset_cmd_mod

    class FakeDataset:
        validations = type("V", (), {"plugins": []})()

    fake_cfg = type("Cfg", (), {"datasets": {"d1": FakeDataset()}})()

    monkeypatch.setattr(
        "src.utils.config.load_config", lambda _: fake_cfg, raising=True,
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text("model: {}\n", encoding="utf-8")

    result = runner.invoke(
        app, ["dataset", "validate", "--config", str(config_path)],
    )
    assert result.exit_code == 2
    combined = result.stdout + (result.stderr or "")
    assert "no validation plugins" in combined.lower()
    _ = dataset_cmd_mod  # silence "unused" — import proves module loads


# ---------------------------------------------------------------------------
# project use / current — context store flow
# ---------------------------------------------------------------------------


def test_project_current_when_unset(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RYOTENKAI_HOME", str(tmp_path))
    result = runner.invoke(app, ["project", "current"])
    assert result.exit_code == 0
    assert "no project selected" in result.stdout


def test_project_current_after_use_dry_run_does_not_persist(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``project use --dry-run`` must NOT touch the context file."""
    from src.api.services import project_service

    monkeypatch.setenv("RYOTENKAI_HOME", str(tmp_path))
    monkeypatch.setattr(
        project_service, "get_detail",
        lambda *a, **k: type("D", (), {"id": "x"})(),
    )

    result = runner.invoke(app, ["project", "use", "alpha", "--dry-run"])
    assert result.exit_code == 0
    assert "dry-run" in result.stdout
    # No file written.
    assert not (tmp_path / "cli-context.json").exists()


def test_project_use_persists_then_current_reads_it(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.api.services import project_service

    monkeypatch.setenv("RYOTENKAI_HOME", str(tmp_path))
    monkeypatch.setattr(
        project_service, "get_detail",
        lambda *a, **k: type("D", (), {"id": "alpha"})(),
    )

    use_result = runner.invoke(app, ["project", "use", "alpha"])
    assert use_result.exit_code == 0
    assert "current project: alpha" in use_result.stdout

    cur_result = runner.invoke(app, ["project", "current"])
    assert cur_result.exit_code == 0
    assert "alpha" in cur_result.stdout


# ---------------------------------------------------------------------------
# plugin validate — wraps validate_manifest_file
# ---------------------------------------------------------------------------


def test_plugin_validate_missing_file(runner: CliRunner, tmp_path: Path) -> None:
    result = runner.invoke(app, ["plugin", "validate", str(tmp_path / "ghost.toml")])
    # File does not exist — Typer's ``exists=True`` rejects before we reach
    # the validator, so exit 2 (Click usage error) is acceptable.
    assert result.exit_code != 0


def test_plugin_validate_valid_manifest(
    runner: CliRunner, tmp_path: Path,
) -> None:
    from src.community.constants import MANIFEST_FILENAME
    from src.community.manifest import LATEST_SCHEMA_VERSION

    folder = tmp_path / "p"
    folder.mkdir()
    (folder / MANIFEST_FILENAME).write_text(
        f"schema_version = {LATEST_SCHEMA_VERSION}\n"
        "[plugin]\n"
        'id = "p"\n'
        'kind = "validation"\n'
        'name = "p"\n'
        'version = "1.0.0"\n'
        'description = "x"\n'
        "\n[plugin.entry_point]\n"
        'module = "plugin"\n'
        'class = "X"\n',
        encoding="utf-8",
    )
    result = runner.invoke(app, ["plugin", "validate", str(folder)])
    assert result.exit_code == 0
    assert "OK" in result.stdout


# ---------------------------------------------------------------------------
# run start — dry-run path (must not start orchestrator)
# ---------------------------------------------------------------------------


def test_run_start_dry_run_succeeds(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``run start --dry-run`` validates config without touching orchestrator."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text("model: {}\n", encoding="utf-8")

    monkeypatch.setattr(
        "src.utils.config.load_config",
        lambda _: type("Cfg", (), {})(),
        raising=True,
    )

    result = runner.invoke(
        app, ["run", "start", "--config", str(config_path), "--dry-run"],
    )
    assert result.exit_code == 0
    assert "dry-run" in result.stdout


# ---------------------------------------------------------------------------
# server status / stop — reserved stubs
# ---------------------------------------------------------------------------


def test_server_status_is_reserved(runner: CliRunner) -> None:
    result = runner.invoke(app, ["server", "status"])
    assert result.exit_code != 0


def test_server_stop_is_reserved(runner: CliRunner) -> None:
    result = runner.invoke(app, ["server", "stop"])
    assert result.exit_code != 0
