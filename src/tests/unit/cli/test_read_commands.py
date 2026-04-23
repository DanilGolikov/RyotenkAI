"""Text/JSON path tests for the migrated read commands."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


# ---------------------------------------------------------------------------
# runs-list
# ---------------------------------------------------------------------------


def test_runs_list_empty_dir_text(runner: CliRunner, tmp_path) -> None:
    from src.main import app

    out = runner.invoke(app, ["runs-list", str(tmp_path)])
    assert out.exit_code == 0, out.output
    assert "No runs found" in out.output


def test_runs_list_empty_dir_json(runner: CliRunner, tmp_path) -> None:
    from src.main import app

    out = runner.invoke(app, ["-o", "json", "runs-list", str(tmp_path)])
    assert out.exit_code == 0, out.output
    assert json.loads(out.output) == []


def test_runs_list_seeded_text(runner: CliRunner, seed_run) -> None:
    from src.main import app

    run_dir = seed_run("run_test_alpha")
    out = runner.invoke(app, ["runs-list", str(run_dir.parent)])
    assert out.exit_code == 0, out.output
    assert "run_test_alpha" in out.output


def test_runs_list_seeded_json(runner: CliRunner, seed_run) -> None:
    from src.main import app

    run_dir = seed_run("run_test_beta")
    out = runner.invoke(app, ["-o", "json", "runs-list", str(run_dir.parent)])
    assert out.exit_code == 0, out.output
    payload = json.loads(out.output)
    assert len(payload) == 1
    row = payload[0]
    # Contract — stable field set
    for key in ("run_id", "status", "attempts", "started_at", "completed_at", "duration_s", "config_name"):
        assert key in row, f"missing key: {key}"
    assert row["run_id"] == "run_test_beta"


# ---------------------------------------------------------------------------
# inspect-run
# ---------------------------------------------------------------------------


def test_inspect_run_missing_dir_errors_cleanly(runner: CliRunner, tmp_path) -> None:
    from src.main import app

    out = runner.invoke(app, ["inspect-run", str(tmp_path / "nope")])
    assert out.exit_code != 0
    combined = out.output + (out.stderr or "")
    # No Rich traceback box — just a single error line with optional hint.
    assert "error:" in combined
    assert "Traceback" not in combined


def test_inspect_run_text(runner: CliRunner, seed_run) -> None:
    from src.main import app

    run_dir = seed_run("run_test_inspect")
    out = runner.invoke(app, ["inspect-run", str(run_dir)])
    assert out.exit_code == 0, out.output
    assert "run_test_inspect" in out.output


def test_inspect_run_json(runner: CliRunner, seed_run) -> None:
    from src.main import app

    run_dir = seed_run("run_test_inspect_json")
    out = runner.invoke(app, ["-o", "json", "inspect-run", str(run_dir)])
    assert out.exit_code == 0, out.output
    payload = json.loads(out.output)
    assert payload["run_id"] == "run_test_inspect_json"
    assert payload["attempts"][0]["attempt_no"] == 1
    assert isinstance(payload["attempts"][0]["stages"], list)


# ---------------------------------------------------------------------------
# run-status
# ---------------------------------------------------------------------------


def test_run_status_once_text(runner: CliRunner, seed_run) -> None:
    from src.main import app

    run_dir = seed_run("run_test_status")
    out = runner.invoke(app, ["run-status", str(run_dir), "--once"])
    assert out.exit_code == 0, out.output
    assert "run_test_status" in out.output


def test_run_status_json_is_snapshot(runner: CliRunner, seed_run) -> None:
    from src.main import app

    run_dir = seed_run("run_test_status_json")
    out = runner.invoke(app, ["-o", "json", "run-status", str(run_dir)])
    assert out.exit_code == 0, out.output
    payload = json.loads(out.output)
    assert payload["run_id"] == "run_test_status_json"
    assert payload["current_attempt"] == 1
    assert isinstance(payload["stages"], list)


# ---------------------------------------------------------------------------
# run-diff
# ---------------------------------------------------------------------------


def test_run_diff_single_attempt_text(runner: CliRunner, seed_run) -> None:
    from src.main import app

    run_dir = seed_run("run_test_diff_single")
    out = runner.invoke(app, ["run-diff", str(run_dir)])
    assert out.exit_code == 0, out.output
    assert "nothing to compare" in out.output.lower()


def test_run_diff_single_attempt_json(runner: CliRunner, seed_run) -> None:
    from src.main import app

    run_dir = seed_run("run_test_diff_single_json")
    out = runner.invoke(app, ["-o", "json", "run-diff", str(run_dir)])
    assert out.exit_code == 0, out.output
    payload = json.loads(out.output)
    assert payload["status"] == "single_attempt"


def test_run_diff_two_attempts_json(runner: CliRunner, seed_run) -> None:
    from src.main import app

    run_dir = seed_run("run_test_diff_two", attempts=2)
    out = runner.invoke(app, ["-o", "json", "run-diff", str(run_dir)])
    assert out.exit_code == 0, out.output
    payload = json.loads(out.output)
    assert payload["attempt_a"] == 1
    assert payload["attempt_b"] == 2
    assert "training_critical_changed" in payload
    assert "late_stage_changed" in payload


# ---------------------------------------------------------------------------
# config-validate — smoke via missing config
# ---------------------------------------------------------------------------


def test_config_validate_missing_config_fails_cleanly(
    runner: CliRunner, tmp_path
) -> None:
    from src.main import app

    out = runner.invoke(app, ["config-validate", "-c", str(tmp_path / "nope.yaml")])
    assert out.exit_code != 0
    combined = out.output + (out.stderr or "")
    assert "Traceback" not in combined
