"""Phase G — CLI tests for ``ryotenkai runs conditions``.

Verifies kubectl-style condition table rendering and JSON output for
the new ``runs conditions`` command. Builds a minimal
``pipeline_state.json`` on disk with seeded conditions and invokes the
CLI through Typer's :class:`CliRunner` — no service mocks.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from ryotenkai_control.cli.app import app


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


def _seed_run_dir(run_dir: Path, *, condition_ts: datetime) -> Path:
    """Write a minimal ``pipeline_state.json`` with two stages, one of
    which has a Progressing=True + Available=False condition."""
    run_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "schema_version": 1,
        "logical_run_id": "test-run",
        "run_directory": str(run_dir),
        "config_path": "",
        "active_attempt_id": None,
        "pipeline_status": "running",
        "training_critical_config_hash": "",
        "late_stage_config_hash": "",
        "model_dataset_config_hash": "",
        "attempts": [
            {
                "attempt_id": "att-1",
                "attempt_no": 1,
                "runtime_name": "test",
                "requested_action": "fresh",
                "effective_action": "fresh",
                "restart_from_stage": None,
                "status": "running",
                "started_at": "2026-05-16T00:00:00+00:00",
                "completed_at": None,
                "enabled_stage_names": ["alpha", "beta"],
                "stage_runs": {
                    "alpha": {
                        "stage_name": "alpha",
                        "status": "running",
                        "conditions": [
                            {
                                "type": "Progressing",
                                "status": "True",
                                "reason": "StageStarted",
                                "message": "alpha is running",
                                "last_transition_time": condition_ts.isoformat(),
                            },
                            {
                                "type": "Available",
                                "status": "False",
                                "reason": "NotYet",
                                "message": None,
                                "last_transition_time": condition_ts.isoformat(),
                            },
                        ],
                    },
                    "beta": {
                        "stage_name": "beta",
                        "status": "pending",
                    },
                },
            }
        ],
        "current_output_lineage": {},
    }
    (run_dir / "pipeline_state.json").write_text(json.dumps(state), encoding="utf-8")
    return run_dir


def test_conditions_text_renders_table(runner: CliRunner, tmp_path: Path) -> None:
    run_dir = _seed_run_dir(tmp_path / "run-1", condition_ts=datetime.now(UTC))
    result = runner.invoke(app, ["runs", "conditions", str(run_dir)])
    assert result.exit_code == 0, result.stdout
    # The header should include all standard types as columns.
    for col in ("STAGE", "AVAILABLE", "PROGRESSING", "DEGRADED", "OOMRISK", "RATELIMITED", "AGE"):
        assert col in result.stdout
    # Stage names render as rows.
    assert "alpha" in result.stdout
    assert "beta" in result.stdout
    # Alpha has Progressing=True.
    assert "True" in result.stdout


def test_conditions_json_payload_shape(runner: CliRunner, tmp_path: Path) -> None:
    ts = datetime(2026, 5, 16, 12, 0, 0, tzinfo=UTC)
    run_dir = _seed_run_dir(tmp_path / "run-2", condition_ts=ts)
    result = runner.invoke(app, ["-o", "json", "runs", "conditions", str(run_dir)])
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["attempt_no"] == 1
    stages = {s["name"]: s for s in payload["stages"]}
    assert set(stages.keys()) == {"alpha", "beta"}
    alpha = stages["alpha"]
    assert len(alpha["conditions"]) == 2
    types = {c["type"]: c for c in alpha["conditions"]}
    assert types["Progressing"]["status"] == "True"
    assert types["Available"]["status"] == "False"
    # beta has no conditions.
    assert stages["beta"]["conditions"] == []


def test_conditions_missing_run_dir(runner: CliRunner, tmp_path: Path) -> None:
    result = runner.invoke(app, ["runs", "conditions", str(tmp_path / "nope")])
    assert result.exit_code != 0
