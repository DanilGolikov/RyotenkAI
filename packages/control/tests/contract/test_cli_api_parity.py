"""Parity tests: every CLI read command must agree with its API counterpart.

Plan B Q-23 / NR-02: the CLI reads ``pipeline_state.json`` directly,
the API goes through services + a process-local cache. Both paths must
emit the same payload modulo cosmetic differences (timestamp precision,
list ordering, path types). When they drift, this gate fires.

Each test wires the CLI to the same ``runs_dir`` / catalog the API
sees, invokes both sides, and compares payloads through
:func:`src.tests.contract._normalize.normalise`.

We deliberately keep the suite small (one test per pair) — the goal is
contract enforcement, not exhaustive coverage. Per-command edge cases
live in unit tests under ``src/tests/unit/cli`` and
``src/tests/integration/api``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from src.tests.contract._normalize import normalise


def _seed_run(runs_dir: Path, run_id: str = "run_parity") -> Path:
    """Materialise a minimal completed-run directory the CLI + API can read."""
    from src.pipeline.state import (
        PipelineAttemptState,
        PipelineState,
        PipelineStateStore,
        StageRunState,
    )

    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    store = PipelineStateStore(run_dir)
    stage_runs = {
        "Dataset Validator": StageRunState(
            stage_name="Dataset Validator",
            status=StageRunState.STATUS_COMPLETED,
            execution_mode=StageRunState.MODE_EXECUTED,
            started_at="2026-04-10T10:00:00+00:00",
            completed_at="2026-04-10T10:05:00+00:00",
        ),
    }
    attempt = PipelineAttemptState(
        attempt_id=f"{run_id}:attempt:1",
        attempt_no=1,
        runtime_name="single_node",
        requested_action="fresh",
        effective_action="fresh",
        restart_from_stage=None,
        status=StageRunState.STATUS_COMPLETED,
        started_at="2026-04-10T10:00:00+00:00",
        completed_at="2026-04-10T10:30:00+00:00",
        training_critical_config_hash="train-hash",
        late_stage_config_hash="late-hash",
        model_dataset_config_hash="md-hash",
        root_mlflow_run_id=None,
        enabled_stage_names=list(stage_runs.keys()),
        stage_runs=stage_runs,
    )
    state = PipelineState(
        schema_version=1,
        logical_run_id=run_id,
        run_directory=str(run_dir),
        config_path="config/pipeline.yaml",
        active_attempt_id=attempt.attempt_id,
        pipeline_status=StageRunState.STATUS_COMPLETED,
        training_critical_config_hash="train-hash",
        late_stage_config_hash="late-hash",
        model_dataset_config_hash="md-hash",
        attempts=[attempt],
        current_output_lineage={},
    )
    store.save(state)
    return run_dir


# ---------------------------------------------------------------------------
# runs ls ↔ GET /runs
# ---------------------------------------------------------------------------


def test_runs_ls_matches_api_runs_list(
    runs_dir: Path,
    cli_runner: CliRunner,
    cli_app_obj,
    api_client: TestClient,
) -> None:
    _seed_run(runs_dir, "run_alpha")
    _seed_run(runs_dir, "run_beta")

    cli_result = cli_runner.invoke(
        cli_app_obj, ["-o", "json", "runs", "ls", str(runs_dir)],
    )
    assert cli_result.exit_code == 0, cli_result.stdout
    cli_payload = json.loads(cli_result.stdout)

    api_response = api_client.get("/api/v1/runs")
    assert api_response.status_code == 200
    api_payload = api_response.json()

    # Project both shapes onto the keys they share. CLI emits a flat
    # list of rows; API wraps in ``{"groups": {<group_name>: [rows]}}``
    # — flatten across all groups.
    api_rows = [r for rows in api_payload.get("groups", {}).values() for r in rows]
    cli_ids = sorted(row["run_id"] for row in cli_payload)
    api_ids = sorted(row["run_id"] for row in api_rows)
    assert cli_ids == api_ids


# ---------------------------------------------------------------------------
# runs inspect <run> ↔ GET /runs/{id}
# ---------------------------------------------------------------------------


def test_runs_inspect_matches_api_get_run(
    runs_dir: Path,
    cli_runner: CliRunner,
    cli_app_obj,
    api_client: TestClient,
) -> None:
    run_dir = _seed_run(runs_dir, "run_gamma")

    cli_result = cli_runner.invoke(
        cli_app_obj, ["-o", "json", "runs", "inspect", str(run_dir)],
    )
    assert cli_result.exit_code == 0, cli_result.stdout
    cli_payload = json.loads(cli_result.stdout)

    api_response = api_client.get(f"/api/v1/runs/{run_dir.name}")
    assert api_response.status_code == 200
    api_payload = api_response.json()

    # Compare a stable subset that both sides are expected to expose
    # identically. Field renames (CLI ``run_id`` ↔ API ``logical_run_id``)
    # are bridged here to stay declarative.
    cli_subset = {
        "logical_run_id": cli_payload["logical_run_id"],
        "status": cli_payload["status"],
        "config_path": cli_payload["config_path"],
        "attempts_count": len(cli_payload["attempts"]),
        "first_attempt_status": cli_payload["attempts"][0]["status"],
    }
    api_subset = {
        "logical_run_id": api_payload["logical_run_id"],
        "status": api_payload["status"],
        "config_path": api_payload["config_path"],
        "attempts_count": len(api_payload["attempts"]),
        "first_attempt_status": api_payload["attempts"][0]["status"],
    }
    assert normalise(cli_subset) == normalise(api_subset)


# ---------------------------------------------------------------------------
# preset ls ↔ GET /config/presets
# ---------------------------------------------------------------------------


def test_preset_ls_matches_api_presets(
    cli_runner: CliRunner,
    cli_app_obj,
    api_client: TestClient,
) -> None:
    cli_result = cli_runner.invoke(cli_app_obj, ["-o", "json", "preset", "ls"])
    assert cli_result.exit_code == 0, cli_result.stdout
    cli_payload = json.loads(cli_result.stdout)

    api_response = api_client.get("/api/v1/config/presets")
    assert api_response.status_code == 200
    api_payload = api_response.json()

    # CLI exposes the manifest field as ``id``; the API schema renames
    # it to ``name`` for the preset dropdown. Both refer to the same
    # ``preset.id`` from manifest.toml.
    cli_ids = sorted(row["id"] for row in cli_payload)
    api_ids = sorted(row["name"] for row in api_payload.get("presets", []))
    assert cli_ids == api_ids
