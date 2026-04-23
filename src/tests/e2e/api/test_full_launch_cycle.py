"""End-to-end: POST /launch -> subprocess writes pipeline_state.json -> GET /runs/{id} reflects result.

We don't spawn a real training process (that needs GPUs / MLflow). We instead
point execute_launch_subprocess at a tiny Python one-liner that writes a
minimal pipeline_state.json and exits. This validates the API + state-store
contract end-to-end without booting the full orchestrator.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api.config import ApiSettings
from src.api.main import create_app
from src.api.services import launch_service
from src.pipeline import launch as pipeline_launch


pytestmark = pytest.mark.slow


def _fake_spawn_writes_state(run_dir: Path):
    """Return a spawn fn that writes a valid pipeline_state.json via a real
    subprocess (so we exercise the detached-process path end-to-end)."""

    def _spawn(request, *, python_executable=None, extra_env=None, **_ignored):  # type: ignore[no-untyped-def]
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "attempts" / "attempt_1").mkdir(parents=True, exist_ok=True)
        state_payload = {
            "schema_version": 1,
            "logical_run_id": run_dir.name,
            "run_directory": str(run_dir),
            "config_path": "config/pipeline.yaml",
            "active_attempt_id": f"{run_dir.name}:attempt:1",
            "pipeline_status": "completed",
            "training_critical_config_hash": "h1",
            "late_stage_config_hash": "h2",
            "model_dataset_config_hash": "h3",
            "root_mlflow_run_id": None,
            "mlflow_runtime_tracking_uri": None,
            "mlflow_ca_bundle_path": None,
            "attempts": [
                {
                    "attempt_id": f"{run_dir.name}:attempt:1",
                    "attempt_no": 1,
                    "runtime_name": "single_node",
                    "requested_action": "fresh",
                    "effective_action": "fresh",
                    "restart_from_stage": None,
                    "status": "completed",
                    "started_at": "2026-04-19T10:00:00+00:00",
                    "completed_at": "2026-04-19T10:00:05+00:00",
                    "error": None,
                    "training_critical_config_hash": "h1",
                    "late_stage_config_hash": "h2",
                    "model_dataset_config_hash": "h3",
                    "root_mlflow_run_id": None,
                    "pipeline_attempt_mlflow_run_id": None,
                    "training_run_id": None,
                    "enabled_stage_names": ["Dataset Validator"],
                    "stage_runs": {
                        "Dataset Validator": {
                            "stage_name": "Dataset Validator",
                            "status": "completed",
                            "execution_mode": "executed",
                            "outputs": {},
                            "error": None,
                            "failure_kind": None,
                            "reuse_from": None,
                            "skip_reason": None,
                            "started_at": "2026-04-19T10:00:00+00:00",
                            "completed_at": "2026-04-19T10:00:02+00:00",
                        }
                    },
                }
            ],
            "current_output_lineage": {},
        }
        state_path = run_dir / "pipeline_state.json"
        # Write via subprocess so we cross a real fork/exec boundary.
        import subprocess

        code = f"""
import json, pathlib, time
payload = {json.dumps(state_payload)!r}
pathlib.Path({str(state_path)!r}).write_text(payload, encoding='utf-8')
pathlib.Path({str(run_dir / 'attempts' / 'attempt_1' / 'pipeline.log')!r}).write_text('e2e test log line\\n', encoding='utf-8')
"""
        proc = subprocess.Popen([sys.executable, "-c", code])
        launcher_log = run_dir / "tui_launch.log"
        launcher_log.write_text("[e2e] spawned\n", encoding="utf-8")
        return proc.pid, (sys.executable, "-c", "..."), launcher_log

    return _spawn


def test_launch_then_poll_state_reflects_completion(tmp_path: Path, monkeypatch) -> None:
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    run_id = "run_e2e_1"
    target_dir = runs_dir / run_id

    settings = ApiSettings(runs_dir=runs_dir, serve_spa=False)
    app = create_app(settings)
    from src.api.dependencies import get_settings

    app.dependency_overrides[get_settings] = lambda: settings

    monkeypatch.setattr(launch_service, "spawn_launch_detached", _fake_spawn_writes_state(target_dir))

    with TestClient(app) as client:
        r_create = client.post("/api/v1/runs", json={"run_id": run_id})
        assert r_create.status_code == 201

        config_file = tmp_path / "config.yaml"
        config_file.write_text("stub", encoding="utf-8")

        r_launch = client.post(
            f"/api/v1/runs/{run_id}/launch",
            json={"mode": "fresh", "config_path": str(config_file)},
        )
        assert r_launch.status_code == 202

        # Wait for subprocess to write state (tight loop, <3s).
        deadline = time.time() + 3.0
        status = None
        while time.time() < deadline:
            r_detail = client.get(f"/api/v1/runs/{run_id}")
            if r_detail.status_code == 200:
                status = r_detail.json()["status"]
                if status == "completed":
                    break
            time.sleep(0.05)

        assert status == "completed"

        # API view must match disk view byte-equal for core fields.
        disk_state = json.loads((target_dir / "pipeline_state.json").read_text(encoding="utf-8"))
        api_state = client.get(f"/api/v1/runs/{run_id}").json()
        assert disk_state["logical_run_id"] == api_state["logical_run_id"]
        assert disk_state["pipeline_status"] == api_state["pipeline_status"]
        assert len(disk_state["attempts"]) == len(api_state["attempts"])


def test_stale_lock_interrupt_cleans_up(tmp_path: Path, monkeypatch) -> None:
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    run_id = "run_e2e_2"
    target = runs_dir / run_id
    target.mkdir(parents=True, exist_ok=True)
    # Minimal state so /interrupt dependency chain works.
    (target / "pipeline_state.json").write_text(
        json.dumps({
            "schema_version": 1,
            "logical_run_id": run_id,
            "run_directory": str(target),
            "config_path": "",
            "active_attempt_id": None,
            "pipeline_status": "running",
            "training_critical_config_hash": "",
            "late_stage_config_hash": "",
            "model_dataset_config_hash": "",
            "attempts": [],
            "current_output_lineage": {},
        }),
        encoding="utf-8",
    )
    (target / "run.lock").write_text("pid=999999\nstarted_at=2026\n", encoding="utf-8")

    monkeypatch.setattr(pipeline_launch, "is_process_alive", lambda pid: False)
    monkeypatch.setattr(launch_service, "is_process_alive", lambda pid: False)

    settings = ApiSettings(runs_dir=runs_dir, serve_spa=False)
    app = create_app(settings)
    from src.api.dependencies import get_settings

    app.dependency_overrides[get_settings] = lambda: settings
    with TestClient(app) as client:
        r = client.post(f"/api/v1/runs/{run_id}/interrupt")
        assert r.status_code == 200
        body = r.json()
        assert body["interrupted"] is False
        assert body["reason"] == "process_not_found"
        assert not (target / "run.lock").exists()
