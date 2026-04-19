from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import pytest
from fastapi.testclient import TestClient

from src.api.config import ApiSettings
from src.api.dependencies import get_settings
from src.api.main import create_app
from src.pipeline.state import (
    PipelineAttemptState,
    PipelineState,
    PipelineStateStore,
    StageRunState,
)


@pytest.fixture
def runs_dir(tmp_path: Path) -> Path:
    target = tmp_path / "runs"
    target.mkdir(parents=True, exist_ok=True)
    return target


@pytest.fixture
def projects_root(tmp_path: Path) -> Path:
    target = tmp_path / "ryotenkai_home"
    target.mkdir(parents=True, exist_ok=True)
    return target


@pytest.fixture
def settings(runs_dir: Path, projects_root: Path) -> ApiSettings:
    return ApiSettings(
        runs_dir=runs_dir,
        projects_root=projects_root,
        serve_spa=False,
        cors_origins=["http://localhost:5173"],
    )


@pytest.fixture
def client(settings: ApiSettings) -> Iterator[TestClient]:
    app = create_app(settings)
    app.dependency_overrides[get_settings] = lambda: settings
    with TestClient(app) as test_client:
        yield test_client


def _make_attempt(
    *,
    attempt_no: int = 1,
    status: str = StageRunState.STATUS_COMPLETED,
    stages: dict[str, str] | None = None,
) -> PipelineAttemptState:
    stage_runs: dict[str, StageRunState] = {}
    stage_statuses = stages or {"Dataset Validator": StageRunState.STATUS_COMPLETED}
    for name, stage_status in stage_statuses.items():
        stage_runs[name] = StageRunState(
            stage_name=name,
            status=stage_status,
            execution_mode=StageRunState.MODE_EXECUTED,
            started_at="2026-04-10T10:00:00+00:00",
            completed_at="2026-04-10T10:05:00+00:00" if stage_status != StageRunState.STATUS_RUNNING else None,
        )
    return PipelineAttemptState(
        attempt_id=f"run_test:attempt:{attempt_no}",
        attempt_no=attempt_no,
        runtime_name="single_node",
        requested_action="fresh",
        effective_action="fresh",
        restart_from_stage=None,
        status=status,
        started_at="2026-04-10T10:00:00+00:00",
        completed_at="2026-04-10T10:30:00+00:00" if status != StageRunState.STATUS_RUNNING else None,
        training_critical_config_hash="train-hash",
        late_stage_config_hash="late-hash",
        model_dataset_config_hash="md-hash",
        root_mlflow_run_id=None,
        enabled_stage_names=list(stage_statuses.keys()),
        stage_runs=stage_runs,
    )


@pytest.fixture
def seed_completed_run(runs_dir: Path):
    """Factory that creates a run directory with a completed attempt."""

    def _seed(run_id: str = "run_test_completed", *, config_path: str = "config/pipeline.yaml") -> Path:
        run_dir = runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        store = PipelineStateStore(run_dir)
        attempt = _make_attempt(attempt_no=1, status=StageRunState.STATUS_COMPLETED)
        state = PipelineState(
            schema_version=1,
            logical_run_id=run_id,
            run_directory=str(run_dir),
            config_path=config_path,
            active_attempt_id=attempt.attempt_id,
            pipeline_status=StageRunState.STATUS_COMPLETED,
            training_critical_config_hash="train-hash",
            late_stage_config_hash="late-hash",
            model_dataset_config_hash="md-hash",
            attempts=[attempt],
            current_output_lineage={},
        )
        store.save(state)
        attempt_dir = store.next_attempt_dir(1)
        attempt_dir.mkdir(parents=True, exist_ok=True)
        (attempt_dir / "pipeline.log").write_text(
            "2026-04-10 10:00:00 | INFO | pipeline start\n"
            "2026-04-10 10:05:00 | INFO | Dataset Validator completed\n",
            encoding="utf-8",
        )
        return run_dir

    return _seed


@pytest.fixture
def seed_running_run(runs_dir: Path):
    """Factory that creates a run with a running attempt and a lock file."""

    def _seed(run_id: str = "run_test_running", *, pid: int = 999999) -> Path:
        run_dir = runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        store = PipelineStateStore(run_dir)
        attempt = _make_attempt(
            attempt_no=1,
            status=StageRunState.STATUS_RUNNING,
            stages={"Dataset Validator": StageRunState.STATUS_RUNNING},
        )
        state = PipelineState(
            schema_version=1,
            logical_run_id=run_id,
            run_directory=str(run_dir),
            config_path="config/pipeline.yaml",
            active_attempt_id=attempt.attempt_id,
            pipeline_status=StageRunState.STATUS_RUNNING,
            training_critical_config_hash="train-hash",
            late_stage_config_hash="late-hash",
            model_dataset_config_hash="md-hash",
            attempts=[attempt],
            current_output_lineage={},
        )
        store.save(state)
        (run_dir / "run.lock").write_text(f"pid={pid}\nstarted_at=2026-04-10T10:00:00+00:00\n", encoding="utf-8")
        return run_dir

    return _seed


@pytest.fixture
def state_json_raw(runs_dir: Path):
    """Helper to read state file directly for assertions."""

    def _read(run_id: str) -> dict:
        return json.loads((runs_dir / run_id / "pipeline_state.json").read_text(encoding="utf-8"))

    return _read
