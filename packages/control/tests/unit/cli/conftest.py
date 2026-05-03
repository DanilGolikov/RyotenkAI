"""Shared fixtures for CLI tests."""

from __future__ import annotations

from pathlib import Path  # noqa: TC003 -- runtime use in factory return type

import pytest

from src.pipeline.state import (
    PipelineAttemptState,
    PipelineState,
    PipelineStateStore,
    StageRunState,
)


def _make_stage(name: str, status: str) -> StageRunState:
    return StageRunState(
        stage_name=name,
        status=status,
        execution_mode=StageRunState.MODE_EXECUTED,
        started_at="2026-04-10T10:00:00+00:00",
        completed_at="2026-04-10T10:05:00+00:00"
            if status != StageRunState.STATUS_RUNNING
            else None,
    )


def _make_attempt(
    *,
    attempt_no: int,
    status: str = StageRunState.STATUS_COMPLETED,
    stages: dict[str, str] | None = None,
) -> PipelineAttemptState:
    stages = stages or {"Dataset Validator": StageRunState.STATUS_COMPLETED}
    stage_runs = {name: _make_stage(name, st) for name, st in stages.items()}
    return PipelineAttemptState(
        attempt_id=f"run_test:attempt:{attempt_no}",
        attempt_no=attempt_no,
        runtime_name="single_node",
        requested_action="fresh" if attempt_no == 1 else "restart",
        effective_action="fresh" if attempt_no == 1 else "restart",
        restart_from_stage=None,
        status=status,
        started_at="2026-04-10T10:00:00+00:00",
        completed_at="2026-04-10T10:05:00+00:00"
            if status != StageRunState.STATUS_RUNNING
            else None,
        training_critical_config_hash="train-hash",
        late_stage_config_hash="late-hash",
        model_dataset_config_hash="md-hash",
        root_mlflow_run_id=None,
        enabled_stage_names=list(stages.keys()),
        stage_runs=stage_runs,
    )


@pytest.fixture()
def seed_run(tmp_path: Path):
    """Factory that drops a pipeline_state.json into a tmp runs dir."""
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()

    def _seed(
        run_id: str,
        *,
        status: str = StageRunState.STATUS_COMPLETED,
        attempts: int = 1,
    ) -> Path:
        run_dir = runs_dir / run_id
        run_dir.mkdir()
        store = PipelineStateStore(run_dir)
        attempt_list = [
            _make_attempt(attempt_no=i, status=status) for i in range(1, attempts + 1)
        ]
        state = PipelineState(
            schema_version=1,
            logical_run_id=run_id,
            run_directory=str(run_dir),
            config_path="config/pipeline.yaml",
            active_attempt_id=attempt_list[-1].attempt_id,
            pipeline_status=status,
            training_critical_config_hash="train-hash",
            late_stage_config_hash="late-hash",
            model_dataset_config_hash="md-hash",
            attempts=attempt_list,
            current_output_lineage={},
        )
        store.save(state)
        return run_dir

    return _seed


@pytest.fixture()
def runs_dir_factory(seed_run):
    """Returns the runs dir after seeding one completed run."""

    def _make(*, run_count: int = 1) -> Path:
        last_dir = None
        for i in range(run_count):
            last_dir = seed_run(f"run_test_{i:03d}")
        return last_dir.parent if last_dir else None  # type: ignore[union-attr]

    return _make
