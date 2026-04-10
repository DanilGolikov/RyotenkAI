"""Tests for pipeline state transition functions (src/pipeline/state/transitioner.py).

Covers all 6 extracted functions:
- mark_stage_running
- mark_stage_completed
- mark_stage_failed
- mark_stage_skipped
- mark_stage_interrupted
- finalize_attempt_state
"""

from __future__ import annotations

import pytest

from src.pipeline.state import PipelineAttemptState, PipelineState, StageRunState
from src.pipeline.state.transitioner import (
    finalize_attempt_state,
    mark_stage_completed,
    mark_stage_failed,
    mark_stage_interrupted,
    mark_stage_running,
    mark_stage_skipped,
)

pytestmark = pytest.mark.unit


def _make_attempt(attempt_id: str = "a1", attempt_no: int = 1) -> PipelineAttemptState:
    return PipelineAttemptState(
        attempt_id=attempt_id,
        attempt_no=attempt_no,
        runtime_name="runpod",
        requested_action="fresh",
        effective_action="fresh",
        restart_from_stage=None,
        status="running",
        started_at="2026-01-01T00:00:00Z",
    )


def _make_state(active_attempt_id: str = "a1") -> PipelineState:
    return PipelineState(
        schema_version=1,
        logical_run_id="run-1",
        run_directory="/tmp/run",
        config_path="/tmp/config.yaml",
        active_attempt_id=active_attempt_id,
        pipeline_status="running",
        training_critical_config_hash="",
        late_stage_config_hash="",
    )


class TestMarkStageRunning:
    def test_creates_running_entry(self) -> None:
        attempt = _make_attempt()
        mark_stage_running(attempt=attempt, stage_name="Dataset Validator", started_at="2026-01-01T00:00:00Z")
        state = attempt.stage_runs["Dataset Validator"]
        assert state.status == StageRunState.STATUS_RUNNING
        assert state.execution_mode == StageRunState.MODE_EXECUTED
        assert state.started_at == "2026-01-01T00:00:00Z"

    def test_overwrites_existing_entry(self) -> None:
        attempt = _make_attempt()
        mark_stage_running(attempt=attempt, stage_name="s1", started_at="t1")
        mark_stage_running(attempt=attempt, stage_name="s1", started_at="t2")
        assert attempt.stage_runs["s1"].started_at == "t2"


class TestMarkStageCompleted:
    def test_marks_completed_with_outputs(self) -> None:
        attempt = _make_attempt()
        mark_stage_running(attempt=attempt, stage_name="GPU Deployer", started_at="t0")
        mark_stage_completed(attempt=attempt, stage_name="GPU Deployer", outputs={"ssh_host": "1.2.3.4"})
        state = attempt.stage_runs["GPU Deployer"]
        assert state.status == StageRunState.STATUS_COMPLETED
        assert state.outputs == {"ssh_host": "1.2.3.4"}
        assert state.error is None
        assert state.failure_kind is None
        assert state.completed_at is not None

    def test_creates_entry_if_missing(self) -> None:
        attempt = _make_attempt()
        mark_stage_completed(attempt=attempt, stage_name="new-stage", outputs={})
        assert "new-stage" in attempt.stage_runs
        assert attempt.stage_runs["new-stage"].status == StageRunState.STATUS_COMPLETED

    def test_clears_prior_skip_reason(self) -> None:
        attempt = _make_attempt()
        mark_stage_skipped(attempt=attempt, stage_name="s1", reason="dry_run")
        mark_stage_completed(attempt=attempt, stage_name="s1", outputs={})
        assert attempt.stage_runs["s1"].skip_reason is None


class TestMarkStageFailed:
    def test_marks_stage_and_attempt_failed(self) -> None:
        attempt = _make_attempt()
        mark_stage_failed(
            attempt=attempt,
            stage_name="Training Monitor",
            error="connection lost",
            failure_kind="SSH_ERROR",
        )
        stage = attempt.stage_runs["Training Monitor"]
        assert stage.status == StageRunState.STATUS_FAILED
        assert stage.error == "connection lost"
        assert stage.failure_kind == "SSH_ERROR"
        # Attempt level propagation
        assert attempt.status == StageRunState.STATUS_FAILED
        assert attempt.error == "connection lost"
        assert attempt.completed_at is not None

    def test_stores_outputs_when_provided(self) -> None:
        attempt = _make_attempt()
        mark_stage_failed(
            attempt=attempt,
            stage_name="s1",
            error="boom",
            failure_kind="ERR",
            outputs={"partial": True},
        )
        assert attempt.stage_runs["s1"].outputs == {"partial": True}

    def test_outputs_defaults_to_empty_dict(self) -> None:
        attempt = _make_attempt()
        mark_stage_failed(attempt=attempt, stage_name="s1", error="e", failure_kind="F")
        assert attempt.stage_runs["s1"].outputs == {}


class TestMarkStageSkipped:
    def test_marks_skipped_with_reason(self) -> None:
        attempt = _make_attempt()
        mark_stage_skipped(attempt=attempt, stage_name="Model Evaluator", reason="eval_disabled")
        state = attempt.stage_runs["Model Evaluator"]
        assert state.status == StageRunState.STATUS_SKIPPED
        assert state.execution_mode == StageRunState.MODE_SKIPPED
        assert state.skip_reason == "eval_disabled"
        assert state.started_at is not None
        assert state.completed_at is not None

    def test_stores_outputs(self) -> None:
        attempt = _make_attempt()
        mark_stage_skipped(attempt=attempt, stage_name="s1", reason="r", outputs={"k": "v"})
        assert attempt.stage_runs["s1"].outputs == {"k": "v"}

    def test_outputs_defaults_to_empty(self) -> None:
        attempt = _make_attempt()
        mark_stage_skipped(attempt=attempt, stage_name="s1", reason="r")
        assert attempt.stage_runs["s1"].outputs == {}


class TestMarkStageInterrupted:
    def test_marks_interrupted(self) -> None:
        attempt = _make_attempt()
        mark_stage_interrupted(attempt=attempt, stage_name="Training Monitor", started_at="t0")
        state = attempt.stage_runs["Training Monitor"]
        assert state.status == StageRunState.STATUS_INTERRUPTED
        assert state.execution_mode == StageRunState.MODE_EXECUTED
        assert state.started_at == "t0"
        assert state.completed_at is not None


class TestFinalizeAttemptState:
    def test_sets_status_and_clears_active_attempt(self) -> None:
        attempt = _make_attempt(attempt_id="a1")
        state = _make_state(active_attempt_id="a1")
        finalize_attempt_state(state=state, attempt=attempt, status="completed")
        assert attempt.status == "completed"
        assert state.pipeline_status == "completed"
        assert state.active_attempt_id is None

    def test_does_not_clear_unrelated_active_attempt(self) -> None:
        attempt = _make_attempt(attempt_id="a1")
        state = _make_state(active_attempt_id="a2")
        finalize_attempt_state(state=state, attempt=attempt, status="failed")
        assert state.active_attempt_id == "a2"

    def test_uses_provided_completed_at(self) -> None:
        attempt = _make_attempt()
        state = _make_state()
        finalize_attempt_state(state=state, attempt=attempt, status="completed", completed_at="2026-01-01T12:00:00Z")
        assert attempt.completed_at == "2026-01-01T12:00:00Z"

    def test_generates_completed_at_if_not_provided(self) -> None:
        attempt = _make_attempt()
        state = _make_state()
        finalize_attempt_state(state=state, attempt=attempt, status="completed")
        assert attempt.completed_at is not None
