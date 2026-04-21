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

from src.pipeline.state import PipelineAttemptState, PipelineState, StageLineageRef, StageRunState
from src.pipeline.state.lineage_manager import invalidate_from as invalidate_lineage_from
from src.pipeline.state.lineage_manager import restore_reused as restore_reused_context
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


_PIPELINE_STAGES = ["s0", "s1", "s2", "s3"]


class TestInvalidateLineageFrom:
    def test_drops_entries_from_start_stage_onward(self) -> None:
        lineage = {
            name: StageLineageRef(attempt_id="a", stage_name=name, outputs={})
            for name in _PIPELINE_STAGES
        }
        new_lineage = invalidate_lineage_from(
            lineage=lineage, stage_names=_PIPELINE_STAGES, start_stage_name="s1"
        )
        assert set(new_lineage.keys()) == {"s0"}

    def test_preserves_input_unchanged(self) -> None:
        lineage = {"s0": StageLineageRef(attempt_id="a", stage_name="s0", outputs={})}
        new_lineage = invalidate_lineage_from(
            lineage=lineage, stage_names=_PIPELINE_STAGES, start_stage_name="s2"
        )
        assert lineage == {"s0": StageLineageRef(attempt_id="a", stage_name="s0", outputs={})}
        assert new_lineage is not lineage

    def test_drops_all_when_start_is_first_stage(self) -> None:
        lineage = {
            name: StageLineageRef(attempt_id="a", stage_name=name, outputs={})
            for name in _PIPELINE_STAGES
        }
        new_lineage = invalidate_lineage_from(
            lineage=lineage, stage_names=_PIPELINE_STAGES, start_stage_name="s0"
        )
        assert new_lineage == {}

    def test_unknown_start_stage_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown stage name"):
            invalidate_lineage_from(lineage={}, stage_names=_PIPELINE_STAGES, start_stage_name="missing")


class TestRestoreReusedContext:
    def test_copies_lineage_outputs_into_context(self) -> None:
        attempt = _make_attempt()
        lineage = {"s0": StageLineageRef(attempt_id="a0", stage_name="s0", outputs={"k": "v"})}
        context: dict = {}
        syncs: list[tuple[str, dict]] = []

        def sync(ctx: dict, name: str, outputs: dict) -> None:
            syncs.append((name, outputs))

        restore_reused_context(
            attempt=attempt,
            lineage=lineage,
            stage_names=_PIPELINE_STAGES,
            start_stage_name="s2",
            enabled_stage_names=_PIPELINE_STAGES,
            context=context,
            sync_root_from_stage=sync,
        )
        assert context["s0"] == {"k": "v"}
        assert syncs == [("s0", {"k": "v"})]
        s0 = attempt.stage_runs["s0"]
        assert s0.status == StageRunState.STATUS_COMPLETED
        assert s0.execution_mode == StageRunState.MODE_REUSED
        assert s0.reuse_from == {"attempt_id": "a0", "stage_name": "s0"}

    def test_disabled_stage_marked_skipped(self) -> None:
        attempt = _make_attempt()
        lineage = {"s0": StageLineageRef(attempt_id="a0", stage_name="s0", outputs={"k": "v"})}
        context: dict = {}
        restore_reused_context(
            attempt=attempt,
            lineage=lineage,
            stage_names=_PIPELINE_STAGES,
            start_stage_name="s2",
            enabled_stage_names=["s1", "s2"],  # s0 disabled
            context=context,
            sync_root_from_stage=lambda *_: None,
        )
        s0 = attempt.stage_runs["s0"]
        assert s0.status == StageRunState.STATUS_SKIPPED
        assert s0.skip_reason == "disabled_by_config"
        assert "s0" not in context  # context not populated for disabled stages

    def test_stops_at_start_stage(self) -> None:
        attempt = _make_attempt()
        lineage = {
            name: StageLineageRef(attempt_id="a", stage_name=name, outputs={"n": name})
            for name in _PIPELINE_STAGES
        }
        restore_reused_context(
            attempt=attempt,
            lineage=lineage,
            stage_names=_PIPELINE_STAGES,
            start_stage_name="s2",
            enabled_stage_names=_PIPELINE_STAGES,
            context={},
            sync_root_from_stage=lambda *_: None,
        )
        # Only s0 and s1 should be restored (s2 is start, s3 is after)
        assert set(attempt.stage_runs.keys()) == {"s0", "s1"}

    def test_missing_lineage_entry_skipped(self) -> None:
        attempt = _make_attempt()
        lineage = {"s1": StageLineageRef(attempt_id="a", stage_name="s1", outputs={})}
        restore_reused_context(
            attempt=attempt,
            lineage=lineage,
            stage_names=_PIPELINE_STAGES,
            start_stage_name="s3",
            enabled_stage_names=_PIPELINE_STAGES,
            context={},
            sync_root_from_stage=lambda *_: None,
        )
        assert "s0" not in attempt.stage_runs  # no lineage → no entry
        assert "s1" in attempt.stage_runs
        assert "s2" not in attempt.stage_runs

    def test_unknown_start_stage_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown stage name"):
            restore_reused_context(
                attempt=_make_attempt(),
                lineage={},
                stage_names=_PIPELINE_STAGES,
                start_stage_name="missing",
                enabled_stage_names=_PIPELINE_STAGES,
                context={},
                sync_root_from_stage=lambda *_: None,
            )
