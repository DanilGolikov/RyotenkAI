"""Comprehensive tests for :mod:`src.pipeline.state.attempt_controller`.

Coverage split by category (required by project policy):

1. **Positive**           — happy-path API calls mutate state and persist.
2. **Negative**           — invariant violations raise ``AttemptControllerError``.
3. **Boundary**           — empty strings, missing keys, None outputs, edge statuses.
4. **Invariants**         — every public mutator triggers exactly one save_fn call.
5. **Dependency errors**  — save_fn raises, lineage_manager edge cases.
6. **Regressions**        — snapshot() is deep-copy; rejected-attempt never emits
                             a transient RUNNING state; active_attempt_id cleared on finalize.
7. **Combinatorial**      — parametrised matrices over status/outputs/flags.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from src.pipeline.state.attempt_controller import (
    AttemptController,
    AttemptControllerError,
)
from src.pipeline.state.models import (
    PipelineAttemptState,
    PipelineState,
    StageLineageRef,
    StageRunState,
)


def _noop_sync(
    _ctx: dict[str, Any], _stage_name: str, _outputs: dict[str, Any]
) -> None:
    """No-op sync callback — matches lineage_manager.restore_reused contract."""


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_state(**overrides: Any) -> PipelineState:
    defaults: dict[str, Any] = {
        "schema_version": 1,
        "logical_run_id": "run_x",
        "run_directory": "/tmp/run_x",
        "config_path": "/tmp/cfg.yaml",
        "active_attempt_id": None,
        "pipeline_status": StageRunState.STATUS_PENDING,
        "training_critical_config_hash": "",
        "late_stage_config_hash": "",
        "model_dataset_config_hash": "",
    }
    defaults.update(overrides)
    return PipelineState(**defaults)


def _make_attempt(attempt_id: str = "attempt_1", attempt_no: int = 1) -> PipelineAttemptState:
    return PipelineAttemptState(
        attempt_id=attempt_id,
        attempt_no=attempt_no,
        runtime_name="runtime_1",
        requested_action="fresh",
        effective_action="fresh",
        restart_from_stage=None,
        status=StageRunState.STATUS_PENDING,
        started_at="2026-04-21T12:00:00+00:00",
    )


@pytest.fixture
def run_ctx() -> SimpleNamespace:
    return SimpleNamespace(name="run_x", run_id="rid_x")


@pytest.fixture
def save_fn() -> MagicMock:
    return MagicMock()


@pytest.fixture
def controller(save_fn: MagicMock, run_ctx: SimpleNamespace) -> AttemptController:
    return AttemptController(save_fn=save_fn, run_ctx=run_ctx)


@pytest.fixture
def primed_controller(
    save_fn: MagicMock, run_ctx: SimpleNamespace
) -> tuple[AttemptController, PipelineState, PipelineAttemptState]:
    """Controller with adopted state and a registered active attempt.

    Callers that want to count persist invocations AFTER priming must reset
    the mock themselves — priming produces 1 save_fn call (``register_attempt``).
    """
    state = _make_state()
    attempt = _make_attempt()
    ctrl = AttemptController(save_fn=save_fn, run_ctx=run_ctx)
    ctrl.adopt_state(state)
    ctrl.register_attempt(attempt)
    return ctrl, state, attempt


# ===========================================================================
# 1. POSITIVE
# ===========================================================================


class TestPositive:
    def test_adopt_state_is_no_persist(
        self, controller: AttemptController, save_fn: MagicMock
    ) -> None:
        state = _make_state()
        controller.adopt_state(state)
        assert controller.has_state
        assert controller.state is state
        save_fn.assert_not_called()  # adoption does NOT persist

    def test_register_attempt_appends_and_activates(
        self, controller: AttemptController, save_fn: MagicMock
    ) -> None:
        state = _make_state()
        controller.adopt_state(state)
        attempt = _make_attempt()
        controller.register_attempt(attempt)

        assert state.attempts == [attempt]
        assert state.active_attempt_id == attempt.attempt_id
        assert state.pipeline_status == StageRunState.STATUS_RUNNING
        assert controller.has_active_attempt
        assert controller.active_attempt is attempt
        save_fn.assert_called_once_with(state)

    def test_record_running(
        self,
        primed_controller: tuple[AttemptController, PipelineState, PipelineAttemptState],
        save_fn: MagicMock,
    ) -> None:
        ctrl, state, attempt = primed_controller
        save_fn.reset_mock()
        ctrl.record_running(stage_name="Stage A", started_at="2026-04-21T12:05:00+00:00")

        stage_state = attempt.stage_runs["Stage A"]
        assert stage_state.status == StageRunState.STATUS_RUNNING
        assert stage_state.started_at == "2026-04-21T12:05:00+00:00"
        save_fn.assert_called_once_with(state)

    def test_record_completed_updates_state_and_lineage(
        self,
        primed_controller: tuple[AttemptController, PipelineState, PipelineAttemptState],
        save_fn: MagicMock,
    ) -> None:
        ctrl, state, attempt = primed_controller
        save_fn.reset_mock()

        ctrl.record_completed(stage_name="Stage A", outputs={"k": "v"})

        stage_state = attempt.stage_runs["Stage A"]
        assert stage_state.status == StageRunState.STATUS_COMPLETED
        assert stage_state.outputs == {"k": "v"}
        assert "Stage A" in state.current_output_lineage
        assert state.current_output_lineage["Stage A"].outputs == {"k": "v"}
        save_fn.assert_called_once_with(state)

    def test_record_failed_drops_lineage(
        self,
        primed_controller: tuple[AttemptController, PipelineState, PipelineAttemptState],
    ) -> None:
        ctrl, state, attempt = primed_controller
        # Pre-seed lineage to verify drop semantics
        state.current_output_lineage["Stage A"] = StageLineageRef(
            attempt_id=attempt.attempt_id, stage_name="Stage A", outputs={"old": "value"}
        )

        ctrl.record_failed(
            stage_name="Stage A",
            error="boom",
            failure_kind="SOME_CODE",
        )

        assert attempt.stage_runs["Stage A"].status == StageRunState.STATUS_FAILED
        assert attempt.stage_runs["Stage A"].error == "boom"
        assert attempt.stage_runs["Stage A"].failure_kind == "SOME_CODE"
        assert attempt.status == StageRunState.STATUS_FAILED
        assert "Stage A" not in state.current_output_lineage

    def test_record_skipped_drops_lineage(
        self,
        primed_controller: tuple[AttemptController, PipelineState, PipelineAttemptState],
    ) -> None:
        ctrl, state, attempt = primed_controller
        state.current_output_lineage["Stage A"] = StageLineageRef(
            attempt_id="old", stage_name="Stage A", outputs={"v": 1}
        )

        ctrl.record_skipped(stage_name="Stage A", reason="disabled_by_config")

        assert attempt.stage_runs["Stage A"].status == StageRunState.STATUS_SKIPPED
        assert attempt.stage_runs["Stage A"].skip_reason == "disabled_by_config"
        assert "Stage A" not in state.current_output_lineage

    def test_record_interrupted(
        self,
        primed_controller: tuple[AttemptController, PipelineState, PipelineAttemptState],
    ) -> None:
        ctrl, _, attempt = primed_controller
        ctrl.record_interrupted(stage_name="Stage A", started_at="2026-04-21T13:00:00+00:00")
        stage_state = attempt.stage_runs["Stage A"]
        assert stage_state.status == StageRunState.STATUS_INTERRUPTED
        assert stage_state.started_at == "2026-04-21T13:00:00+00:00"

    def test_finalize_clears_active_attempt(
        self,
        primed_controller: tuple[AttemptController, PipelineState, PipelineAttemptState],
    ) -> None:
        ctrl, state, attempt = primed_controller
        ctrl.finalize(status=StageRunState.STATUS_COMPLETED, completed_at="2026-04-21T14:00:00+00:00")

        assert state.pipeline_status == StageRunState.STATUS_COMPLETED
        assert state.active_attempt_id is None
        assert attempt.status == StageRunState.STATUS_COMPLETED
        assert attempt.completed_at == "2026-04-21T14:00:00+00:00"

    def test_record_rejected_attempt_is_atomic(
        self, controller: AttemptController, save_fn: MagicMock
    ) -> None:
        state = _make_state()
        controller.adopt_state(state)
        attempt = _make_attempt()
        controller.record_rejected_attempt(
            attempt=attempt,
            status=StageRunState.STATUS_FAILED,
            completed_at="2026-04-21T12:30:00+00:00",
        )
        assert state.attempts == [attempt]
        assert state.pipeline_status == StageRunState.STATUS_FAILED
        assert state.active_attempt_id is None  # never activated
        assert attempt.status == StageRunState.STATUS_FAILED
        assert attempt.completed_at == "2026-04-21T12:30:00+00:00"
        # Exactly ONE write — no transient RUNNING snapshot
        save_fn.assert_called_once_with(state)

    def test_invalidate_lineage_from(
        self,
        primed_controller: tuple[AttemptController, PipelineState, PipelineAttemptState],
    ) -> None:
        ctrl, state, attempt = primed_controller
        state.current_output_lineage = {
            "S1": StageLineageRef(attempt_id=attempt.attempt_id, stage_name="S1", outputs={}),
            "S2": StageLineageRef(attempt_id=attempt.attempt_id, stage_name="S2", outputs={}),
            "S3": StageLineageRef(attempt_id=attempt.attempt_id, stage_name="S3", outputs={}),
        }
        new_lineage = ctrl.invalidate_lineage_from(
            stage_names=["S1", "S2", "S3"], start_stage_name="S2"
        )
        assert set(new_lineage.keys()) == {"S1"}
        assert new_lineage is state.current_output_lineage

    def test_set_mlflow_run_ids(
        self,
        primed_controller: tuple[AttemptController, PipelineState, PipelineAttemptState],
    ) -> None:
        ctrl, state, attempt = primed_controller
        ctrl.set_mlflow_run_ids(root_run_id="root_123", attempt_run_id="att_456")
        assert state.root_mlflow_run_id == "root_123"
        assert attempt.root_mlflow_run_id == "root_123"
        assert attempt.pipeline_attempt_mlflow_run_id == "att_456"

    def test_save_explicit(
        self,
        primed_controller: tuple[AttemptController, PipelineState, PipelineAttemptState],
        save_fn: MagicMock,
    ) -> None:
        ctrl, state, _ = primed_controller
        save_fn.reset_mock()
        ctrl.save()
        save_fn.assert_called_once_with(state)


# ===========================================================================
# 2. NEGATIVE
# ===========================================================================


class TestNegative:
    def test_register_attempt_without_state_raises(
        self, controller: AttemptController
    ) -> None:
        with pytest.raises(AttemptControllerError, match="state has not been adopted"):
            controller.register_attempt(_make_attempt())

    def test_record_running_without_attempt_raises(
        self, controller: AttemptController
    ) -> None:
        controller.adopt_state(_make_state())
        with pytest.raises(AttemptControllerError, match="no active attempt"):
            controller.record_running(stage_name="S", started_at="t")

    def test_record_completed_without_attempt_raises(
        self, controller: AttemptController
    ) -> None:
        controller.adopt_state(_make_state())
        with pytest.raises(AttemptControllerError, match="no active attempt"):
            controller.record_completed(stage_name="S", outputs={})

    def test_record_failed_without_state_raises(
        self, controller: AttemptController
    ) -> None:
        with pytest.raises(AttemptControllerError, match="state has not been adopted"):
            controller.record_failed(stage_name="S", error="x", failure_kind="Y")

    def test_record_skipped_without_state_raises(
        self, controller: AttemptController
    ) -> None:
        with pytest.raises(AttemptControllerError, match="state has not been adopted"):
            controller.record_skipped(stage_name="S", reason="x")

    def test_record_interrupted_without_attempt_raises(
        self, controller: AttemptController
    ) -> None:
        controller.adopt_state(_make_state())
        with pytest.raises(AttemptControllerError, match="no active attempt"):
            controller.record_interrupted(stage_name="S", started_at="t")

    def test_finalize_without_state_raises(
        self, controller: AttemptController
    ) -> None:
        with pytest.raises(AttemptControllerError, match="state has not been adopted"):
            controller.finalize(status=StageRunState.STATUS_COMPLETED)

    def test_invalidate_lineage_without_state_raises(
        self, controller: AttemptController
    ) -> None:
        with pytest.raises(AttemptControllerError, match="state has not been adopted"):
            controller.invalidate_lineage_from(stage_names=["A"], start_stage_name="A")

    def test_restore_reused_without_attempt_raises(
        self, controller: AttemptController
    ) -> None:
        controller.adopt_state(_make_state())
        with pytest.raises(AttemptControllerError, match="no active attempt"):
            controller.restore_reused_context(
                stage_names=["A"],
                start_stage_name="A",
                enabled_stage_names=["A"],
                context={},
                sync_root_from_stage=_noop_sync,
            )

    def test_record_rejected_without_state_raises(
        self, controller: AttemptController
    ) -> None:
        with pytest.raises(AttemptControllerError, match="state has not been adopted"):
            controller.record_rejected_attempt(
                attempt=_make_attempt(),
                status=StageRunState.STATUS_FAILED,
                completed_at="t",
            )

    def test_active_attempt_property_without_attempt_raises(
        self, controller: AttemptController
    ) -> None:
        controller.adopt_state(_make_state())
        with pytest.raises(AttemptControllerError):
            _ = controller.active_attempt

    def test_state_property_without_adoption_raises(
        self, controller: AttemptController
    ) -> None:
        with pytest.raises(AttemptControllerError):
            _ = controller.state


# ===========================================================================
# 3. BOUNDARY
# ===========================================================================


class TestBoundary:
    def test_empty_outputs(
        self,
        primed_controller: tuple[AttemptController, PipelineState, PipelineAttemptState],
    ) -> None:
        ctrl, state, _ = primed_controller
        ctrl.record_completed(stage_name="Stage A", outputs={})
        assert state.current_output_lineage["Stage A"].outputs == {}

    def test_none_outputs_in_failed(
        self,
        primed_controller: tuple[AttemptController, PipelineState, PipelineAttemptState],
    ) -> None:
        ctrl, _, attempt = primed_controller
        ctrl.record_failed(stage_name="S", error="e", failure_kind="C", outputs=None)
        assert attempt.stage_runs["S"].outputs == {}

    def test_none_outputs_in_skipped(
        self,
        primed_controller: tuple[AttemptController, PipelineState, PipelineAttemptState],
    ) -> None:
        ctrl, _, attempt = primed_controller
        ctrl.record_skipped(stage_name="S", reason="r", outputs=None)
        assert attempt.stage_runs["S"].outputs == {}

    def test_finalize_without_completed_at_uses_default(
        self,
        primed_controller: tuple[AttemptController, PipelineState, PipelineAttemptState],
    ) -> None:
        ctrl, _, attempt = primed_controller
        ctrl.finalize(status=StageRunState.STATUS_COMPLETED)
        assert attempt.completed_at is not None  # transitioner fills a default

    def test_invalidate_empty_lineage_is_noop(
        self,
        primed_controller: tuple[AttemptController, PipelineState, PipelineAttemptState],
    ) -> None:
        ctrl, state, _ = primed_controller
        state.current_output_lineage = {}
        new_lineage = ctrl.invalidate_lineage_from(
            stage_names=["S1", "S2"], start_stage_name="S1"
        )
        assert new_lineage == {}

    def test_invalidate_from_unknown_stage_raises(
        self,
        primed_controller: tuple[AttemptController, PipelineState, PipelineAttemptState],
    ) -> None:
        ctrl, _, _ = primed_controller
        with pytest.raises(ValueError, match="Unknown stage name"):
            ctrl.invalidate_lineage_from(stage_names=["S1"], start_stage_name="MISSING")

    def test_set_mlflow_run_ids_all_none_is_noop(
        self,
        primed_controller: tuple[AttemptController, PipelineState, PipelineAttemptState],
        save_fn: MagicMock,
    ) -> None:
        ctrl, state, _ = primed_controller
        save_fn.reset_mock()
        root_before = state.root_mlflow_run_id
        ctrl.set_mlflow_run_ids()  # no args
        assert state.root_mlflow_run_id == root_before
        save_fn.assert_called_once()  # still persists even though nothing changed

    def test_record_stage_log_paths_missing_stage_is_noop(
        self,
        primed_controller: tuple[AttemptController, PipelineState, PipelineAttemptState],
        save_fn: MagicMock,
    ) -> None:
        ctrl, _, _ = primed_controller
        save_fn.reset_mock()
        ctrl.record_stage_log_paths(stage_name="nonexistent", log_paths={"k": "v"})
        save_fn.assert_not_called()

    def test_mark_attempt_completed_at_without_attempt_is_noop(
        self, controller: AttemptController
    ) -> None:
        # No raise, just no-op
        controller.mark_attempt_completed_at(completed_at="t")

    def test_record_attempt_error_without_attempt_is_noop(
        self, controller: AttemptController
    ) -> None:
        controller.record_attempt_error(error="x")  # no raise


# ===========================================================================
# 4. INVARIANTS
# ===========================================================================


class TestInvariants:
    """Structural guarantees the controller MUST uphold.

    The most important one: every public mutator produces a persist — this is
    what makes "every mutation is durable" a type-level property, not a
    discipline-level one. Every violation below would be a silent data-loss
    bug, so we lock each mutator down with a dedicated assertion.
    """

    def test_register_attempt_persists_exactly_once(
        self, controller: AttemptController, save_fn: MagicMock
    ) -> None:
        controller.adopt_state(_make_state())
        controller.register_attempt(_make_attempt())
        assert save_fn.call_count == 1

    def test_record_running_persists_exactly_once(
        self,
        primed_controller: tuple[AttemptController, PipelineState, PipelineAttemptState],
        save_fn: MagicMock,
    ) -> None:
        ctrl, _, _ = primed_controller
        save_fn.reset_mock()
        ctrl.record_running(stage_name="S", started_at="t")
        assert save_fn.call_count == 1

    def test_record_completed_persists_exactly_once(
        self,
        primed_controller: tuple[AttemptController, PipelineState, PipelineAttemptState],
        save_fn: MagicMock,
    ) -> None:
        ctrl, _, _ = primed_controller
        save_fn.reset_mock()
        ctrl.record_completed(stage_name="S", outputs={})
        assert save_fn.call_count == 1

    def test_record_failed_persists_exactly_once(
        self,
        primed_controller: tuple[AttemptController, PipelineState, PipelineAttemptState],
        save_fn: MagicMock,
    ) -> None:
        ctrl, _, _ = primed_controller
        save_fn.reset_mock()
        ctrl.record_failed(stage_name="S", error="x", failure_kind="Y")
        assert save_fn.call_count == 1

    def test_record_skipped_persists_exactly_once(
        self,
        primed_controller: tuple[AttemptController, PipelineState, PipelineAttemptState],
        save_fn: MagicMock,
    ) -> None:
        ctrl, _, _ = primed_controller
        save_fn.reset_mock()
        ctrl.record_skipped(stage_name="S", reason="r")
        assert save_fn.call_count == 1

    def test_record_interrupted_persists_exactly_once(
        self,
        primed_controller: tuple[AttemptController, PipelineState, PipelineAttemptState],
        save_fn: MagicMock,
    ) -> None:
        ctrl, _, _ = primed_controller
        save_fn.reset_mock()
        ctrl.record_interrupted(stage_name="S", started_at="t")
        assert save_fn.call_count == 1

    def test_finalize_persists_exactly_once(
        self,
        primed_controller: tuple[AttemptController, PipelineState, PipelineAttemptState],
        save_fn: MagicMock,
    ) -> None:
        ctrl, _, _ = primed_controller
        save_fn.reset_mock()
        ctrl.finalize(status=StageRunState.STATUS_COMPLETED)
        assert save_fn.call_count == 1

    def test_record_rejected_persists_exactly_once(
        self, controller: AttemptController, save_fn: MagicMock
    ) -> None:
        controller.adopt_state(_make_state())
        controller.record_rejected_attempt(
            attempt=_make_attempt(),
            status=StageRunState.STATUS_FAILED,
            completed_at="t",
        )
        assert save_fn.call_count == 1

    def test_invalidate_lineage_persists_exactly_once(
        self,
        primed_controller: tuple[AttemptController, PipelineState, PipelineAttemptState],
        save_fn: MagicMock,
    ) -> None:
        ctrl, _, _ = primed_controller
        save_fn.reset_mock()
        ctrl.invalidate_lineage_from(stage_names=["S"], start_stage_name="S")
        assert save_fn.call_count == 1

    def test_mark_completed_at_does_not_persist(
        self,
        primed_controller: tuple[AttemptController, PipelineState, PipelineAttemptState],
        save_fn: MagicMock,
    ) -> None:
        """Dedicated no-persist primitive so "stamp then finalize" is 1 write, not 2."""
        ctrl, _, _ = primed_controller
        save_fn.reset_mock()
        ctrl.mark_attempt_completed_at(completed_at="t")
        save_fn.assert_not_called()

    def test_record_attempt_error_does_not_persist(
        self,
        primed_controller: tuple[AttemptController, PipelineState, PipelineAttemptState],
        save_fn: MagicMock,
    ) -> None:
        ctrl, _, _ = primed_controller
        save_fn.reset_mock()
        ctrl.record_attempt_error(error="x")
        save_fn.assert_not_called()

    def test_adopt_state_does_not_persist(
        self, controller: AttemptController, save_fn: MagicMock
    ) -> None:
        controller.adopt_state(_make_state())
        save_fn.assert_not_called()

    def test_single_active_attempt_throughout_lifecycle(
        self,
        primed_controller: tuple[AttemptController, PipelineState, PipelineAttemptState],
    ) -> None:
        ctrl, _, attempt_1 = primed_controller
        attempt_1_id = attempt_1.attempt_id
        ctrl.record_running(stage_name="S1", started_at="t1")
        ctrl.record_completed(stage_name="S1", outputs={})
        assert ctrl.active_attempt_id() == attempt_1_id


# ===========================================================================
# 5. DEPENDENCY ERRORS
# ===========================================================================


class TestDependencyErrors:
    def test_save_fn_exception_propagates(
        self, run_ctx: SimpleNamespace
    ) -> None:
        def failing_save(_state: PipelineState) -> None:
            raise OSError("disk full")

        ctrl = AttemptController(save_fn=failing_save, run_ctx=run_ctx)
        ctrl.adopt_state(_make_state())

        with pytest.raises(OSError, match="disk full"):
            ctrl.register_attempt(_make_attempt())

    def test_save_fn_exception_on_finalize_propagates(
        self, run_ctx: SimpleNamespace
    ) -> None:
        call_count = {"n": 0}

        def flaky_save(_state: PipelineState) -> None:
            call_count["n"] += 1
            if call_count["n"] == 2:
                raise RuntimeError("mid-run failure")

        ctrl = AttemptController(save_fn=flaky_save, run_ctx=run_ctx)
        ctrl.adopt_state(_make_state())
        ctrl.register_attempt(_make_attempt())  # 1st save: success

        with pytest.raises(RuntimeError, match="mid-run failure"):
            ctrl.finalize(status=StageRunState.STATUS_COMPLETED)  # 2nd save: fail

    def test_invalidate_unknown_stage_propagates_value_error(
        self,
        primed_controller: tuple[AttemptController, PipelineState, PipelineAttemptState],
    ) -> None:
        ctrl, _, _ = primed_controller
        with pytest.raises(ValueError):
            ctrl.invalidate_lineage_from(stage_names=["A"], start_stage_name="MISSING")

    def test_restore_reused_unknown_stage_propagates_value_error(
        self,
        primed_controller: tuple[AttemptController, PipelineState, PipelineAttemptState],
    ) -> None:
        ctrl, _, _ = primed_controller
        with pytest.raises(ValueError):
            ctrl.restore_reused_context(
                stage_names=["A"],
                start_stage_name="MISSING",
                enabled_stage_names=["A"],
                context={},
                sync_root_from_stage=_noop_sync,
            )


# ===========================================================================
# 6. REGRESSIONS (specific bugs we are locking down)
# ===========================================================================


class TestRegressions:
    def test_snapshot_is_deep_copy(
        self,
        primed_controller: tuple[AttemptController, PipelineState, PipelineAttemptState],
    ) -> None:
        """Red flag #3: snapshot() returning a live ref collapses invariants."""
        ctrl, state, attempt = primed_controller
        state.current_output_lineage["X"] = StageLineageRef(
            attempt_id=attempt.attempt_id, stage_name="X", outputs={"k": [1, 2, 3]}
        )

        snap = ctrl.snapshot()

        # Snapshot is independent — mutating it must NOT leak back.
        snap.current_output_lineage["X"].outputs["k"].append(999)
        assert state.current_output_lineage["X"].outputs["k"] == [1, 2, 3]

        snap.attempts[0].status = "MUTATED"
        assert state.attempts[0].status != "MUTATED"

    def test_rejected_attempt_never_writes_running_status(
        self, controller: AttemptController, save_fn: MagicMock
    ) -> None:
        """Regression: record_rejected_attempt must not emit a transient
        RUNNING snapshot — on crash recovery we could otherwise see a "live"
        attempt that actually failed to launch."""
        state = _make_state()
        controller.adopt_state(state)

        observed_statuses: list[str] = []

        def spy(s: PipelineState) -> None:
            observed_statuses.append(s.pipeline_status)

        controller._save_fn = spy  # type: ignore[assignment]  # redirect save capture

        controller.record_rejected_attempt(
            attempt=_make_attempt(),
            status=StageRunState.STATUS_FAILED,
            completed_at="t",
        )
        # Exactly one persist, already at FAILED — never RUNNING.
        assert observed_statuses == [StageRunState.STATUS_FAILED]
        _ = save_fn  # silence unused-fixture linter

    def test_finalize_with_orphan_active_attempt_id_leaves_others_alone(
        self,
        primed_controller: tuple[AttemptController, PipelineState, PipelineAttemptState],
    ) -> None:
        """If active_attempt_id does not match any appended attempt, finalize
        still writes a coherent pipeline_status."""
        ctrl, state, attempt = primed_controller
        state.active_attempt_id = "OTHER"  # simulated corruption
        ctrl.finalize(status=StageRunState.STATUS_FAILED)
        assert state.pipeline_status == StageRunState.STATUS_FAILED
        # attempt.status was still updated because the controller knows _its_
        # active attempt regardless of what's on state.active_attempt_id.
        assert attempt.status == StageRunState.STATUS_FAILED

    def test_adopt_second_state_replaces_but_not_reset_attempt(
        self, controller: AttemptController
    ) -> None:
        """If adopt_state is called twice (edge, but possible), active attempt
        is NOT silently nullified — caller must explicitly re-register."""
        state1 = _make_state(logical_run_id="run_1")
        controller.adopt_state(state1)
        attempt = _make_attempt()
        controller.register_attempt(attempt)

        state2 = _make_state(logical_run_id="run_2")
        controller.adopt_state(state2)
        # active_attempt still refers to old handle — controller does not
        # auto-reset. This is intentional: it would hide a caller bug.
        assert controller.has_active_attempt

    def test_has_state_property_reflects_adoption(
        self, controller: AttemptController
    ) -> None:
        assert not controller.has_state
        controller.adopt_state(_make_state())
        assert controller.has_state

    def test_has_active_attempt_reflects_registration(
        self, controller: AttemptController
    ) -> None:
        controller.adopt_state(_make_state())
        assert not controller.has_active_attempt
        controller.register_attempt(_make_attempt())
        assert controller.has_active_attempt

    def test_explicit_save_matches_state_identity(
        self,
        primed_controller: tuple[AttemptController, PipelineState, PipelineAttemptState],
        save_fn: MagicMock,
    ) -> None:
        """Regression: save() must write the controller's live state, not a snapshot."""
        ctrl, state, _ = primed_controller
        save_fn.reset_mock()
        ctrl.save()
        passed = save_fn.call_args.args[0]
        assert passed is state  # same object ref

    def test_record_completed_after_restart_uses_new_attempt_id(
        self, controller: AttemptController
    ) -> None:
        """Regression: after a restart, lineage entries must carry the new attempt_id."""
        controller.adopt_state(_make_state())
        controller.register_attempt(_make_attempt(attempt_id="a1"))
        controller.record_completed(stage_name="S", outputs={"v": 1})
        ref = controller.state.current_output_lineage["S"]
        assert ref.attempt_id == "a1"


# ===========================================================================
# 7. COMBINATORIAL
# ===========================================================================


@pytest.mark.parametrize("stage_name", ["S1", "Stage 2", "stage-3", ""])
@pytest.mark.parametrize(
    "outputs",
    [{}, {"k": "v"}, {"nested": {"deep": [1, 2, 3]}}, {"k": None}],
)
def test_record_completed_matrix(
    stage_name: str,
    outputs: dict[str, Any],
    primed_controller: tuple[AttemptController, PipelineState, PipelineAttemptState],
) -> None:
    ctrl, state, _ = primed_controller
    ctrl.record_completed(stage_name=stage_name, outputs=outputs)
    assert stage_name in state.current_output_lineage
    assert state.current_output_lineage[stage_name].outputs == outputs


@pytest.mark.parametrize(
    "status",
    [
        StageRunState.STATUS_COMPLETED,
        StageRunState.STATUS_FAILED,
        StageRunState.STATUS_INTERRUPTED,
    ],
)
@pytest.mark.parametrize("with_completed_at", [True, False])
def test_finalize_matrix(
    status: str,
    with_completed_at: bool,
    primed_controller: tuple[AttemptController, PipelineState, PipelineAttemptState],
) -> None:
    ctrl, state, _ = primed_controller
    ts = "2026-04-21T99:99:99+00:00" if with_completed_at else None
    ctrl.finalize(status=status, completed_at=ts)
    assert state.pipeline_status == status


@pytest.mark.parametrize("n_stages_before_restart", [0, 1, 2, 3])
def test_invalidate_lineage_matrix(
    n_stages_before_restart: int,
    primed_controller: tuple[AttemptController, PipelineState, PipelineAttemptState],
) -> None:
    ctrl, state, attempt = primed_controller
    stage_names = [f"S{i}" for i in range(4)]
    for name in stage_names:
        state.current_output_lineage[name] = StageLineageRef(
            attempt_id=attempt.attempt_id, stage_name=name, outputs={}
        )

    start = stage_names[n_stages_before_restart]
    ctrl.invalidate_lineage_from(stage_names=stage_names, start_stage_name=start)
    remaining = set(state.current_output_lineage.keys())
    expected = set(stage_names[:n_stages_before_restart])
    assert remaining == expected


@pytest.mark.parametrize(
    ("root_id", "attempt_id"),
    [
        (None, None),
        ("r1", None),
        (None, "a1"),
        ("r1", "a1"),
    ],
)
def test_set_mlflow_run_ids_matrix(
    root_id: str | None,
    attempt_id: str | None,
    primed_controller: tuple[AttemptController, PipelineState, PipelineAttemptState],
) -> None:
    ctrl, state, attempt = primed_controller
    ctrl.set_mlflow_run_ids(root_run_id=root_id, attempt_run_id=attempt_id)
    if root_id is not None:
        assert state.root_mlflow_run_id == root_id
        assert attempt.root_mlflow_run_id == root_id
    if attempt_id is not None:
        assert attempt.pipeline_attempt_mlflow_run_id == attempt_id
