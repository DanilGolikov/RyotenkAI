"""Comprehensive tests for :mod:`stage_execution_loop` (Phase A2 Batch 6).

Phase A2 Batch 6 migrated the loop from ``Result[T, AppError]`` to
raise-based: ``run_attempt`` returns :class:`PipelineContext` on success
and raises :class:`RyotenkAIError` on failure. Tests use ``pytest.raises``
+ assert the typed exception's ``code`` / ``detail`` / ``context``.

Coverage split by category (project policy — 7-class):

1. **Positive**          — happy-path success; per-stage lifecycle; skipped-config.
2. **Negative**          — stage raise; prereq error; unexpected exception; state I/O.
3. **Boundary**          — empty stage list; start_idx at last; stop_idx mid-way.
4. **Invariants**        — controller is the only writer; summary runs once.
5. **Dependency errors** — MLflow None; collector already flushed; state_store fails.
6. **Regressions**       — SystemExit re-raised; hooks called; log_layout used.
7. **Combinatorial**     — skip_reason × stage_name × collector presence matrix.
"""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

import pytest

from ryotenkai_control.pipeline.execution import StageExecutionLoop
from ryotenkai_control.pipeline.launch import PreparedAttempt
from ryotenkai_control.pipeline.stages.constants import StageNames
from ryotenkai_control.pipeline.state import (
    AttemptController,
    PipelineAttemptState,
    PipelineState,
    PipelineStateError,
    PipelineStateStore,
    StageRunState,
    build_attempt_state,
)
from ryotenkai_shared.errors import (
    InternalError,
    PipelineStageFailedError,
    RyotenkAIError,
)
from ryotenkai_shared.utils.logs_layout import LogLayout

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_stage(name: str, *, output: dict[str, Any] | None = None, raises: RyotenkAIError | None = None) -> MagicMock:
    """Build a MagicMock stage. ``raises`` overrides ``output``."""
    stage = MagicMock()
    stage.stage_name = name
    if raises is not None:
        stage.run.side_effect = raises
    else:
        stage.run.return_value = output if output is not None else {"ran": True}
    return stage


STAGE_A = StageNames.DATASET_VALIDATOR
STAGE_B = StageNames.GPU_DEPLOYER
STAGE_C = StageNames.TRAINING_MONITOR


def _make_state(tmp_path: Path) -> PipelineState:
    store = PipelineStateStore(tmp_path)
    return store.init_state(
        logical_run_id="run_x",
        config_path=str(tmp_path / "cfg.yaml"),
        training_critical_config_hash="t",
        late_stage_config_hash="l",
    )


def _make_controller(
    state: PipelineState,
    save_fn: MagicMock | None = None,
) -> AttemptController:
    save_fn = save_fn or MagicMock()
    ctrl = AttemptController(
        save_fn=save_fn, run_ctx=SimpleNamespace(name="run_x", run_id="rid")
    )
    ctrl.adopt_state(state)
    attempt = build_attempt_state(
        state=state,
        run_ctx=SimpleNamespace(name="run_x", run_id="rid"),
        requested_action="fresh",
        effective_action="fresh",
        restart_from_stage=STAGE_A,
        enabled_stage_names=[STAGE_A, STAGE_B, STAGE_C],
        training_critical_config_hash="t",
        late_stage_config_hash="l",
        model_dataset_config_hash="",
    )
    ctrl.register_attempt(attempt)
    return ctrl


def _make_prepared(tmp_path: Path, stages: list[MagicMock]) -> PreparedAttempt:
    state = _make_state(tmp_path)
    attempt_dir = tmp_path / "attempt_1"
    attempt_dir.mkdir(parents=True, exist_ok=True)
    attempt = PipelineAttemptState(
        attempt_id="a1",
        attempt_no=1,
        runtime_name="rt",
        requested_action="fresh",
        effective_action="fresh",
        restart_from_stage=stages[0].stage_name,
        status=StageRunState.STATUS_PENDING,
        started_at="2026-04-21T00:00:00+00:00",
    )
    return PreparedAttempt(
        state=state,
        attempt=attempt,
        state_store=PipelineStateStore(tmp_path),
        run_directory=tmp_path,
        attempt_directory=attempt_dir,
        log_layout=LogLayout(attempt_dir),
        logical_run_id="run_x",
        requested_action="fresh",
        effective_action="fresh",
        start_stage_name=stages[0].stage_name,
        start_idx=0,
        stop_idx=len(stages),
        enabled_stage_names=tuple(s.stage_name for s in stages),
        forced_stage_names=frozenset(),
    )


def _make_stage_planner() -> MagicMock:
    p = MagicMock()
    p.validate_stage_prerequisites.return_value = None
    return p


def _make_context_propagator() -> MagicMock:
    p = MagicMock()
    p.extract_restart_outputs.return_value = {}
    p.get_stage_skip_reason.return_value = None
    p.fill_collector_from_context.return_value = None
    return p


def _build_loop(
    tmp_path: Path,
    *,
    stages: list[MagicMock] | None = None,
    state: PipelineState | None = None,
    save_fn: MagicMock | None = None,
    planner: MagicMock | None = None,
    propagator: MagicMock | None = None,
    on_stage_completed: MagicMock | None = None,
    on_shutdown_signal: MagicMock | None = None,
) -> tuple[
    StageExecutionLoop,
    AttemptController,
    list[MagicMock],
    dict[str, MagicMock],
    MagicMock,
]:
    stages = stages or [_make_stage(STAGE_A), _make_stage(STAGE_B)]
    state = state or _make_state(tmp_path)
    controller = _make_controller(state, save_fn=save_fn)
    collectors = {s.stage_name: MagicMock(is_flushed=False, _started_at=None) for s in stages}
    for c in collectors.values():
        c.set_started_at = MagicMock()
        c.flush_ok = MagicMock()
        c.flush_error = MagicMock()
    stage_info_logger = MagicMock()
    validation_artifact_mgr = MagicMock()
    validation_artifact_mgr.build_dataset_validation_state_outputs.return_value = {}
    summary_reporter = MagicMock()

    loop = StageExecutionLoop(
        stages=stages,
        collectors=collectors,
        attempt_controller=controller,
        stage_planner=planner or _make_stage_planner(),
        context_propagator=propagator or _make_context_propagator(),
        stage_info_logger=stage_info_logger,
        validation_artifact_mgr=validation_artifact_mgr,
        summary_reporter=summary_reporter,
        on_stage_completed=on_stage_completed,
        on_shutdown_signal=on_shutdown_signal,
    )
    return loop, controller, stages, collectors, summary_reporter


def _mk_context() -> dict[str, Any]:
    return {"initial": True}


# ===========================================================================
# 1. POSITIVE
# ===========================================================================


class TestPositive:
    def test_all_stages_succeed(self, tmp_path: Path) -> None:
        loop, controller, stages, _, summary_reporter = _build_loop(tmp_path)
        prepared = _make_prepared(tmp_path, stages)
        context = _mk_context()

        out = loop.run_attempt(
            prepared=prepared,
            context=context,
            log_layout=prepared.log_layout,
        )
        assert out is context
        for s in stages:
            s.run.assert_called_once()
        assert controller.state.pipeline_status == StageRunState.STATUS_COMPLETED
        summary_reporter.print_summary.assert_called_once()

    def test_stage_result_merged_into_context(self, tmp_path: Path) -> None:
        stage = _make_stage(STAGE_A, output={"key_from_stage": 42})
        loop, _, _, _, _ = _build_loop(tmp_path, stages=[stage])
        prepared = _make_prepared(tmp_path, [stage])
        context = _mk_context()

        loop.run_attempt(
            prepared=prepared, context=context,
            log_layout=prepared.log_layout,
        )
        assert context["key_from_stage"] == 42

    def test_on_stage_completed_hook_fires(self, tmp_path: Path) -> None:
        hook = MagicMock()
        loop, _, stages, _, _ = _build_loop(tmp_path, on_stage_completed=hook)
        prepared = _make_prepared(tmp_path, stages)

        loop.run_attempt(
            prepared=prepared, context=_mk_context(),
            log_layout=prepared.log_layout,
        )
        assert hook.call_args_list == [((s.stage_name,),) for s in stages]

    def test_disabled_stage_is_skipped(self, tmp_path: Path) -> None:
        loop, controller, stages, _, _ = _build_loop(tmp_path)
        prepared = _make_prepared(tmp_path, stages)
        prepared = dataclasses.replace(prepared, enabled_stage_names=(STAGE_A,))

        loop.run_attempt(
            prepared=prepared, context=_mk_context(),
            log_layout=prepared.log_layout,
        )
        assert controller.state.attempts[-1].stage_runs[STAGE_B].status == StageRunState.STATUS_SKIPPED
        stages[1].run.assert_not_called()

    def test_stage_execution_completes_without_wide_mlflow_manager(
        self, tmp_path: Path
    ) -> None:
        """After the wide ``IMLflowManager`` retirement, the loop no
        longer calls ``log_stage_*`` on a manager -- per-stage signals
        flow via :class:`StageStartedEvent` / :class:`StageCompletedEvent`
        on the typed journal. We pin only that the run completes.
        """
        loop, _, stages, _, _ = _build_loop(tmp_path)
        prepared = _make_prepared(tmp_path, stages)

        ctx = loop.run_attempt(
            prepared=prepared, context=_mk_context(),
            log_layout=prepared.log_layout,
        )
        assert ctx is not None


# ===========================================================================
# 2. NEGATIVE
# ===========================================================================


class TestNegative:
    def test_stage_raise_routes_through_failure_handler(self, tmp_path: Path) -> None:
        # Use GPU_DEPLOYER (non-validator) — DATASET_VALIDATOR has a
        # different artifact-flushing path through ``ValidationArtifactManager``.
        stage_exc = PipelineStageFailedError(detail="boom", context={"reason": "X"})
        stages = [_make_stage(STAGE_B, raises=stage_exc)]
        loop, controller, _, collectors, _ = _build_loop(tmp_path, stages=stages)
        prepared = _make_prepared(tmp_path, stages)

        with pytest.raises(PipelineStageFailedError) as ei:
            loop.run_attempt(
                prepared=prepared, context=_mk_context(),
                log_layout=prepared.log_layout,
            )
        # Same typed exception propagates upward.
        assert ei.value is stage_exc
        # Loop transitioned pipeline to FAILED via _handle_stage_failure.
        assert controller.state.pipeline_status == StageRunState.STATUS_FAILED
        # Non-validator collector flushed via flush_error with the typed detail.
        collectors[STAGE_B].flush_error.assert_called_once()
        flush_kwargs = collectors[STAGE_B].flush_error.call_args.kwargs
        assert flush_kwargs["error"] == "boom"

    def test_dataset_validator_failure_uses_validation_artifact_manager(
        self, tmp_path: Path
    ) -> None:
        """DatasetValidator failures route through ValidationArtifactManager."""
        stage_exc = PipelineStageFailedError(detail="boom")
        stages = [_make_stage(STAGE_A, raises=stage_exc)]
        loop, controller, _, _, _ = _build_loop(tmp_path, stages=stages)
        prepared = _make_prepared(tmp_path, stages)

        with pytest.raises(PipelineStageFailedError):
            loop.run_attempt(
                prepared=prepared, context=_mk_context(),
                log_layout=prepared.log_layout,
            )
        assert controller.state.pipeline_status == StageRunState.STATUS_FAILED

    def test_prereq_error_raises_pipeline_stage_failed_error(
        self, tmp_path: Path
    ) -> None:
        planner = _make_stage_planner()
        planner.validate_stage_prerequisites.return_value = PipelineStageFailedError(
            detail="missing dep", context={"legacy_code": "PREREQ"}
        )
        stage = _make_stage(STAGE_A)
        loop, controller, _, _, _ = _build_loop(tmp_path, stages=[stage], planner=planner)
        prepared = _make_prepared(tmp_path, [stage])

        with pytest.raises(PipelineStageFailedError) as ei:
            loop.run_attempt(
                prepared=prepared, context=_mk_context(),
                log_layout=prepared.log_layout,
            )
        assert ei.value.context["legacy_code"] == "PREREQ"
        assert ei.value.context["prereq_failure"] is True
        stage.run.assert_not_called()
        assert controller.state.pipeline_status == StageRunState.STATUS_FAILED

    def test_unexpected_plain_exception_raises_internal_error(
        self, tmp_path: Path
    ) -> None:
        stage = _make_stage(STAGE_A)
        stage.run.side_effect = RuntimeError("surprise!")
        loop, controller, _, _, _ = _build_loop(tmp_path, stages=[stage])
        prepared = _make_prepared(tmp_path, [stage])

        with pytest.raises(InternalError) as ei:
            loop.run_attempt(
                prepared=prepared, context=_mk_context(),
                log_layout=prepared.log_layout,
            )
        assert ei.value.context["legacy_code"] == "UNEXPECTED_ERROR"
        assert ei.value.context["exception_type"] == "RuntimeError"
        assert controller.state.pipeline_status == StageRunState.STATUS_FAILED

    def test_pipeline_state_error_wrapped_as_internal_error(
        self, tmp_path: Path
    ) -> None:
        stage = _make_stage(STAGE_A)
        save_fn = MagicMock()
        loop, controller, _, _, _ = _build_loop(
            tmp_path, stages=[stage], save_fn=save_fn
        )
        prepared = _make_prepared(tmp_path, [stage])

        def _save_raises(_state: PipelineState) -> None:
            raise PipelineStateError("disk full")

        controller._save_fn = _save_raises  # type: ignore[assignment]

        with pytest.raises(InternalError) as ei:
            loop.run_attempt(
                prepared=prepared, context=_mk_context(),
                log_layout=prepared.log_layout,
            )
        assert ei.value.context["legacy_code"] == "PIPELINE_STATE_ERROR"


# ===========================================================================
# 3. BOUNDARY
# ===========================================================================


class TestBoundary:
    def test_empty_stage_range_returns_context_immediately(self, tmp_path: Path) -> None:
        stages = [_make_stage(STAGE_A)]
        loop, controller, _, _, _ = _build_loop(tmp_path, stages=stages)
        prepared = _make_prepared(tmp_path, stages)
        prepared = dataclasses.replace(prepared, start_idx=1, stop_idx=1)
        ctx = _mk_context()

        out = loop.run_attempt(
            prepared=prepared, context=ctx,
            log_layout=prepared.log_layout,
        )
        assert out is ctx
        stages[0].run.assert_not_called()
        assert controller.state.pipeline_status == StageRunState.STATUS_COMPLETED

    def test_start_idx_skips_early_stages(self, tmp_path: Path) -> None:
        stages = [_make_stage(s) for s in [STAGE_A, STAGE_B, STAGE_C]]
        loop, _, _, _, _ = _build_loop(tmp_path, stages=stages)
        prepared = _make_prepared(tmp_path, stages)
        prepared = dataclasses.replace(prepared, start_idx=1)

        loop.run_attempt(
            prepared=prepared, context=_mk_context(),
            log_layout=prepared.log_layout,
        )
        stages[0].run.assert_not_called()
        stages[1].run.assert_called_once()
        stages[2].run.assert_called_once()

    def test_stop_idx_stops_mid_list(self, tmp_path: Path) -> None:
        stages = [_make_stage(s) for s in [STAGE_A, STAGE_B, STAGE_C]]
        loop, _, _, _, _ = _build_loop(tmp_path, stages=stages)
        prepared = _make_prepared(tmp_path, stages)
        prepared = dataclasses.replace(prepared, stop_idx=2)

        loop.run_attempt(
            prepared=prepared, context=_mk_context(),
            log_layout=prepared.log_layout,
        )
        stages[0].run.assert_called_once()
        stages[1].run.assert_called_once()
        stages[2].run.assert_not_called()

    def test_single_stage_run(self, tmp_path: Path) -> None:
        stages = [_make_stage(STAGE_A)]
        loop, controller, _, _, _ = _build_loop(tmp_path, stages=stages)
        prepared = _make_prepared(tmp_path, stages)

        out = loop.run_attempt(
            prepared=prepared, context=_mk_context(),
            log_layout=prepared.log_layout,
        )
        assert out is not None
        assert controller.state.pipeline_status == StageRunState.STATUS_COMPLETED


# ===========================================================================
# 4. INVARIANTS
# ===========================================================================


class TestInvariants:
    def test_every_stage_completion_persists(self, tmp_path: Path) -> None:
        stages = [_make_stage(s) for s in [STAGE_A, STAGE_B]]
        save_fn = MagicMock()
        loop, _, _, _, _ = _build_loop(tmp_path, stages=stages, save_fn=save_fn)
        prepared = _make_prepared(tmp_path, stages)
        save_fn.reset_mock()

        loop.run_attempt(
            prepared=prepared, context=_mk_context(),
            log_layout=prepared.log_layout,
        )
        n_stages = len(stages)
        # Each stage: record_running + record_condition(Progressing=True) +
        # record_stage_log_paths + record_completed +
        # record_condition(Progressing=False) + record_condition(Available=True).
        # Plus finalize at the end. (Phase G — k8s/Operator-style
        # conditions[] emissions on stage entry and completion.)
        per_stage_saves = 6
        assert save_fn.call_count == per_stage_saves * n_stages + 1

    def test_summary_reporter_called_exactly_once_on_success(
        self, tmp_path: Path
    ) -> None:
        stages = [_make_stage(s) for s in [STAGE_A, STAGE_B]]
        loop, _, _, _, summary = _build_loop(tmp_path, stages=stages)
        prepared = _make_prepared(tmp_path, stages)

        loop.run_attempt(
            prepared=prepared, context=_mk_context(),
            log_layout=prepared.log_layout,
        )
        summary.print_summary.assert_called_once()

    def test_summary_reporter_not_called_on_failure(self, tmp_path: Path) -> None:
        stage_exc = PipelineStageFailedError(detail="x")
        stages = [_make_stage(STAGE_A, raises=stage_exc)]
        loop, _, _, _, summary = _build_loop(tmp_path, stages=stages)
        prepared = _make_prepared(tmp_path, stages)

        with pytest.raises(PipelineStageFailedError):
            loop.run_attempt(
                prepared=prepared, context=_mk_context(),
                log_layout=prepared.log_layout,
            )
        summary.print_summary.assert_not_called()

    def test_interrupt_marks_attempt_interrupted_and_raises_typed(
        self, tmp_path: Path
    ) -> None:
        stage = _make_stage(STAGE_A)
        stage.run.side_effect = KeyboardInterrupt()
        loop, controller, _, _, _ = _build_loop(tmp_path, stages=[stage])
        prepared = _make_prepared(tmp_path, [stage])

        with pytest.raises(PipelineStageFailedError) as ei:
            loop.run_attempt(
                prepared=prepared, context=_mk_context(),
                log_layout=prepared.log_layout,
            )
        assert ei.value.context["legacy_code"] == "PIPELINE_INTERRUPTED"
        assert controller.state.pipeline_status == StageRunState.STATUS_INTERRUPTED

    def test_on_shutdown_signal_hook_fires_on_interrupt(self, tmp_path: Path) -> None:
        stage = _make_stage(STAGE_A)
        stage.run.side_effect = KeyboardInterrupt()
        hook = MagicMock()
        loop, _, _, _, _ = _build_loop(tmp_path, stages=[stage], on_shutdown_signal=hook)
        prepared = _make_prepared(tmp_path, [stage])

        with pytest.raises(PipelineStageFailedError):
            loop.run_attempt(
                prepared=prepared, context=_mk_context(),
                log_layout=prepared.log_layout,
            )
        hook.assert_called_once_with("SIGINT")


# ===========================================================================
# 5. DEPENDENCY ERRORS
# ===========================================================================


class TestDependencyErrors:
    def test_mlflow_manager_none_is_tolerated(self, tmp_path: Path) -> None:
        stages = [_make_stage(STAGE_A)]
        loop, _, _, _, _ = _build_loop(tmp_path, stages=stages)
        prepared = _make_prepared(tmp_path, stages)

        out = loop.run_attempt(
            prepared=prepared, context=_mk_context(),
            log_layout=prepared.log_layout,
        )
        assert out is not None

    def test_already_flushed_collector_not_flushed_twice(self, tmp_path: Path) -> None:
        stage_exc = PipelineStageFailedError(detail="x")
        stages = [_make_stage(STAGE_A, raises=stage_exc)]
        loop, _, _, collectors, _ = _build_loop(tmp_path, stages=stages)
        prepared = _make_prepared(tmp_path, stages)
        collectors[STAGE_A].is_flushed = True  # pre-flushed

        with pytest.raises(PipelineStageFailedError):
            loop.run_attempt(
                prepared=prepared, context=_mk_context(),
                log_layout=prepared.log_layout,
            )
        collectors[STAGE_A].flush_error.assert_not_called()

    def test_save_fn_failure_during_running_transition_raises_internal_error(
        self, tmp_path: Path
    ) -> None:
        stages = [_make_stage(STAGE_A)]
        loop, controller, _, _, _ = _build_loop(tmp_path, stages=stages)
        prepared = _make_prepared(tmp_path, stages)

        def _flaky(_state: PipelineState) -> None:
            raise PipelineStateError("save failed")

        controller._save_fn = _flaky  # type: ignore[assignment]

        with pytest.raises(InternalError) as ei:
            loop.run_attempt(
                prepared=prepared, context=_mk_context(),
                log_layout=prepared.log_layout,
            )
        assert ei.value.context["legacy_code"] == "PIPELINE_STATE_ERROR"


# ===========================================================================
# 6. REGRESSIONS
# ===========================================================================


class TestRegressions:
    def test_system_exit_is_reraised(self, tmp_path: Path) -> None:
        stage = _make_stage(STAGE_A)
        stage.run.side_effect = SystemExit(1)
        loop, _, _, _, _ = _build_loop(tmp_path, stages=[stage])
        prepared = _make_prepared(tmp_path, [stage])

        with pytest.raises(SystemExit):
            loop.run_attempt(
                prepared=prepared, context=_mk_context(),
                log_layout=prepared.log_layout,
            )

    def test_no_default_hooks_required_for_success(self, tmp_path: Path) -> None:
        """Regression: loop runs without on_stage_completed/on_shutdown_signal."""
        stages = [_make_stage(STAGE_A)]
        loop, _, _, _, _ = _build_loop(tmp_path, stages=stages)  # no hooks
        prepared = _make_prepared(tmp_path, stages)

        out = loop.run_attempt(
            prepared=prepared, context=_mk_context(),
            log_layout=prepared.log_layout,
        )
        assert out is not None

    def test_log_layout_consulted_for_each_stage(self, tmp_path: Path) -> None:
        stages = [_make_stage(s) for s in [STAGE_A, STAGE_B]]
        loop, controller, _, _, _ = _build_loop(tmp_path, stages=stages)
        prepared = _make_prepared(tmp_path, stages)

        loop.run_attempt(
            prepared=prepared, context=_mk_context(),
            log_layout=prepared.log_layout,
        )
        for s in stages:
            assert s.stage_name in controller.state.attempts[-1].stage_runs

    def test_outside_loop_interrupt_helper_marks_attempt(self, tmp_path: Path) -> None:
        loop, controller, _, _, _ = _build_loop(tmp_path)
        loop.handle_interrupt_outside_loop()
        assert controller.state.pipeline_status == StageRunState.STATUS_INTERRUPTED

    def test_outside_loop_unexpected_error_helper_returns_typed_exc(
        self, tmp_path: Path
    ) -> None:
        loop, controller, _, _, _ = _build_loop(tmp_path)
        err = RuntimeError("bang")
        typed_exc = loop.handle_unexpected_error_outside_loop(err)
        assert isinstance(typed_exc, InternalError)
        assert typed_exc.context["legacy_code"] == "UNEXPECTED_ERROR"
        assert typed_exc.context["exception_type"] == "RuntimeError"
        assert controller.state.pipeline_status == StageRunState.STATUS_FAILED

    def test_stage_typed_exc_preserves_identity_through_loop(
        self, tmp_path: Path
    ) -> None:
        """Pin: the SAME instance bubbles up — no wrapping/rewrapping."""
        original = PipelineStageFailedError(
            detail="propagate me", context={"k": "v"}
        )
        stages = [_make_stage(STAGE_A, raises=original)]
        loop, _, _, _, _ = _build_loop(tmp_path, stages=stages)
        prepared = _make_prepared(tmp_path, stages)

        with pytest.raises(RyotenkAIError) as ei:
            loop.run_attempt(
                prepared=prepared, context=_mk_context(),
                log_layout=prepared.log_layout,
            )
        assert ei.value is original


# ===========================================================================
# 7. COMBINATORIAL
# ===========================================================================


@pytest.mark.parametrize("skip_reason", [None, "late_stage_disabled"])
@pytest.mark.parametrize("stage_name", [STAGE_A, STAGE_B, StageNames.MODEL_EVALUATOR])
@pytest.mark.parametrize("collector_present", [True, False])
def test_success_matrix(
    tmp_path: Path,
    skip_reason: str | None,
    stage_name: str,
    collector_present: bool,
) -> None:
    stage = _make_stage(stage_name)
    propagator = _make_context_propagator()
    propagator.get_stage_skip_reason.return_value = skip_reason

    collectors = (
        {stage_name: MagicMock(is_flushed=False, _started_at=None)}
        if collector_present
        else {}
    )
    if collector_present:
        collectors[stage_name].set_started_at = MagicMock()
        collectors[stage_name].flush_ok = MagicMock()

    state = _make_state(tmp_path)
    controller = _make_controller(state)

    loop = StageExecutionLoop(
        stages=[stage],
        collectors=collectors,
        attempt_controller=controller,
        stage_planner=_make_stage_planner(),
        context_propagator=propagator,
        stage_info_logger=MagicMock(),
        validation_artifact_mgr=MagicMock(),
        summary_reporter=MagicMock(),
    )
    prepared = _make_prepared(tmp_path, [stage])

    out = loop.run_attempt(
        prepared=prepared, context=_mk_context(),
        log_layout=prepared.log_layout,
    )
    assert out is not None

    stage_status = controller.state.attempts[-1].stage_runs[stage_name].status
    if skip_reason is not None:
        assert stage_status == StageRunState.STATUS_SKIPPED
    else:
        assert stage_status == StageRunState.STATUS_COMPLETED
