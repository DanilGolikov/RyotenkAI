"""Stage execution loop — the heart of the pipeline runtime.

This module owns everything that happens between "launch prepared" and
"pipeline finished": the for-loop over stages, per-stage bookkeeping,
outcome handlers for success/failure/interrupt, and the exception
boundary that converts raw exceptions into ``Result[..., AppError]``.

Architecture
------------
The orchestrator's job shrinks to:

    prepared = launch_preparator.prepare(...)
    # ... wire attempt into controller, fork context, setup MLflow ...
    return stage_execution_loop.run_attempt(prepared, context, log_layout, mlflow_manager)

The loop knows nothing about config loading, secret resolution, or
signal-handler registration. It receives its collaborators via ctor and
its per-run data via ``run_attempt`` arguments — test isolation falls out
for free.

State mutation policy
---------------------
The loop **never** writes to ``PipelineState`` directly. Every transition
(RUNNING → COMPLETED/FAILED/SKIPPED/INTERRUPTED, finalize) goes through
the injected :class:`AttemptController` so the "every mutation is
durable" invariant from PR-A4 still holds.

Error boundary
--------------
``run_attempt`` converts every exception class below into an appropriate
``Result[..., AppError]``:

* ``KeyboardInterrupt``      → ``Err(PIPELINE_INTERRUPTED)`` + records interrupt.
* ``SystemExit``             → records interrupt, then **re-raises** so the
                                runner's exit code is preserved.
* ``PipelineStateError``     → ``Err(PIPELINE_STATE_ERROR)`` (state_store I/O).
* any other ``Exception``    → ``Err(UNEXPECTED_ERROR)`` + finalize as FAILED.

``LaunchPreparationError`` is *deliberately* NOT caught here — it can only
be raised by the preparator, which runs before the loop. Letting it
propagate keeps the rejection-recording path on the orchestrator where it
belongs.

Orchestrator-level hooks
------------------------
Two hooks cover cross-cutting concerns the loop cannot know about:

* ``on_stage_completed``      — fired after every successful stage with
                                 ``stage_name``. Used by the orchestrator to
                                 trigger ``_maybe_early_release_gpu`` after
                                 MODEL_RETRIEVER.
* ``on_shutdown_signal``      — fired when interrupt is detected, with the
                                 signal name (default ``"SIGINT"``). The
                                 orchestrator uses it to populate
                                 ``_shutdown_signal_name`` for the cleanup
                                 phase's "skip GPU disconnect on SIGINT"
                                 policy.

Both hooks default to no-ops — the loop is fully operational with neither.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from src.pipeline.artifacts.base import utc_now_iso
from src.pipeline.constants import (
    MLFLOW_CATEGORY_PIPELINE,
    MLFLOW_SOURCE_ORCHESTRATOR,
    SEPARATOR_CHAR,
    SEPARATOR_LINE_WIDTH,
)
from src.pipeline.stages import StageNames
from src.pipeline.state import PipelineStateError, StageRunState
from src.utils.logger import logger, stage_logging_context
from src.utils.result import AppError, Err, Ok, Result

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from src.pipeline.artifacts import StageArtifactCollector
    from src.pipeline.context import ContextPropagator, PipelineContext, StageInfoLogger
    from src.pipeline.execution import StagePlanner
    from src.pipeline.launch import PreparedAttempt
    from src.pipeline.reporting import ExecutionSummaryReporter
    from src.pipeline.stages.base import PipelineStage
    from src.pipeline.state import AttemptController
    from src.pipeline.validation.artifact_manager import ValidationArtifactManager
    from src.training.managers.mlflow_manager import MLflowManager
    from src.utils.logs_layout import LogLayout

_STATUS_FAILED = "failed"


def _noop(*_args: Any, **_kwargs: Any) -> None:
    """Default for optional hooks — guarantees the loop never calls ``None()``."""


class StageExecutionLoop:
    """Runs the configured stages of a single prepared attempt.

    Instantiated once per orchestrator (dependencies are stable), then
    invoked per run via :meth:`run_attempt`.
    """

    __slots__ = (
        "_attempt_controller",
        "_collectors",
        "_context_propagator",
        "_on_shutdown_signal",
        "_on_stage_completed",
        "_stage_info_logger",
        "_stage_planner",
        "_stages",
        "_summary_reporter",
        "_validation_artifact_mgr",
    )

    def __init__(
        self,
        *,
        stages: Sequence[PipelineStage],
        collectors: Mapping[str, StageArtifactCollector],
        attempt_controller: AttemptController,
        stage_planner: StagePlanner,
        context_propagator: ContextPropagator,
        stage_info_logger: StageInfoLogger,
        validation_artifact_mgr: ValidationArtifactManager,
        summary_reporter: ExecutionSummaryReporter,
        on_stage_completed: Callable[[str], None] | None = None,
        on_shutdown_signal: Callable[[str], None] | None = None,
    ) -> None:
        self._stages = stages
        self._collectors = collectors
        self._attempt_controller = attempt_controller
        self._stage_planner = stage_planner
        self._context_propagator = context_propagator
        self._stage_info_logger = stage_info_logger
        self._validation_artifact_mgr = validation_artifact_mgr
        self._summary_reporter = summary_reporter
        self._on_stage_completed = on_stage_completed or _noop
        self._on_shutdown_signal = on_shutdown_signal or _noop

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run_attempt(
        self,
        *,
        prepared: PreparedAttempt,
        context: PipelineContext,
        mlflow_manager: MLflowManager | None,
        log_layout: LogLayout,
    ) -> Result[PipelineContext, AppError]:
        """Execute stages ``[start_idx, stop_idx)`` for the prepared attempt.

        Returns ``Ok(context)`` on full success or ``Err(app_error)`` on any
        failure / interrupt / prereq violation. ``SystemExit`` propagates.
        """
        pipeline_start_time = time.time()
        current_stage_name: str | None = None
        current_stage_started_at: str | None = None

        try:
            for i, stage in enumerate(self._stages):
                stage_name = stage.stage_name

                if i < prepared.start_idx:
                    continue
                if i >= prepared.stop_idx:
                    break

                if stage_name not in prepared.enabled_stage_names:
                    self._attempt_controller.record_skipped(
                        stage_name=stage_name,
                        reason="disabled_by_config",
                    )
                    continue

                prereq_error = self._stage_planner.validate_stage_prerequisites(
                    stage_name=stage_name,
                    start_stage_name=prepared.start_stage_name,
                    context=context,
                )
                if prereq_error is not None:
                    self._attempt_controller.record_failed(
                        stage_name=stage_name,
                        error=prereq_error.message,
                        failure_kind=prereq_error.code,
                    )
                    self._attempt_controller.finalize(status=StageRunState.STATUS_FAILED)
                    return Err(prereq_error)

                current_stage_name = stage_name
                current_stage_started_at = utc_now_iso()
                current_stage_start_time = time.time()

                collector = self._collectors.get(stage_name)
                if collector:
                    collector.set_started_at(current_stage_started_at)

                with stage_logging_context(stage_name, log_layout):
                    self._attempt_controller.record_running(
                        stage_name=stage_name, started_at=current_stage_started_at
                    )
                    self._record_stage_log_paths(
                        stage_name=stage_name, log_layout=log_layout
                    )

                    logger.info(SEPARATOR_CHAR * SEPARATOR_LINE_WIDTH)
                    logger.info(f"Stage {i + 1}/{len(self._stages)}: {stage_name}")
                    logger.info(SEPARATOR_CHAR * SEPARATOR_LINE_WIDTH)

                    if mlflow_manager:
                        mlflow_manager.log_stage_start(
                            stage_name=stage_name,
                            stage_idx=i,
                            total_stages=len(self._stages),
                        )

                    result = stage.run(context)
                    stage_duration = time.time() - current_stage_start_time

                    if result.is_failure():
                        stage_err = result.unwrap_err()  # type: ignore[union-attr]
                        return self._handle_stage_failure(
                            context=context,
                            mlflow_manager=mlflow_manager,
                            stage_name=stage_name,
                            stage_idx=i,
                            stage_err=stage_err,
                            collector=collector,
                            started_at=current_stage_started_at,
                            duration_seconds=stage_duration,
                        )

                    stage_result = result.unwrap()
                    if stage_result is not None:
                        context.update(stage_result)

                    # Orchestrator-level hook (e.g. early GPU release after
                    # MODEL_RETRIEVER). Runs BEFORE the success handler so
                    # any side effect it has is visible in logs/collectors.
                    self._on_stage_completed(stage_name)

                    self._handle_stage_success(
                        context=context,
                        mlflow_manager=mlflow_manager,
                        stage_name=stage_name,
                        stage_idx=i,
                        collector=collector,
                        started_at=current_stage_started_at,
                        duration_seconds=stage_duration,
                    )

                current_stage_name = None
                current_stage_started_at = None

            pipeline_duration = time.time() - pipeline_start_time
            self._finalize_successful_run(
                context=context,
                pipeline_duration=pipeline_duration,
                mlflow_manager=mlflow_manager,
            )
            return Ok(context)

        except (KeyboardInterrupt, SystemExit) as exc:
            is_system_exit = isinstance(exc, SystemExit)
            self._handle_interrupt(
                mlflow_manager=mlflow_manager,
                current_stage_name=current_stage_name,
                current_stage_started_at=current_stage_started_at,
            )
            if is_system_exit:
                raise
            return Err(
                AppError(message="Pipeline interrupted by user", code="PIPELINE_INTERRUPTED")
            )
        except PipelineStateError as e:
            return Err(AppError(message=str(e), code="PIPELINE_STATE_ERROR"))
        except Exception as e:
            return self._handle_unexpected_error(e, mlflow_manager=mlflow_manager)

    # ------------------------------------------------------------------
    # Public exception helpers (used by the orchestrator outside the loop)
    # ------------------------------------------------------------------

    def handle_interrupt_outside_loop(
        self,
        *,
        mlflow_manager: MLflowManager | None,
    ) -> None:
        """Mark the (non-started) attempt interrupted when KBI is raised
        during ``_prepare_stateful_attempt`` — i.e. before the loop gets a
        chance to own the exception boundary.

        Behaviour mirrors :meth:`_handle_interrupt` but with no current-stage
        context (no stage was running).
        """
        self._handle_interrupt(
            mlflow_manager=mlflow_manager,
            current_stage_name=None,
            current_stage_started_at=None,
        )

    def handle_unexpected_error_outside_loop(
        self,
        e: Exception,
        *,
        mlflow_manager: MLflowManager | None,
    ) -> Result[PipelineContext, AppError]:
        """Convert a non-loop exception into an UNEXPECTED_ERROR ``Err``.

        Used when ``_prepare_stateful_attempt`` itself raises something the
        orchestrator doesn't want to let propagate. Delegates to the same
        logic as in-loop handling for consistency.
        """
        return self._handle_unexpected_error(e, mlflow_manager=mlflow_manager)

    # ------------------------------------------------------------------
    # Outcome handlers
    # ------------------------------------------------------------------

    def _handle_stage_failure(
        self,
        *,
        context: PipelineContext,
        mlflow_manager: MLflowManager | None,
        stage_name: str,
        stage_idx: int,
        stage_err: AppError,
        collector: StageArtifactCollector | None,
        started_at: str,
        duration_seconds: float,
    ) -> Result[PipelineContext, AppError]:
        """Log, flush artifacts, mark failed, finalize; return STAGE_FAILED Err."""
        logger.error(f"Pipeline failed at stage {stage_idx + 1}: {stage_name}")
        logger.error(f"Error: {stage_err}")
        if mlflow_manager:
            mlflow_manager.log_stage_failed(
                stage_name=stage_name, stage_idx=stage_idx, error=str(stage_err)
            )
        if collector and not collector.is_flushed:
            if stage_name == StageNames.DATASET_VALIDATOR:
                self._validation_artifact_mgr.flush_validation_artifact(
                    started_at=started_at, duration_seconds=duration_seconds
                )
            else:
                collector.flush_error(
                    error=str(stage_err),
                    started_at=started_at,
                    duration_seconds=duration_seconds,
                    context=context,
                )
        self._attempt_controller.record_failed(
            stage_name=stage_name,
            error=str(stage_err),
            failure_kind=getattr(stage_err, "code", "STAGE_FAILED"),
            outputs=(
                self._validation_artifact_mgr.build_dataset_validation_state_outputs(
                    error=str(stage_err)
                )
                if stage_name == StageNames.DATASET_VALIDATOR
                else None
            ),
        )
        self._attempt_controller.finalize(status=StageRunState.STATUS_FAILED)
        return Err(
            AppError(
                message=f"Stage '{stage_name}' failed: {stage_err.message}",
                code="STAGE_FAILED",
                details={
                    **stage_err.to_log_dict(),
                    "stage_name": stage_name,
                    "stage_idx": stage_idx,
                },
            )
        )

    def _handle_stage_success(
        self,
        *,
        context: PipelineContext,
        mlflow_manager: MLflowManager | None,
        stage_name: str,
        stage_idx: int,
        collector: StageArtifactCollector | None,
        started_at: str,
        duration_seconds: float,
    ) -> None:
        """Flush artifacts, log MLflow completion, record COMPLETED or SKIPPED."""
        outputs = self._context_propagator.extract_restart_outputs(
            context=context, stage_name=stage_name
        )
        skip_reason = self._context_propagator.get_stage_skip_reason(
            context=context, stage_name=stage_name
        )

        if collector and not collector.is_flushed:
            if stage_name == StageNames.DATASET_VALIDATOR:
                self._validation_artifact_mgr.flush_validation_artifact(
                    started_at=started_at, duration_seconds=duration_seconds
                )
            else:
                self._context_propagator.fill_collector_from_context(
                    context=context, stage_name=stage_name, collector=collector
                )
                collector.flush_ok(
                    started_at=started_at,
                    duration_seconds=duration_seconds,
                    context=context,
                )

        if mlflow_manager:
            self._stage_info_logger.log(
                mlflow_manager=mlflow_manager,
                context=context,
                stage_name=stage_name,
            )
            mlflow_manager.log_stage_complete(
                stage_name=stage_name,
                stage_idx=stage_idx,
                duration_seconds=duration_seconds,
            )

        if skip_reason is not None:
            self._attempt_controller.record_skipped(
                stage_name=stage_name, reason=skip_reason, outputs=outputs
            )
        else:
            self._attempt_controller.record_completed(
                stage_name=stage_name, outputs=outputs
            )

        logger.info(
            f"Stage {stage_idx + 1} completed successfully ({duration_seconds:.1f}s)"
        )

    def _finalize_successful_run(
        self,
        *,
        context: PipelineContext,
        pipeline_duration: float,
        mlflow_manager: MLflowManager | None,
    ) -> None:
        """Terminal happy path: mark attempt COMPLETED, emit summary + MLflow."""
        self._attempt_controller.finalize(
            status=StageRunState.STATUS_COMPLETED,
            completed_at=utc_now_iso(),
        )

        logger.info("\n" + SEPARATOR_CHAR * SEPARATOR_LINE_WIDTH)
        logger.info("Pipeline completed successfully!")
        logger.info(SEPARATOR_CHAR * SEPARATOR_LINE_WIDTH)

        if mlflow_manager:
            mlflow_manager.log_event_complete(
                f"Pipeline completed successfully in {pipeline_duration:.1f}s",
                category=MLFLOW_CATEGORY_PIPELINE,
                source=MLFLOW_SOURCE_ORCHESTRATOR,
                duration_seconds=pipeline_duration,
            )
            mlflow_manager.set_tags({"pipeline.status": "completed"})
            mlflow_manager.log_params({"pipeline.duration_seconds": pipeline_duration})

        # Summary printing reads the live context — context is a PipelineContext
        # which is dict-compatible, so summary_reporter gets its data straight
        # from the stage execution surface.
        self._summary_reporter.print_summary(context=context)

    def _handle_interrupt(
        self,
        *,
        mlflow_manager: MLflowManager | None,
        current_stage_name: str | None,
        current_stage_started_at: str | None,
    ) -> None:
        """Mark attempt/stage interrupted, log warning, notify shutdown hook."""
        # Orchestrator uses this to populate _shutdown_signal_name for the
        # cleanup phase. Default "SIGINT" matches the pre-refactor behaviour.
        self._on_shutdown_signal("SIGINT")
        logger.warning("\nPipeline interrupted by user")
        if self._attempt_controller.has_active_attempt:
            completed_at = utc_now_iso()
            self._attempt_controller.mark_attempt_completed_at(completed_at=completed_at)
            if current_stage_name and current_stage_started_at:
                self._attempt_controller.record_interrupted(
                    stage_name=current_stage_name,
                    started_at=current_stage_started_at,
                )
            if self._attempt_controller.has_state:
                self._attempt_controller.finalize(
                    status=StageRunState.STATUS_INTERRUPTED,
                    completed_at=completed_at,
                )
        if mlflow_manager:
            mlflow_manager.log_event_warning(
                "Pipeline interrupted by user",
                category=MLFLOW_CATEGORY_PIPELINE,
                source=MLFLOW_SOURCE_ORCHESTRATOR,
            )
            mlflow_manager.set_tags({"pipeline.status": "interrupted"})

    def _handle_unexpected_error(
        self,
        e: Exception,
        *,
        mlflow_manager: MLflowManager | None,
    ) -> Result[PipelineContext, AppError]:
        """Mark attempt FAILED on an unexpected exception and surface UNEXPECTED_ERROR."""
        logger.exception(f"Unexpected error in pipeline: {e}")
        completed_at = utc_now_iso()
        if self._attempt_controller.has_active_attempt:
            self._attempt_controller.mark_attempt_completed_at(completed_at=completed_at)
        if self._attempt_controller.has_state:
            self._attempt_controller.finalize(
                status=StageRunState.STATUS_FAILED,
                completed_at=completed_at,
            )
        if mlflow_manager:
            mlflow_manager.log_event_error(
                f"Unexpected error: {e!s}",
                category=MLFLOW_CATEGORY_PIPELINE,
                source=MLFLOW_SOURCE_ORCHESTRATOR,
                error_type=type(e).__name__,
            )
            mlflow_manager.set_tags({"pipeline.status": _STATUS_FAILED})
        return Err(
            AppError(
                message=f"Unexpected error: {e!s}",
                code="UNEXPECTED_ERROR",
                details={"exception_type": type(e).__name__},
            )
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _record_stage_log_paths(
        self, *, stage_name: str, log_layout: LogLayout
    ) -> None:
        """Attach log-file registry for ``stage_name`` via AttemptController."""
        include_remote_training = stage_name == StageNames.TRAINING_MONITOR
        log_paths = log_layout.stage_log_registry(
            stage_name,
            include_remote_training=include_remote_training,
        )
        self._attempt_controller.record_stage_log_paths(
            stage_name=stage_name, log_paths=log_paths
        )


__all__ = ["StageExecutionLoop"]
