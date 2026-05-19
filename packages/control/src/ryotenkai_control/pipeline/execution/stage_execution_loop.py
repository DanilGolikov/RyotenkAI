"""Stage execution loop — the heart of the pipeline runtime.

This module owns everything that happens between "launch prepared" and
"pipeline finished": the for-loop over stages, per-stage bookkeeping,
outcome handlers for success/failure/interrupt, and the exception
boundary that surfaces typed :class:`RyotenkAIError` to the orchestrator.

Architecture
------------
The orchestrator's job shrinks to:

    prepared = launch_preparator.prepare(...)
    # ... wire attempt into controller, fork context, setup MLflow ...
    return stage_execution_loop.run_attempt(prepared, context, log_layout)

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
Phase A2 Batch 6 migrated the loop from ``Result[T, AppError]`` to raise-
based: ``run_attempt`` returns :class:`PipelineContext` on full success
and raises a typed :class:`RyotenkAIError` on failure. The orchestrator
returns a plain dict and re-raises :class:`RyotenkAIError` -- Result is
fully gone (Phase A2 finale, commit ``e27619b``).

* ``KeyboardInterrupt``      → raises :class:`PipelineStageFailedError`
                                with ``context["legacy_code"] = "PIPELINE_INTERRUPTED"``
                                + records interrupt.
* ``SystemExit``             → records interrupt, then **re-raises** so the
                                runner's exit code is preserved.
* ``PipelineStateError``     → raises :class:`InternalError` with
                                ``context["legacy_code"] = "PIPELINE_STATE_ERROR"``.
* any other ``Exception``    → raises :class:`InternalError` +
                                finalize as FAILED.
* ``RyotenkAIError`` from a stage → routed to ``_handle_stage_failure``
                                then re-raised.

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

from ryotenkai_control.pipeline.artifacts.base import utc_now_iso
from ryotenkai_control.pipeline.constants import (
    SEPARATOR_CHAR,
    SEPARATOR_LINE_WIDTH,
)
from ryotenkai_control.pipeline.stages import StageNames
from ryotenkai_control.pipeline.state import PipelineStateError, StageRunState
from ryotenkai_control.pipeline.state.models import AttemptFailure
from ryotenkai_shared.contracts.pipeline_conditions import ConditionStatus
from ryotenkai_shared.errors import (
    InternalError,
    PipelineStageFailedError,
    RyotenkAIError,
)
from ryotenkai_shared.utils.logger import logger, stage_logging_context

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from ryotenkai_control.pipeline.artifacts import StageArtifactCollector
    from ryotenkai_control.pipeline.context import ContextPropagator, PipelineContext, StageInfoLogger
    from ryotenkai_control.pipeline.execution import StagePlanner
    from ryotenkai_control.pipeline.launch import PreparedAttempt
    from ryotenkai_control.pipeline.reporting import ExecutionSummaryReporter
    from ryotenkai_control.pipeline.stages.base import PipelineStage
    from ryotenkai_control.pipeline.stages.dataset_validator.artifact_manager import ValidationArtifactManager
    from ryotenkai_control.pipeline.state import AttemptController
    from ryotenkai_shared.utils.logs_layout import LogLayout

def _noop(*_args: Any, **_kwargs: Any) -> None:
    """Default for optional hooks — guarantees the loop never calls ``None()``."""


def _current_request_id_or_none() -> str | None:
    """Phase H2 — best-effort request_id lookup for ``AttemptFailure``.

    Imported lazily so the loop module doesn't pull in
    ``ryotenkai_shared.api`` unconditionally (the API surface depends on
    FastAPI/Starlette which is not always loaded — e.g. ``ryotenkai
    runs ls`` doesn't need it).
    """
    try:
        from ryotenkai_shared.api.request_id import current_request_id

        return current_request_id()
    except Exception:  # noqa: BLE001 — defensive
        return None


def _stamp_attempt_context(
    exc: RyotenkAIError,
    *,
    stage_name: str,
    stage_idx: int,
    stage_total: int,
    attempt_no: int | None,
    started_at: str | None,
) -> None:
    """Phase H1 — stamp attempt + stage identity onto ``exc.context``.

    Worker-level outcome logger (``worker._log_pipeline_outcome``) and
    the H2 ``AttemptFailure`` recorder read this so they don't need a
    contextvar lookup. Stamped fields are public (no underscore) — they
    are useful in problem+json output too. We never overwrite a value
    that the raise-site explicitly set; ``setdefault`` semantics.
    """
    ctx = exc.context
    if not isinstance(ctx, dict):
        return
    ctx.setdefault("stage_name", stage_name)
    ctx.setdefault("stage_idx", stage_idx)
    ctx.setdefault("stage_total", stage_total)
    if attempt_no is not None:
        ctx.setdefault("attempt_no", attempt_no)
    if started_at:
        ctx.setdefault("stage_started_at", started_at)


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
        log_layout: LogLayout,
    ) -> PipelineContext:
        """Execute stages ``[start_idx, stop_idx)`` for the prepared attempt.

        Returns ``context`` on full success.

        Raises:
            RyotenkAIError: on stage failure / prereq violation / state I/O
                / unexpected exception / KeyboardInterrupt. The orchestrator
                catches it, finalises the run, and re-raises.
            SystemExit: re-raised verbatim so the runner's exit code is
                preserved.
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
                    legacy_code = ""
                    if isinstance(prereq_error.context, dict):
                        legacy_code = str(prereq_error.context.get("legacy_code") or "")
                    failure_kind = legacy_code or prereq_error.code.value
                    detail = prereq_error.detail or str(prereq_error)
                    self._attempt_controller.record_failed(
                        stage_name=stage_name,
                        error=detail,
                        failure_kind=failure_kind,
                    )
                    # Surface prereq violation as typed exception so callers
                    # can match RyotenkAIError; legacy code preserved via
                    # context["legacy_code"] for observability.
                    prereq_exc = PipelineStageFailedError(
                        detail=detail,
                        context={
                            "legacy_code": failure_kind,
                            "stage_name": stage_name,
                            "stage_idx": i,
                            "stage_total": len(self._stages),
                            "attempt_no": self._current_attempt_no(),
                            "prereq_failure": True,
                        },
                    )
                    # Phase H2 — typed AttemptFailure for the prereq path.
                    try:
                        self._attempt_controller.record_failure(
                            AttemptFailure.from_exception(
                                prereq_exc,
                                stage_name=stage_name,
                                stage_idx=i,
                                stage_total=len(self._stages),
                                request_id=_current_request_id_or_none(),
                            )
                        )
                    except Exception:  # noqa: BLE001 — defensive
                        pass
                    self._attempt_controller.finalize(status=StageRunState.STATUS_FAILED)
                    raise prereq_exc

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
                    # Phase G — emit Progressing=True on stage entry.
                    # ``reason="StageStarted"`` matches k8s
                    # convention for in-progress positive states (cf.
                    # ``"AsExpected"``).
                    self._attempt_controller.record_condition(
                        stage_name=stage_name,
                        type="Progressing",
                        status=ConditionStatus.TRUE,
                        reason="StageStarted",
                        message=f"Stage {stage_name} started",
                    )
                    self._record_stage_log_paths(
                        stage_name=stage_name, log_layout=log_layout
                    )

                    logger.info(SEPARATOR_CHAR * SEPARATOR_LINE_WIDTH)
                    logger.info(f"Stage {i + 1}/{len(self._stages)}: {stage_name}")
                    logger.info(SEPARATOR_CHAR * SEPARATOR_LINE_WIDTH)

                    try:
                        stage_result = stage.run(context)
                    except RyotenkAIError as stage_exc:
                        stage_duration = time.time() - current_stage_start_time
                        self._handle_stage_failure(
                            context=context,
                            stage_name=stage_name,
                            stage_idx=i,
                            stage_exc=stage_exc,
                            collector=collector,
                            started_at=current_stage_started_at,
                            duration_seconds=stage_duration,
                        )
                        # Phase H1 — stamp attempt context onto the
                        # exception so the worker-level outcome logger
                        # (worker.py) and the H2 ``AttemptFailure``
                        # recorder can read attempt metadata without
                        # a contextvar lookup.
                        _stamp_attempt_context(
                            stage_exc,
                            stage_name=stage_name,
                            stage_idx=i,
                            stage_total=len(self._stages),
                            attempt_no=self._current_attempt_no(),
                            started_at=current_stage_started_at,
                        )
                        raise
                    stage_duration = time.time() - current_stage_start_time

                    if stage_result is not None:
                        context.update(stage_result)

                    # Phase 11.C-2 — capture pod identity for resume.
                    # Runs unconditionally (no-op when context lacks
                    # ``resource_id`` / ``provider_name`` — e.g. mock
                    # provider runs and stages other than GPUDeployer).
                    self._capture_pod_metadata_if_present(context)
                    # Phase 11.C-2 — propagate pod terminal state hint
                    # from TrainingMonitor (when present) so CLI / UI
                    # show the right "stopped" badge without a probe
                    # round-trip.
                    self._capture_pod_status_if_present(context)

                    # Orchestrator-level hook (e.g. early GPU release after
                    # MODEL_RETRIEVER). Runs BEFORE the success handler so
                    # any side effect it has is visible in logs/collectors.
                    self._on_stage_completed(stage_name)

                    self._handle_stage_success(
                        context=context,
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
            )
            return context

        except RyotenkAIError:
            # Stage / prereq failure — already recorded via
            # _handle_stage_failure or the prereq branch above. Re-raise
            # to the orchestrator boundary.
            raise
        except (KeyboardInterrupt, SystemExit) as exc:
            is_system_exit = isinstance(exc, SystemExit)
            self._handle_interrupt(
                current_stage_name=current_stage_name,
                current_stage_started_at=current_stage_started_at,
            )
            if is_system_exit:
                raise
            raise PipelineStageFailedError(
                detail="Pipeline interrupted by user",
                context={"legacy_code": "PIPELINE_INTERRUPTED"},
            ) from exc
        except PipelineStateError as exc:
            raise InternalError(
                detail=str(exc),
                context={"legacy_code": "PIPELINE_STATE_ERROR"},
                cause=exc,
            ) from exc
        except Exception as exc:
            self._handle_unexpected_error(exc)
            raise InternalError(
                detail=f"Unexpected error: {exc!s}",
                context={
                    "legacy_code": "UNEXPECTED_ERROR",
                    "exception_type": type(exc).__name__,
                },
                cause=exc,
            ) from exc

    # ------------------------------------------------------------------
    # Public exception helpers (used by the orchestrator outside the loop)
    # ------------------------------------------------------------------

    def handle_interrupt_outside_loop(self) -> None:
        """Mark the (non-started) attempt interrupted when KBI is raised
        during ``_prepare_stateful_attempt`` — i.e. before the loop gets a
        chance to own the exception boundary.

        Behaviour mirrors :meth:`_handle_interrupt` but with no current-stage
        context (no stage was running).
        """
        self._handle_interrupt(
            current_stage_name=None,
            current_stage_started_at=None,
        )

    def handle_unexpected_error_outside_loop(
        self,
        e: Exception,
    ) -> RyotenkAIError:
        """Record + return a typed :class:`RyotenkAIError` for a non-loop exception.

        Used when ``_prepare_stateful_attempt`` itself raises something the
        orchestrator doesn't want to let propagate. Delegates to the same
        recording logic as in-loop handling for consistency, then returns
        a typed exception the orchestrator can wrap or raise as needed.
        """
        self._handle_unexpected_error(e)
        return InternalError(
            detail=f"Unexpected error: {e!s}",
            context={
                "legacy_code": "UNEXPECTED_ERROR",
                "exception_type": type(e).__name__,
            },
            cause=e,
        )

    # ------------------------------------------------------------------
    # Outcome handlers
    # ------------------------------------------------------------------

    def _handle_stage_failure(
        self,
        *,
        context: PipelineContext,
        stage_name: str,
        stage_idx: int,
        stage_exc: RyotenkAIError,
        collector: StageArtifactCollector | None,
        started_at: str,
        duration_seconds: float,
    ) -> None:
        """Log, flush artifacts, mark failed, finalize.

        Called from the loop's inner try/except when ``stage.run`` raises
        a :class:`RyotenkAIError`. The caller re-raises after this returns
        so the exception propagates to the orchestrator boundary.
        """
        # Prefer the typed code's wire identifier; ``detail`` for messages.
        error_code = stage_exc.code.value
        error_message = stage_exc.detail or str(stage_exc)
        logger.error(f"Pipeline failed at stage {stage_idx + 1}: {stage_name}")
        logger.error(f"Error: {error_message}")
        if collector and not collector.is_flushed:
            if stage_name == StageNames.DATASET_VALIDATOR:
                self._validation_artifact_mgr.flush_validation_artifact(
                    started_at=started_at, duration_seconds=duration_seconds
                )
            else:
                collector.flush_error(
                    error=error_message,
                    started_at=started_at,
                    duration_seconds=duration_seconds,
                    context=context,
                )
        self._attempt_controller.record_failed(
            stage_name=stage_name,
            error=error_message,
            failure_kind=error_code,
            outputs=(
                self._validation_artifact_mgr.build_dataset_validation_state_outputs(
                    error=error_message
                )
                if stage_name == StageNames.DATASET_VALIDATOR
                else None
            ),
        )
        # Phase G — emit Degraded=True with the ErrorCode-derived reason
        # and flip Progressing=False so dashboards see the stage stopped
        # making progress. Reason is a stable CamelCase alias
        # (``"StageFailed"``) rather than the raw ErrorCode UPPER_SNAKE
        # value so the conditions side-channel stays consistent with
        # k8s tooling that pins on CamelCase.
        self._attempt_controller.record_condition(
            stage_name=stage_name,
            type="Progressing",
            status=ConditionStatus.FALSE,
            reason="StageFailed",
            message=error_message,
        )
        self._attempt_controller.record_condition(
            stage_name=stage_name,
            type="Degraded",
            status=ConditionStatus.TRUE,
            reason="StageFailed",
            message=f"{error_code}: {error_message}",
        )
        # Phase H2 — persist a typed AttemptFailure on the attempt
        # BEFORE finalize so resume tooling / web UI can read the
        # structured failure record from pipeline_state.json. Best-
        # effort: any persistence hiccup must not mask the original
        # exception path.
        try:
            failure = AttemptFailure.from_exception(
                stage_exc,
                stage_name=stage_name,
                stage_idx=stage_idx,
                stage_total=len(self._stages),
                request_id=_current_request_id_or_none(),
            )
            self._attempt_controller.record_failure(failure)
        except Exception:  # noqa: BLE001 — defensive persistence
            pass
        self._attempt_controller.finalize(status=StageRunState.STATUS_FAILED)

    def _handle_stage_success(
        self,
        *,
        context: PipelineContext,
        stage_name: str,
        stage_idx: int,
        collector: StageArtifactCollector | None,
        started_at: str,
        duration_seconds: float,
    ) -> None:
        """Flush artifacts, record COMPLETED or SKIPPED."""
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

        if skip_reason is not None:
            self._attempt_controller.record_skipped(
                stage_name=stage_name, reason=skip_reason, outputs=outputs
            )
        else:
            self._attempt_controller.record_completed(
                stage_name=stage_name, outputs=outputs
            )

        # Phase G — terminal positive emission: Progressing flips to
        # False (no longer making progress because we're done), and
        # Available=True signals the stage outputs are ready for
        # downstream consumers.
        self._attempt_controller.record_condition(
            stage_name=stage_name,
            type="Progressing",
            status=ConditionStatus.FALSE,
            reason="StageCompleted",
            message=f"Stage {stage_name} finished",
        )
        self._attempt_controller.record_condition(
            stage_name=stage_name,
            type="Available",
            status=ConditionStatus.TRUE,
            reason="AsExpected",
            message=None,
        )

        logger.info(
            f"Stage {stage_idx + 1} completed successfully ({duration_seconds:.1f}s)"
        )

    def _finalize_successful_run(
        self,
        *,
        context: PipelineContext,
        pipeline_duration: float,
    ) -> None:
        """Terminal happy path: mark attempt COMPLETED, emit summary."""
        self._attempt_controller.finalize(
            status=StageRunState.STATUS_COMPLETED,
            completed_at=utc_now_iso(),
        )

        logger.info("\n" + SEPARATOR_CHAR * SEPARATOR_LINE_WIDTH)
        logger.info("Pipeline completed successfully!")
        logger.info(SEPARATOR_CHAR * SEPARATOR_LINE_WIDTH)

        # Summary printing reads the live context — context is a PipelineContext
        # which is dict-compatible, so summary_reporter gets its data straight
        # from the stage execution surface.
        self._summary_reporter.print_summary(context=context)

    def _handle_interrupt(
        self,
        *,
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

    def _handle_unexpected_error(
        self,
        e: Exception,
    ) -> None:
        """Mark attempt FAILED on an unexpected exception.

        The caller (loop's outer except, or
        :meth:`handle_unexpected_error_outside_loop`) raises a typed
        :class:`InternalError` after this returns — keeping the recording
        side effects in one place.
        """
        logger.exception(f"Unexpected error in pipeline: {e}")
        completed_at = utc_now_iso()
        if self._attempt_controller.has_active_attempt:
            self._attempt_controller.mark_attempt_completed_at(completed_at=completed_at)
            # Phase H2 — typed AttemptFailure for the unexpected-error
            # path. Synthesise an InternalError-like record so the
            # consumer sees a stable ``code="INTERNAL_ERROR"``.
            try:
                self._attempt_controller.record_failure(
                    AttemptFailure(
                        code="INTERNAL_ERROR",
                        title="Internal error",
                        detail=f"Unexpected error: {e!s}",
                        context={"exception_type": type(e).__name__},
                        failed_at=completed_at,
                        request_id=_current_request_id_or_none(),
                    )
                )
            except Exception:  # noqa: BLE001 — defensive
                pass
        if self._attempt_controller.has_state:
            self._attempt_controller.finalize(
                status=StageRunState.STATUS_FAILED,
                completed_at=completed_at,
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _current_attempt_no(self) -> int | None:
        """Phase H1 — best-effort attempt-no lookup for outcome stamping.

        Returns ``None`` when no attempt is active (pre-stage error path
        that races the controller setup). Never raises — the caller is
        in an exception path and a None here just means the outcome
        block omits the ``attempts:`` line.
        """
        try:
            if self._attempt_controller.has_active_attempt:
                return int(self._attempt_controller.active_attempt.attempt_no)
        except Exception:  # noqa: BLE001 — defensive
            return None
        return None

    def _record_stage_log_paths(
        self, *, stage_name: str, log_layout: LogLayout
    ) -> None:
        """Attach log-file registry for ``stage_name`` via AttemptController."""
        include_remote_trainer_stdio = stage_name == StageNames.TRAINING_MONITOR
        log_paths = log_layout.stage_log_registry(
            stage_name,
            include_remote_trainer_stdio=include_remote_trainer_stdio,
        )
        self._attempt_controller.record_stage_log_paths(
            stage_name=stage_name, log_paths=log_paths
        )

    def _capture_pod_status_if_present(self, context: dict[str, Any]) -> None:
        """Phase 11.C-2 — propagate ``pod_terminal_state`` to attempt.

        :class:`TrainingMonitor` writes ``"stopped"`` into the context
        when the trainer exited cleanly (Phase 11.B's natural-completion
        path runs ``podStop``). Stages that don't deal with the pod
        leave the key absent ⇒ no-op.

        ``PodAvailabilityProbe`` is the live source of truth on resume
        — this stored hint just lets ``ryotenkai runs ls`` show the
        right badge without a RunPod GraphQL round-trip.
        """
        new_status = context.get("pod_terminal_state")
        if not isinstance(new_status, str) or not new_status:
            return
        try:
            self._attempt_controller.update_pod_status(
                last_known_status=new_status,
            )
        except Exception:
            pass

    def _capture_pod_metadata_if_present(self, context: dict[str, Any]) -> None:
        """Phase 11.C-2 — persist pod identity to the active attempt.

        Reads the GPUDeployer's contributions to ``context`` (``resource_id``
        + ``provider_name``) and forwards them to
        :meth:`AttemptController.set_pod_metadata`. The attempt's
        ``pod_metadata`` field becomes available to ``ryotenkai run resume``
        on Mac wake (see ``PodAvailabilityProbe`` in
        ``src/pipeline/launch/pod_availability.py``).

        Best-effort by design:

        * No ``resource_id`` in context → silent skip. Stages other than
          ``GPUDeployer`` don't put one there; mock-provider runs also
          don't.
        * ``set_pod_metadata`` errors don't propagate — the pipeline
          continues even if state persistence has a hiccup. The
          attempt loses the resume hint, but training proceeds.

        Idempotent: subsequent calls with the same pod_id overwrite
        identical data; calling on a non-GPUDeployer stage is a no-op.
        """
        resource_id = context.get("resource_id")
        provider_name = context.get("provider_name")
        if not isinstance(resource_id, str) or not resource_id:
            return
        if not isinstance(provider_name, str) or not provider_name:
            return
        try:
            self._attempt_controller.set_pod_metadata(
                pod_id=resource_id,
                provider=provider_name,
                last_known_status="running",
            )
        except Exception:
            # Persistence failure must not break the pipeline. Operator
            # forensics still see provider events; the resume hint is
            # the only thing lost.
            pass


__all__ = ["StageExecutionLoop"]
