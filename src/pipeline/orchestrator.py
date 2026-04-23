"""
Pipeline Orchestrator
Manages the execution of all pipeline stages in sequence.
Implements the Chain of Responsibility pattern.

Features:
- Stage-by-stage execution with error handling
- MLflow integration for pipeline event logging
- Automatic cleanup on errors
- Summary generation with all pipeline context
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.config.runtime import RuntimeSettings, load_runtime_settings
from src.pipeline.bootstrap import PipelineBootstrap
from src.pipeline.constants import (
    EXIT_CODE_SIGINT,
    SEPARATOR_CHAR,
    SEPARATOR_LINE_WIDTH,
)
from src.pipeline.domain import RunContext
from src.pipeline.launch import LaunchPreparationError, PreparedAttempt
from src.pipeline.reporting import ExecutionSummaryReporter
from src.pipeline.stages import StageNames
from src.pipeline.state import (
    AttemptController,
    PipelineAttemptState,
    PipelineState,
    PipelineStateError,
    PipelineStateStore,
)
from src.pipeline.state.run_lock_guard import RunLockGuard
from src.utils.logger import logger
from src.utils.result import AppError, Err, Result

if TYPE_CHECKING:
    from src.pipeline.stages.base import PipelineStage
    from src.training.managers.mlflow_manager import MLflowManager
    from src.utils.logs_layout import LogLayout

# Re-export from the launch package so downstream test imports keep working
# without needing to know that the exception moved. Orchestrator.run() still
# catches it by this name.
__all__ = ["LaunchPreparationError", "PipelineOrchestrator"]


class PipelineOrchestrator:
    """
    Orchestrates the execution of all pipeline stages.
    Manages context flow between stages and handles errors.

    Integrates with MLflow for:
    - Pipeline event logging (stage start/complete/fail)
    - Summary generation from centralized MLflow data
    """

    def __init__(
        self,
        config_path: Path,
        run_directory: Path | None = None,
        settings: RuntimeSettings | None = None,
    ):
        """Initialize the orchestrator.

        Construction is two-phase:

        1. Declare per-run mutable state (run_ctx, _run_lock_guard, etc.) and
           the AttemptController — its save_fn closes over ``self._state_store``
           so the controller must exist before anything that might mutate state.
        2. :meth:`PipelineBootstrap.build` does the heavy wiring (config load,
           validation, component construction). The frozen result is copied
           onto ``self.*`` fields to keep backward compatibility with callers
           that read ``orch.config`` / ``orch.stages`` / etc.
        """
        # ----- Phase 1: per-run mutable state + single-writer controller -----
        self.settings: RuntimeSettings = settings or load_runtime_settings()
        # Do not name this attribute `run` — it would shadow
        # PipelineOrchestrator.run().
        self.run_ctx: RunContext = RunContext.create()
        self.logical_run_id: str | None = None
        self.run_directory: Path | None = run_directory
        self.attempt_directory: Path | None = None
        self._log_layout: LogLayout | None = None
        self._state_store: PipelineStateStore | None = None
        self._shutdown_signal_name: str | None = None
        self._run_lock_guard: RunLockGuard | None = None
        # Single writer of PipelineState / active attempt / lineage.
        # save_fn is a closure that reads ``self._state_store`` at call time —
        # that way ``_state_store`` can be bound by LaunchPreparator later
        # without re-creating the controller.
        self._attempt_controller: AttemptController = AttemptController(
            save_fn=self._persist_state,
            run_ctx=self.run_ctx,
        )

        # ----- Phase 2: delegate wiring to PipelineBootstrap -----
        bootstrap = PipelineBootstrap.build(
            config_path=config_path,
            run_ctx=self.run_ctx,
            settings=self.settings,
            attempt_controller=self._attempt_controller,
            on_stage_completed=self._on_stage_completed,
            on_shutdown_signal=self._on_shutdown_signal,
        )
        # Unpack onto ``self.*`` for backward compatibility — downstream
        # callers (and tests) still read ``orch.config``, ``orch.stages``,
        # ``orch._registry`` etc.
        self.config_path = bootstrap.config_path
        self.config = bootstrap.config
        self.secrets = bootstrap.secrets
        self.context = bootstrap.context
        self.stages = bootstrap.stages
        self._collectors = bootstrap.collectors
        self._validation_artifact_mgr = bootstrap.validation_artifact_mgr
        self._context_propagator = bootstrap.context_propagator
        self._stage_info_logger = bootstrap.stage_info_logger
        self._config_drift = bootstrap.config_drift
        self._summary_reporter = bootstrap.summary_reporter
        self._mlflow_attempt = bootstrap.mlflow_attempt
        self._registry = bootstrap.registry
        self._stage_planner = bootstrap.stage_planner
        self._launch_preparator = bootstrap.launch_preparator
        self._restart_inspector = bootstrap.restart_inspector
        self._stage_execution_loop = bootstrap.stage_execution_loop

    def notify_signal(self, *, signal_name: str) -> None:
        """Notify orchestrator about an external shutdown signal (SIGINT/SIGTERM)."""
        self._shutdown_signal_name = str(signal_name or "").upper()

    @property
    def _mlflow_manager(self) -> MLflowManager | None:
        """Backward-compat alias — many call sites read ``self._mlflow_manager`` directly."""
        return self._mlflow_attempt.manager

    @_mlflow_manager.setter
    def _mlflow_manager(self, value: MLflowManager | None) -> None:
        """Backward-compat setter — some tests assign to ``orchestrator._mlflow_manager``.

        Safe to call before ``_mlflow_attempt`` is initialised (tests sometimes
        partially build the orchestrator): in that case the assignment is a no-op.
        """
        attempt_mgr = self.__dict__.get("_mlflow_attempt")
        if attempt_mgr is not None:
            attempt_mgr._manager = value

    def _setup_mlflow(self) -> MLflowManager | None:
        """Setup MLflow for pipeline event logging (delegates to MLflowAttemptManager)."""
        return self._mlflow_attempt.bootstrap()

    def run(
        self,
        *,
        run_dir: Path | None = None,
        resume: bool = False,
        restart_from_stage: str | int | None = None,
    ) -> Result[dict[str, Any], AppError]:
        """Run the pipeline using persisted state and attempt-aware restart logic."""
        return self._run_stateful(
            run_dir=run_dir,
            resume=resume,
            restart_from_stage=restart_from_stage,
        )

    def _run_stateful(
        self,
        *,
        run_dir: Path | None,
        resume: bool,
        restart_from_stage: str | int | None,
    ) -> Result[dict[str, Any], AppError]:
        logger.info(SEPARATOR_CHAR * SEPARATOR_LINE_WIDTH)
        logger.info("Starting RyotenkAI Training Pipeline (stateful flow)")
        logger.info(SEPARATOR_CHAR * SEPARATOR_LINE_WIDTH)

        pipeline_success = False
        config_hashes = self._build_config_hashes()

        try:
            prepared = self._prepare_stateful_attempt(
                run_dir=run_dir,
                resume=resume,
                restart_from_stage=restart_from_stage,
                config_hashes=config_hashes,
            )
            assert self._log_layout is not None
            result = self._stage_execution_loop.run_attempt(
                prepared=prepared,
                context=self.context,
                mlflow_manager=self._mlflow_manager,
                log_layout=self._log_layout,
            )
            pipeline_success = result.is_ok()
            return result  # type: ignore[return-value]
        except LaunchPreparationError as exc:
            logger.error(exc.app_error.message)
            # Surface the run_directory+state_store that the preparator
            # resolved before raising, so teardown callers that read them
            # still see the right paths.
            if self._launch_preparator.last_state_store is not None:
                self._state_store = self._launch_preparator.last_state_store
            if self._launch_preparator.last_run_directory is not None:
                self.run_directory = self._launch_preparator.last_run_directory
            if (
                exc.state is not None
                and exc.requested_action is not None
                and exc.effective_action is not None
                and exc.start_stage_name is not None
            ):
                self._launch_preparator.record_launch_rejection(
                    launch_error=exc,
                    config_hashes=config_hashes,
                )
            return Err(exc.app_error)
        except (KeyboardInterrupt, SystemExit) as exc:
            # Interrupt during prepare — loop never got to own the boundary.
            # Delegate to the loop's public helper so the record-interrupted
            # + MLflow-warning semantics stay in one place.
            self._stage_execution_loop.handle_interrupt_outside_loop(
                mlflow_manager=self._mlflow_manager
            )
            if isinstance(exc, SystemExit):
                raise
            return Err(
                AppError(message="Pipeline interrupted by user", code="PIPELINE_INTERRUPTED")
            )
        except PipelineStateError as e:
            # Corrupted state file / persistence failure during bootstrap.
            # Distinct error code keeps observability clean.
            return Err(AppError(message=str(e), code="PIPELINE_STATE_ERROR"))
        except Exception as e:
            # Unexpected during prepare — same conversion as in-loop so the
            # caller sees a consistent AppError shape.
            return self._stage_execution_loop.handle_unexpected_error_outside_loop(
                e, mlflow_manager=self._mlflow_manager
            )
        finally:
            # Each teardown step is wrapped independently: a failure in one must
            # not skip the others. Critically, this guarantees run_lock.release()
            # always runs, otherwise subsequent pipeline launches get stuck.
            for step_name, step in (
                ("flush_pending_collectors", lambda: self._flush_pending_collectors()),
                ("cleanup_resources", lambda: self._cleanup_resources(success=pipeline_success)),
                ("teardown_mlflow_attempt", lambda: self._teardown_mlflow_attempt(pipeline_success=pipeline_success)),
            ):
                try:
                    step()
                except Exception:
                    logger.exception(f"[CLEANUP] step '{step_name}' failed")
            # RunLockGuard.__exit__ is already defensive (swallows release errors,
            # logs them). Still wrap defensively here in case the guard itself
            # raises unexpectedly — run_lock release must NEVER skip.
            guard = self._run_lock_guard
            self._run_lock_guard = None
            if guard is not None:
                try:
                    guard.__exit__(None, None, None)
                except Exception:
                    logger.exception("[CLEANUP] run lock guard exit failed")

    def _prepare_stateful_attempt(
        self,
        *,
        run_dir: Path | None,
        resume: bool,
        restart_from_stage: str | int | None,
        config_hashes: dict[str, str],
    ) -> PreparedAttempt:
        """Prepare an attempt + execute the non-pure launch steps.

        Delegates to :class:`LaunchPreparator` for the pure "compute
        PreparedAttempt from disk state" part, then does the three cross-
        cutting setup steps the preparator deliberately doesn't touch:
        acquiring the run lock, forking the execution context, registering
        the attempt into the controller + restoring lineage, and opening
        MLflow runs. Returns the PreparedAttempt so the loop can drive it.
        """
        prepared = self._launch_preparator.prepare(
            run_dir=run_dir,
            resume=resume,
            restart_from_stage=restart_from_stage,
            config_hashes=config_hashes,
        )

        # Mirror the prepared attempt onto orchestrator fields — downstream
        # code still reads ``self.run_directory``/``self._state_store`` etc.
        # These will be removed once RunSession (PR-A11) takes over.
        self.run_directory = prepared.run_directory
        self._state_store = prepared.state_store
        self.attempt_directory = prepared.attempt_directory
        self._log_layout = prepared.log_layout
        self.logical_run_id = prepared.logical_run_id

        # Acquire the run.lock via RunLockGuard so release is impossible
        # to forget in the finally block (Invariant #1 of the architecture).
        self._run_lock_guard = RunLockGuard(prepared.state_store.lock_path)
        self._run_lock_guard.__enter__()

        # Fork the run-scoped context into per-attempt context. fork()
        # guarantees no state leaks back to the original.
        self.context = self.context.fork(
            attempt_id=prepared.attempt.attempt_id,
            attempt_no=prepared.attempt.attempt_no,
            attempt_directory=prepared.attempt_directory,
            logical_run_id=prepared.logical_run_id,
            run_directory=prepared.run_directory,
            forced_stages=set(prepared.forced_stage_names),
        )

        # Register the attempt — flips state.attempts + active_attempt_id +
        # pipeline_status=RUNNING and persists atomically.
        self._attempt_controller.register_attempt(prepared.attempt)

        # Invalidate lineage from restart point first, then restore context
        # from the (already-trimmed) remaining lineage.
        stage_names = [s.stage_name for s in self.stages]
        self._attempt_controller.invalidate_lineage_from(
            stage_names=stage_names,
            start_stage_name=prepared.start_stage_name,
        )
        self._attempt_controller.restore_reused_context(
            stage_names=stage_names,
            start_stage_name=prepared.start_stage_name,
            enabled_stage_names=list(prepared.enabled_stage_names),
            context=self.context,
            sync_root_from_stage=self._sync_root_context_from_stage_outputs,
        )

        self._setup_mlflow_for_attempt(
            state=prepared.state, attempt=prepared.attempt, start_stage_idx=prepared.start_idx
        )
        self._ensure_mlflow_preflight(state=prepared.state)

        return prepared

    # ------------------------------------------------------------------
    # Hooks consumed by StageExecutionLoop (PR-A6)
    # ------------------------------------------------------------------

    def _on_stage_completed(self, stage_name: str) -> None:
        """Orchestrator-level side effects after a successful stage.

        Currently fires early GPU release after MODEL_RETRIEVER. Kept here
        because the policy depends on ``self.config`` (provider config), which
        the loop deliberately does not see.
        """
        if stage_name == StageNames.MODEL_RETRIEVER:
            self._maybe_early_release_gpu()

    def _on_shutdown_signal(self, signal_name: str) -> None:
        """Populate ``_shutdown_signal_name`` when the loop detects an interrupt.

        Used by the cleanup phase's "skip GPU disconnect on SIGINT" policy —
        without this, cleanup would always run disconnect even for user
        cancellations. Preserves the pre-refactor behaviour of defaulting to
        "SIGINT" when the external signal hook has not already set a name.
        """
        self._shutdown_signal_name = self._shutdown_signal_name or signal_name

    def _build_config_hashes(self) -> dict[str, str]:
        return self._config_drift.build_config_hashes()

    def _sync_root_context_from_stage_outputs(
        self, ctx: dict[str, Any], stage_name: str, outputs: dict[str, Any]
    ) -> None:
        """AttemptController-compatible sync callback.

        Accepts ``ctx`` as first positional arg (the controller passes the
        context it received, not ``self.context``) to match the generic
        ``lineage_manager.restore_reused`` callback contract.
        """
        self._context_propagator.sync_root_from_stage(
            context=ctx, stage_name=stage_name, outputs=outputs
        )

    def _persist_state(self, state: PipelineState) -> None:
        """Save callback injected into ``AttemptController``.

        Reads ``self._state_store`` at call time so the bootstrap flow can
        bind the store before the first mutation. If the store hasn't been
        created yet (pre-bootstrap), the save is a no-op — matches the old
        ``_save_state`` contract.
        """
        if self._state_store is None:
            return
        self._state_store.save(state)

    def _record_stage_log_paths(self, *, stage_name: str) -> None:
        """Attach the log file registry for ``stage_name`` to its StageRunState."""
        if self._log_layout is None:
            return
        include_remote_training = stage_name == StageNames.TRAINING_MONITOR
        log_paths = self._log_layout.stage_log_registry(
            stage_name,
            include_remote_training=include_remote_training,
        )
        self._attempt_controller.record_stage_log_paths(
            stage_name=stage_name, log_paths=log_paths
        )

    def _setup_mlflow_for_attempt(
        self, *, state: PipelineState, attempt: PipelineAttemptState, start_stage_idx: int
    ) -> None:
        # Bootstrap via the thin delegate so tests that patch ``_setup_mlflow``
        # can inject a mock MLflowManager without reaching into MLflowAttemptManager.
        manager = self._setup_mlflow()
        self._mlflow_attempt.setup_for_attempt(
            state=state,
            attempt=attempt,
            start_stage_idx=start_stage_idx,
            context=self.context,
            total_stages=len(self.stages),
            run_directory=self.run_directory,
            manager=manager,
        )

    def _ensure_mlflow_preflight(self, *, state: PipelineState) -> None:
        """Fail fast when mandatory MLflow setup/connectivity is not available."""
        app_error = self._mlflow_attempt.ensure_preflight()
        if app_error is not None:
            raise LaunchPreparationError(app_error, state=state)
        self._mlflow_attempt.log_config_artifact()
        # MLflowAttemptManager may have written run ids onto state/attempt
        # out-of-band; persist those through the controller.
        self._attempt_controller.save()

    def _teardown_mlflow_attempt(self, *, pipeline_success: bool) -> None:
        attempt_run_id = (
            self._attempt_controller.active_attempt.pipeline_attempt_mlflow_run_id
            if self._attempt_controller.has_active_attempt
            else None
        )

        def _before_end() -> None:
            self._aggregate_training_metrics()

        def _sync_state_and_return_path() -> Path | None:
            if self._state_store is None:
                return None
            self._attempt_controller.save()
            return self._state_store.state_path

        def _after_end(run_id: str | None) -> None:
            self._generate_experiment_report(run_id=run_id)

        self._mlflow_attempt.teardown_attempt(
            pipeline_success=pipeline_success,
            attempt_run_id=attempt_run_id,
            on_before_end=_before_end,
            state_path_supplier=_sync_state_and_return_path,
            on_after_end=_after_end,
        )

    def list_restart_points(self, run_dir: Path) -> list[dict[str, Any]]:
        """Delegate to :class:`RestartPointsInspector`."""
        return self._restart_inspector.inspect(run_dir)

    def _flush_pending_collectors(self) -> None:
        """Delegate: flush still-open collectors via the registry."""
        self._registry.flush_pending_collectors(self.context)

    def _print_summary(self) -> None:
        """Print a comprehensive summary of the pipeline execution."""
        self._summary_reporter.print_summary(context=self.context)

    def _maybe_early_release_gpu(self) -> None:
        """Delegate: early-release GPU via the registry."""
        self._registry.maybe_early_release_gpu()

    def _cleanup_resources(self, *, success: bool = False) -> None:
        """Delegate: reverse-order cleanup via the registry."""
        self._registry.cleanup_in_reverse(
            success=success,
            shutdown_signal_name=self._shutdown_signal_name,
        )

    def _aggregate_training_metrics(self) -> None:
        """Delegate: aggregate per-phase MLflow metrics into the parent attempt run."""
        self._summary_reporter.aggregate_training_metrics(
            mlflow_manager=self._mlflow_manager,
            collect_fn=lambda: self._collect_descendant_metrics(max_depth=2),
        )

    def _collect_descendant_metrics(self, max_depth: int = 2) -> list[dict[str, float]]:
        return ExecutionSummaryReporter.collect_descendant_metrics(
            mlflow_manager=self._mlflow_manager, max_depth=max_depth
        )

    def _generate_experiment_report(self, run_id: str | None = None) -> None:
        """Delegate: generate the post-pipeline experiment Markdown report.

        Passes ``PipelineConfig.reports.sections`` through so users control
        which sections appear and in what order directly from their YAML.
        """
        ExecutionSummaryReporter.generate_experiment_report(
            run_id=run_id,
            mlflow_manager=self._mlflow_manager,
            sections=self.config.reports.sections,
        )

    def get_stage_by_name(self, name: str) -> PipelineStage | None:
        """Delegate: look up a stage by name via the registry."""
        return self._registry.get_stage_by_name(name)

    def list_stages(self) -> list[str]:
        """Delegate: list stage names via the registry."""
        return self._registry.list_stage_names()


def run_pipeline(config_path: str) -> int:
    """
    Main entry point for running the pipeline.

    Args:
        config_path: Path to pipeline configuration file

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        orchestrator = PipelineOrchestrator(Path(config_path))
        result = orchestrator.run()

        if result.is_success():
            logger.info("Pipeline execution completed successfully")
            return 0
        else:
            logger.error(f"Pipeline execution failed: {result.unwrap_err()}")  # type: ignore[union-attr]
            return 1

    except KeyboardInterrupt:
        # Fallback: direct KBI without a signal handler registered (rare / legacy path).
        logger.warning("\nPipeline interrupted by user")
        return EXIT_CODE_SIGINT
    except Exception as e:
        logger.exception(f"Unexpected error in pipeline: {e}")
        return 1


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.pipeline.orchestrator <config_path>")
        sys.exit(2)
    sys.exit(run_pipeline(sys.argv[1]))
