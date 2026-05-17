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

import time
from typing import TYPE_CHECKING, Any

from ryotenkai_control.pipeline.bootstrap import PipelineBootstrap
from ryotenkai_control.pipeline.constants import (
    EXIT_CODE_SIGINT,
    SEPARATOR_CHAR,
    SEPARATOR_LINE_WIDTH,
)
from ryotenkai_control.pipeline.launch import LaunchPreparationError, PreparedAttempt
from ryotenkai_control.pipeline.launch.run_lock_guard import RunLockGuard
from ryotenkai_control.pipeline.reporting import ExecutionSummaryReporter
from ryotenkai_control.pipeline.run_lifecycle_coordinator import (
    RunLifecycleCoordinator,
)
from ryotenkai_control.pipeline.stages import StageNames
from ryotenkai_control.pipeline.state import (
    AttemptController,
    PipelineAttemptState,
    PipelineState,
    PipelineStateError,
    PipelineStateStore,
)
from ryotenkai_shared.config.runtime import RuntimeSettings, load_runtime_settings
from ryotenkai_shared.errors import (
    InternalError,
    PipelineStageFailedError,
    RyotenkAIError,
)
from ryotenkai_shared.pipeline_context import RunContext
from ryotenkai_shared.utils.logger import logger

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from ryotenkai_control.events import ControlEventEmitter
    from ryotenkai_control.pipeline.stages.base import PipelineStage
    from ryotenkai_shared.config.pipeline.schema import PipelineConfig
    from ryotenkai_shared.infrastructure.mlflow.protocol import IMLflowManager
    from ryotenkai_shared.utils.logs_layout import LogLayout

# Re-export from the launch package so downstream test imports keep working
# without needing to know that the exception moved. Orchestrator.run() still
# catches it by this name.
__all__ = ["LaunchPreparationError", "PipelineOrchestrator"]


def _wrap_as_launch_error(
    exc: RyotenkAIError,
    *,
    state: PipelineState,
    legacy_code: str | None = None,
) -> LaunchPreparationError:
    """Wrap a typed exception (e.g. from MLflow preflight) as ``LaunchPreparationError``.

    Used by the orchestrator's preflight path so the run()-level
    ``except LaunchPreparationError`` block continues to drive the
    rejection-recording flow uniformly. The original exception is
    attached via ``__cause__`` (``raise ... from``) by the caller.
    """
    detail = exc.detail or str(exc)
    context: dict[str, Any] = {"typed_error_class": type(exc).__name__}
    if isinstance(exc.context, dict):
        for key, value in exc.context.items():
            context.setdefault(key, value)
    if legacy_code is not None:
        context["legacy_code"] = legacy_code
    err = LaunchPreparationError(detail, context=context)
    err.state = state  # type: ignore[attr-defined]
    err.requested_action = None  # type: ignore[attr-defined]
    err.effective_action = None  # type: ignore[attr-defined]
    err.start_stage_name = None  # type: ignore[attr-defined]
    return err


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
        *,
        config: PipelineConfig,
        run_directory: Path | None = None,
        settings: RuntimeSettings | None = None,
        mlflow_manager: IMLflowManager | None = None,
        stages_override: Sequence[PipelineStage] | None = None,
    ):
        """Initialize the orchestrator from a pre-loaded config.

        Caller (worker subprocess) loads the YAML via
        :func:`src.workspace.integrations.loader.load_pipeline_config`,
        which returns a validated :class:`PipelineConfig` with
        ``_source_path`` set. Run-level metadata
        (``project_id`` / ``actor`` / ``config_version_hash``) and
        per-project secrets reach the orchestrator through
        ``os.environ`` — the launcher (CLI / Web API) merges ``env.json``
        and sets ``RYOTENKAI_*`` env vars before forking the worker.

        Args:
            config: Fully-loaded :class:`PipelineConfig` (must have
                ``_source_path`` set — the loader handles this).
            run_directory: Optional explicit run dir for resume/restart;
                a fresh dir under ``settings.runs_base_dir`` is created
                when ``None``.
            settings: Optional :class:`RuntimeSettings`; read from env
                via :func:`load_runtime_settings` when ``None``.
            mlflow_manager: Optional pre-built :class:`IMLflowManager`
                that short-circuits :meth:`_setup_mlflow` (tests).
            stages_override: Optional pre-built stage list that replaces
                :meth:`StageRegistry._build_stages` (tests only).

        Construction is two-phase: declare per-run mutable state +
        :class:`AttemptController` first (its ``save_fn`` closure reads
        ``self._state_store`` at call time), then delegate the heavy
        wiring to :meth:`PipelineBootstrap.build`. The frozen result is
        copied onto ``self.*`` fields so downstream callers can read
        ``orch.config`` / ``orch.stages`` / etc.
        """
        # ----- Phase 1: per-run mutable state + single-writer controller -----
        self.settings: RuntimeSettings = settings or load_runtime_settings()
        # Do not name this attribute `run` — it would shadow
        # PipelineOrchestrator.run().
        self.run_ctx: RunContext = RunContext.create()
        self.run_directory: Path | None = run_directory
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
            config=config,
            run_ctx=self.run_ctx,
            settings=self.settings,
            attempt_controller=self._attempt_controller,
            on_stage_completed=self._on_stage_completed,
            on_shutdown_signal=self._on_shutdown_signal,
            stages_override=stages_override,
            run_directory=run_directory,
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

        # Phase 8: hand the event-emitter lifecycle to
        # :class:`RunLifecycleCoordinator`. The coordinator owns lazy
        # construction once LaunchPreparator resolves the run dir, the
        # registry register/deregister, the four ``emit_run_*``
        # terminal events, and the MlflowFinalizer upload in finalize.
        self._coord = RunLifecycleCoordinator(
            run_ctx=self.run_ctx,
            algorithm_supplier=self._derive_algorithm,
            dataset_id_supplier=self._derive_dataset_id,
            model_id_supplier=self._derive_model_id,
            mlflow_run_id_supplier=self._safe_attempt_mlflow_run_id,
            active_stage_supplier=self._safe_active_stage_name,
            shutdown_signal_supplier=lambda: self._shutdown_signal_name,
            mlflow_manager_supplier=lambda: self._mlflow_manager,
            pre_built_emitter=bootstrap.emitter,
        )

        # Optional injection: tests / advanced callers can supply a pre-built
        # IMLflowManager (e.g. FakeMLflowManager). When provided, _setup_mlflow
        # returns it directly without invoking _mlflow_attempt.bootstrap() —
        # this replaces the patch.object(*, "_setup_mlflow") test scaffold.
        if mlflow_manager is not None:
            self._mlflow_manager = mlflow_manager

    # ------------------------------------------------------------------
    # Backward-compat: ``orch.emitter`` — direct attribute reads (and
    # the rare write in ``test_orchestrator_emitter_wiring``) now route
    # through the coordinator. Stages read the emitter through bootstrap
    # result; the orchestrator itself never touches it directly outside
    # the property below.
    # ------------------------------------------------------------------

    @property
    def emitter(self) -> ControlEventEmitter | None:
        """Active event emitter, or ``None`` until the run dir is resolved."""
        return self._coord.emitter

    @emitter.setter
    def emitter(self, value: ControlEventEmitter | None) -> None:
        """Allow tests to swap the emitter without reaching into the coord."""
        self._coord._emitter = value
        if value is not None:
            self._coord._register_emitter()

    def notify_signal(self, *, signal_name: str) -> None:
        """Notify orchestrator about an external shutdown signal (SIGINT/SIGTERM)."""
        self._shutdown_signal_name = str(signal_name or "").upper()

    @property
    def _mlflow_manager(self) -> IMLflowManager | None:
        """Backward-compat alias — many call sites read ``self._mlflow_manager`` directly."""
        return self._mlflow_attempt.manager

    @_mlflow_manager.setter
    def _mlflow_manager(self, value: IMLflowManager | None) -> None:
        """Backward-compat setter — some tests assign to ``orchestrator._mlflow_manager``.

        Safe to call before ``_mlflow_attempt`` is initialised (tests sometimes
        partially build the orchestrator): in that case the assignment is a no-op.
        """
        attempt_mgr = self.__dict__.get("_mlflow_attempt")
        if attempt_mgr is not None:
            attempt_mgr._manager = value

    def _setup_mlflow(self) -> IMLflowManager | None:
        """Setup MLflow for pipeline event logging (delegates to MLflowAttemptManager).

        Idempotent: if a manager was already injected (via constructor kwarg
        ``mlflow_manager`` or assignment to ``_mlflow_manager``), returns it
        without re-running bootstrap. This is the test-friendly path.
        """
        existing = self._mlflow_attempt.manager
        if existing is not None:
            return existing
        return self._mlflow_attempt.bootstrap()

    def run(
        self,
        *,
        run_dir: Path | None = None,
        resume: bool = False,
        restart_from_stage: str | int | None = None,
    ) -> dict[str, Any]:
        """Run the pipeline using persisted state and attempt-aware restart logic.

        Returns the final pipeline context on full success.

        Raises:
            RyotenkAIError: stage failure / prereq violation / state I/O
                error / unexpected exception / KeyboardInterrupt /
                LaunchPreparationError. Callers (``worker.py`` /
                ``run_pipeline``) catch and map to exit codes.
            SystemExit: re-raised verbatim so the runner's exit code
                is preserved.
        """
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
    ) -> dict[str, Any]:
        logger.info(SEPARATOR_CHAR * SEPARATOR_LINE_WIDTH)
        logger.info("Starting RyotenkAI Training Pipeline (stateful flow)")
        logger.info(SEPARATOR_CHAR * SEPARATOR_LINE_WIDTH)

        pipeline_success = False
        config_hashes = self._build_config_hashes()
        run_start_monotonic = time.monotonic()

        try:
            prepared = self._prepare_stateful_attempt(
                run_dir=run_dir,
                resume=resume,
                restart_from_stage=restart_from_stage,
                config_hashes=config_hashes,
            )
            assert self._log_layout is not None
            # Phase 3-4: build the journal under <run_directory>/events.jsonl,
            # propagate the emitter into every stage that opts in via
            # ``set_emitter``, then emit RunStartedEvent.
            self._ensure_event_emitter()
            self._wire_emitter_into_stages()
            self._coord.emit_run_started(config_hashes=config_hashes)
            # Loop is raise-based: returns context on success, raises
            # ``RyotenkAIError`` on failure (Phase A2 Batch 6).
            context = self._stage_execution_loop.run_attempt(
                prepared=prepared,
                context=self.context,
                mlflow_manager=self._mlflow_manager,
                log_layout=self._log_layout,
            )
            pipeline_success = True
            self._coord.emit_run_completed(
                duration_s=time.monotonic() - run_start_monotonic,
                status="success",
            )
            return context
        except LaunchPreparationError as exc:
            logger.error(exc.detail or str(exc))
            # Surface the preparator-resolved paths so teardown sees them.
            if self._launch_preparator.last_state_store is not None:
                self._state_store = self._launch_preparator.last_state_store
            if self._launch_preparator.last_run_directory is not None:
                self.run_directory = self._launch_preparator.last_run_directory
            # Rejection recording requires the four context fields populated
            # by the preparator's ``_make_launch_error`` helper — preflight
            # wraps via ``_wrap_as_launch_error`` and intentionally leaves
            # requested_action/effective_action/start_stage_name unset.
            exc_state = getattr(exc, "state", None)
            exc_requested = getattr(exc, "requested_action", None)
            exc_effective = getattr(exc, "effective_action", None)
            exc_start_stage = getattr(exc, "start_stage_name", None)
            if (
                exc_state is not None
                and exc_requested is not None
                and exc_effective is not None
                and exc_start_stage is not None
            ):
                self._launch_preparator.record_launch_rejection(
                    launch_error=exc,
                    config_hashes=config_hashes,
                )
            # Re-raise the typed LaunchPreparationError (a RyotenkAIError
            # subclass) so worker.py catches the typed boundary directly.
            raise
        except (KeyboardInterrupt, SystemExit) as exc:
            # Interrupt during prepare — loop never got to own the boundary.
            self._stage_execution_loop.handle_interrupt_outside_loop(
                mlflow_manager=self._mlflow_manager,
            )
            self._coord.emit_run_cancelled(reason="user_interrupt")
            if isinstance(exc, SystemExit):
                raise
            raise PipelineStageFailedError(
                detail="Pipeline interrupted by user",
                context={"legacy_code": "PIPELINE_INTERRUPTED"},
            ) from exc
        except PipelineStateError as e:
            # Corrupted state file / persistence failure during bootstrap.
            raise InternalError(
                detail=str(e),
                context={"legacy_code": "PIPELINE_STATE_ERROR"},
                cause=e,
            ) from e
        except RyotenkAIError as ryot_exc:
            # Stage/loop failure: emit terminal RunFailedEvent before propagating.
            self._coord.emit_run_failed(ryot_exc)
            raise
        except Exception as e:
            # Unexpected: route through the loop helper for consistent recording,
            # then raise the typed exception it produced.
            typed_exc = self._stage_execution_loop.handle_unexpected_error_outside_loop(
                e, mlflow_manager=self._mlflow_manager,
            )
            self._coord.emit_run_failed(typed_exc)
            raise typed_exc from e
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
            # Phase 8: hand emitter close + MLflow finalize + registry
            # deregister to the coordinator. ``finalize`` is idempotent
            # and never raises.
            try:
                self._coord.finalize(pipeline_success=pipeline_success)
            except Exception:
                logger.exception("[CLEANUP] run lifecycle finalize failed")

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

        # Keep error-recovery readable fields in sync with the preparator.
        self.run_directory = prepared.run_directory
        self._state_store = prepared.state_store
        self._log_layout = prepared.log_layout

        # Acquire run.lock so release is impossible to forget (Invariant #1).
        self._run_lock_guard = RunLockGuard(prepared.state_store.lock_path)
        self._run_lock_guard.__enter__()

        # Fork the run-scoped context into per-attempt context — no leaks back.
        self.context = self.context.fork(
            attempt_id=prepared.attempt.attempt_id,
            attempt_no=prepared.attempt.attempt_no,
            attempt_directory=prepared.attempt_directory,
            logical_run_id=prepared.logical_run_id,
            run_directory=prepared.run_directory,
            forced_stages=set(prepared.forced_stage_names),
        )

        # Register attempt — flips state.attempts/active_attempt_id atomically.
        self._attempt_controller.register_attempt(prepared.attempt)

        # Invalidate lineage from restart point, then restore remaining lineage.
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
        """Fire early GPU release after MODEL_RETRIEVER (provider-config dependent)."""
        if stage_name == StageNames.MODEL_RETRIEVER:
            self._maybe_early_release_gpu()

    def _on_shutdown_signal(self, signal_name: str) -> None:
        """Record interrupt name so cleanup can skip GPU disconnect on SIGINT."""
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
        self._context_propagator.sync_root_from_stage(context=ctx, stage_name=stage_name, outputs=outputs)

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
        """Fail fast when mandatory MLflow setup/connectivity is not available.

        Phase A2 Batch 7: ``ensure_preflight`` now raises typed exceptions
        (:class:`ConfigInvalidError` for missing manager,
        :class:`ProviderUnavailableError` for connectivity failures).
        Wrap them in a :class:`LaunchPreparationError` so the run() catch
        path's rejection-recording works uniformly.
        """
        try:
            self._mlflow_attempt.ensure_preflight()
        except RyotenkAIError as exc:
            legacy_code = exc.context.get("legacy_code") if isinstance(exc.context, dict) else None
            raise _wrap_as_launch_error(exc, state=state, legacy_code=legacy_code) from exc
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

    # ------------------------------------------------------------------
    # Event-emitter wiring (Phase 3-8) — thin delegates to coordinator.
    # The methods below preserve the orchestrator's public surface used
    # by ``test_orchestrator_emitter_wiring`` and any caller that still
    # invokes the per-emission helpers by name. The actual lifecycle
    # work lives in :class:`RunLifecycleCoordinator`.
    # ------------------------------------------------------------------

    def _wire_emitter_into_stages(self) -> None:
        """Push the active emitter onto every stage that exposes ``set_emitter``.

        Stages are constructed inside :class:`PipelineBootstrap` before
        the canonical run directory is known; the coordinator only
        builds the emitter once :class:`LaunchPreparator` resolves
        the directory. Stages that don't expose ``set_emitter`` keep
        their construction-time defaults (no emitter wired).
        """
        emitter = self._coord.emitter
        if emitter is None:
            return
        for stage in self.stages:
            setter = getattr(stage, "set_emitter", None)
            if callable(setter):
                try:
                    setter(emitter)
                except Exception as exc:
                    logger.warning(
                        "[Orchestrator] failed to wire emitter into %s: %s",
                        getattr(stage, "stage_name", type(stage).__name__),
                        exc,
                    )

    def _ensure_event_emitter(self) -> None:
        """Build the :class:`ControlEventEmitter` once ``run_directory`` is known.

        Falls back to ``settings.runs_base_dir / run_ctx.name`` when no
        run directory was ever resolved — the directory is created on
        demand by the coordinator.
        """
        run_directory = self.run_directory
        if run_directory is None:
            run_directory = self.settings.runs_base_dir / self.run_ctx.name
        self._coord.bind_run_directory(run_directory)

    def _emit_run_started(self, *, config_hashes: dict[str, str]) -> None:
        self._coord.emit_run_started(config_hashes=config_hashes)

    def _emit_run_completed(self, *, duration_s: float, status: str) -> None:
        self._coord.emit_run_completed(duration_s=duration_s, status=status)

    def _emit_run_failed(self, exc: BaseException) -> None:
        self._coord.emit_run_failed(exc)

    def _emit_run_cancelled(self, *, reason: str) -> None:
        self._coord.emit_run_cancelled(reason=reason)

    # ----- Suppliers consumed by RunLifecycleCoordinator -------------

    def _derive_algorithm(self) -> str:
        """Algorithm string for :class:`RunStartedPayload`; falls back to ``sft``."""
        try:
            strategies = getattr(self.config.training, "strategies", None) or []
            if strategies:
                value = getattr(strategies[0], "strategy_type", None)
                if isinstance(value, str) and value in {"sft", "cpt", "dpo", "grpo", "sapo"}:
                    return value
        except Exception:  # pragma: no cover — defensive
            pass
        return "sft"

    def _derive_dataset_id(self) -> str:
        """Dataset identifier from the configured registry (first key)."""
        try:
            datasets = getattr(self.config, "datasets", None) or {}
            if datasets:
                return next(iter(datasets.keys()))
        except Exception:  # pragma: no cover — defensive
            pass
        return "unknown"

    def _derive_model_id(self) -> str:
        """Model identifier from ``config.model.name``."""
        try:
            model_obj = getattr(self.config, "model", None)
            value = getattr(model_obj, "name", None) if model_obj is not None else None
            if isinstance(value, str) and value:
                return value
        except Exception:  # pragma: no cover — defensive
            pass
        return "unknown"

    def _safe_attempt_mlflow_run_id(self) -> str | None:
        try:
            if self._attempt_controller.has_active_attempt:
                return self._attempt_controller.active_attempt.pipeline_attempt_mlflow_run_id
        except Exception:  # pragma: no cover — defensive
            pass
        return None

    def _safe_active_stage_name(self) -> str | None:
        """Name of the currently-executing stage from :class:`StageExecutionLoop`."""
        loop = getattr(self, "_stage_execution_loop", None)
        if loop is None:
            return None
        candidate = getattr(loop, "current_stage_name", None)
        if isinstance(candidate, str):
            return candidate
        return None


def run_pipeline(config_path: str) -> int:
    """
    Developer-convenience entry point (``python -m src.pipeline.orchestrator``).

    The user-facing CLI (``ryotenkai run start``) goes through the
    project adapter; this helper exists only for ad-hoc debugging.

    Args:
        config_path: Path to pipeline configuration file

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    from ryotenkai_shared.config.loader import load_pipeline_config

    try:
        cfg = load_pipeline_config(config_path)
        orchestrator = PipelineOrchestrator(config=cfg)
        orchestrator.run()
        logger.info("Pipeline execution completed successfully")
        return 0

    except KeyboardInterrupt:
        # Fallback: direct KBI without a signal handler registered (rare / legacy path).
        logger.warning("\nPipeline interrupted by user")
        return EXIT_CODE_SIGINT
    except RyotenkAIError as exc:
        logger.error(f"Pipeline execution failed: {exc.detail or exc!s}")
        return 1
    except Exception as e:
        logger.exception(f"Unexpected error in pipeline: {e}")
        return 1


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.pipeline.orchestrator <config_path>")
        sys.exit(2)
    sys.exit(run_pipeline(sys.argv[1]))
