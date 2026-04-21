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
from src.constants import PROVIDER_RUNPOD
from src.pipeline.artifacts import (
    StageArtifactCollector,
)
from src.pipeline.config_drift import ConfigDriftValidator
from src.pipeline.constants import (
    EXIT_CODE_SIGINT,
    SEPARATOR_CHAR,
    SEPARATOR_LINE_WIDTH,
)
from src.pipeline.context import ContextPropagator, PipelineContext, StageInfoLogger
from src.pipeline.domain import RunContext
from src.pipeline.execution import StageExecutionLoop
from src.pipeline.executor import StagePlanner, is_inference_runtime_healthy
from src.pipeline.launch import LaunchPreparationError, LaunchPreparator, PreparedAttempt
from src.pipeline.mlflow_attempt import MLflowAttemptManager
from src.pipeline.reporting import ExecutionSummaryReporter
from src.pipeline.stages import (
    DatasetValidator,
    GPUDeployer,
    InferenceDeployer,
    ModelEvaluator,
    ModelRetriever,
    PipelineContextKeys,
    StageNames,
    TrainingMonitor,
)
from src.pipeline.stages.gpu_deployer import IEarlyReleasable
from src.pipeline.state import (
    AttemptController,
    PipelineAttemptState,
    PipelineState,
    PipelineStateError,
    PipelineStateStore,
)
from src.pipeline.state.run_lock_guard import RunLockGuard
from src.pipeline.validation.artifact_manager import ValidationArtifactManager

# Runtime re-export — several tests patch ``src.pipeline.orchestrator.MLflowManager``.
# Keeping this symbol here lets those tests keep working without migrating them yet.
from src.training.managers.mlflow_manager import MLflowManager  # noqa: TC001
from src.utils.config import PipelineConfig, Secrets, load_config, load_secrets, validate_strategy_chain
from src.utils.logger import logger
from src.utils.result import AppError, Err, Result

if TYPE_CHECKING:
    from src.pipeline.stages.base import PipelineStage
    from src.utils.logs_layout import LogLayout

# Status literals used in MLflow event attributes.
_STATUS_FAILED = "failed"
_STATUS_PASSED = "passed"
_STATUS_RUNNING = "running"
_STATUS_STARTED = "started"


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
        """
        Initialize the orchestrator with configuration.

        Args:
            config_path: Path to pipeline configuration YAML file
            run_directory: Explicit run directory (resume / restart).
            settings: Runtime settings (env vars snapshot). Auto-loaded if not provided.
        """
        self.settings: RuntimeSettings = settings or load_runtime_settings()
        # Single source of truth for run naming (no env fallbacks, no legacy).
        # NOTE: Do not name this attribute `run` — it would shadow PipelineOrchestrator.run().
        self.run_ctx: RunContext = RunContext.create()
        self.logical_run_id: str | None = None
        self.run_directory: Path | None = run_directory
        self.attempt_directory: Path | None = None
        self._log_layout: LogLayout | None = None
        self._state_store: PipelineStateStore | None = None
        # Single writer of PipelineState / active attempt / lineage.
        # save_fn is a closure that reads ``self._state_store`` at call time —
        # that way ``_state_store`` can be bound by LaunchPreparator later
        # without re-creating the controller.
        self._attempt_controller: AttemptController = AttemptController(
            save_fn=self._persist_state,
            run_ctx=self.run_ctx,
        )
        # Ownership of the on-disk run.lock. Kept as a RunLockGuard so the
        # finally block can rely on context-manager semantics (Invariant #1).
        self._run_lock_guard: RunLockGuard | None = None

        logger.info("Initializing Pipeline Orchestrator")

        # Load configuration
        try:
            self.config_path = config_path  # Save for later use
            self.config: PipelineConfig = load_config(config_path)
            self.secrets: Secrets = load_secrets()

            # Canonical HuggingFace auth token for the entire process.
            # All HuggingFace integrations must rely on HF_TOKEN only.
            import os

            # os.environ requires str values; tests may inject MagicMock secrets.
            os.environ["HF_TOKEN"] = str(self.secrets.hf_token)

            # Provider-specific secrets (required only when provider is active)
            try:
                active_provider = self.config.get_active_provider_name()
            except (ValueError, AttributeError):
                # Config may be incomplete in tests or during initialization
                active_provider = None

            if active_provider == PROVIDER_RUNPOD and not getattr(self.secrets, "runpod_api_key", None):
                raise ValueError(
                    f"RUNPOD_API_KEY is required when using provider {PROVIDER_RUNPOD!r}. "
                    "Set it via environment variable RUNPOD_API_KEY or in config/secrets.env."
                )

            # Inference-only provider: RunPod Serverless (training provider may be different).
            inference_cfg = getattr(self.config, "inference", None)
            if (
                getattr(inference_cfg, "enabled", False) is True
                and getattr(inference_cfg, "provider", None) in {PROVIDER_RUNPOD}
                and not getattr(self.secrets, "runpod_api_key", None)
            ):
                raise ValueError(
                    f"RUNPOD_API_KEY is required when using inference.provider={getattr(inference_cfg, 'provider', None)!r}. "
                    "Set it via environment variable RUNPOD_API_KEY or in config/secrets.env."
                )

            # Fail-fast: validate that all secrets required by enabled evaluation plugins
            # are present in secrets.env before any pipeline stage runs.
            from src.config.validators.runtime import validate_eval_plugin_secrets

            validate_eval_plugin_secrets(self.config, self.secrets)

            logger.info("Configuration loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise

        # Check strategy chain EARLY (before any stages run)
        strategies = self.config.training.strategies
        if strategies:
            validation = validate_strategy_chain(strategies)
            if validation.is_failure():
                error = validation.unwrap_err()
                chain_str = " -> ".join(s.strategy_type.upper() for s in strategies)
                logger.error(f"Invalid strategy chain: {chain_str}")
                logger.error(f"   Error: {error}")
                raise ValueError(f"Invalid strategy chain: {error}")
            chain_str = " -> ".join(s.strategy_type.upper() for s in strategies)
            logger.info(f"Strategy chain checked: {chain_str}")

        # Pipeline context (shared data between stages).
        # PipelineContext inherits from dict — existing stages that accept
        # ``dict[str, Any]`` keep working without changes.
        self.context: PipelineContext = PipelineContext(
            {
                PipelineContextKeys.CONFIG_PATH: str(config_path),
                PipelineContextKeys.RUN: self.run_ctx,
            }
        )

        # MLflow lifecycle manager (extracted from orchestrator).
        # Owns MLflowManager + root/attempt runs; orchestrator keeps a single
        # collaborator instead of four scattered attributes.
        self._mlflow_attempt = MLflowAttemptManager(self.config, self.config_path)
        self._shutdown_signal_name: str | None = None
        self._cleanup_done: bool = False

        # Stage artifact collectors — one per stage, keyed by StageNames value
        self._collectors: dict[str, StageArtifactCollector] = self._init_collectors()
        # Validation artifact accumulation (extracted from orchestrator)
        self._validation_artifact_mgr = ValidationArtifactManager(
            collectors=self._collectors,
            context=self.context,
        )
        # Stage context propagation (extracted from orchestrator).
        self._context_propagator = ContextPropagator(self._validation_artifact_mgr)
        # Post-stage MLflow info logging (extracted from orchestrator).
        self._stage_info_logger = StageInfoLogger()
        # Config hash / drift validation (extracted from orchestrator).
        self._config_drift = ConfigDriftValidator(self.config)
        # End-of-pipeline reporting (extracted from orchestrator).
        self._summary_reporter = ExecutionSummaryReporter(self.config)

        # Initialize stages (after context + collectors, since stages may reference validation mgr)
        self.stages: list[PipelineStage] = self._init_stages()
        logger.info(f"Initialized {len(self.stages)} pipeline stages")

        # Stage planner: pure stage-ordering logic (extracted from orchestrator).
        # Depends on the finalised self.stages + self.config, so instantiate here.
        self._stage_planner = StagePlanner(self.stages, self.config)

        # Launch preparator: owns state-store creation, drift validation, and
        # per-attempt dir/log layout. Returns a frozen PreparedAttempt — the
        # orchestrator's _prepare_stateful_attempt reads that rather than
        # mutating a dozen instance fields.
        self._launch_preparator = LaunchPreparator(
            config_path=self.config_path,
            run_ctx=self.run_ctx,
            settings=self.settings,
            stages=self.stages,
            stage_planner=self._stage_planner,
            config_drift=self._config_drift,
            attempt_controller=self._attempt_controller,
        )

        # Stage execution loop: the for-loop + outcome handlers + exception
        # boundary for mid-run failures. Runs AFTER _prepare_stateful_attempt
        # produces a PreparedAttempt.
        self._stage_execution_loop = StageExecutionLoop(
            stages=self.stages,
            collectors=self._collectors,
            attempt_controller=self._attempt_controller,
            stage_planner=self._stage_planner,
            context_propagator=self._context_propagator,
            stage_info_logger=self._stage_info_logger,
            validation_artifact_mgr=self._validation_artifact_mgr,
            summary_reporter=self._summary_reporter,
            on_stage_completed=self._on_stage_completed,
            on_shutdown_signal=self._on_shutdown_signal,
        )

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

    def _init_stages(self) -> list[PipelineStage]:
        """Initialize all pipeline stages in execution order."""
        from src.pipeline.stages.dataset_validator import DatasetValidatorEventCallbacks

        # Create callbacks for DatasetValidator (MLflow integration)
        vam = self._validation_artifact_mgr
        validator_callbacks = DatasetValidatorEventCallbacks(
            on_dataset_scheduled=vam.on_dataset_scheduled,
            on_dataset_loaded=vam.on_dataset_loaded,
            on_validation_completed=vam.on_validation_completed,
            on_validation_failed=vam.on_validation_failed,
            on_plugin_start=vam.on_plugin_start,
            on_plugin_complete=vam.on_plugin_complete,
            on_plugin_failed=vam.on_plugin_failed,
        )

        stages: list[PipelineStage] = [
            DatasetValidator(self.config, secrets=self.secrets, callbacks=validator_callbacks),
            GPUDeployer(self.config, self.secrets),  # Universal provider-based deployer
            TrainingMonitor(self.config, secrets=self.secrets),
            ModelRetriever(self.config, self.secrets),
            InferenceDeployer(self.config, self.secrets),
            ModelEvaluator(self.config, self.secrets),
        ]

        return stages

    def _init_collectors(self) -> dict[str, StageArtifactCollector]:
        """Create one StageArtifactCollector per pipeline stage."""
        from src.pipeline.stages.constants import StageNames

        return {
            StageNames.DATASET_VALIDATOR: StageArtifactCollector(
                stage="dataset_validator",
                artifact_name="dataset_validator_results.json",
            ),
            StageNames.GPU_DEPLOYER: StageArtifactCollector(
                stage="gpu_deployer",
                artifact_name="gpu_deployer_results.json",
            ),
            StageNames.TRAINING_MONITOR: StageArtifactCollector(
                stage="training_monitor",
                artifact_name="training_monitor_results.json",
            ),
            StageNames.MODEL_RETRIEVER: StageArtifactCollector(
                stage="model_retriever",
                artifact_name="model_retriever_results.json",
            ),
            StageNames.INFERENCE_DEPLOYER: StageArtifactCollector(
                stage="inference_deployer",
                artifact_name="inference_deployer_results.json",
            ),
            StageNames.MODEL_EVALUATOR: StageArtifactCollector(
                stage="model_evaluator",
                artifact_name="evaluation_results.json",
            ),
        }

    def _setup_mlflow(self) -> MLflowManager | None:
        """Setup MLflow for pipeline event logging (delegates to MLflowAttemptManager)."""
        return self._mlflow_attempt.bootstrap()

    def _get_mlflow_run_id(self) -> str | None:
        """Best-effort MLflow run_id (delegates to MLflowAttemptManager)."""
        return self._mlflow_attempt.get_run_id()

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

    def _normalize_stage_ref(self, stage_ref: str | int | None) -> str:
        return self._stage_planner.normalize_stage_ref(stage_ref)

    def _get_stage_index(self, stage_name: str) -> int:
        return self._stage_planner.get_stage_index(stage_name)

    def _forced_stage_names(self, *, start_stage_name: str) -> set[str]:
        return self._stage_planner.forced_stage_names(start_stage_name=start_stage_name)

    def _compute_enabled_stage_names(self, *, start_stage_name: str) -> list[str]:
        return self._stage_planner.compute_enabled_stage_names(start_stage_name=start_stage_name)

    def _derive_resume_stage(self, state: PipelineState) -> str | None:
        return self._stage_planner.derive_resume_stage(state)

    def _validate_config_drift(
        self,
        *,
        state: PipelineState,
        start_stage_name: str,
        config_hashes: dict[str, str],
        resume: bool,
    ) -> AppError | None:
        return self._config_drift.validate_drift(
            state=state,
            start_stage_name=start_stage_name,
            config_hashes=config_hashes,
            resume=resume,
        )

    def _sync_root_context_from_stage(self, stage_name: str, outputs: dict[str, Any]) -> None:
        self._context_propagator.sync_root_from_stage(
            context=self.context, stage_name=stage_name, outputs=outputs
        )

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

    def _extract_restart_outputs(self, stage_name: str) -> dict[str, Any]:
        return self._context_propagator.extract_restart_outputs(
            context=self.context, stage_name=stage_name
        )

    def _get_stage_skip_reason(self, stage_name: str) -> str | None:
        return self._context_propagator.get_stage_skip_reason(
            context=self.context, stage_name=stage_name
        )

    # ------------------------------------------------------------------
    # AttemptController backplane
    # ------------------------------------------------------------------
    # Read-only views onto the controller-owned state, kept for backward
    # compatibility with callers (and tests) that still reference the old
    # private attribute names. Writes must go through the controller, not
    # these properties.

    @property
    def _pipeline_state(self) -> PipelineState | None:
        """Read-only view of the controller-owned pipeline state."""
        return self._attempt_controller.state if self._attempt_controller.has_state else None

    @property
    def _current_attempt(self) -> PipelineAttemptState | None:
        """Read-only view of the controller-owned active attempt."""
        if self._attempt_controller.has_active_attempt:
            return self._attempt_controller.active_attempt
        return None

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

    def _open_existing_root_run(self, root_run_id: str) -> Any:
        return self._mlflow_attempt.open_existing_root_run(root_run_id)

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

    def _validate_stage_prerequisites(self, *, stage_name: str, start_stage_name: str) -> AppError | None:
        return self._stage_planner.validate_stage_prerequisites(
            stage_name=stage_name,
            start_stage_name=start_stage_name,
            context=self.context,
        )

    def _is_inference_runtime_healthy(self, inference_ctx: dict[str, Any] | None = None) -> bool:
        ctx = inference_ctx if inference_ctx is not None else self.context.get(StageNames.INFERENCE_DEPLOYER, {})
        return is_inference_runtime_healthy(ctx if isinstance(ctx, dict) else None)

    def list_restart_points(self, run_dir: Path) -> list[dict[str, Any]]:
        store = PipelineStateStore(run_dir.expanduser().resolve())
        state = store.load()
        # Purely read-only inspection — do not pollute the controller's live
        # state with a snapshot from another run. This used to set
        # ``self._pipeline_state = state`` but that value was never read.
        config_hashes = self._build_config_hashes()
        points: list[dict[str, Any]] = []
        for stage in self.stages:
            stage_name = stage.stage_name
            available = True
            reason = "restart_allowed"
            mode = "fresh_only"

            if stage_name == StageNames.TRAINING_MONITOR:
                mode = "reconnect_only"
                ref = state.current_output_lineage.get(StageNames.GPU_DEPLOYER)
                gpu_outputs = ref.outputs if ref else {}
                if not all(gpu_outputs.get(key) for key in ("ssh_host", "ssh_port", "workspace_path")):
                    available = False
                    reason = "missing_gpu_deployer_outputs"
            elif stage_name == StageNames.MODEL_RETRIEVER:
                mode = "fresh_or_resume"
                ref = state.current_output_lineage.get(StageNames.GPU_DEPLOYER)
                if ref is None:
                    available = False
                    reason = "missing_gpu_deployer_outputs"
            elif stage_name == StageNames.INFERENCE_DEPLOYER:
                mode = "fresh_or_resume"
                ref = state.current_output_lineage.get(StageNames.MODEL_RETRIEVER)
                outputs = ref.outputs if ref else {}
                if not (outputs.get("hf_repo_id") or outputs.get("local_model_path")):
                    available = False
                    reason = "missing_model_retriever_outputs"
            elif stage_name == StageNames.MODEL_EVALUATOR:
                mode = "live_runtime_only"
                ref = state.current_output_lineage.get(StageNames.INFERENCE_DEPLOYER)
                if ref is None:
                    available = False
                    reason = "missing_inference_outputs"
                else:
                    if not self._is_inference_runtime_healthy(inference_ctx=dict(ref.outputs)):
                        available = False
                        reason = "inference_runtime_not_healthy"

            if state.model_dataset_config_hash:
                if state.model_dataset_config_hash != config_hashes["model_dataset"]:
                    available = False
                    reason = "training_critical_config_changed"
            elif state.training_critical_config_hash != config_hashes["training_critical"]:
                available = False
                reason = "training_critical_config_changed"

            if state.late_stage_config_hash != config_hashes["late_stage"] and stage_name not in {
                StageNames.INFERENCE_DEPLOYER,
                StageNames.MODEL_EVALUATOR,
            }:
                available = False
                reason = "late_stage_config_changed"

            points.append(
                {
                    "stage": stage_name,
                    "available": available,
                    "mode": mode,
                    "reason": reason,
                }
            )
        return points

    def _log_stage_specific_info(self, stage_name: str) -> None:
        """Log stage-specific info to MLflow after stage completion."""
        self._stage_info_logger.log(
            mlflow_manager=self._mlflow_manager,
            context=self.context,
            stage_name=stage_name,
        )

    def _fill_from_context(self, stage_name: str, collector: StageArtifactCollector) -> None:
        """Populate collector with stage-specific data read from pipeline context."""
        self._context_propagator.fill_collector_from_context(
            context=self.context, stage_name=stage_name, collector=collector
        )

    def _flush_pending_collectors(self) -> None:
        """Flush all collectors that are still open (interrupted / exception).

        Called in the finally block so every stage gets at least a status
        artifact even when the pipeline is killed mid-run.

        Stages that never started (set_started_at was never called) are skipped:
        they have no data to preserve and writing an empty interrupted artifact
        only adds noise to the report issues section.
        """
        for stage_name, collector in self._collectors.items():
            if collector.is_flushed:
                continue
            if collector._started_at is None:
                # Stage never started — skip writing empty artifact
                logger.debug("[ARTIFACT] skip flush for not-started stage %s", stage_name)
                continue
            try:
                collector.flush_interrupted(
                    started_at=collector._started_at,
                    duration_seconds=0.0,
                    context=self.context,
                )
                logger.debug("[ARTIFACT] flush_interrupted for %s", stage_name)
            except Exception as exc:
                logger.warning("[ARTIFACT] flush_interrupted failed for %s: %s", stage_name, exc)

    def _print_summary(self) -> None:
        """Print a comprehensive summary of the pipeline execution."""
        self._summary_reporter.print_summary(context=self.context)

    def _maybe_early_release_gpu(self) -> None:
        """Terminate training pod early if terminate_after_retrieval=true.

        Called by run() right after ModelRetriever completes successfully.
        Frees the training pod before InferenceDeployer / ModelEvaluator stages
        run, avoiding unnecessary GPU billing.

        Uses IEarlyReleasable protocol so any future GPU stage can opt in
        without changes here.
        """
        try:
            provider_cfg = self.config.get_provider_config()
            cleanup_cfg = provider_cfg.get("cleanup") if isinstance(provider_cfg, dict) else None
            if not (isinstance(cleanup_cfg, dict) and cleanup_cfg.get("terminate_after_retrieval") is True):
                return
        except Exception:
            return

        for stage in self.stages:
            if isinstance(stage, IEarlyReleasable):
                logger.info("[ORCHESTRATOR] terminate_after_retrieval=true: releasing training pod early.")
                stage.release()
                return

    def _cleanup_resources(self, *, success: bool = False) -> None:
        """
        Cleanup resources for ALL stages.
        Called automatically even if pipeline fails.

        Calls cleanup() on all stages in reverse order (last-first).
        This ensures proper cleanup of dependent resources.

        Args:
            success: Whether pipeline completed successfully (reserved for future use)
        """
        if self._cleanup_done:
            logger.debug("Cleanup already done, skipping duplicate call")
            return
        self._cleanup_done = True
        logger.info("Cleaning up pipeline resources...")

        if not success:
            for stage in self.stages:
                if hasattr(stage, "notify_pipeline_failure"):
                    try:
                        stage.notify_pipeline_failure()
                    except Exception as e:
                        logger.debug(f"Error notifying pipeline failure to {stage.stage_name}: {e}")

        # Policy: optionally skip provider disconnect on Ctrl+C (SIGINT).
        skip_gpu_deployer_cleanup = False
        if self._shutdown_signal_name == "SIGINT":
            try:
                provider_name = self.config.get_active_provider_name()
                provider_cfg = self.config.get_provider_config()
                cleanup_cfg = provider_cfg.get("cleanup") if isinstance(provider_cfg, dict) else None
                if isinstance(cleanup_cfg, dict) and cleanup_cfg.get("on_interrupt") is False:
                    skip_gpu_deployer_cleanup = True
                    logger.warning(
                        f"[CLEANUP] Skipping GPU provider disconnect on SIGINT "
                        f"(providers.{provider_name}.cleanup.on_interrupt=false)"
                    )
            except Exception:
                # Best-effort: never crash during cleanup due to config inspection.
                pass

        # Cleanup stages in reverse order (last-first)
        # This ensures dependent resources are cleaned up properly
        for stage in reversed(self.stages):
            try:
                if hasattr(stage, "cleanup"):
                    if skip_gpu_deployer_cleanup and getattr(stage, "stage_name", None) == StageNames.GPU_DEPLOYER:
                        continue
                    stage.cleanup()
                    logger.debug(f"Cleanup complete: {stage.stage_name}")
            except KeyboardInterrupt:
                # Second Ctrl+C during cleanup — log and continue so all stages are cleaned up.
                logger.warning(f"[CLEANUP] Interrupted during cleanup of {stage.stage_name}, continuing...")
            except Exception as e:
                logger.warning(f"Error during cleanup of {stage.stage_name}: {e}")

        logger.info("Pipeline cleanup complete")

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
        """Delegate: generate the post-pipeline experiment Markdown report."""
        ExecutionSummaryReporter.generate_experiment_report(
            run_id=run_id, mlflow_manager=self._mlflow_manager
        )

    def get_stage_by_name(self, name: str) -> PipelineStage | None:
        """Get a stage by its name."""
        for stage in self.stages:
            if stage.stage_name == name:
                return stage
        return None

    def list_stages(self) -> list[str]:
        """List all available stage names."""
        return [stage.stage_name for stage in self.stages]


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
