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
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.config.runtime import RuntimeSettings, load_runtime_settings
from src.constants import PROVIDER_RUNPOD
from src.pipeline.artifacts import (
    StageArtifactCollector,
)
from src.pipeline.artifacts.base import utc_now_iso
from src.pipeline.config_drift import ConfigDriftValidator
from src.pipeline.constants import (
    EXIT_CODE_SIGINT,
    MLFLOW_CATEGORY_PIPELINE,
    MLFLOW_SOURCE_ORCHESTRATOR,
    SEPARATOR_CHAR,
    SEPARATOR_LINE_WIDTH,
)
from src.pipeline.context import ContextPropagator, PipelineContext, StageInfoLogger
from src.pipeline.domain import RunContext
from src.pipeline.executor import StagePlanner, is_inference_runtime_healthy
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
    PipelineAttemptState,
    PipelineState,
    PipelineStateError,
    PipelineStateLoadError,
    PipelineStateStore,
    StageLineageRef,
    StageRunState,
    build_attempt_state,
    update_lineage,
)
from src.pipeline.state.run_lock_guard import RunLockGuard
from src.pipeline.state.transitioner import (
    finalize_attempt_state,
    invalidate_lineage_from,
    mark_stage_completed,
    mark_stage_failed,
    mark_stage_interrupted,
    mark_stage_running,
    mark_stage_skipped,
    restore_reused_context,
)
from src.pipeline.validation.artifact_manager import ValidationArtifactManager

# Runtime re-export — several tests patch ``src.pipeline.orchestrator.MLflowManager``.
# Keeping this symbol here lets those tests keep working without migrating them yet.
from src.training.managers.mlflow_manager import MLflowManager  # noqa: TC001
from src.utils.config import PipelineConfig, Secrets, load_config, load_secrets, validate_strategy_chain
from src.utils.logger import init_run_logging, logger, stage_logging_context
from src.utils.logs_layout import LogLayout
from src.utils.result import AppError, Err, Ok, Result

if TYPE_CHECKING:
    from src.pipeline.stages.base import PipelineStage

# Status literals used in MLflow event attributes.
_STATUS_FAILED = "failed"
_STATUS_PASSED = "passed"
_STATUS_RUNNING = "running"
_STATUS_STARTED = "started"


class LaunchPreparationError(Exception):
    """Internal error used to reject a launch before stage execution begins."""

    def __init__(
        self,
        app_error: AppError,
        *,
        state: PipelineState | None = None,
        requested_action: str | None = None,
        effective_action: str | None = None,
        start_stage_name: str | None = None,
    ) -> None:
        super().__init__(app_error.message)
        self.app_error = app_error
        self.state = state
        self.requested_action = requested_action
        self.effective_action = effective_action
        self.start_stage_name = start_stage_name


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
        self._pipeline_state: PipelineState | None = None
        self._current_attempt: PipelineAttemptState | None = None
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
        pipeline_start_time = time.time()
        current_stage_name: str | None = None
        current_stage_started_at: str | None = None
        current_stage_start_time: float | None = None

        config_hashes = self._build_config_hashes()

        try:
            state, attempt, start_idx, stop_idx, start_stage_name, enabled_stage_names = (
                self._prepare_stateful_attempt(
                    run_dir=run_dir,
                    resume=resume,
                    restart_from_stage=restart_from_stage,
                    config_hashes=config_hashes,
                )
            )

            for i, stage in enumerate(self.stages):
                stage_name = stage.stage_name

                if i < start_idx:
                    continue
                if i >= stop_idx:
                    break

                if stage_name not in enabled_stage_names:
                    self._mark_stage_skipped(
                        attempt=attempt,
                        stage_name=stage_name,
                        reason="disabled_by_config",
                    )
                    state.current_output_lineage = update_lineage(
                        state.current_output_lineage,
                        stage_name=stage_name,
                        attempt_id=attempt.attempt_id,
                        remove=True,
                    )
                    self._save_state()
                    continue

                prereq_error = self._validate_stage_prerequisites(
                    stage_name=stage_name, start_stage_name=start_stage_name
                )
                if prereq_error is not None:
                    self._mark_stage_failed(
                        attempt=attempt,
                        stage_name=stage_name,
                        error=prereq_error.message,
                        failure_kind=prereq_error.code,
                    )
                    self._finalize_attempt_state(
                        state=state,
                        attempt=attempt,
                        status=StageRunState.STATUS_FAILED,
                    )
                    self._save_state()
                    return Err(prereq_error)

                current_stage_name = stage_name
                current_stage_started_at = utc_now_iso()
                current_stage_start_time = time.time()

                collector = self._collectors.get(stage_name)
                if collector:
                    collector.set_started_at(current_stage_started_at)

                assert self._log_layout is not None
                with stage_logging_context(stage_name, self._log_layout):
                    self._mark_stage_running(attempt=attempt, stage_name=stage_name, started_at=current_stage_started_at)
                    self._record_stage_log_paths(attempt=attempt, stage_name=stage_name)
                    self._save_state()

                    logger.info(SEPARATOR_CHAR * SEPARATOR_LINE_WIDTH)
                    logger.info(f"Stage {i + 1}/{len(self.stages)}: {stage_name}")
                    logger.info(SEPARATOR_CHAR * SEPARATOR_LINE_WIDTH)

                    if self._mlflow_manager:
                        self._mlflow_manager.log_stage_start(
                            stage_name=stage_name, stage_idx=i, total_stages=len(self.stages)
                        )

                    result = stage.run(self.context)
                    stage_duration = time.time() - current_stage_start_time

                    if result.is_failure():
                        stage_err = result.unwrap_err()  # type: ignore[union-attr]
                        return self._handle_stage_failure(
                            state=state,
                            attempt=attempt,
                            stage_name=stage_name,
                            stage_idx=i,
                            stage_err=stage_err,
                            collector=collector,
                            started_at=current_stage_started_at,
                            duration_seconds=stage_duration,
                        )

                    stage_result = result.unwrap()
                    if stage_result is not None:
                        self.context.update(stage_result)

                    if stage_name == StageNames.MODEL_RETRIEVER:
                        self._maybe_early_release_gpu()

                    self._handle_stage_success(
                        state=state,
                        attempt=attempt,
                        stage_name=stage_name,
                        stage_idx=i,
                        collector=collector,
                        started_at=current_stage_started_at,
                        duration_seconds=stage_duration,
                    )

                current_stage_name = None
                current_stage_started_at = None
                current_stage_start_time = None

            pipeline_success = True
            pipeline_duration = time.time() - pipeline_start_time
            self._finalize_successful_run(state=state, attempt=attempt, pipeline_duration=pipeline_duration)
            return Ok(self.context)

        except (KeyboardInterrupt, SystemExit) as exc:
            is_system_exit = isinstance(exc, SystemExit)
            self._handle_interrupt(
                current_stage_name=current_stage_name,
                current_stage_started_at=current_stage_started_at,
            )
            if is_system_exit:
                raise
            return Err(AppError(message="Pipeline interrupted by user", code="PIPELINE_INTERRUPTED"))
        except LaunchPreparationError as exc:
            logger.error(exc.app_error.message)
            if (
                exc.state is not None
                and exc.requested_action is not None
                and exc.effective_action is not None
                and exc.start_stage_name is not None
            ):
                self._record_launch_rejection_attempt(
                    state=exc.state,
                    requested_action=exc.requested_action,
                    effective_action=exc.effective_action,
                    start_stage_name=exc.start_stage_name,
                    config_hashes=config_hashes,
                    app_error=exc.app_error,
                )
            return Err(exc.app_error)
        except PipelineStateError as e:
            return Err(AppError(message=str(e), code="PIPELINE_STATE_ERROR"))
        except Exception as e:
            return self._handle_unexpected_error(e)
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
    ) -> tuple[PipelineState, PipelineAttemptState, int, int, str, list[str]]:
        """Boot state, acquire run lock, materialise attempt, restore lineage, open MLflow.

        Runs end-to-end before the stage loop. Leaves the orchestrator in a
        fully-configured state ready to execute stages. All side effects are
        captured in ``self._state_store``, ``self.run_directory``,
        ``self._current_attempt``, ``self.attempt_directory``, ``self._log_layout``,
        ``self.context``, and ``self._run_lock_guard``.
        """
        state, requested_action, effective_action, start_stage_name = self._bootstrap_pipeline_state(
            run_dir=run_dir,
            resume=resume,
            restart_from_stage=restart_from_stage,
            config_hashes=config_hashes,
        )
        assert self._state_store is not None
        assert self.run_directory is not None
        state.training_critical_config_hash = config_hashes["training_critical"]
        state.late_stage_config_hash = config_hashes["late_stage"]
        state.model_dataset_config_hash = config_hashes["model_dataset"]

        start_idx = self._get_stage_index(start_stage_name)
        stop_idx = len(self.stages)

        # Acquire the run.lock via RunLockGuard so release is impossible
        # to forget in the finally block (Invariant #1 of the architecture).
        self._run_lock_guard = RunLockGuard(self._state_store.lock_path)
        self._run_lock_guard.__enter__()

        enabled_stage_names = self._compute_enabled_stage_names(start_stage_name=start_stage_name)
        attempt = build_attempt_state(
            state=state,
            run_ctx=self.run_ctx,
            requested_action=requested_action,
            effective_action=effective_action,
            restart_from_stage=start_stage_name,
            enabled_stage_names=enabled_stage_names,
            training_critical_config_hash=config_hashes["training_critical"],
            late_stage_config_hash=config_hashes["late_stage"],
            model_dataset_config_hash=config_hashes["model_dataset"],
        )
        self._current_attempt = attempt
        self.attempt_directory = self._state_store.next_attempt_dir(attempt.attempt_no)
        self._log_layout = LogLayout(self.attempt_directory)
        init_run_logging(self.run_ctx.name, log_dir=self.attempt_directory)

        # Fork the run-scoped context (CONFIG_PATH, RUN) into a per-attempt
        # context with fresh attempt-scoped keys. PipelineContext.fork
        # guarantees no state leaks back to the original.
        assert self.run_directory is not None
        self.context = self.context.fork(
            attempt_id=attempt.attempt_id,
            attempt_no=attempt.attempt_no,
            attempt_directory=self.attempt_directory,
            logical_run_id=state.logical_run_id,
            run_directory=self.run_directory,
            forced_stages=set(self._forced_stage_names(start_stage_name=start_stage_name)),
        )

        state.attempts.append(attempt)
        state.active_attempt_id = attempt.attempt_id
        state.pipeline_status = StageRunState.STATUS_RUNNING

        current_lineage = self._invalidate_lineage_from(
            lineage=state.current_output_lineage,
            start_stage_name=start_stage_name,
        )
        self._restore_reused_context(
            attempt=attempt,
            lineage=state.current_output_lineage,
            start_stage_name=start_stage_name,
            enabled_stage_names=enabled_stage_names,
        )
        state.current_output_lineage = current_lineage
        self._save_state()

        self._setup_mlflow_for_attempt(state=state, attempt=attempt, start_stage_idx=start_idx)
        self._ensure_mlflow_preflight(state=state)

        return state, attempt, start_idx, stop_idx, start_stage_name, enabled_stage_names

    def _finalize_successful_run(
        self,
        *,
        state: PipelineState,
        attempt: PipelineAttemptState,
        pipeline_duration: float,
    ) -> None:
        """Mark attempt completed, emit summary + MLflow completion event."""
        self._finalize_attempt_state(
            state=state,
            attempt=attempt,
            status=StageRunState.STATUS_COMPLETED,
            completed_at=utc_now_iso(),
        )
        self._save_state()

        logger.info("\n" + SEPARATOR_CHAR * SEPARATOR_LINE_WIDTH)
        logger.info("Pipeline completed successfully!")
        logger.info(SEPARATOR_CHAR * SEPARATOR_LINE_WIDTH)

        if self._mlflow_manager:
            self._mlflow_manager.log_event_complete(
                f"Pipeline completed successfully in {pipeline_duration:.1f}s",
                category=MLFLOW_CATEGORY_PIPELINE,
                source=MLFLOW_SOURCE_ORCHESTRATOR,
                duration_seconds=pipeline_duration,
            )
            self._mlflow_manager.set_tags({"pipeline.status": "completed"})
            self._mlflow_manager.log_params({"pipeline.duration_seconds": pipeline_duration})

        self._print_summary()

    def _handle_interrupt(
        self,
        *,
        current_stage_name: str | None,
        current_stage_started_at: str | None,
    ) -> None:
        """Mark the attempt/stage interrupted and record the MLflow warning event."""
        self._shutdown_signal_name = self._shutdown_signal_name or "SIGINT"
        logger.warning("\nPipeline interrupted by user")
        if self._current_attempt:
            completed_at = utc_now_iso()
            self._current_attempt.completed_at = completed_at
            if current_stage_name and current_stage_started_at:
                self._mark_stage_interrupted(
                    attempt=self._current_attempt,
                    stage_name=current_stage_name,
                    started_at=current_stage_started_at,
                )
            if self._pipeline_state:
                self._finalize_attempt_state(
                    state=self._pipeline_state,
                    attempt=self._current_attempt,
                    status=StageRunState.STATUS_INTERRUPTED,
                    completed_at=completed_at,
                )
                self._save_state()
        if self._mlflow_manager:
            self._mlflow_manager.log_event_warning(
                "Pipeline interrupted by user",
                category=MLFLOW_CATEGORY_PIPELINE,
                source=MLFLOW_SOURCE_ORCHESTRATOR,
            )
            self._mlflow_manager.set_tags({"pipeline.status": "interrupted"})

    def _handle_unexpected_error(self, e: Exception) -> Result[dict[str, Any], AppError]:
        """Mark the attempt failed on an unexpected exception and surface it as an Err."""
        logger.exception(f"Unexpected error in pipeline: {e}")
        completed_at = utc_now_iso()
        if self._current_attempt:
            self._current_attempt.completed_at = completed_at
        if self._pipeline_state:
            if self._current_attempt is not None:
                self._finalize_attempt_state(
                    state=self._pipeline_state,
                    attempt=self._current_attempt,
                    status=StageRunState.STATUS_FAILED,
                    completed_at=completed_at,
                )
            else:
                self._pipeline_state.pipeline_status = StageRunState.STATUS_FAILED
            self._save_state()
        if self._mlflow_manager:
            self._mlflow_manager.log_event_error(
                f"Unexpected error: {e!s}",
                category=MLFLOW_CATEGORY_PIPELINE,
                source=MLFLOW_SOURCE_ORCHESTRATOR,
                error_type=type(e).__name__,
            )
            self._mlflow_manager.set_tags({"pipeline.status": _STATUS_FAILED})
        return Err(
            AppError(
                message=f"Unexpected error: {e!s}",
                code="UNEXPECTED_ERROR",
                details={"exception_type": type(e).__name__},
            )
        )

    def _handle_stage_failure(
        self,
        *,
        state: PipelineState,
        attempt: PipelineAttemptState,
        stage_name: str,
        stage_idx: int,
        stage_err: AppError,
        collector: StageArtifactCollector | None,
        started_at: str,
        duration_seconds: float,
    ) -> Result[dict[str, Any], AppError]:
        """Handle a Failure result from a stage: log, flush artifacts, mark failed, return Err."""
        logger.error(f"Pipeline failed at stage {stage_idx + 1}: {stage_name}")
        logger.error(f"Error: {stage_err}")
        if self._mlflow_manager:
            self._mlflow_manager.log_stage_failed(
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
                    context=self.context,
                )
        self._mark_stage_failed(
            attempt=attempt,
            stage_name=stage_name,
            error=str(stage_err),
            failure_kind=getattr(stage_err, "code", "STAGE_FAILED"),
            outputs=(
                self._validation_artifact_mgr.build_dataset_validation_state_outputs(error=str(stage_err))
                if stage_name == StageNames.DATASET_VALIDATOR
                else None
            ),
        )
        self._finalize_attempt_state(
            state=state, attempt=attempt, status=StageRunState.STATUS_FAILED
        )
        state.current_output_lineage = update_lineage(
            state.current_output_lineage,
            stage_name=stage_name,
            attempt_id=attempt.attempt_id,
            remove=True,
        )
        self._save_state()
        return Err(
            AppError(
                message=f"Stage '{stage_name}' failed: {stage_err.message}",
                code="STAGE_FAILED",
                details={**stage_err.to_log_dict(), "stage_name": stage_name, "stage_idx": stage_idx},
            )
        )

    def _handle_stage_success(
        self,
        *,
        state: PipelineState,
        attempt: PipelineAttemptState,
        stage_name: str,
        stage_idx: int,
        collector: StageArtifactCollector | None,
        started_at: str,
        duration_seconds: float,
    ) -> None:
        """Handle a successful stage completion: flush artifacts, log, update lineage/state."""
        outputs = self._extract_restart_outputs(stage_name)
        skip_reason = self._get_stage_skip_reason(stage_name)

        if collector and not collector.is_flushed:
            if stage_name == StageNames.DATASET_VALIDATOR:
                self._validation_artifact_mgr.flush_validation_artifact(
                    started_at=started_at, duration_seconds=duration_seconds
                )
            else:
                self._fill_from_context(stage_name, collector)
                collector.flush_ok(
                    started_at=started_at,
                    duration_seconds=duration_seconds,
                    context=self.context,
                )

        if self._mlflow_manager:
            self._log_stage_specific_info(stage_name)
            self._mlflow_manager.log_stage_complete(
                stage_name=stage_name, stage_idx=stage_idx, duration_seconds=duration_seconds
            )

        if skip_reason is not None:
            self._mark_stage_skipped(
                attempt=attempt, stage_name=stage_name, reason=skip_reason, outputs=outputs
            )
            state.current_output_lineage = update_lineage(
                state.current_output_lineage,
                stage_name=stage_name,
                attempt_id=attempt.attempt_id,
                remove=True,
            )
        else:
            self._mark_stage_completed(attempt=attempt, stage_name=stage_name, outputs=outputs)
            state.current_output_lineage = update_lineage(
                state.current_output_lineage,
                stage_name=stage_name,
                attempt_id=attempt.attempt_id,
                outputs=outputs,
            )

        self._save_state()
        logger.info(f"Stage {stage_idx + 1} completed successfully ({duration_seconds:.1f}s)")

    def _bootstrap_pipeline_state(
        self,
        *,
        run_dir: Path | None,
        resume: bool,
        restart_from_stage: str | int | None,
        config_hashes: dict[str, str],
    ) -> tuple[PipelineState, str, str, str]:
        requested_run_dir = run_dir or self.run_directory
        normalized_restart = self._normalize_stage_ref(restart_from_stage) if restart_from_stage is not None else None

        if requested_run_dir is not None:
            resolved_run_dir = requested_run_dir.expanduser().resolve()
        else:
            runs_base = self.settings.runs_base_dir
            resolved_run_dir = (runs_base / self.run_ctx.name).resolve()

        self.run_directory = resolved_run_dir
        self._state_store = PipelineStateStore(resolved_run_dir)

        if self._state_store.exists():
            state = self._state_store.load()
            self._pipeline_state = state
            self.logical_run_id = state.logical_run_id
            requested_action = "resume" if resume else ("restart" if normalized_restart else "fresh")
            start_stage_name = normalized_restart or (
                self._derive_resume_stage(state) if resume else self.stages[0].stage_name
            )
            if start_stage_name is None:
                raise LaunchPreparationError(
                    AppError(
                        message="No resumable stage found in pipeline_state.json",
                        code="RESUME_NOT_AVAILABLE",
                    ),
                    state=state,
                    requested_action=requested_action,
                    effective_action=("auto_resume" if resume and normalized_restart is None else requested_action),
                    start_stage_name=self.stages[0].stage_name,
                )
            drift_error = self._validate_config_drift(
                state=state,
                start_stage_name=start_stage_name,
                config_hashes=config_hashes,
                resume=resume,
            )
            if drift_error is not None:
                raise LaunchPreparationError(
                    drift_error,
                    state=state,
                    requested_action=requested_action,
                    effective_action=("auto_resume" if resume and normalized_restart is None else requested_action),
                    start_stage_name=start_stage_name,
                )
            return (
                state,
                requested_action,
                ("auto_resume" if resume and normalized_restart is None else requested_action),
                start_stage_name,
            )

        if resume or normalized_restart is not None:
            raise PipelineStateLoadError(f"Missing pipeline_state.json in run directory: {resolved_run_dir}")

        logical_run_id = resolved_run_dir.name if run_dir is not None else self.run_ctx.name
        resolved_run_dir.mkdir(parents=True, exist_ok=True)
        state = self._state_store.init_state(
            logical_run_id=logical_run_id,
            config_path=str(self.config_path.expanduser().resolve()),
            training_critical_config_hash=config_hashes["training_critical"],
            late_stage_config_hash=config_hashes["late_stage"],
            model_dataset_config_hash=config_hashes["model_dataset"],
        )
        self._pipeline_state = state
        self.logical_run_id = logical_run_id
        return state, "fresh", "fresh", self.stages[0].stage_name

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

    def _record_launch_rejection_attempt(
        self,
        *,
        state: PipelineState,
        requested_action: str,
        effective_action: str,
        start_stage_name: str,
        config_hashes: dict[str, str],
        app_error: AppError,
    ) -> None:
        if self._state_store is None:
            return
        enabled_stage_names = self._compute_enabled_stage_names(start_stage_name=start_stage_name)
        attempt = build_attempt_state(
            state=state,
            run_ctx=self.run_ctx,
            requested_action=requested_action,
            effective_action=effective_action,
            restart_from_stage=start_stage_name,
            enabled_stage_names=enabled_stage_names,
            training_critical_config_hash=config_hashes["training_critical"],
            late_stage_config_hash=config_hashes["late_stage"],
            model_dataset_config_hash=config_hashes["model_dataset"],
        )
        self._current_attempt = attempt
        self.attempt_directory = self._state_store.next_attempt_dir(attempt.attempt_no)
        init_run_logging(self.run_ctx.name, log_dir=self.attempt_directory)
        logger.info(SEPARATOR_CHAR * SEPARATOR_LINE_WIDTH)
        logger.error(f"Launch rejected before stage execution: {app_error}")
        logger.info(SEPARATOR_CHAR * SEPARATOR_LINE_WIDTH)
        attempt.error = app_error.message
        state.attempts.append(attempt)
        state.active_attempt_id = attempt.attempt_id
        self._finalize_attempt_state(
            state=state,
            attempt=attempt,
            status=StageRunState.STATUS_FAILED,
            completed_at=utc_now_iso(),
        )
        self._save_state()

    def _invalidate_lineage_from(
        self,
        *,
        lineage: dict[str, StageLineageRef],
        start_stage_name: str,
    ) -> dict[str, StageLineageRef]:
        return invalidate_lineage_from(
            lineage=lineage,
            stage_names=[s.stage_name for s in self.stages],
            start_stage_name=start_stage_name,
        )

    def _restore_reused_context(
        self,
        *,
        attempt: PipelineAttemptState,
        lineage: dict[str, StageLineageRef],
        start_stage_name: str,
        enabled_stage_names: list[str],
    ) -> None:
        propagator = self._context_propagator

        def _sync(ctx: dict[str, Any], stage_name: str, outputs: dict[str, Any]) -> None:
            propagator.sync_root_from_stage(context=ctx, stage_name=stage_name, outputs=outputs)

        restore_reused_context(
            attempt=attempt,
            lineage=lineage,
            stage_names=[s.stage_name for s in self.stages],
            start_stage_name=start_stage_name,
            enabled_stage_names=enabled_stage_names,
            context=self.context,
            sync_root_from_stage=_sync,
        )

    def _sync_root_context_from_stage(self, stage_name: str, outputs: dict[str, Any]) -> None:
        self._context_propagator.sync_root_from_stage(
            context=self.context, stage_name=stage_name, outputs=outputs
        )

    def _extract_restart_outputs(self, stage_name: str) -> dict[str, Any]:
        return self._context_propagator.extract_restart_outputs(
            context=self.context, stage_name=stage_name
        )

    def _get_stage_skip_reason(self, stage_name: str) -> str | None:
        return self._context_propagator.get_stage_skip_reason(
            context=self.context, stage_name=stage_name
        )

    # Stage state transitions — delegated to src.pipeline.state.transitioner
    _mark_stage_running = staticmethod(mark_stage_running)  # type: ignore[assignment]
    _mark_stage_completed = staticmethod(mark_stage_completed)  # type: ignore[assignment]
    _mark_stage_failed = staticmethod(mark_stage_failed)  # type: ignore[assignment]
    _mark_stage_skipped = staticmethod(mark_stage_skipped)  # type: ignore[assignment]
    _mark_stage_interrupted = staticmethod(mark_stage_interrupted)  # type: ignore[assignment]
    _finalize_attempt_state = staticmethod(finalize_attempt_state)  # type: ignore[assignment]

    def _save_state(self) -> None:
        if self._state_store is None or self._pipeline_state is None:
            return
        self._state_store.save(self._pipeline_state)

    def _record_stage_log_paths(self, *, attempt: PipelineAttemptState, stage_name: str) -> None:
        """Attach the log file registry for ``stage_name`` to its StageRunState."""
        if self._log_layout is None:
            return
        stage_state = attempt.stage_runs.get(stage_name)
        if stage_state is None:
            return
        include_remote_training = stage_name == StageNames.TRAINING_MONITOR
        stage_state.log_paths = self._log_layout.stage_log_registry(
            stage_name,
            include_remote_training=include_remote_training,
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
        self._save_state()

    def _open_existing_root_run(self, root_run_id: str) -> Any:
        return self._mlflow_attempt.open_existing_root_run(root_run_id)

    def _teardown_mlflow_attempt(self, *, pipeline_success: bool) -> None:
        attempt_run_id = (
            self._current_attempt.pipeline_attempt_mlflow_run_id if self._current_attempt else None
        )

        def _before_end() -> None:
            self._aggregate_training_metrics()

        def _sync_state_and_return_path() -> Path | None:
            if self._state_store is None:
                return None
            self._save_state()
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
        self._pipeline_state = state
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
