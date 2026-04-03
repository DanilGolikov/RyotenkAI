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

import contextlib
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.request import urlopen

from src.constants import PROVIDER_RUNPOD
from src.pipeline.artifacts import (
    StageArtifactCollector,
    ValidationArtifactData,
    ValidationDatasetData,
    ValidationPluginData,
)
from src.pipeline.artifacts.base import utc_now_iso
from src.pipeline.constants import (
    CTX_PROVIDER_NAME_UNKNOWN,
    CTX_PROVIDER_TYPE_UNKNOWN,
    CTX_RUNTIME_SECONDS,
    CTX_TRAINING_DURATION,
    CTX_TRAINING_INFO,
    EXIT_CODE_SIGINT,
    MLFLOW_CATEGORY_PIPELINE,
    MLFLOW_SOURCE_ORCHESTRATOR,
    SECONDS_PER_HOUR,
    SEPARATOR_CHAR,
    SEPARATOR_LINE_WIDTH,
    SUMMARY_LINE_WIDTH,
)
from src.pipeline.domain import RunContext
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
    PipelineRunLock,
    PipelineState,
    PipelineStateError,
    PipelineStateLoadError,
    PipelineStateStore,
    StageLineageRef,
    StageRunState,
    acquire_run_lock,
    build_attempt_state,
    hash_payload,
    update_lineage,
)
from src.reports import ExperimentReportGenerator
from src.training.managers.mlflow_manager import MLflowManager
from src.config.runtime import RuntimeSettings, load_runtime_settings
from src.utils.config import AdaLoraConfig, PipelineConfig, Secrets, load_config, load_secrets, validate_strategy_chain
from src.utils.logger import console, get_run_log_dir, init_run_logging, logger
from src.utils.result import AppError, ConfigDriftError, Err, Ok, Result

if TYPE_CHECKING:
    from src.pipeline.stages.base import PipelineStage

# Status literals used in MLflow event attributes (> 3 uses → extract to avoid WPS226)
_STATUS_FAILED = "failed"
_STATUS_PASSED = "passed"
_STATUS_RUNNING = "running"
_STATUS_STARTED = "started"

# Dict key literals reused across context reads (> 3 uses → WPS226)
_KEY_DESCRIPTION = "description"
_KEY_UPLOAD_DURATION = "upload_duration_seconds"

# HTTP status thresholds for health checks
_HTTP_OK_MIN = 200
_HTTP_ERROR_MIN = 400
_VALIDATION_STATUS_FAILED = "failed"
_VALIDATION_STATUS_PASSED = "passed"
_VALIDATION_ARTIFACT_REF = "dataset_validator_results.json"


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
        self._state_store: PipelineStateStore | None = None
        self._pipeline_state: PipelineState | None = None
        self._current_attempt: PipelineAttemptState | None = None
        self._run_lock: PipelineRunLock | None = None

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

        # Validate strategy chain EARLY (before any stages run)
        strategies = self.config.training.strategies
        if strategies:
            is_valid, error_msg = validate_strategy_chain(strategies)
            if not is_valid:
                chain_str = " -> ".join(s.strategy_type.upper() for s in strategies)
                logger.error(f"Invalid strategy chain: {chain_str}")
                logger.error(f"   Error: {error_msg}")
                raise ValueError(f"Invalid strategy chain: {error_msg}")
            chain_str = " -> ".join(s.strategy_type.upper() for s in strategies)
            logger.info(f"Strategy chain validated: {chain_str}")

        # Initialize stages
        self.stages: list[PipelineStage] = self._init_stages()
        logger.info(f"Initialized {len(self.stages)} pipeline stages")

        # Pipeline context (shared data between stages)
        # Include config_path so deployment knows which config to upload
        self.context: dict[str, Any] = {
            PipelineContextKeys.CONFIG_PATH: str(config_path),
            PipelineContextKeys.RUN: self.run_ctx,
        }

        # MLflow manager for pipeline event logging
        self._mlflow_manager: MLflowManager | None = None
        self._mlflow_run_context: Any = None
        self._mlflow_root_run: Any = None
        self._mlflow_attempt_run: Any = None
        self._shutdown_signal_name: str | None = None
        self._cleanup_done: bool = False

        # Stage artifact collectors — one per stage, keyed by StageNames value
        self._collectors: dict[str, StageArtifactCollector] = self._init_collectors()
        # Per-path accumulator for DatasetValidator plugin results
        # key: dataset_path, value: ValidationDatasetData being built
        self._validation_accumulator: dict[str, ValidationDatasetData] = {}
        # Validation plugin descriptions captured on start callback.
        # key: (dataset_path, plugin_id) -> description
        self._validation_plugin_descriptions: dict[tuple[str, str], str] = {}

    def notify_signal(self, *, signal_name: str) -> None:
        """Notify orchestrator about an external shutdown signal (SIGINT/SIGTERM)."""
        self._shutdown_signal_name = str(signal_name or "").upper()

    def _init_stages(self) -> list[PipelineStage]:
        """Initialize all pipeline stages in execution order."""
        from src.pipeline.stages.dataset_validator import DatasetValidatorEventCallbacks

        # Create callbacks for DatasetValidator (MLflow integration)
        validator_callbacks = DatasetValidatorEventCallbacks(
            on_dataset_scheduled=self._on_dataset_scheduled,
            on_dataset_loaded=self._on_dataset_loaded,
            on_validation_completed=self._on_validation_completed,
            on_validation_failed=self._on_validation_failed,
            on_plugin_start=self._on_plugin_start,
            on_plugin_complete=self._on_plugin_complete,
            on_plugin_failed=self._on_plugin_failed,
        )

        stages: list[PipelineStage] = [
            DatasetValidator(self.config, secrets=self.secrets, callbacks=validator_callbacks),
            GPUDeployer(self.config, self.secrets),  # Universal provider-based deployer
            TrainingMonitor(self.config),
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
        """Setup MLflow for pipeline event logging."""
        mlflow_config = self.config.experiment_tracking.mlflow
        try:
            # Explicitly disable system metrics for pipeline orchestrator (Control Plane)
            # We don't want to log Mac/Host CPU/RAM metrics to the main run
            # because it confuses the dashboard (mixes with GPU metrics from provider)

            # 1. Force config setting
            config_copy = self.config
            config_copy.experiment_tracking.mlflow.system_metrics_callback_enabled = False

            # 2. Force environment variable (critical for MLflow internals)
            import os

            os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "false"

            manager = MLflowManager(config_copy)

            # 3. Disable system metrics logging in MLflow client directly
            # Important: Call disable_system_metrics_logging() globally first
            try:
                import mlflow

                mlflow.disable_system_metrics_logging()
            except Exception:
                # Best-effort: MLflow may be missing, API may differ, or call can fail.
                # We still proceed with manager.setup(disable_system_metrics=True).
                pass

            manager.setup(disable_system_metrics=True)
            return manager
        except Exception as e:
            logger.warning(f"MLflow setup failed: {e}")
            return None

    def _get_mlflow_run_id(self) -> str | None:
        """
        Best-effort get current MLflow run_id (works for real MLflowManager and test mocks).

        - Prefer `run_id` property when it's a non-empty string.
        - Fall back to legacy `_run_id` attribute used in older code/tests.
        """
        if not self._mlflow_manager:
            return None

        run_id = getattr(self._mlflow_manager, "run_id", None)
        if isinstance(run_id, str) and run_id:
            return run_id

        legacy_run_id = getattr(self._mlflow_manager, "_run_id", None)
        if isinstance(legacy_run_id, str) and legacy_run_id:
            return legacy_run_id

        return None

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

            start_idx = self._get_stage_index(start_stage_name)
            stop_idx = len(self.stages)

            self._run_lock = acquire_run_lock(self._state_store.lock_path)

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
            )
            self._current_attempt = attempt
            self.attempt_directory = self._state_store.next_attempt_dir(attempt.attempt_no)
            init_run_logging(self.run_ctx.name, log_dir=self.attempt_directory)

            self.context = {
                PipelineContextKeys.CONFIG_PATH: str(self.config_path),
                PipelineContextKeys.RUN: self.run_ctx,
                PipelineContextKeys.LOGICAL_RUN_ID: state.logical_run_id,
                PipelineContextKeys.ATTEMPT_ID: attempt.attempt_id,
                PipelineContextKeys.ATTEMPT_NO: attempt.attempt_no,
                PipelineContextKeys.RUN_DIRECTORY: str(self.run_directory),
                PipelineContextKeys.ATTEMPT_DIRECTORY: str(self.attempt_directory),
                PipelineContextKeys.FORCED_STAGES: set(self._forced_stage_names(start_stage_name=start_stage_name)),
            }

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

                self._mark_stage_running(attempt=attempt, stage_name=stage_name, started_at=current_stage_started_at)
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
                    logger.error(f"Pipeline failed at stage {i + 1}: {stage_name}")
                    logger.error(f"Error: {stage_err}")
                    if self._mlflow_manager:
                        self._mlflow_manager.log_stage_failed(stage_name=stage_name, stage_idx=i, error=str(stage_err))
                    if collector and not collector.is_flushed:
                        if stage_name == StageNames.DATASET_VALIDATOR:
                            self._flush_validation_artifact(
                                started_at=current_stage_started_at,
                                duration_seconds=stage_duration,
                            )
                        else:
                            collector.flush_error(
                                error=str(stage_err),
                                started_at=current_stage_started_at,
                                duration_seconds=stage_duration,
                                context=self.context,
                            )
                    self._mark_stage_failed(
                        attempt=attempt,
                        stage_name=stage_name,
                        error=str(stage_err),
                        failure_kind=getattr(stage_err, "code", "STAGE_FAILED"),
                        outputs=(
                            self._build_dataset_validation_state_outputs(error=str(stage_err))
                            if stage_name == StageNames.DATASET_VALIDATOR
                            else None
                        ),
                    )
                    self._finalize_attempt_state(
                        state=state,
                        attempt=attempt,
                        status=StageRunState.STATUS_FAILED,
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
                            details={**stage_err.to_log_dict(), "stage_name": stage_name, "stage_idx": i},
                        )
                    )

                stage_result = result.unwrap()
                if stage_result is not None:
                    self.context.update(stage_result)

                if stage_name == StageNames.MODEL_RETRIEVER:
                    self._maybe_early_release_gpu()

                outputs = self._extract_restart_outputs(stage_name)
                skip_reason = self._get_stage_skip_reason(stage_name)

                if collector and not collector.is_flushed:
                    if stage_name == StageNames.DATASET_VALIDATOR:
                        self._flush_validation_artifact(
                            started_at=current_stage_started_at,
                            duration_seconds=stage_duration,
                        )
                    else:
                        self._fill_from_context(stage_name, collector)
                        collector.flush_ok(
                            started_at=current_stage_started_at,
                            duration_seconds=stage_duration,
                            context=self.context,
                        )

                if self._mlflow_manager:
                    self._log_stage_specific_info(stage_name)
                    self._mlflow_manager.log_stage_complete(
                        stage_name=stage_name,
                        stage_idx=i,
                        duration_seconds=stage_duration,
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
                logger.info(f"Stage {i + 1} completed successfully ({stage_duration:.1f}s)")
                current_stage_name = None
                current_stage_started_at = None
                current_stage_start_time = None

            pipeline_success = True
            pipeline_duration = time.time() - pipeline_start_time
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
            return Ok(self.context)

        except (KeyboardInterrupt, SystemExit) as exc:
            is_system_exit = isinstance(exc, SystemExit)
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
            logger.exception(f"Unexpected error in pipeline: {e}")
            if self._current_attempt:
                completed_at = utc_now_iso()
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
        finally:
            self._flush_pending_collectors()
            self._cleanup_resources(success=pipeline_success)
            self._teardown_mlflow_attempt(pipeline_success=pipeline_success)
            if self._run_lock:
                self._run_lock.release()
                self._run_lock = None

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
        )
        self._pipeline_state = state
        self.logical_run_id = logical_run_id
        return state, "fresh", "fresh", self.stages[0].stage_name

    def _build_config_hashes(self) -> dict[str, str]:
        training_provider_name = self.config.get_active_provider_name()
        training_provider_cfg = self.config.get_provider_config()
        training_payload = {
            "model": self.config.model.model_dump(mode="json"),
            "training": self.config.training.model_dump(mode="json"),
            "datasets": {name: cfg.model_dump(mode="json") for name, cfg in self.config.datasets.items()},
            "provider_name": training_provider_name,
            "provider": training_provider_cfg,
        }
        late_payload = {
            "inference": self.config.inference.model_dump(mode="json"),
            "evaluation": self.config.evaluation.model_dump(mode="json"),
        }
        return {
            "training_critical": hash_payload(training_payload),
            "late_stage": hash_payload(late_payload),
        }

    def _normalize_stage_ref(self, stage_ref: str | int | None) -> str:
        """Resolve stage ref to canonical stage name.

        Accepts:
        - str name: "Inference Deployer", "inference_deployer" (case/underscore-insensitive)
        - int or str digit: 1-based human index (1 = first stage, N = last stage)
        """
        if stage_ref is None:
            raise ValueError("Stage reference is required")
        n = len(self.stages)
        if isinstance(stage_ref, int):
            if 1 <= stage_ref <= n:
                return self.stages[stage_ref - 1].stage_name
            raise ValueError(f"Stage index {stage_ref} out of range 1–{n}")
        stage_value = str(stage_ref).strip()
        if stage_value == "":
            raise ValueError("Stage reference is empty")
        if stage_value.isdigit():
            idx = int(stage_value)
            if 1 <= idx <= n:
                return self.stages[idx - 1].stage_name
            raise ValueError(f"Stage index {stage_value} out of range 1–{n}")
        lowered = stage_value.casefold()
        aliases = {stage.stage_name.casefold(): stage.stage_name for stage in self.stages}
        normalized_aliases = {stage.stage_name.casefold().replace(" ", "_"): stage.stage_name for stage in self.stages}
        if lowered in aliases:
            return aliases[lowered]
        if lowered.replace(" ", "_") in normalized_aliases:
            return normalized_aliases[lowered.replace(" ", "_")]
        raise ValueError(f"Unknown stage reference: {stage_ref!r}. Use a name or 1–{n}")

    def _get_stage_index(self, stage_name: str) -> int:
        for idx, stage in enumerate(self.stages):
            if stage.stage_name == stage_name:
                return idx
        raise ValueError(f"Unknown stage name: {stage_name}")

    def _forced_stage_names(self, *, start_stage_name: str) -> set[str]:
        forced: set[str] = set()
        if start_stage_name == StageNames.INFERENCE_DEPLOYER and not self.config.inference.enabled:
            forced.add(StageNames.INFERENCE_DEPLOYER)
        if start_stage_name == StageNames.MODEL_EVALUATOR and not self.config.evaluation.enabled:
            forced.add(StageNames.MODEL_EVALUATOR)
        return forced

    def _compute_enabled_stage_names(self, *, start_stage_name: str) -> list[str]:
        enabled = [stage.stage_name for stage in self.stages[:4]]
        if self.config.inference.enabled:
            enabled.append(StageNames.INFERENCE_DEPLOYER)
        if self.config.evaluation.enabled:
            enabled.append(StageNames.MODEL_EVALUATOR)
        for forced_name in self._forced_stage_names(start_stage_name=start_stage_name):
            if forced_name not in enabled:
                enabled.append(forced_name)
        return enabled

    def _derive_resume_stage(self, state: PipelineState) -> str | None:
        if not state.attempts:
            return self.stages[0].stage_name
        latest = state.attempts[-1]
        for stage in self.stages:
            stage_state = latest.stage_runs.get(stage.stage_name)
            if stage_state is None:
                return stage.stage_name
            if stage_state.status in {
                StageRunState.STATUS_FAILED,
                StageRunState.STATUS_INTERRUPTED,
                StageRunState.STATUS_PENDING,
                StageRunState.STATUS_RUNNING,
                StageRunState.STATUS_STALE,
            }:
                return stage.stage_name
        return None

    def _validate_config_drift(
        self,
        *,
        state: PipelineState,
        start_stage_name: str,
        config_hashes: dict[str, str],
        resume: bool,
    ) -> AppError | None:
        training_changed = state.training_critical_config_hash != config_hashes["training_critical"]
        late_changed = state.late_stage_config_hash != config_hashes["late_stage"]
        if training_changed:
            return ConfigDriftError(
                message=(
                    "training_critical config changed for existing logical run; "
                    "resume/restart is blocked. Use the original config or start a new run."
                ),
                details={
                    "scope": "training_critical",
                    "start_stage_name": start_stage_name,
                    "resume": resume,
                },
            )
        if late_changed and (
            resume or start_stage_name not in {StageNames.INFERENCE_DEPLOYER, StageNames.MODEL_EVALUATOR}
        ):
            return ConfigDriftError(
                message=(
                    "late_stage config changed; only manual restart from "
                    "Inference Deployer or Model Evaluator is allowed."
                ),
                details={
                    "scope": "late_stage",
                    "start_stage_name": start_stage_name,
                    "resume": resume,
                },
            )
        return None

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
        start_idx = self._get_stage_index(start_stage_name)
        new_lineage = dict(lineage)
        for stage in self.stages[start_idx:]:
            new_lineage.pop(stage.stage_name, None)
        return new_lineage

    def _restore_reused_context(
        self,
        *,
        attempt: PipelineAttemptState,
        lineage: dict[str, StageLineageRef],
        start_stage_name: str,
        enabled_stage_names: list[str],
    ) -> None:
        start_idx = self._get_stage_index(start_stage_name)
        for i, stage in enumerate(self.stages):
            if i >= start_idx:
                break
            ref = lineage.get(stage.stage_name)
            if ref is None:
                continue
            if stage.stage_name not in enabled_stage_names:
                self._mark_stage_skipped(
                    attempt=attempt,
                    stage_name=stage.stage_name,
                    reason="disabled_by_config",
                    outputs=ref.outputs,
                )
                continue
            self.context[stage.stage_name] = dict(ref.outputs)
            self._sync_root_context_from_stage(stage.stage_name, ref.outputs)
            attempt.stage_runs[stage.stage_name] = StageRunState(
                stage_name=stage.stage_name,
                status=StageRunState.STATUS_COMPLETED,
                execution_mode=StageRunState.MODE_REUSED,
                outputs=dict(ref.outputs),
                started_at=utc_now_iso(),
                completed_at=utc_now_iso(),
                reuse_from={"attempt_id": ref.attempt_id, "stage_name": ref.stage_name},
            )

    def _sync_root_context_from_stage(self, stage_name: str, outputs: dict[str, Any]) -> None:
        if stage_name == StageNames.INFERENCE_DEPLOYER:
            if "endpoint_url" in outputs:
                self.context["endpoint_url"] = outputs["endpoint_url"]
            if "inference_model_name" in outputs:
                self.context["inference_model_name"] = outputs["inference_model_name"]

    def _extract_restart_outputs(self, stage_name: str) -> dict[str, Any]:
        stage_ctx = self.context.get(stage_name, {})
        if not isinstance(stage_ctx, dict):
            return {}
        if stage_name == StageNames.DATASET_VALIDATOR:
            return self._build_dataset_validation_state_outputs(stage_ctx=stage_ctx)
        if stage_name == StageNames.GPU_DEPLOYER:
            keys: tuple[str, ...] = (
                "resource_id",
                "ssh_host",
                "ssh_port",
                "ssh_user",
                "ssh_key_path",
                "workspace_path",
                "provider_name",
                "provider_type",
                "provider_info",
                "is_alias_mode",
            )
            return {key: stage_ctx.get(key) for key in keys if stage_ctx.get(key) is not None}
        if stage_name == StageNames.TRAINING_MONITOR:
            outputs = {
                "status": stage_ctx.get("status"),
                "training_duration_seconds": stage_ctx.get("training_duration_seconds"),
                "training_info": stage_ctx.get("training_info"),
            }
            gpu_ctx = self.context.get(StageNames.GPU_DEPLOYER, {})
            if isinstance(gpu_ctx, dict) and gpu_ctx.get("workspace_path"):
                outputs["remote_output_dir"] = f"{gpu_ctx['workspace_path']}/output"
            return {key: value for key, value in outputs.items() if value is not None}
        if stage_name == StageNames.MODEL_RETRIEVER:
            keys = ("hf_repo_id", "local_model_path", "hf_uploaded", "model_size_mb", "upload_duration_seconds")
            return {key: stage_ctx.get(key) for key in keys if stage_ctx.get(key) is not None}
        if stage_name == StageNames.INFERENCE_DEPLOYER:
            endpoint_info = stage_ctx.get("endpoint_info")
            outputs = {
                "endpoint_url": stage_ctx.get("endpoint_url"),
                "inference_endpoint_url": stage_ctx.get("inference_endpoint_url"),
                "inference_model_name": stage_ctx.get("inference_model_name"),
                "endpoint_info": endpoint_info if isinstance(endpoint_info, dict) else {},
                "inference_manifest_path": stage_ctx.get("inference_manifest_path"),
                "inference_scripts": stage_ctx.get("inference_scripts"),
            }
            return {key: value for key, value in outputs.items() if value is not None}
        if stage_name == StageNames.MODEL_EVALUATOR:
            keys = ("eval_passed", "eval_summary", "evaluation_completed_at")
            return {key: stage_ctx.get(key) for key in keys if stage_ctx.get(key) is not None}
        return dict(stage_ctx)

    def _get_stage_skip_reason(self, stage_name: str) -> str | None:
        stage_ctx = self.context.get(stage_name, {})
        if not isinstance(stage_ctx, dict):
            return None
        if stage_ctx.get("inference_skipped"):
            return str(stage_ctx.get("reason", "inference_skipped"))
        if stage_ctx.get("evaluation_skipped"):
            return str(stage_ctx.get("reason", "evaluation_skipped"))
        return None

    def _mark_stage_running(self, *, attempt: PipelineAttemptState, stage_name: str, started_at: str) -> None:
        attempt.stage_runs[stage_name] = StageRunState(
            stage_name=stage_name,
            status=StageRunState.STATUS_RUNNING,
            execution_mode=StageRunState.MODE_EXECUTED,
            started_at=started_at,
        )

    def _mark_stage_completed(
        self,
        *,
        attempt: PipelineAttemptState,
        stage_name: str,
        outputs: dict[str, Any],
    ) -> None:
        stage_state = attempt.stage_runs.get(stage_name) or StageRunState(stage_name=stage_name)
        stage_state.status = StageRunState.STATUS_COMPLETED
        stage_state.execution_mode = StageRunState.MODE_EXECUTED
        stage_state.outputs = dict(outputs)
        stage_state.completed_at = utc_now_iso()
        stage_state.error = None
        stage_state.failure_kind = None
        stage_state.skip_reason = None
        attempt.stage_runs[stage_name] = stage_state

    def _mark_stage_failed(
        self,
        *,
        attempt: PipelineAttemptState,
        stage_name: str,
        error: str,
        failure_kind: str,
        outputs: dict[str, Any] | None = None,
    ) -> None:
        stage_state = attempt.stage_runs.get(stage_name) or StageRunState(stage_name=stage_name)
        stage_state.status = StageRunState.STATUS_FAILED
        stage_state.execution_mode = StageRunState.MODE_EXECUTED
        stage_state.outputs = dict(outputs or {})
        stage_state.error = error
        stage_state.failure_kind = failure_kind
        stage_state.completed_at = utc_now_iso()
        attempt.stage_runs[stage_name] = stage_state
        attempt.status = StageRunState.STATUS_FAILED
        attempt.completed_at = utc_now_iso()
        attempt.error = error

    def _mark_stage_skipped(
        self,
        *,
        attempt: PipelineAttemptState,
        stage_name: str,
        reason: str,
        outputs: dict[str, Any] | None = None,
    ) -> None:
        attempt.stage_runs[stage_name] = StageRunState(
            stage_name=stage_name,
            status=StageRunState.STATUS_SKIPPED,
            execution_mode=StageRunState.MODE_SKIPPED,
            outputs=dict(outputs or {}),
            skip_reason=reason,
            started_at=utc_now_iso(),
            completed_at=utc_now_iso(),
        )

    def _mark_stage_interrupted(
        self,
        *,
        attempt: PipelineAttemptState,
        stage_name: str,
        started_at: str,
    ) -> None:
        attempt.stage_runs[stage_name] = StageRunState(
            stage_name=stage_name,
            status=StageRunState.STATUS_INTERRUPTED,
            execution_mode=StageRunState.MODE_EXECUTED,
            started_at=started_at,
            completed_at=utc_now_iso(),
        )

    def _finalize_attempt_state(
        self,
        *,
        state: PipelineState,
        attempt: PipelineAttemptState,
        status: str,
        completed_at: str | None = None,
    ) -> None:
        attempt.status = status
        attempt.completed_at = completed_at or attempt.completed_at or utc_now_iso()
        state.pipeline_status = status
        if state.active_attempt_id == attempt.attempt_id:
            state.active_attempt_id = None

    def _save_state(self) -> None:
        if self._state_store is None or self._pipeline_state is None:
            return
        self._state_store.save(self._pipeline_state)

    def _setup_mlflow_for_attempt(
        self, *, state: PipelineState, attempt: PipelineAttemptState, start_stage_idx: int
    ) -> None:
        self._mlflow_manager = self._setup_mlflow()
        if not self._mlflow_manager or not self._mlflow_manager.is_active:
            return
        try:
            import mlflow

            mlflow.disable_system_metrics_logging()
        except Exception:
            pass

        if state.root_mlflow_run_id:
            self._mlflow_root_run = self._open_existing_root_run(state.root_mlflow_run_id)
            attempt.root_mlflow_run_id = state.root_mlflow_run_id
        else:
            self._mlflow_run_context = self._mlflow_manager.start_run(run_name=state.logical_run_id)
            self._mlflow_root_run = self._mlflow_run_context.__enter__()
            state.root_mlflow_run_id = self._get_mlflow_run_id()
            attempt.root_mlflow_run_id = state.root_mlflow_run_id

        attempt_name = f"{state.logical_run_id}_attempt_{attempt.attempt_no}"
        attempt_tags = {
            "pipeline.logical_run_id": state.logical_run_id,
            "pipeline.attempt_id": attempt.attempt_id,
            "pipeline.attempt_no": str(attempt.attempt_no),
        }
        self._mlflow_attempt_run = self._mlflow_manager.start_nested_run(run_name=attempt_name, tags=attempt_tags)
        self._mlflow_attempt_run.__enter__()
        attempt.pipeline_attempt_mlflow_run_id = self._get_mlflow_run_id()
        self.context[PipelineContextKeys.MLFLOW_PARENT_RUN_ID] = attempt.pipeline_attempt_mlflow_run_id
        self.context[PipelineContextKeys.MLFLOW_MANAGER] = self._mlflow_manager
        self._mlflow_manager.log_event_start(
            "Pipeline attempt started",
            category=MLFLOW_CATEGORY_PIPELINE,
            source=MLFLOW_SOURCE_ORCHESTRATOR,
        )
        self._mlflow_manager.log_pipeline_config(self.config)
        self._mlflow_manager.log_dataset_config(self.config)
        self._mlflow_manager.log_params(
            {
                "pipeline.total_stages": len(self.stages),
                "pipeline.start_stage": start_stage_idx,
                "pipeline.run_directory": str(self.run_directory),
            }
        )

    def _ensure_mlflow_preflight(self, *, state: PipelineState) -> None:
        """Fail fast when mandatory MLflow setup/connectivity is not available."""
        tracking_uri = self.config.experiment_tracking.mlflow.tracking_uri
        if self._mlflow_manager is None or not self._mlflow_manager.is_active:
            raise LaunchPreparationError(
                AppError(
                    code="MLFLOW_SETUP_FAILED",
                    message=f"MLflow setup failed: {tracking_uri}",
                ),
                state=state,
            )
        if not self._mlflow_manager.check_mlflow_connectivity():
            raise LaunchPreparationError(
                AppError(
                    code="MLFLOW_UNREACHABLE",
                    message=f"MLflow not reachable: {tracking_uri}",
                ),
                state=state,
            )
        if self.config_path.exists():
            self._mlflow_manager.log_artifact(str(self.config_path))
        self._save_state()

    def _open_existing_root_run(self, root_run_id: str) -> Any:
        assert self._mlflow_manager is not None
        if self._mlflow_manager._mlflow is None:
            return None
        run = self._mlflow_manager._mlflow.start_run(run_id=root_run_id, nested=False, log_system_metrics=False)
        self._mlflow_manager._run = run
        self._mlflow_manager._run_id = root_run_id
        self._mlflow_manager._parent_run_id = root_run_id
        return run

    def _teardown_mlflow_attempt(self, *, pipeline_success: bool) -> None:
        if self._mlflow_manager:
            try:
                self._aggregate_training_metrics()
            except Exception as e:
                logger.warning(f"Failed to aggregate training metrics: {e}")

            attempt_run_id = self._current_attempt.pipeline_attempt_mlflow_run_id if self._current_attempt else None

            if self._mlflow_attempt_run:
                with contextlib.suppress(Exception):
                    if pipeline_success:
                        self._mlflow_attempt_run.__exit__(None, None, None)
                    else:
                        _exc = RuntimeError("Pipeline attempt failed")
                        self._mlflow_attempt_run.__exit__(type(_exc), _exc, None)
                self._mlflow_attempt_run = None

            # Log final pipeline_state.json to the root MLflow run.
            # The nested attempt run is already closed above, so log_artifact
            # goes to the root run context.
            if self._state_store is not None:
                with contextlib.suppress(Exception):
                    self._save_state()
                    if self._state_store.state_path.exists():
                        self._mlflow_manager.log_artifact(str(self._state_store.state_path))

            root_status = "FINISHED" if pipeline_success else "FAILED"
            self._mlflow_manager.end_run(status=root_status)
            self._generate_experiment_report(run_id=attempt_run_id)

        if self._mlflow_run_context:
            with contextlib.suppress(Exception):
                self._mlflow_run_context.__exit__(None, None, None)
            self._mlflow_run_context = None

        self._mlflow_root_run = None

        if self._mlflow_manager:
            self._mlflow_manager.cleanup()

    def _validate_stage_prerequisites(self, *, stage_name: str, start_stage_name: str) -> AppError | None:
        if stage_name == StageNames.TRAINING_MONITOR and start_stage_name == StageNames.TRAINING_MONITOR:
            gpu_ctx = self.context.get(StageNames.GPU_DEPLOYER, {})
            if not isinstance(gpu_ctx, dict) or not all(
                gpu_ctx.get(key) for key in ("ssh_host", "ssh_port", "workspace_path")
            ):
                return AppError(
                    message="Training Monitor restart requires persisted GPU deploy outputs and workspace_path",
                    code="MISSING_TRAINING_MONITOR_PREREQUISITES",
                )
        if stage_name == StageNames.INFERENCE_DEPLOYER and start_stage_name == StageNames.INFERENCE_DEPLOYER:
            retriever_ctx = self.context.get(StageNames.MODEL_RETRIEVER, {})
            if not isinstance(retriever_ctx, dict) or not (
                retriever_ctx.get("hf_repo_id") or retriever_ctx.get("local_model_path")
            ):
                return AppError(
                    message="Inference Deployer restart requires Model Retriever outputs",
                    code="MISSING_INFERENCE_PREREQUISITES",
                )
        if (
            stage_name == StageNames.MODEL_EVALUATOR
            and start_stage_name == StageNames.MODEL_EVALUATOR
            and not self._is_inference_runtime_healthy()
        ):
            return AppError(
                message="Model Evaluator restart requires a live inference runtime; restart from Inference Deployer",
                code="INFERENCE_RUNTIME_NOT_HEALTHY",
            )
        return None

    def _is_inference_runtime_healthy(self, inference_ctx: dict[str, Any] | None = None) -> bool:
        ctx = inference_ctx if inference_ctx is not None else self.context.get(StageNames.INFERENCE_DEPLOYER, {})
        if not isinstance(ctx, dict):
            return False
        endpoint_info = ctx.get("endpoint_info")
        if not isinstance(endpoint_info, dict):
            endpoint_info = {}
        health_url = endpoint_info.get("health_url") or ctx.get("endpoint_url")
        if not isinstance(health_url, str) or not health_url:
            return False
        try:
            with urlopen(health_url, timeout=5) as response:
                return _HTTP_OK_MIN <= int(getattr(response, "status", _HTTP_OK_MIN)) < _HTTP_ERROR_MIN
        except Exception:
            return False

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

            if state.training_critical_config_hash != config_hashes["training_critical"]:
                available = False
                reason = "training_critical_config_changed"
            elif state.late_stage_config_hash != config_hashes["late_stage"] and stage_name not in {
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
        if not self._mlflow_manager:
            return

        # After GPU Deployer: log provider info + durations
        if stage_name == StageNames.GPU_DEPLOYER and StageNames.GPU_DEPLOYER in self.context:
            deployer_ctx = self.context[StageNames.GPU_DEPLOYER]
            self._mlflow_manager.log_provider_info(
                provider_name=deployer_ctx.get("provider_name", CTX_PROVIDER_NAME_UNKNOWN),
                provider_type=deployer_ctx.get("provider_type", CTX_PROVIDER_TYPE_UNKNOWN),
                gpu_type=deployer_ctx.get("gpu_type"),
                resource_id=deployer_ctx.get("resource_id"),
            )
            # Log upload and deps durations as events
            upload_dur = deployer_ctx.get(_KEY_UPLOAD_DURATION)
            deps_dur = deployer_ctx.get("deps_duration_seconds")
            if upload_dur:
                self._mlflow_manager.log_event_info(
                    f"Files uploaded ({upload_dur:.1f}s)",
                    category="deployment",
                    source=StageNames.GPU_DEPLOYER,
                    upload_duration_seconds=upload_dur,
                )
            if deps_dur:
                self._mlflow_manager.log_event_info(
                    f"Dependencies installed ({deps_dur:.1f}s)",
                    category="deployment",
                    source=StageNames.GPU_DEPLOYER,
                    deps_duration_seconds=deps_dur,
                )

        # After Dataset Validator: log dataset validation metrics
        if stage_name == StageNames.DATASET_VALIDATOR and StageNames.DATASET_VALIDATOR in self.context:
            validator_ctx = self.context[StageNames.DATASET_VALIDATOR]
            metrics = validator_ctx.get("metrics", {})  # noqa: WPS226
            validation_mode = validator_ctx.get("validation_mode", "legacy")

            if metrics:
                # NEW: Plugin-based validation logs plugin-specific metrics
                if validation_mode == "plugin":
                    # Log all plugin metrics as params
                    params_to_log: dict[str, float | str] = {}
                    for key, value in metrics.items():
                        # Convert to float/int for MLflow
                        try:
                            params_to_log[f"dataset.{key}"] = float(value)
                        except (ValueError, TypeError):
                            params_to_log[f"dataset.{key}"] = str(value)

                    self._mlflow_manager.log_params(params_to_log)

                else:
                    # LEGACY: Log as before (backward compatibility)
                    self._mlflow_manager.log_params(
                        {
                            "dataset.sample_count": validator_ctx.get("sample_count", 0),
                            "dataset.avg_length": metrics.get("avg_length", 0),
                            "dataset.empty_ratio": metrics.get("empty_ratio", 0),
                            "dataset.diversity_score": metrics.get("diversity_score", 0),
                        }
                    )

        # After Training Monitor: log training metrics + duration
        if stage_name == StageNames.TRAINING_MONITOR and StageNames.TRAINING_MONITOR in self.context:
            monitor_ctx = self.context[StageNames.TRAINING_MONITOR]

            # Log training duration as event
            training_dur = monitor_ctx.get(CTX_TRAINING_DURATION)
            if training_dur:
                self._mlflow_manager.log_event_info(
                    f"Training completed ({training_dur:.1f}s)",
                    category="training",
                    source=StageNames.TRAINING_MONITOR,
                    training_duration_seconds=training_dur,
                )

            # Log mock training info if present
            training_info = monitor_ctx.get(CTX_TRAINING_INFO, {})
            if training_info:
                metrics_to_log = {}
                if training_info.get(CTX_RUNTIME_SECONDS):
                    metrics_to_log[f"training.{CTX_RUNTIME_SECONDS}"] = training_info[CTX_RUNTIME_SECONDS]
                if training_info.get("final_loss"):
                    metrics_to_log["training.final_loss"] = training_info["final_loss"]
                if training_info.get("final_accuracy"):
                    metrics_to_log["training.final_accuracy"] = training_info["final_accuracy"]
                if training_info.get("total_steps"):
                    metrics_to_log["training.total_steps"] = float(training_info["total_steps"])

                if metrics_to_log:
                    self._mlflow_manager.log_metrics(metrics_to_log)

        # After Model Retriever: log model info
        if stage_name == StageNames.MODEL_RETRIEVER and StageNames.MODEL_RETRIEVER in self.context:
            retriever_ctx = self.context[StageNames.MODEL_RETRIEVER]

            model_size = retriever_ctx.get("model_size_mb")
            if model_size:
                self._mlflow_manager.log_event_info(
                    f"Model size: {model_size:.1f} MB",
                    category="model",
                    source=StageNames.MODEL_RETRIEVER,
                    model_size_mb=model_size,
                )

            hf_uploaded = retriever_ctx.get("hf_uploaded")
            upload_dur = retriever_ctx.get(_KEY_UPLOAD_DURATION)
            if hf_uploaded and upload_dur:
                hf_repo = retriever_ctx.get("hf_repo_id", CTX_PROVIDER_NAME_UNKNOWN)
                self._mlflow_manager.log_event_info(
                    f"Model uploaded to HF: {hf_repo} ({upload_dur:.1f}s)",
                    category="model",
                    source=StageNames.MODEL_RETRIEVER,
                    hf_repo_id=hf_repo,
                    upload_duration_seconds=upload_dur,
                )

    def _fill_from_context(self, stage_name: str, collector: StageArtifactCollector) -> None:
        """Populate collector with stage-specific data read from pipeline context.

        Called for simple stages where all data is available after stage.run()
        completes. Complex stages (DatasetValidator) populate the collector
        themselves via callbacks, so this method is never called for them.
        """
        ctx = self.context.get(stage_name, {})
        if not isinstance(ctx, dict):
            return

        if stage_name == StageNames.GPU_DEPLOYER:
            collector.put(
                upload_duration_seconds=ctx.get(_KEY_UPLOAD_DURATION),
                deps_duration_seconds=ctx.get("deps_duration_seconds"),
                provider_name=ctx.get("provider_name"),
                provider_type=ctx.get("provider_type"),
                gpu_type=ctx.get("gpu_type"),
                resource_id=ctx.get("resource_id"),
            )

        elif stage_name == StageNames.TRAINING_MONITOR:
            collector.put(
                training_duration_seconds=ctx.get(CTX_TRAINING_DURATION),
            )

        elif stage_name == StageNames.MODEL_RETRIEVER:
            collector.put(
                model_size_mb=ctx.get("model_size_mb"),
                hf_repo_id=ctx.get("hf_repo_id"),
                upload_duration_seconds=ctx.get(_KEY_UPLOAD_DURATION),
            )

        elif stage_name == StageNames.INFERENCE_DEPLOYER:
            collector.put(
                endpoint_url=ctx.get("endpoint_url"),
                model_name=ctx.get("model_name"),
                provider=ctx.get("provider"),
            )

        elif stage_name == StageNames.MODEL_EVALUATOR:
            eval_summary = ctx.get("eval_summary", {})
            if isinstance(eval_summary, dict):
                collector.put(**eval_summary)
            else:
                collector.put(eval_summary=str(eval_summary))

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
        console.print("\n" + SEPARATOR_CHAR * SUMMARY_LINE_WIDTH)
        console.print("[bold green]PIPELINE EXECUTION SUMMARY[/bold green]")
        console.print(SEPARATOR_CHAR * SUMMARY_LINE_WIDTH)

        # Configuration section
        console.print("\n[bold cyan]Configuration:[/bold cyan]")
        console.print(f"   Model: {self.config.model.name}")
        console.print(f"   Training Type: {self.config.training.type}")
        console.print(f"   4-bit Quantization: {self.config.training.get_effective_load_in_4bit()}")
        try:
            adapter_cfg = self.config.get_adapter_config()
            if isinstance(adapter_cfg, AdaLoraConfig):
                console.print(f"   AdaLoRA init_r/target_r: {adapter_cfg.init_r}/{adapter_cfg.target_r}")
            else:
                console.print(f"   LoRA r/alpha: {adapter_cfg.r}/{adapter_cfg.lora_alpha}")
        except ValueError:
            pass  # Not qlora/lora training type
        console.print(f"   Batch Size: {self.config.training.hyperparams.per_device_train_batch_size}")
        # Show strategy chain
        strategies = self.config.training.get_strategy_chain()
        if strategies:
            console.print(f"   Strategies: {' -> '.join(s.strategy_type.upper() for s in strategies)}")

        # Dataset section
        default_ds = self.config.get_primary_dataset()
        console.print("\n[bold cyan]Dataset:[/bold cyan]")
        if default_ds.get_source_type() == "huggingface" and default_ds.source_hf is not None:
            console.print(f"   Train (HF): {default_ds.source_hf.train_id}")
            if default_ds.source_hf.eval_id:
                console.print(f"   Eval  (HF): {default_ds.source_hf.eval_id}")
        elif default_ds.source_local is not None:
            console.print(f"   Train (local): {default_ds.source_local.local_paths.train}")
            if default_ds.source_local.local_paths.eval:
                console.print(f"   Eval  (local): {default_ds.source_local.local_paths.eval}")
        else:
            console.print("   [dim]Dataset source not configured[/dim]")
        if StageNames.DATASET_VALIDATOR in self.context:
            validator_ctx = self.context[StageNames.DATASET_VALIDATOR]
            console.print(f"   Samples: {validator_ctx.get('sample_count', 'N/A')}")
            if validator_ctx.get("avg_length"):
                console.print(f"   Avg Length: {validator_ctx.get('avg_length', 0):.0f} chars")
        adapter_type = default_ds.adapter_type or "auto-detect"
        console.print(f"   Adapter: {adapter_type}")

        # Grab deployer context upfront — used across multiple sections below
        deployer_ctx: dict = self.context.get(StageNames.GPU_DEPLOYER, {})

        # Training metrics
        console.print("\n[bold cyan]Training:[/bold cyan]")
        if StageNames.TRAINING_MONITOR in self.context:
            monitor_ctx = self.context[StageNames.TRAINING_MONITOR]
            # Prod stores a flat key; mock wraps it inside training_info
            training_duration = monitor_ctx.get(CTX_TRAINING_DURATION) or monitor_ctx.get(CTX_TRAINING_INFO, {}).get(
                CTX_RUNTIME_SECONDS, 0
            )
            console.print(f"   Duration: {training_duration / 60:.1f} minutes")

            # Deployment timings (available when GPUDeployer ran)
            pod_startup = deployer_ctx.get("pod_startup_seconds")
            upload_dur = deployer_ctx.get(_KEY_UPLOAD_DURATION)
            if pod_startup is not None:
                console.print(f"   Pod ready: {pod_startup:.0f}s")
            if upload_dur is not None:
                console.print(f"   Files upload: {upload_dur:.0f}s")

            # Loss / accuracy from mock training_info (real runs get these from MLflow)
            training_info = monitor_ctx.get(CTX_TRAINING_INFO, {})
            if training_info.get("final_loss"):
                console.print(f"   Final Loss: {training_info['final_loss']:.4f}")
            if training_info.get("final_accuracy"):
                console.print(f"   Accuracy: {training_info['final_accuracy']:.2%}")
        else:
            console.print("   [dim]Training info not available[/dim]")

        # Provider info
        console.print("\n[bold cyan]GPU Provider:[/bold cyan]")
        if deployer_ctx:
            provider_name = deployer_ctx.get("provider_name", CTX_PROVIDER_NAME_UNKNOWN)
            provider_type = deployer_ctx.get("provider_type", CTX_PROVIDER_TYPE_UNKNOWN)
            gpu_type = deployer_ctx.get("gpu_type")
            gpu_count = deployer_ctx.get("gpu_count")

            provider_label = provider_name
            if gpu_type:
                gpu_label = f"{gpu_count}x {gpu_type}" if gpu_count and gpu_count > 1 else gpu_type
                provider_label = f"{provider_name} ({gpu_label})"
            console.print(f"   Provider: {provider_label} [{provider_type}]")

            if provider_type == "cloud":
                cost_per_hr = deployer_ctx.get("cost_per_hr") or 0

                if cost_per_hr > 0:
                    console.print(f"   Rate: ${cost_per_hr:.4f}/hr")

                    # Use prod key (training_duration_seconds) with mock fallback
                    training_sec = 0
                    if StageNames.TRAINING_MONITOR in self.context:
                        monitor_ctx = self.context[StageNames.TRAINING_MONITOR]
                        training_sec = monitor_ctx.get(CTX_TRAINING_DURATION) or monitor_ctx.get(
                            CTX_TRAINING_INFO, {}
                        ).get(CTX_RUNTIME_SECONDS, 0)
                    training_hours = training_sec / SECONDS_PER_HOUR
                    total_cost = cost_per_hr * training_hours
                    console.print(f"   Training time: {training_hours:.2f}h")
                    console.print(f"   [bold yellow]Training cost: ${total_cost:.4f}[/bold yellow]")
                else:
                    console.print("   Rate: [dim]N/A[/dim]")
            else:
                console.print("   Cost: $0 (local)")
        else:
            console.print("   [dim]Provider info not available[/dim]")

        # Model location
        console.print("\n[bold cyan]Model Output:[/bold cyan]")
        if StageNames.MODEL_RETRIEVER in self.context:
            retriever_ctx = self.context[StageNames.MODEL_RETRIEVER]
            console.print(f"   Local Path: {retriever_ctx.get('local_model_path', 'N/A')}")
            hf_repo = retriever_ctx.get("hf_repo_id")
            if hf_repo:
                console.print(f"   HuggingFace: {hf_repo}")
            else:
                console.print("   HuggingFace: [dim]Not uploaded[/dim]")
            model_size_mb = retriever_ctx.get("model_size_mb")
            if model_size_mb:
                console.print(f"   Size: {model_size_mb:.0f} MB")
        else:
            console.print("   Output Dir: output/ (hardcoded inside remote run workspace)")

        # Evaluation metrics
        if StageNames.MODEL_EVALUATOR in self.context:
            eval_ctx = self.context[StageNames.MODEL_EVALUATOR]
            metrics = eval_ctx.get("metrics", {})
            if metrics:
                console.print("\n[bold cyan]Evaluation:[/bold cyan]")
                for metric_name, metric_value in metrics.items():
                    if isinstance(metric_value, float):
                        console.print(f"   {metric_name}: {metric_value:.4f}")
                    else:
                        console.print(f"   {metric_name}: {metric_value}")

        # Launch command — only shown when InferenceDeployer ran and produced artifacts
        if StageNames.INFERENCE_DEPLOYER in self.context:
            chat_script = self.context[StageNames.INFERENCE_DEPLOYER].get("inference_scripts", {}).get("chat")
            if chat_script:
                console.print("\n[bold cyan]Launch Command:[/bold cyan]")
                console.print(f"   [yellow]python {chat_script}[/yellow]")

        console.print("\n" + SEPARATOR_CHAR * SUMMARY_LINE_WIDTH)

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
        """
        Aggregate training metrics from child/grandchild runs into parent run.

        MLflow Best Practice: Log aggregated/summary metrics to parent run
        while keeping detailed metrics in child runs.

        Hierarchy:
        - pipeline_* (parent) ← metrics aggregated HERE
            └── strategy_run (child)
                    └── phase_0_* (grandchild) ← metrics collected from HERE

        Metrics aggregated:
        - All non-system metrics from phase runs (prefixed with phase_{idx}_)
        - Summary: final_train_loss, total_train_steps, total_train_runtime
        """
        if not self._mlflow_manager:
            return

        # Collect metrics from all descendant runs (depth=2)
        all_phase_metrics = self._collect_descendant_metrics(max_depth=2)

        if not all_phase_metrics:
            logger.debug("[METRICS] No metrics found in descendant runs")
            return

        logger.info(f"[METRICS] Aggregating metrics from {len(all_phase_metrics)} phase runs...")

        aggregated_metrics: dict[str, float] = {}
        total_steps = 0
        total_runtime = 0.0
        final_loss = None

        for phase_metrics in all_phase_metrics:
            # Track summary values only
            if train_loss := phase_metrics.get("train_loss"):
                final_loss = train_loss

            if runtime := phase_metrics.get("train_runtime"):
                total_runtime += runtime

            if steps := phase_metrics.get("global_step"):
                total_steps += int(steps)

        # Summary metrics (without phase prefix)
        if final_loss is not None:
            aggregated_metrics["final_train_loss"] = final_loss
        if total_steps > 0:
            aggregated_metrics["total_train_steps"] = float(total_steps)
        if total_runtime > 0:
            aggregated_metrics["total_train_runtime"] = total_runtime

        # Log to parent run
        if aggregated_metrics:
            self._mlflow_manager.log_metrics(aggregated_metrics)
            logger.info(f"[METRICS] Aggregated {len(aggregated_metrics)} metrics to parent run")
            if final_loss:
                logger.info(f"[METRICS] Final train_loss: {final_loss:.4f}")

    def _collect_descendant_metrics(self, max_depth: int = 2) -> list[dict[str, float]]:
        """
        Collect metrics from all descendant runs (children, grandchildren).

        Args:
            max_depth: Maximum depth to search (1=children, 2=grandchildren)

        Returns:
            List of metrics dicts from phase runs (runs starting with 'phase_')
        """
        if not self._mlflow_manager:
            return []

        try:
            client = self._mlflow_manager.client
            parent_run_id = self._get_mlflow_run_id()
            if not parent_run_id:
                return []

            # Get experiment ID
            run = client.get_run(parent_run_id)
            experiment_id = run.info.experiment_id

            # BFS to find all descendant runs
            phase_metrics: list[dict[str, float]] = []
            visited: set[str] = set()
            queue: list[tuple[str, int]] = [(parent_run_id, 0)]  # (run_id, depth)

            while queue:
                current_id, depth = queue.pop(0)

                if current_id in visited or depth > max_depth:
                    continue
                visited.add(current_id)

                # Find children of current run
                children = client.search_runs(
                    experiment_ids=[experiment_id],
                    filter_string=f"tags.`mlflow.parentRunId` = '{current_id}'",
                )

                for child in children:
                    child_name = child.info.run_name or ""

                    # If this is a phase run, collect its metrics
                    if child_name.startswith("phase_"):
                        metrics = dict(child.data.metrics)
                        if metrics:
                            phase_metrics.append(metrics)
                            logger.debug(f"[METRICS] Found phase run: {child_name} ({len(metrics)} metrics)")

                    # Add to queue for further exploration
                    if depth + 1 <= max_depth:
                        queue.append((child.info.run_id, depth + 1))

            return phase_metrics

        except Exception as e:
            logger.warning(f"[METRICS] Failed to collect descendant metrics: {e}")
            return []

    def _generate_experiment_report(self, run_id: str | None = None) -> None:
        """
        Generate full experiment report after pipeline completion.

        Uses ExperimentReportGenerator to:
        - Fetch data from MLflow (parent + child runs)
        - Extract metrics, events, config
        - Render Markdown report
        - Upload to MLflow as artifact
        - Save locally to logs directory

        Args:
            run_id: MLflow run ID (passed explicitly since run may be ended)
        """
        if not run_id:
            logger.warning("[REPORT] Cannot generate report: no run_id provided")
            return

        try:
            tracking_uri = self._mlflow_manager._gateway.uri if self._mlflow_manager else ""
            local_logs_dir = get_run_log_dir()

            logger.info(f"[REPORT] Generating experiment report for run {run_id[:8]}...")

            generator = ExperimentReportGenerator(tracking_uri)
            report = generator.generate(
                run_id=run_id,
                local_logs_dir=local_logs_dir,
            )

            logger.info(f"[REPORT] Report generated ({len(report)} chars)")
            logger.info(f"[REPORT] Saved to: {local_logs_dir / 'experiment_report.md'}")

        except Exception as e:
            logger.warning(f"[REPORT] Failed to generate report: {e}")

    # =========================================================================
    # DATASET VALIDATOR CALLBACKS (NEW: Plugin System)
    # =========================================================================

    def _on_dataset_scheduled(self, dataset_name: str, dataset_path: str, _validation_mode: str) -> None:
        """Callback: dataset scheduled for validation — initialize accumulator entry."""
        acc: ValidationDatasetData = {
            "name": dataset_name,  # noqa: WPS226
            "path": dataset_path,
            "sample_count": None,
            "status": "scheduled",  # noqa: WPS226
            "critical_failures": 0,
            "plugins": [],
        }
        self._validation_accumulator[dataset_path] = acc

    def _on_dataset_loaded(
        self, _dataset_name: str, dataset_path: str, sample_count: int, critical_failures: int
    ) -> None:
        """Callback: dataset loaded — update accumulator with sample count."""
        entry = self._validation_accumulator.get(dataset_path)
        if entry:
            entry["sample_count"] = sample_count
            entry["critical_failures"] = critical_failures

    def _on_validation_completed(
        self, _dataset_name: str, dataset_path: str, _metrics: dict, _warnings: list[str]
    ) -> None:
        """Callback: validation completed successfully — mark dataset as passed."""
        entry = self._validation_accumulator.get(dataset_path)
        if entry:
            entry["status"] = _STATUS_PASSED

    def _on_validation_failed(self, _dataset_name: str, dataset_path: str, _errors: list[str]) -> None:
        """Callback: validation failed — mark dataset as failed."""
        entry = self._validation_accumulator.get(dataset_path)
        if entry:
            entry["status"] = _STATUS_FAILED

    def _on_plugin_start(
        self,
        _dataset_name: str,
        dataset_path: str,
        plugin_id: str,
        _plugin_name: str,
        description: str,
    ) -> None:
        """Callback: validation plugin started — cache plugin description for artifact writing."""
        self._validation_plugin_descriptions[(dataset_path, plugin_id)] = description

    def _on_plugin_complete(
        self,
        _dataset_name: str,
        dataset_path: str,
        plugin_id: str,
        plugin_name: str,
        params: dict,
        thresholds: dict,
        metrics: dict,
        duration_ms: float,
    ) -> None:
        """Callback: validation plugin completed — append to accumulator."""
        description = self._validation_plugin_descriptions.pop((dataset_path, plugin_id), "")
        plugin_data: ValidationPluginData = {
            "id": plugin_id,  # noqa: WPS226
            "plugin_name": plugin_name,  # noqa: WPS226
            "passed": True,
            "duration_ms": duration_ms,
            "description": description,
            "metrics": metrics,  # noqa: WPS226
            "params": params,
            "thresholds": thresholds,
            "errors": [],
            "recommendations": [],
        }
        entry = self._validation_accumulator.get(dataset_path)
        if entry:
            entry["plugins"].append(plugin_data)

    def _on_plugin_failed(
        self,
        _dataset_name: str,
        dataset_path: str,
        plugin_id: str,
        plugin_name: str,
        params: dict,
        thresholds: dict,
        metrics: dict,
        duration_ms: float,
        errors: list[str],
        recommendations: list[str],
    ) -> None:
        """Callback: validation plugin failed — append to accumulator."""
        description = self._validation_plugin_descriptions.pop((dataset_path, plugin_id), "")
        plugin_data: ValidationPluginData = {
            "id": plugin_id,  # noqa: WPS226
            "plugin_name": plugin_name,  # noqa: WPS226
            "passed": False,
            "duration_ms": duration_ms,
            "description": description,
            "metrics": metrics,  # noqa: WPS226
            "params": params,
            "thresholds": thresholds,
            "errors": errors,
            "recommendations": recommendations,
        }
        entry = self._validation_accumulator.get(dataset_path)
        if entry:
            entry["plugins"].append(plugin_data)

    def _flush_validation_artifact(self, started_at: str, duration_seconds: float) -> None:
        """Write dataset_validator_results.json from accumulated data.

        Called after DatasetValidator stage finishes (success or failure).
        The orchestrator's main loop calls flush_ok/flush_error AFTER this
        method has already been called via the validation-completed callback.
        We piggy-back on the final state to write the artifact.
        """
        collector = self._collectors.get(StageNames.DATASET_VALIDATOR)
        if not collector or collector.is_flushed:
            return

        datasets: list[ValidationDatasetData] = list(self._validation_accumulator.values())
        artifact_data: ValidationArtifactData = {"datasets": datasets}
        collector.put(**artifact_data)

        all_passed = all(d.get("status") in {_STATUS_PASSED, "scheduled"} for d in datasets)  # noqa: WPS226
        if all_passed:
            collector.flush_ok(
                started_at=started_at,
                duration_seconds=duration_seconds,
                context=self.context,
            )
        else:
            failed = [d["name"] for d in datasets if d.get("status") == _STATUS_FAILED]  # noqa: WPS226
            collector.flush_error(
                error=f"Dataset validation failed: {', '.join(failed)}",
                started_at=started_at,
                duration_seconds=duration_seconds,
                context=self.context,
            )

    def _build_dataset_validation_state_outputs(
        self,
        *,
        stage_ctx: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> dict[str, Any]:
        datasets = list(self._validation_accumulator.values())
        failed_datasets = [dataset["name"] for dataset in datasets if dataset.get("status") == _STATUS_FAILED]
        passed_count = sum(1 for dataset in datasets if dataset.get("status") == _STATUS_PASSED)
        outputs: dict[str, Any] = {
            "validation_artifact_ref": _VALIDATION_ARTIFACT_REF,
        }

        if datasets:
            outputs.update(
                {
                    "datasets_validated": len(datasets),
                    "datasets_passed": passed_count,
                    "datasets_failed": len(failed_datasets),
                }
            )
            if failed_datasets:
                outputs["failed_datasets"] = failed_datasets

        if stage_ctx is not None:
            validation_status = stage_ctx.get("validation_status")
            if isinstance(validation_status, str) and validation_status:
                outputs["validation_status"] = validation_status
            warnings = stage_ctx.get("warnings")
            if isinstance(warnings, list):
                outputs["validation_warning_count"] = len(warnings)
            message = stage_ctx.get("message")
            if message:
                outputs["validation_message"] = str(message)
        elif error is not None:
            outputs["validation_status"] = _VALIDATION_STATUS_FAILED
            outputs["validation_message"] = error

        if "validation_status" not in outputs:
            outputs["validation_status"] = _VALIDATION_STATUS_FAILED if failed_datasets else _VALIDATION_STATUS_PASSED
        return outputs

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
