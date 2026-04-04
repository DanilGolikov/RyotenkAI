"""
PhaseExecutor - Execute single training phase.

Thin orchestrator that delegates to:
- AdapterCacheManager  (adapter_cache.py)
- MlflowPhaseLogger    (mlflow_logger.py)
- PhaseTrainingRunner  (training_runner.py)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.training.constants import CATEGORY_TRAINING
from src.utils.logger import logger
from src.utils.result import Err, Ok, Result, TrainingError

from src.training.orchestrator.phase_executor.adapter_cache import (
    AdapterCacheManager,
    _retry_call,
)
from src.training.orchestrator.phase_executor.mlflow_logger import MlflowPhaseLogger
from src.training.orchestrator.phase_executor.training_runner import PhaseTrainingRunner

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer

    from src.training.managers.data_buffer import DataBuffer
    from src.training.orchestrator.metrics_collector import MetricsCollector
    from src.training.orchestrator.shutdown_handler import ShutdownHandler
    from src.utils.config import PipelineConfig, StrategyPhaseConfig
    from src.utils.container import IDatasetLoader, IMLflowManager, IStrategyFactory, ITrainerFactory


class PhaseExecutor:
    """
    Executes a single training phase with full error handling.

    Thin orchestrator:
        shutdown-check → cache-check → mlflow-start → training → cache-upload → mlflow-end

    Supports Dependency Injection for all major sub-components (testability).
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        config: PipelineConfig,
        memory_manager: Any,
        dataset_loader: IDatasetLoader,
        metrics_collector: MetricsCollector,
        *,
        shutdown_handler: ShutdownHandler | None = None,
        strategy_factory: IStrategyFactory | None = None,
        trainer_factory: ITrainerFactory | None = None,
        mlflow_manager: IMLflowManager | None = None,
        # DI for sub-components (useful in tests)
        cache_manager: AdapterCacheManager | None = None,
        mlflow_logger: MlflowPhaseLogger | None = None,
        training_runner: PhaseTrainingRunner | None = None,
    ):
        self.tokenizer = tokenizer
        self.config = config
        self.shutdown_handler = shutdown_handler
        self._mlflow_manager: IMLflowManager | None = mlflow_manager

        # Sub-components (injected or built from args)
        self._cache_manager: AdapterCacheManager = cache_manager or AdapterCacheManager(config)
        self._mlflow_logger: MlflowPhaseLogger = mlflow_logger or MlflowPhaseLogger(
            mlflow_manager, config
        )
        self._training_runner: PhaseTrainingRunner = training_runner or PhaseTrainingRunner(
            tokenizer=tokenizer,
            config=config,
            memory_manager=memory_manager,
            dataset_loader=dataset_loader,
            metrics_collector=metrics_collector,
            strategy_factory=strategy_factory,
            trainer_factory=trainer_factory,
            mlflow_manager=mlflow_manager,
            shutdown_handler=shutdown_handler,
        )

        # Expose factories for backward compatibility
        self.memory_manager = memory_manager
        self.dataset_loader: IDatasetLoader = dataset_loader
        self.metrics_collector = metrics_collector
        self.strategy_factory = self._training_runner.strategy_factory
        self.trainer_factory = self._training_runner.trainer_factory

        # Internal tracking (for backward-compat access from tests/callbacks)
        self._current_trainer: Any | None = None
        self._current_output_dir: str | None = None
        self._current_phase_idx: int | None = None

        logger.debug(
            f"[PE:INIT] PhaseExecutor initialized "
            f"(sf_injected={strategy_factory is not None}, tf_injected={trainer_factory is not None}, "
            f"dl_type={type(dataset_loader).__name__}, shutdown_handler={shutdown_handler is not None}, "
            f"mlflow={mlflow_manager is not None})"
        )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def execute(
        self,
        phase_idx: int,
        phase: StrategyPhaseConfig,
        model: PreTrainedModel,
        buffer: DataBuffer,
        *,
        upstream_retrained: bool = False,
    ) -> Result[PreTrainedModel, TrainingError]:
        """
        Execute a single training phase.

        Steps:
        1. Check for pending shutdown
        2. [Adapter cache] If enabled and not upstream_retrained: check HF for cached adapter
        3. Mark phase started
        4-12. Delegate training to PhaseTrainingRunner
        13. [Adapter cache] Upload trained adapter (soft-fail)

        Returns:
            Result[PreTrainedModel, TrainingError]
        """
        self._current_trainer = None
        self._current_output_dir = None
        self._current_phase_idx = phase_idx

        # 0. CHECK FOR PENDING SHUTDOWN
        if self._should_stop():
            logger.warning(
                f"[PE:SHUTDOWN_BEFORE_START] phase={phase_idx} - shutdown requested before phase start"
            )
            buffer.mark_phase_interrupted(
                phase_idx,
                reason="Shutdown requested before phase start",
            )
            return Err(
                TrainingError(
                    message="Shutdown requested before phase start",
                    code="TRAINING_SHUTDOWN_BEFORE_START",
                )
            )

        # 1. ADAPTER CACHE CHECK
        dataset_fingerprint: str | None = None
        if phase.adapter_cache.enabled:
            dataset_fingerprint = self._compute_dataset_fingerprint_safe(phase_idx, phase)
            if not upstream_retrained and dataset_fingerprint is not None:
                cache_result = self._try_adapter_cache_hit(
                    phase_idx, phase, model, buffer, dataset_fingerprint
                )
                if cache_result is not None:
                    return cache_result
            elif upstream_retrained:
                logger.warning(
                    f"[PE:ADAPTER_CACHE_MISS_FORCED] phase={phase_idx} "
                    f"({phase.strategy_type}): upstream phase was retrained — skipping cache lookup"
                )

        # 2. MARK PHASE STARTED
        buffer.mark_phase_started(phase_idx)
        logger.debug(f"[PE:START] phase={phase_idx}, strategy={phase.strategy_type}")

        if self._mlflow_manager:
            self._mlflow_manager.log_event_start(
                f"Phase {phase_idx} ({phase.strategy_type.upper()}) started",
                category=CATEGORY_TRAINING,
                source=f"PhaseExecutor:{phase_idx}",
                phase_idx=phase_idx,
                strategy_type=phase.strategy_type,
            )

        # 3. START MLFLOW NESTED RUN
        nested_run_ctx = self._mlflow_logger.start_nested_run(phase_idx, phase)
        phase_succeeded = False

        try:
            self._mlflow_logger.log_phase_start(phase_idx, phase)

            # 4-12. TRAINING (includes dataset loading, trainer creation, training, checkpointing)
            run_result = self._training_runner.run(phase_idx, phase, model, buffer)
            if run_result.is_failure():
                return run_result

            trained_model, final_checkpoint, metrics = run_result.unwrap()

            # 13. MLFLOW COMPLETION LOGGING
            self._mlflow_logger.log_completion(
                phase_idx, metrics, str(final_checkpoint)
            )

            # 14. ADAPTER CACHE UPLOAD (soft-fail)
            if phase.adapter_cache.enabled and dataset_fingerprint is not None:
                self._upload_adapter_to_cache(
                    phase_idx, phase, final_checkpoint, buffer, dataset_fingerprint
                )

            logger.debug(f"[PE:COMPLETE] phase={phase_idx}, strategy={phase.strategy_type}")
            phase_succeeded = True
            return Ok(trained_model)

        finally:
            mlflow_status = "FINISHED" if phase_succeeded else "FAILED"
            self._mlflow_logger.end_nested_run(nested_run_ctx, status=mlflow_status)
            self._current_trainer = None
            self._current_output_dir = None
            self._current_phase_idx = None

    def _should_stop(self) -> bool:
        """Check if shutdown was requested."""
        if self.shutdown_handler is not None:
            return self.shutdown_handler.should_stop()
        return False

    # ------------------------------------------------------------------
    # Backward-compat delegates for tests that call these directly
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_dataset_fingerprint(dataset_config: Any) -> str:
        """Backward-compat alias — delegates to AdapterCacheManager."""
        return AdapterCacheManager._compute_dataset_fingerprint(dataset_config)

    def _compute_dataset_fingerprint_safe(
        self, phase_idx: int, phase: StrategyPhaseConfig
    ) -> str | None:
        """Backward-compat alias — delegates to cache manager."""
        return self._cache_manager.compute_fingerprint_safe(phase_idx, phase)

    @staticmethod
    def _retry_call(fn: Any, retries: int = 3, delay_s: int = 10, label: str = "") -> Any:
        """Backward-compat alias — delegates to adapter_cache._retry_call."""
        return _retry_call(fn, retries=retries, delay_s=delay_s, label=label)

    def _try_adapter_cache_hit(
        self,
        phase_idx: int,
        phase: StrategyPhaseConfig,
        model: PreTrainedModel,
        buffer: DataBuffer,
        fingerprint: str,
    ) -> Result[PreTrainedModel, TrainingError] | None:
        """Backward-compat alias — delegates to cache manager."""
        return self._cache_manager.try_hit(phase_idx, phase, model, buffer, fingerprint)

    def _upload_adapter_to_cache(
        self,
        phase_idx: int,
        phase: StrategyPhaseConfig,
        checkpoint_path: Path,
        buffer: DataBuffer,
        fingerprint: str,
    ) -> None:
        """Backward-compat alias — delegates to cache manager."""
        return self._cache_manager.upload(phase_idx, phase, checkpoint_path, buffer, fingerprint)

    def _save_checkpoint(self, trainer: Any, output_dir: str) -> Path:
        """Backward-compat alias — delegates to training runner."""
        return self._training_runner._save_checkpoint(trainer, output_dir)

    def _handle_graceful_shutdown(
        self,
        buffer: DataBuffer,
        phase_idx: int,
        trainer: Any | None,
        output_dir: str | None,
    ) -> Result[Any, TrainingError]:
        """Backward-compat alias — delegates to training runner."""
        return self._training_runner.handle_graceful_shutdown(buffer, phase_idx, trainer, output_dir)

    def _handle_error(
        self,
        buffer: DataBuffer,
        phase_idx: int,
        error_type: str,
        error: Exception,
    ) -> Result[Any, TrainingError]:
        """Backward-compat alias — delegates to training runner."""
        return self._training_runner.handle_error(buffer, phase_idx, error_type, error)

    # MLflow helpers — backward-compat for any external callers
    def _mlflow_start_nested_run(self, phase_idx: int, phase: StrategyPhaseConfig) -> Any:
        return self._mlflow_logger.start_nested_run(phase_idx, phase)

    def _mlflow_end_nested_run(self, run: Any, status: str = "FINISHED") -> None:
        self._mlflow_logger.end_nested_run(run, status=status)

    def _mlflow_log_phase_start(self, phase_idx: int, phase: StrategyPhaseConfig) -> None:
        self._mlflow_logger.log_phase_start(phase_idx, phase)

    def _mlflow_log_dataset(self, phase_idx: int, dataset: Any, phase: StrategyPhaseConfig) -> None:
        self._mlflow_logger.log_dataset(phase_idx, dataset, phase)

    def _mlflow_log_completion(
        self, phase_idx: int, metrics: dict[str, Any], checkpoint_path: str
    ) -> None:
        self._mlflow_logger.log_completion(phase_idx, metrics, checkpoint_path)

    def _mlflow_log_error(self, phase_idx: int, error_type: str, error_msg: str) -> None:
        self._mlflow_logger.log_error(phase_idx, error_type, error_msg)


__all__ = ["PhaseExecutor"]
