"""
PhaseExecutor - Execute single training phase.

Thin orchestrator that delegates to:
- AdapterCacheManager  (adapter_cache.py)
- MlflowPhaseLogger    (mlflow_logger.py)
- PhaseTrainingRunner  (training_runner.py)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ryotenkai_pod.trainer.orchestrator.phase_executor.adapter_cache import (
    AdapterCacheManager,
    _retry_call,
)
from ryotenkai_pod.trainer.orchestrator.phase_executor.training_runner import (
    TRAINING_INTERRUPTED_LEGACY_CODE,
    PhaseTrainingRunner,
)


class _NoOpMlflowPhaseLogger:
    """Stub for the deleted ``MlflowPhaseLogger``.

    Pattern A (M4) makes the HF Trainer's MLflow callback adopt the
    parent run via ``MLFLOW_RUN_ID`` + ``MLFLOW_NESTED_RUN=TRUE``, so
    starting nested runs from the trainer subprocess is no longer
    needed. Phase-level observability now flows through typed events
    on the journal. All methods are inert.
    """

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    def start_nested_run(self, *_args: Any, **_kwargs: Any) -> Any:
        return None

    def end_nested_run(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def log_phase_start(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def log_cache_hit(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def log_completion(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def log_dataset(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def log_error(self, *_args: Any, **_kwargs: Any) -> None:
        return None


# Public alias preserved for callers that still type-annotate against
# ``MlflowPhaseLogger`` (no production code should rely on the legacy
# concrete API any more).
MlflowPhaseLogger = _NoOpMlflowPhaseLogger
from ryotenkai_shared.errors import RyotenkAIError, TrainingFailedError
from ryotenkai_shared.utils.logger import logger

if TYPE_CHECKING:
    from pathlib import Path

    from transformers import PreTrainedModel, PreTrainedTokenizer

    from ryotenkai_pod.trainer.container import IDatasetLoader, IStrategyFactory, ITrainerFactory
    from ryotenkai_pod.trainer.managers.data_buffer import DataBuffer
    from ryotenkai_pod.trainer.orchestrator.metrics_collector import MetricsCollector
    from ryotenkai_pod.trainer.orchestrator.shutdown_handler import ShutdownHandler
    from ryotenkai_shared.config import PipelineConfig, StrategyPhaseConfig


# Phase 9.A: error code emitted by ``PhaseTrainingRunner.handle_graceful_shutdown``.
# Pinned here as a constant rather than imported because the symbol is the
# *contract* between training_runner and executor — neither side owns it.
_TRAINING_INTERRUPTED_CODE = TRAINING_INTERRUPTED_LEGACY_CODE


def _is_cancellation_error(exc: BaseException) -> bool:
    """Was this exception raised because the user cancelled training?

    Returns True only when the failure carries the cancellation legacy
    code (``TRAINING_INTERRUPTED``) — not for generic training failures.
    Used by the finally block in :meth:`PhaseExecutor.execute` to pick
    MLflow ``RunStatus.KILLED`` over ``FAILED`` so cancelled runs are
    visually distinct on the MLflow UI from genuine crashes.

    Post-Batch-14 contract: the cancellation signal flows through
    :class:`TrainingFailedError` instances whose ``context["legacy_code"]``
    is ``"TRAINING_INTERRUPTED"``. The truth-table at the call site is
    unchanged.
    """
    if not isinstance(exc, RyotenkAIError):
        return False
    legacy = exc.context.get("legacy_code") if exc.context else None
    return legacy == _TRAINING_INTERRUPTED_CODE


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
        # DI for sub-components (useful in tests)
        cache_manager: AdapterCacheManager | None = None,
        mlflow_logger: MlflowPhaseLogger | None = None,
        training_runner: PhaseTrainingRunner | None = None,
    ):
        self.tokenizer = tokenizer
        self.config = config
        self.shutdown_handler = shutdown_handler

        # Sub-components (injected or built from args)
        self._cache_manager: AdapterCacheManager = cache_manager or AdapterCacheManager(config)
        self._mlflow_logger: MlflowPhaseLogger = mlflow_logger or MlflowPhaseLogger(config)
        self._training_runner: PhaseTrainingRunner = training_runner or PhaseTrainingRunner(
            tokenizer=tokenizer,
            config=config,
            memory_manager=memory_manager,
            dataset_loader=dataset_loader,
            metrics_collector=metrics_collector,
            trainer_factory=trainer_factory,
            shutdown_handler=shutdown_handler,
        )

        # Expose factories for backward compatibility
        self.memory_manager = memory_manager
        self.dataset_loader: IDatasetLoader = dataset_loader
        self.metrics_collector = metrics_collector
        # strategy_factory kept as attribute for backward compat; no longer used in training loop
        if strategy_factory is None:
            from ryotenkai_pod.trainer.strategies.factory import StrategyFactory

            self.strategy_factory = StrategyFactory()
        else:
            self.strategy_factory = strategy_factory  # type: ignore[assignment]
        self.trainer_factory = self._training_runner.trainer_factory

        # Internal tracking (for backward-compat access from tests/callbacks)
        self._current_trainer: Any | None = None
        self._current_output_dir: str | None = None
        self._current_phase_idx: int | None = None

        logger.debug(
            f"[PE:INIT] PhaseExecutor initialized "
            f"(sf_injected={strategy_factory is not None}, tf_injected={trainer_factory is not None}, "
            f"dl_type={type(dataset_loader).__name__}, shutdown_handler={shutdown_handler is not None})"
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
    ) -> PreTrainedModel:
        """
        Execute a single training phase.

        Steps:
        1. Check for pending shutdown
        2. [Adapter cache] If enabled and not upstream_retrained: check HF for cached adapter
        3. Mark phase started
        4-12. Delegate training to PhaseTrainingRunner
        13. [Adapter cache] Upload trained adapter (soft-fail)

        Returns:
            PreTrainedModel: trained or cached adapter model.

        Raises:
            TrainingFailedError: on shutdown-before-start, cancellation, or
                generic phase failure.
            TrainingOOMError: OOM recovery exhausted.
            DatasetLoadFailedError: dataset loader failed.
        """
        self._current_trainer = None
        self._current_output_dir = None
        self._current_phase_idx = phase_idx

        # 0. CHECK FOR PENDING SHUTDOWN
        if self._should_stop():
            logger.warning(f"[PE:SHUTDOWN_BEFORE_START] phase={phase_idx} - shutdown requested before phase start")
            buffer.mark_phase_interrupted(
                phase_idx,
                reason="Shutdown requested before phase start",
            )
            raise TrainingFailedError(
                detail="Shutdown requested before phase start",
                context={
                    "legacy_code": "TRAINING_SHUTDOWN_BEFORE_START",
                    "phase_idx": phase_idx,
                },
            )

        # 1. START MLFLOW NESTED RUN (before cache check so cache hits are tracked)
        nested_run_ctx = self._mlflow_logger.start_nested_run(phase_idx, phase)
        phase_succeeded = False
        # Phase 9.A: track user-initiated cancellation separately from
        # genuine failure. Without this flag the finally block below
        # closes the nested MLflow run as ``FAILED`` whenever
        # ``phase_succeeded == False`` — including the cancelled
        # branch, where ``training_runner.handle_graceful_shutdown``
        # raises ``TrainingFailedError`` with
        # ``context["legacy_code"]="TRAINING_INTERRUPTED"``. The flag
        # turns into ``RunStatus.KILLED`` in the finally block, the
        # canonical MLflow signal for "stopped by user".
        was_cancelled = False

        try:
            self._mlflow_logger.log_phase_start(phase_idx, phase)

            # 2. ADAPTER CACHE CHECK
            dataset_fingerprint: str | None = None
            if phase.adapter_cache.enabled:
                dataset_fingerprint = self._compute_dataset_fingerprint_safe(phase_idx, phase)
                if not upstream_retrained and dataset_fingerprint is not None:
                    cached_model = self._try_adapter_cache_hit(
                        phase_idx, phase, model, buffer, dataset_fingerprint
                    )
                    if cached_model is not None:
                        self._mlflow_logger.log_cache_hit(phase_idx, phase)
                        phase_succeeded = True
                        return cached_model
                elif upstream_retrained:
                    logger.warning(
                        f"[PE:ADAPTER_CACHE_MISS_FORCED] phase={phase_idx} "
                        f"({phase.strategy_type}): upstream phase was retrained — skipping cache lookup"
                    )

            # 3. MARK PHASE STARTED
            buffer.mark_phase_started(phase_idx)
            logger.debug(f"[PE:START] phase={phase_idx}, strategy={phase.strategy_type}")

            # Phase 7: ``log_event_start`` removed; phase start is
            # captured via :class:`TrainingStartedEvent` from the
            # trainer entrypoint.

            # 4-12. TRAINING (includes dataset loading, trainer creation, training, checkpointing)
            trained_model, final_checkpoint, metrics = self._training_runner.run(
                phase_idx, phase, model, buffer
            )

            # 13. MLFLOW COMPLETION LOGGING
            self._mlflow_logger.log_completion(
                phase_idx,
                metrics,  # type: ignore[arg-type]
                str(final_checkpoint),
            )

            # 14. ADAPTER CACHE UPLOAD (soft-fail)
            if phase.adapter_cache.enabled and dataset_fingerprint is not None:
                self._upload_adapter_to_cache(phase_idx, phase, final_checkpoint, buffer, dataset_fingerprint)

            logger.debug(f"[PE:COMPLETE] phase={phase_idx}, strategy={phase.strategy_type}")
            phase_succeeded = True
            return trained_model

        except RyotenkAIError as exc:
            # Distinguish user-initiated cancellation from a genuine
            # training failure. The training runner tags graceful
            # shutdown with ``context["legacy_code"]="TRAINING_INTERRUPTED"``
            # (see ``handle_graceful_shutdown``); anything else is a
            # real failure (OOM, dataset corrupt, plugin error, ...).
            was_cancelled = _is_cancellation_error(exc)
            raise

        finally:
            # MLflow ``RunStatus`` is the single source of truth for
            # how an operator distinguishes "user cancelled" from
            # "training failed" on the MLflow UI. Phase 9.1.C decision.
            if was_cancelled:
                mlflow_status = "KILLED"
            elif phase_succeeded:
                mlflow_status = "FINISHED"
            else:
                mlflow_status = "FAILED"
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

    def _compute_dataset_fingerprint_safe(self, phase_idx: int, phase: StrategyPhaseConfig) -> str | None:
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
    ) -> PreTrainedModel | None:
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
    ) -> None:
        """Backward-compat alias — delegates to training runner. Always raises."""
        self._training_runner.handle_graceful_shutdown(buffer, phase_idx, trainer, output_dir)

    def _handle_error(
        self,
        buffer: DataBuffer,
        phase_idx: int,
        error_type: str,
        error: Exception,
    ) -> None:
        """Backward-compat alias — delegates to training runner. Always raises."""
        self._training_runner.handle_error(buffer, phase_idx, error_type, error)

    # MLflow helpers — backward-compat for any external callers
    def _mlflow_start_nested_run(self, phase_idx: int, phase: StrategyPhaseConfig) -> Any:
        return self._mlflow_logger.start_nested_run(phase_idx, phase)

    def _mlflow_end_nested_run(self, run: Any, status: str = "FINISHED") -> None:
        self._mlflow_logger.end_nested_run(run, status=status)

    def _mlflow_log_phase_start(self, phase_idx: int, phase: StrategyPhaseConfig) -> None:
        self._mlflow_logger.log_phase_start(phase_idx, phase)

    def _mlflow_log_dataset(self, phase_idx: int, dataset: Any, phase: StrategyPhaseConfig) -> None:
        self._mlflow_logger.log_dataset(phase_idx, dataset, phase)

    def _mlflow_log_completion(self, phase_idx: int, metrics: dict[str, Any], checkpoint_path: str) -> None:
        self._mlflow_logger.log_completion(phase_idx, metrics, checkpoint_path)  # type: ignore[arg-type]

    def _mlflow_log_error(self, phase_idx: int, error_type: str, error_msg: str) -> None:
        self._mlflow_logger.log_error(phase_idx, error_type, error_msg)


__all__ = ["PhaseExecutor"]
