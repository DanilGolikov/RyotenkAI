"""
PhaseTrainingRunner — Core training loop for a single pipeline phase.

Handles:
- Dataset loading
- Trainer creation with OOM protection
- Training execution with graceful shutdown support
- Checkpoint saving
- Metrics extraction
- Error handling (OOM, validation, unexpected, graceful shutdown)
"""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import TYPE_CHECKING, Any

from tqdm.contrib.logging import logging_redirect_tqdm

from src.training.constants import (
    CATEGORY_TRAINING,
    TRUNCATE_ERROR_MSG,
    TRUNCATE_ERROR_SHORT,
    TAG_PHASE_IDX,
    TAG_STRATEGY_TYPE,
)
from src.training.trainers.factory import TrainerFactory
from src.utils.logger import logger
from src.utils.memory_manager import MemoryManager, OOMRecoverableError
from src.utils.result import Err, Ok, Result, TrainingError

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer

    from src.training.managers.data_buffer import DataBuffer
    from src.training.orchestrator.shutdown_handler import ShutdownHandler
    from src.utils.config import PipelineConfig, StrategyPhaseConfig
    from src.utils.container import IDatasetLoader, IMLflowManager, ITrainerFactory


class PhaseTrainingRunner:
    """
    Executes the core training workflow for a single phase.

    Responsibilities:
    - Dataset loading (format already validated by Stage 0)
    - Trainer lifecycle (create, train, save)
    - OOM recovery via MemoryManager
    - Graceful shutdown checkpoint saving
    - Error classification and marking phase failed
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        config: PipelineConfig,
        memory_manager: MemoryManager,
        dataset_loader: IDatasetLoader,
        metrics_collector: Any,
        trainer_factory: ITrainerFactory | None = None,
        mlflow_manager: IMLflowManager | None = None,
        shutdown_handler: ShutdownHandler | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.config = config
        self.memory_manager = memory_manager
        self.dataset_loader: IDatasetLoader = dataset_loader
        self.metrics_collector = metrics_collector
        self._mlflow_manager = mlflow_manager
        self.shutdown_handler = shutdown_handler

        self.trainer_factory: ITrainerFactory = (
            trainer_factory if trainer_factory is not None else TrainerFactory()
        )

        # Tracking for emergency checkpoints
        self._current_trainer: Any | None = None
        self._current_output_dir: str | None = None

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(
        self,
        phase_idx: int,
        phase: StrategyPhaseConfig,
        model: PreTrainedModel,
        buffer: DataBuffer,
    ) -> Result[tuple[PreTrainedModel, Path, dict], TrainingError]:
        """
        Execute the training workflow for one phase.

        Returns:
            Ok((trained_model, final_checkpoint_path, metrics)) on success
            Err(TrainingError) on failure or interruption
        """
        self._current_trainer = None
        self._current_output_dir = None

        try:
            output_dir = buffer.get_phase_output_dir(phase_idx)
            self._current_output_dir = output_dir
            logger.info(f"   Output: {output_dir}")

            dataset_result = self.dataset_loader.load_for_phase(phase)
            if dataset_result.is_failure():
                buffer.mark_phase_failed(phase_idx, dataset_result.error)  # type: ignore[union-attr]
                return dataset_result

            train_dataset, eval_dataset = dataset_result.unwrap()
            logger.info(f"   Dataset loaded: {len(train_dataset) if hasattr(train_dataset, '__len__') else '?'} samples")

            if self._mlflow_manager:
                self._mlflow_manager.log_event_info(
                    f"Dataset prepared: {len(train_dataset)} samples",
                    category=CATEGORY_TRAINING,
                    source=f"PhaseExecutor:{phase_idx}",
                    phase_idx=phase_idx,
                    samples=len(train_dataset),
                )

            self._log_dataset(phase_idx, train_dataset, phase)

            trainer = self._create_trainer(
                phase_idx,
                phase,
                model,
                train_dataset,
                output_dir=output_dir,
                eval_dataset=eval_dataset,
            )
            self._current_trainer = trainer

            resume_checkpoint = buffer.get_resume_checkpoint(phase_idx)
            if resume_checkpoint:
                logger.info(f"   Resuming from checkpoint: {resume_checkpoint}")

            trained_model = self._run_training(phase_idx, trainer, resume_checkpoint, buffer)

            if self._should_stop():
                return self.handle_graceful_shutdown(buffer, phase_idx, trainer, output_dir)

            final_checkpoint = self._save_checkpoint(trainer, output_dir)

            metrics = self.metrics_collector.extract_from_trainer(trainer)

            buffer.mark_phase_completed(
                phase_idx,
                checkpoint_path=str(final_checkpoint),
                metrics=metrics,
            )

            if self._mlflow_manager:
                train_loss = metrics.train_loss
                self._mlflow_manager.log_event_complete(
                    f"Phase {phase_idx} ({phase.strategy_type.upper()}) completed"
                    + (f", loss={train_loss:.4f}" if train_loss else ""),
                    category=CATEGORY_TRAINING,
                    source=f"PhaseExecutor:{phase_idx}",
                    phase_idx=phase_idx,
                    strategy_type=phase.strategy_type,
                    checkpoint=str(final_checkpoint),
                    **metrics.numeric_kwargs(),
                )

            buffer.cleanup_old_checkpoints(keep_last=2)

            return Ok((trained_model, final_checkpoint, metrics))

        except OOMRecoverableError as e:
            if self._mlflow_manager:
                self._mlflow_manager.log_oom(
                    operation=f"phase_{phase_idx}_{phase.strategy_type}",
                    free_mb=None,
                )
            return self.handle_error(buffer, phase_idx, "OOM", e)

        except ValueError as e:
            return self.handle_error(buffer, phase_idx, "Validation", e)

        except KeyboardInterrupt:
            return self.handle_graceful_shutdown(
                buffer, phase_idx, self._current_trainer, self._current_output_dir
            )

        except Exception as e:
            if self._should_stop():
                return self.handle_graceful_shutdown(
                    buffer, phase_idx, self._current_trainer, self._current_output_dir
                )
            return self.handle_error(buffer, phase_idx, "Unexpected", e)

        finally:
            self._current_trainer = None
            self._current_output_dir = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _should_stop(self) -> bool:
        if self.shutdown_handler is not None:
            return self.shutdown_handler.should_stop()
        return False

    def _create_trainer(
        self,
        phase_idx: int,
        phase_config: StrategyPhaseConfig,
        model: PreTrainedModel,
        train_dataset: Any,
        *,
        output_dir: str,
        eval_dataset: Any | None = None,
    ) -> Any:
        """Create TRL trainer with memory protection."""
        logger.debug(f"[PE:CREATE_TRAINER] phase={phase_idx}")

        def extract_trainer_context(_self, _phase_idx, phase_config, model, _train_dataset, **_kwargs):
            return {
                "phase": _phase_idx,
                "strategy": phase_config.strategy_type,
                "model_params": sum(p.numel() for p in model.parameters()),
            }

        @self.memory_manager.with_memory_protection(
            f"create_trainer_phase_{phase_idx}",
            max_retries=2,
            context_factory=extract_trainer_context,
        )
        def protected_create(_self, _phase_idx, phase_config, model, train_dataset, **kwargs):
            return _self.trainer_factory.create_from_phase(
                phase=phase_config,
                model=model,
                tokenizer=_self.tokenizer,
                train_dataset=train_dataset,
                output_dir=kwargs.get("output_dir"),
                eval_dataset=kwargs.get("eval_dataset"),
                config=_self.config,
                mlflow_manager=_self._mlflow_manager,
            )

        trainer = protected_create(
            self,
            phase_idx,
            phase_config,
            model,
            train_dataset,
            output_dir=output_dir,
            eval_dataset=eval_dataset,
        )

        logger.debug(f"[PE:TRAINER_CREATED] class={trainer.__class__.__name__}")
        return trainer

    def _run_training(
        self,
        phase_idx: int,
        trainer: Any,
        resume_checkpoint: str | None,
        _buffer: DataBuffer,
    ) -> PreTrainedModel:
        """Run training with memory protection and graceful shutdown support."""
        logger.info("   \U0001f3c3 Training started...")
        logger.debug(f"[PE:TRAIN_START] phase={phase_idx}")

        def extract_training_context(_self, phase_idx, trainer, _resume_checkpoint, __buffer):
            return {
                "phase": phase_idx,
                "batch_size": getattr(trainer.args, "per_device_train_batch_size", "unknown"),
            }

        @self.memory_manager.with_memory_protection(
            f"train_phase_{phase_idx}", context_factory=extract_training_context
        )
        def protected_train(_self, _phase_idx, trainer, resume_checkpoint, __buffer):
            loggers = [logger] if logger.handlers else None
            tqdm_logging = logging_redirect_tqdm(loggers=loggers) if loggers else nullcontext()
            with tqdm_logging:
                trainer.train(resume_from_checkpoint=resume_checkpoint)
            return trainer.model

        trained_model = protected_train(self, phase_idx, trainer, resume_checkpoint, _buffer)

        if self._should_stop():
            logger.warning(
                f"[PE:TRAIN_INTERRUPTED] phase={phase_idx} - shutdown requested during training"
            )
        else:
            logger.debug(f"[PE:TRAIN_COMPLETE] phase={phase_idx}")

        return trained_model

    def _save_checkpoint(self, trainer: Any, output_dir: str) -> Path:
        """Save final checkpoint."""
        final_checkpoint = Path(output_dir) / "checkpoint-final"
        trainer.save_model(str(final_checkpoint))
        logger.info(f"   Checkpoint saved: {final_checkpoint}")
        logger.debug(f"[PE:CHECKPOINT_SAVED] path={final_checkpoint}")

        if self._mlflow_manager:
            self._mlflow_manager.log_event_checkpoint(
                f"Checkpoint saved: {final_checkpoint.name}",
                category=CATEGORY_TRAINING,
                source="PhaseExecutor",
                path=str(final_checkpoint),
            )

        return final_checkpoint

    def _log_dataset(
        self,
        phase_idx: int,
        dataset: Any,
        phase: StrategyPhaseConfig,
    ) -> None:
        """Log dataset to MLflow with experiment → dataset → run linking."""
        if self._mlflow_manager is None:
            return

        try:
            num_samples = len(dataset) if hasattr(dataset, "__len__") else 0
            dataset_name = phase.dataset or f"phase_{phase_idx}_dataset"

            dataset_config = self.config.get_dataset_for_strategy(phase)
            source_uri = dataset_config.get_source_uri()

            mlflow_dataset = self._mlflow_manager.create_mlflow_dataset(
                data=dataset,
                name=dataset_name,
                source=source_uri,
            )

            if mlflow_dataset is not None:
                self._mlflow_manager.log_dataset_input(mlflow_dataset, context="training")
                logger.debug(
                    f"[PE:MLFLOW_DATASET_LINKED] name={dataset_name}, "
                    f"source={source_uri}, samples={num_samples}"
                )
            else:
                self._mlflow_manager.log_dataset_info(
                    name=dataset_name,
                    source=dataset_config.get_source_type(),
                    num_rows=num_samples,
                    extra_tags={
                        TAG_PHASE_IDX: str(phase_idx),
                        TAG_STRATEGY_TYPE: phase.strategy_type,
                        "dataset.source_uri": source_uri,
                    },
                )
                logger.debug(f"[PE:MLFLOW_DATASET_INFO] name={dataset_name}, samples={num_samples}")

        except Exception as e:
            logger.debug(f"[PE:MLFLOW_DATASET_LOG_FAILED] {e}")

    def handle_graceful_shutdown(
        self,
        buffer: DataBuffer,
        phase_idx: int,
        trainer: Any | None,
        output_dir: str | None,
    ) -> Result[Any, TrainingError]:
        """Handle graceful shutdown: save checkpoint, mark phase interrupted."""
        shutdown_info = ""
        if self.shutdown_handler is not None:
            info = self.shutdown_handler.get_shutdown_info()
            shutdown_info = f" (reason={info.get('reason', 'unknown')})"

        logger.warning(f"[PE:GRACEFUL_SHUTDOWN] phase={phase_idx}{shutdown_info}")

        if self._mlflow_manager:
            self._mlflow_manager.log_event_warning(
                f"Phase {phase_idx} interrupted{shutdown_info}",
                category=CATEGORY_TRAINING,
                source=f"PhaseExecutor:{phase_idx}",
                phase_idx=phase_idx,
            )

        checkpoint_path: str | None = None

        if trainer is not None and output_dir is not None:
            if self.shutdown_handler is not None:
                checkpoint_path = self.shutdown_handler.save_emergency_checkpoint(
                    trainer=trainer,
                    output_dir=output_dir,
                    phase_idx=phase_idx,
                )
            else:
                try:
                    checkpoint_name = f"checkpoint-interrupted-phase{phase_idx}"
                    cp_path = Path(output_dir) / checkpoint_name
                    cp_path.mkdir(parents=True, exist_ok=True)
                    trainer.save_model(str(cp_path))
                    checkpoint_path = str(cp_path)
                    logger.info(f"\U0001f4be Emergency checkpoint saved: {checkpoint_path}")
                except Exception as e:
                    logger.error(f"[PE:EMERGENCY_CHECKPOINT_FAILED] {e}")

        buffer.mark_phase_interrupted(
            phase_idx=phase_idx,
            reason=f"Training interrupted by user{shutdown_info}",
            checkpoint_path=checkpoint_path,
        )

        return Err(
            TrainingError(
                message=f"Training interrupted at phase {phase_idx}{shutdown_info}",
                code="TRAINING_INTERRUPTED",
            )
        )

    def handle_error(
        self,
        buffer: DataBuffer,
        phase_idx: int,
        error_type: str,
        error: Exception,
    ) -> Result[Any, TrainingError]:
        """Handle and log error, mark phase failed."""
        error_msg = f"{error_type} error in phase {phase_idx}: {error}"
        logger.error(f"[PE:ERROR] type={error_type}, phase={phase_idx}, error={error}")
        logger.exception(error_msg) if error_type == "Unexpected" else logger.error(error_msg)
        buffer.mark_phase_failed(phase_idx, error_msg)

        if self._mlflow_manager:
            self._mlflow_manager.log_event_error(
                f"Phase {phase_idx} failed: {error_type}",
                category=CATEGORY_TRAINING,
                source=f"PhaseExecutor:{phase_idx}",
                phase_idx=phase_idx,
                error_type=error_type,
                error=str(error)[:TRUNCATE_ERROR_SHORT],
            )
            self._mlflow_manager.set_tags(
                {
                    TAG_PHASE_IDX: str(phase_idx),
                    "status": "failed",
                    "error_type": error_type,
                    "error_msg": error_msg[:TRUNCATE_ERROR_MSG],
                }
            )

        code = "TRAINING_OOM_ERROR" if error_type == "OOM" else "TRAINING_PHASE_FAILED"
        return Err(TrainingError(message=error_msg, code=code))


__all__ = ["PhaseTrainingRunner"]
