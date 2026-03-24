"""
PhaseExecutor - Execute single training phase.

Single Responsibility: Execute one phase of training with full error handling.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.training.constants import (
    CATEGORY_TRAINING,
    TAG_PHASE_IDX,
    TAG_STRATEGY_TYPE,
    TRUNCATE_ERROR_MSG,
    TRUNCATE_ERROR_SHORT,
)
from src.training.strategies.factory import StrategyFactory
from src.training.trainers.factory import TrainerFactory
from src.utils.logger import logger
from src.utils.memory_manager import MemoryManager, OOMRecoverableError
from src.utils.result import Err, Ok, Result, TrainingError

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

    Responsibilities:
    - Create strategy for data preparation
    - Load and validate dataset
    - Create TRL trainer
    - Run training with OOM protection
    - Save checkpoint
    - Update state in DataBuffer
    - Handle graceful shutdown (SIGINT/SIGTERM)

    Supports Dependency Injection for factories (testability).

    Example:
        executor = PhaseExecutor(
            tokenizer=tokenizer,
            config=config,
            memory_manager=memory_manager,
            dataset_loader=dataset_loader,
            metrics_collector=metrics_collector,
            shutdown_handler=shutdown_handler,  # optional, for graceful shutdown
            strategy_factory=strategy_factory,  # optional, DI
            trainer_factory=trainer_factory,    # optional, DI
        )
        result = executor.execute(
            phase_idx=0,
            phase=phase_config,
            model=model,
            buffer=buffer,
        )
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        config: PipelineConfig,
        memory_manager: MemoryManager,
        dataset_loader: IDatasetLoader,
        metrics_collector: MetricsCollector,
        *,
        shutdown_handler: ShutdownHandler | None = None,
        strategy_factory: IStrategyFactory | None = None,
        trainer_factory: ITrainerFactory | None = None,
        mlflow_manager: IMLflowManager | None = None,
    ):
        """
        Initialize PhaseExecutor.

        Args:
            tokenizer: Tokenizer for the model
            config: Pipeline configuration
            memory_manager: MemoryManager for OOM protection
            dataset_loader: IDatasetLoader for loading data (supports JSON, HuggingFace, etc.)
            metrics_collector: MetricsCollector for extracting metrics
            shutdown_handler: Optional ShutdownHandler for graceful shutdown (SIGINT/SIGTERM)
            strategy_factory: Optional StrategyFactory instance (for DI/testing)
            trainer_factory: Optional TrainerFactory instance (for DI/testing)
            mlflow_manager: Optional MLflowManager for experiment tracking and event logging
        """
        self.tokenizer = tokenizer
        self.config = config
        self.memory_manager = memory_manager
        self.dataset_loader: IDatasetLoader = dataset_loader
        self.metrics_collector = metrics_collector
        self.shutdown_handler = shutdown_handler
        self._mlflow_manager: IMLflowManager | None = mlflow_manager
        self._current_trainer: Any | None = None
        self._current_output_dir: str | None = None
        self._current_phase_idx: int | None = None

        # Factories: use injected or create default instances
        self.strategy_factory: IStrategyFactory = (
            strategy_factory if strategy_factory is not None else StrategyFactory()
        )
        self.trainer_factory: ITrainerFactory = trainer_factory if trainer_factory is not None else TrainerFactory()

        logger.debug(
            f"[PE:INIT] PhaseExecutor initialized "
            f"(sf_injected={strategy_factory is not None}, tf_injected={trainer_factory is not None}, "
            f"dl_type={type(dataset_loader).__name__}, shutdown_handler={shutdown_handler is not None}, "
            f"mlflow={mlflow_manager is not None})"
        )

    def execute(
        self,
        phase_idx: int,
        phase: StrategyPhaseConfig,
        model: PreTrainedModel,
        buffer: DataBuffer,
    ) -> Result[PreTrainedModel, TrainingError]:
        """
        Execute a single training phase.

        Steps:
        1. Check for pending shutdown
        2. Mark phase started
        3. Create strategy (for data preparation)
        4. Load and validate dataset
        5. Prepare dataset with strategy
        6. Create trainer
        7. Run training (with shutdown check)
        8. Save checkpoint
        9. Extract metrics
        10. Mark phase completed

        Args:
            phase_idx: Phase index (0-based)
            phase: Strategy phase configuration
            model: Model to train
            buffer: DataBuffer for state management

        Returns:
            Result[PreTrainedModel, TrainingError]: Trained model or error
        """
        # Reset tracking attributes for this execution
        self._current_trainer = None
        self._current_output_dir = None
        self._current_phase_idx = phase_idx

        # 0. CHECK FOR PENDING SHUTDOWN BEFORE STARTING
        if self._should_stop():
            logger.warning(f"[PE:SHUTDOWN_BEFORE_START] phase={phase_idx} - shutdown requested before phase start")
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

        # 1. MARK PHASE STARTED
        buffer.mark_phase_started(phase_idx)
        logger.debug(f"[PE:START] phase={phase_idx}, strategy={phase.strategy_type}")

        # Log event: phase started
        if self._mlflow_manager:
            self._mlflow_manager.log_event_start(
                f"Phase {phase_idx} ({phase.strategy_type.upper()}) started",
                category=CATEGORY_TRAINING,
                source=f"PhaseExecutor:{phase_idx}",
                phase_idx=phase_idx,
                strategy_type=phase.strategy_type,
            )

        # Start MLflow nested run for this phase
        nested_run_ctx = self._mlflow_start_nested_run(phase_idx, phase)
        phase_succeeded = False  # Track success for MLflow status

        try:
            # Log phase start params to MLflow
            self._mlflow_log_phase_start(phase_idx, phase)

            # 2. GET OUTPUT DIR
            output_dir = buffer.get_phase_output_dir(phase_idx)
            self._current_output_dir = output_dir
            logger.info(f"   Output: {output_dir}")

            # 3. CREATE STRATEGY
            strategy = self._create_strategy(phase)

            # 4. LOAD DATASET
            dataset_result = self.dataset_loader.load_for_phase(phase)
            if dataset_result.is_failure():
                buffer.mark_phase_failed(phase_idx, dataset_result.error)  # type: ignore[union-attr]
                return dataset_result

            raw_train_dataset, raw_eval_dataset = dataset_result.unwrap()

            # 5. VALIDATE DATASET
            validation_result = strategy.validate_dataset(raw_train_dataset)
            if validation_result.is_failure():
                buffer.mark_phase_failed(phase_idx, validation_result.error)
                return validation_result

            if raw_eval_dataset is not None:
                eval_validation = strategy.validate_dataset(raw_eval_dataset)
                if eval_validation.is_failure():
                    buffer.mark_phase_failed(phase_idx, eval_validation.error)
                    return eval_validation

            # 6. PREPARE DATASET
            prepare_result = strategy.prepare_dataset(raw_train_dataset, self.tokenizer)
            if prepare_result.is_failure():
                buffer.mark_phase_failed(phase_idx, prepare_result.error)
                return prepare_result

            train_dataset = prepare_result.unwrap()
            logger.info(f"   Dataset prepared: {len(train_dataset)} samples")

            eval_dataset = None
            if raw_eval_dataset is not None:
                eval_prepare = strategy.prepare_dataset(raw_eval_dataset, self.tokenizer)
                if eval_prepare.is_failure():
                    buffer.mark_phase_failed(phase_idx, eval_prepare.error)
                    return eval_prepare
                eval_dataset = eval_prepare.unwrap()
                logger.info(f"   Eval dataset prepared: {len(eval_dataset)} samples")

            # Log event: dataset prepared
            if self._mlflow_manager:
                self._mlflow_manager.log_event_info(
                    f"Dataset prepared: {len(train_dataset)} samples",
                    category=CATEGORY_TRAINING,
                    source=f"PhaseExecutor:{phase_idx}",
                    phase_idx=phase_idx,
                    samples=len(train_dataset),
                )

            # Log dataset info to MLflow
            self._mlflow_log_dataset(phase_idx, train_dataset, phase)

            # 7. CREATE TRAINER (per-phase output_dir is hardcoded by DataBuffer)
            trainer = self._create_trainer(
                phase_idx,
                phase,
                model,
                train_dataset,
                output_dir=output_dir,
                eval_dataset=eval_dataset,
            )
            self._current_trainer = trainer  # Store for emergency checkpoint

            # 9. CHECK FOR MID-PHASE RESUME
            resume_checkpoint = buffer.get_resume_checkpoint(phase_idx)
            if resume_checkpoint:
                logger.info(f"   Resuming from checkpoint: {resume_checkpoint}")

            # 10. TRAIN (with graceful shutdown support)
            trained_model = self._run_training(phase_idx, trainer, resume_checkpoint, buffer)

            # 11. CHECK IF INTERRUPTED DURING TRAINING
            if self._should_stop():
                return self._handle_graceful_shutdown(buffer, phase_idx, trainer, output_dir)

            # 12. SAVE CHECKPOINT
            final_checkpoint = self._save_checkpoint(trainer, output_dir)

            # 13. EXTRACT METRICS
            metrics = self.metrics_collector.extract_from_trainer(trainer)

            # 14. MARK COMPLETED
            buffer.mark_phase_completed(
                phase_idx,
                checkpoint_path=str(final_checkpoint),
                metrics=metrics,
            )

            # Log event: phase completed
            if self._mlflow_manager:
                train_loss = metrics.get("train_loss")
                self._mlflow_manager.log_event_complete(
                    f"Phase {phase_idx} ({phase.strategy_type.upper()}) completed"
                    + (f", loss={train_loss:.4f}" if train_loss else ""),
                    category=CATEGORY_TRAINING,
                    source=f"PhaseExecutor:{phase_idx}",
                    phase_idx=phase_idx,
                    strategy_type=phase.strategy_type,
                    checkpoint=str(final_checkpoint),
                    **{k: v for k, v in metrics.items() if isinstance(v, int | float)},
                )

            # 15. CLEANUP OLD CHECKPOINTS
            buffer.cleanup_old_checkpoints(keep_last=2)

            # Log completion to MLflow
            self._mlflow_log_completion(phase_idx, metrics, str(final_checkpoint))

            logger.debug(f"[PE:COMPLETE] phase={phase_idx}, strategy={phase.strategy_type}")
            phase_succeeded = True
            return Ok(trained_model)

        except OOMRecoverableError as e:
            # Log event: OOM error via MLflow
            if self._mlflow_manager:
                self._mlflow_manager.log_oom(
                    operation=f"phase_{phase_idx}_{phase.strategy_type}",
                    free_mb=None,
                )
            return self._handle_error(buffer, phase_idx, "OOM", e)

        except ValueError as e:
            return self._handle_error(buffer, phase_idx, "Validation", e)

        except KeyboardInterrupt:
            # Handle Ctrl+C during training
            return self._handle_graceful_shutdown(
                buffer,
                phase_idx,
                self._current_trainer,
                self._current_output_dir,
            )

        except Exception as e:
            # Check if it's an interrupt-related exception
            if self._should_stop():
                return self._handle_graceful_shutdown(
                    buffer,
                    phase_idx,
                    self._current_trainer,
                    self._current_output_dir,
                )
            return self._handle_error(buffer, phase_idx, "Unexpected", e)

        finally:
            # Close MLflow nested run with correct status
            mlflow_status = "FINISHED" if phase_succeeded else "FAILED"
            self._mlflow_end_nested_run(nested_run_ctx, status=mlflow_status)

            # Clear current trainer reference
            self._current_trainer = None
            self._current_output_dir = None
            self._current_phase_idx = None

    def _should_stop(self) -> bool:
        """Check if shutdown was requested."""
        if self.shutdown_handler is not None:
            return self.shutdown_handler.should_stop()
        return False

    def _handle_graceful_shutdown(
        self,
        buffer: DataBuffer,
        phase_idx: int,
        trainer: Any | None,
        output_dir: str | None,
    ) -> Result[Any, TrainingError]:
        """
        Handle graceful shutdown: save checkpoint, mark phase interrupted.

        Args:
            buffer: DataBuffer for state management
            phase_idx: Current phase index
            trainer: TRL trainer (may be None if not created yet)
            output_dir: Output directory for checkpoint

        Returns:
            Err result with shutdown message
        """
        shutdown_info = ""
        if self.shutdown_handler is not None:
            info = self.shutdown_handler.get_shutdown_info()
            shutdown_info = f" (reason={info.get('reason', 'unknown')})"

        logger.warning(f"[PE:GRACEFUL_SHUTDOWN] phase={phase_idx}{shutdown_info}")

        # Log event: phase interrupted via MLflow
        if self._mlflow_manager:
            self._mlflow_manager.log_event_warning(
                f"Phase {phase_idx} interrupted{shutdown_info}",
                category=CATEGORY_TRAINING,
                source=f"PhaseExecutor:{phase_idx}",
                phase_idx=phase_idx,
            )

        checkpoint_path: str | None = None

        # Try to save emergency checkpoint
        if trainer is not None and output_dir is not None:
            if self.shutdown_handler is not None:
                checkpoint_path = self.shutdown_handler.save_emergency_checkpoint(
                    trainer=trainer,
                    output_dir=output_dir,
                    phase_idx=phase_idx,
                )
            else:
                # Fallback: save checkpoint without ShutdownHandler
                try:
                    from pathlib import Path

                    checkpoint_name = f"checkpoint-interrupted-phase{phase_idx}"
                    cp_path = Path(output_dir) / checkpoint_name
                    cp_path.mkdir(parents=True, exist_ok=True)
                    trainer.save_model(str(cp_path))
                    checkpoint_path = str(cp_path)
                    logger.info(f"💾 Emergency checkpoint saved: {checkpoint_path}")
                except Exception as e:
                    logger.error(f"[PE:EMERGENCY_CHECKPOINT_FAILED] {e}")

        # Mark phase as interrupted in DataBuffer
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

    def _create_strategy(self, phase: StrategyPhaseConfig) -> Any:
        """Create training strategy for data preparation."""
        logger.debug(f"[PE:CREATE_STRATEGY] type={phase.strategy_type}")
        strategy = self.strategy_factory.create_from_phase(phase, self.config)
        logger.debug(f"[PE:STRATEGY_CREATED] class={strategy.__class__.__name__}")
        return strategy

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

        # Context factory for trainer creation
        def extract_trainer_context(_self, _phase_idx, phase_config, model, _train_dataset, **_kwargs):
            return {
                "phase": _phase_idx,
                "strategy": phase_config.strategy_type,
                "model_params": sum(p.numel() for p in model.parameters()),
            }

        # Wrap trainer creation with memory protection (lower retries for creation)
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
        """
        Run training with memory protection and graceful shutdown support.

        Note: TRL trainers handle SIGINT internally during train() call.
        The shutdown handler will catch the signal and we check should_stop()
        after training completes (or if interrupted).

        Args:
            phase_idx: Phase index for logging
            trainer: TRL trainer instance
            resume_checkpoint: Path to resume checkpoint (if any)
            _buffer: DataBuffer (unused here, but passed for future callback support)

        Returns:
            Trained model
        """
        logger.info("   🏃 Training started...")
        logger.debug(f"[PE:TRAIN_START] phase={phase_idx}")

        # Context factory to extract batch size from trainer
        def extract_training_context(_self, phase_idx, trainer, _resume_checkpoint, __buffer):
            return {
                "phase": phase_idx,
                "batch_size": getattr(trainer.args, "per_device_train_batch_size", "unknown"),
            }

        # Wrap training with memory protection and auto-retry
        @self.memory_manager.with_memory_protection(
            f"train_phase_{phase_idx}", context_factory=extract_training_context
        )
        def protected_train(_self, _phase_idx, trainer, resume_checkpoint, __buffer):
            trainer.train(resume_from_checkpoint=resume_checkpoint)
            return trainer.model

        trained_model = protected_train(self, phase_idx, trainer, resume_checkpoint, _buffer)

        # Check if shutdown was requested during training
        if self._should_stop():
            logger.warning(f"[PE:TRAIN_INTERRUPTED] phase={phase_idx} - shutdown requested during training")
            # Don't raise here - let execute() handle it
        else:
            logger.debug(f"[PE:TRAIN_COMPLETE] phase={phase_idx}")

        return trained_model

    def _save_checkpoint(self, trainer: Any, output_dir: str) -> Path:
        """Save final checkpoint."""
        final_checkpoint = Path(output_dir) / "checkpoint-final"
        trainer.save_model(str(final_checkpoint))
        logger.info(f"   Checkpoint saved: {final_checkpoint}")
        logger.debug(f"[PE:CHECKPOINT_SAVED] path={final_checkpoint}")

        # Log event: checkpoint saved via MLflow
        if self._mlflow_manager:
            self._mlflow_manager.log_event_checkpoint(
                f"Checkpoint saved: {final_checkpoint.name}",
                category=CATEGORY_TRAINING,
                source="PhaseExecutor",
                path=str(final_checkpoint),
            )

        return final_checkpoint

    def _handle_error(
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

        # Log event: phase failed via MLflow
        if self._mlflow_manager:
            self._mlflow_manager.log_event_error(
                f"Phase {phase_idx} failed: {error_type}",
                category=CATEGORY_TRAINING,
                source=f"PhaseExecutor:{phase_idx}",
                phase_idx=phase_idx,
                error_type=error_type,
                error=str(error)[:TRUNCATE_ERROR_SHORT],
            )

        # Log error to MLflow tags
        self._mlflow_log_error(phase_idx, error_type, str(error))

        code = "TRAINING_OOM_ERROR" if error_type == "OOM" else "TRAINING_PHASE_FAILED"
        return Err(TrainingError(message=error_msg, code=code))

    # =========================================================================
    # MLFLOW INTEGRATION HELPERS
    # =========================================================================

    def _mlflow_start_nested_run(
        self,
        phase_idx: int,
        phase: StrategyPhaseConfig,
    ) -> Any:
        """
        Start MLflow nested run for phase.

        Returns run object or None if MLflow is disabled.
        Uses mlflow.start_run(nested=True) directly for proper status handling.
        """
        if self._mlflow_manager is None or not self._mlflow_manager.is_active:
            return None

        try:
            import mlflow

            run_name = f"phase_{phase_idx}_{phase.strategy_type}"

            # 1. Stop system metrics logging for Parent Run
            # This ensures we can restart it for the Nested Run
            with contextlib.suppress(Exception):
                mlflow.disable_system_metrics_logging()

            # 2. Start nested run directly (not via context manager)
            run = mlflow.start_run(run_name=run_name, nested=True)

            # 3. Start system metrics logging for Nested Run
            mlflow_cfg = self.config.experiment_tracking.mlflow
            if mlflow_cfg and mlflow_cfg.system_metrics_callback_enabled:
                with contextlib.suppress(Exception):
                    mlflow.enable_system_metrics_logging()

            # Set tags
            mlflow.set_tags(
                {
                    TAG_PHASE_IDX: str(phase_idx),
                    TAG_STRATEGY_TYPE: phase.strategy_type,
                }
            )

            logger.debug(f"[PE:MLFLOW_NESTED_RUN_STARTED] {run_name}")
            return run
        except Exception as e:
            logger.debug(f"[PE:MLFLOW_NESTED_RUN_START_FAILED] {e}")
            return None

    def _mlflow_end_nested_run(self, run: Any, status: str = "FINISHED") -> None:
        """
        End MLflow nested run with explicit status and restore parent run.

        Args:
            run: Run object from _mlflow_start_nested_run
            status: Run status - "FINISHED" for success, "FAILED" for error
        """
        if run is None:
            return

        try:
            import mlflow

            # Get parent run ID before closing nested run
            parent_run_id = self._mlflow_manager.parent_run_id if self._mlflow_manager else None

            # 1. Stop system metrics logging for Nested Run
            with contextlib.suppress(Exception):
                mlflow.disable_system_metrics_logging()

            # 2. Close nested run
            mlflow.end_run(status=status)
            logger.debug(f"[PE:MLFLOW_NESTED_RUN_ENDED] status={status}")

            # CRITICAL: Restore parent run as active (mlflow.end_run doesn't do this automatically)
            if parent_run_id:
                mlflow.start_run(run_id=parent_run_id, nested=False)
                logger.debug(f"[PE:MLFLOW_PARENT_RUN_RESTORED] run_id={parent_run_id}")

                # 3. Resume system metrics logging for Parent Run
                mlflow_cfg = self.config.experiment_tracking.mlflow
                if mlflow_cfg and mlflow_cfg.system_metrics_callback_enabled:
                    with contextlib.suppress(Exception):
                        mlflow.enable_system_metrics_logging()

        except Exception as e:
            logger.debug(f"[PE:MLFLOW_NESTED_RUN_END_FAILED] {e}")

    def _mlflow_log_phase_start(
        self,
        phase_idx: int,
        phase: StrategyPhaseConfig,
    ) -> None:
        """Log phase start params to MLflow nested run."""
        if self._mlflow_manager is None:
            return

        # Merge global and phase hyperparams to log effective config
        global_hp = self.config.training.hyperparams
        phase_hp = phase.hyperparams

        # Start with global defaults
        effective_params = global_hp.model_dump(exclude_none=True)

        # Override with phase specific values
        if phase_hp:
            effective_params.update(phase_hp.model_dump(exclude_none=True))

        params_to_log = {
            TAG_PHASE_IDX: phase_idx,
            TAG_STRATEGY_TYPE: phase.strategy_type,
            "dataset": phase.dataset or "default",
        }

        # Log effective hyperparams under training.hyperparams.actual.*
        for k, v in effective_params.items():
            params_to_log[f"training.hyperparams.actual.{k}"] = v

        self._mlflow_manager.log_params(params_to_log)

        self._mlflow_manager.set_tags(
            {
                TAG_STRATEGY_TYPE: phase.strategy_type,
                TAG_PHASE_IDX: str(phase_idx),
            }
        )

    def _mlflow_log_dataset(
        self,
        phase_idx: int,
        dataset: Any,
        phase: StrategyPhaseConfig,
    ) -> None:
        """
        Log dataset to MLflow with proper experiment → dataset → run linking.

        Uses MLflow Dataset API (mlflow.log_input) for proper tracking.
        Falls back to log_dataset_info if Dataset API fails.
        """
        if self._mlflow_manager is None:
            return

        try:
            num_samples = len(dataset) if hasattr(dataset, "__len__") else 0
            dataset_name = phase.dataset or f"phase_{phase_idx}_dataset"

            # Get source URI from dataset config
            dataset_config = self.config.get_dataset_for_strategy(phase)
            source_uri = dataset_config.get_source_uri()

            # Try to create MLflow Dataset and link to run (proper API)
            mlflow_dataset = self._mlflow_manager.create_mlflow_dataset(
                data=dataset,
                name=dataset_name,
                source=source_uri,
            )

            if mlflow_dataset is not None:
                # Link dataset to current run
                self._mlflow_manager.log_dataset_input(mlflow_dataset, context="training")
                logger.debug(
                    f"[PE:MLFLOW_DATASET_LINKED] name={dataset_name}, source={source_uri}, samples={num_samples}"
                )
            else:
                # Fallback: log as params (older method)
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

    def _mlflow_log_completion(
        self,
        phase_idx: int,
        metrics: dict[str, Any],
        checkpoint_path: str,
    ) -> None:
        """Log phase completion to MLflow."""
        if self._mlflow_manager is None:
            return

        # Log final metrics
        float_metrics = {k: float(v) for k, v in metrics.items() if isinstance(v, int | float)}
        if float_metrics:
            self._mlflow_manager.log_metrics(float_metrics)

        self._mlflow_manager.log_params({"checkpoint_path": checkpoint_path})
        self._mlflow_manager.set_tags({TAG_PHASE_IDX: str(phase_idx), "status": "completed"})

    def _mlflow_log_error(
        self,
        phase_idx: int,
        error_type: str,
        error_msg: str,
    ) -> None:
        """Log error to MLflow."""
        if self._mlflow_manager is None:
            return

        self._mlflow_manager.set_tags(
            {
                TAG_PHASE_IDX: str(phase_idx),
                "status": "failed",
                "error_type": error_type,
                "error_msg": error_msg[:TRUNCATE_ERROR_MSG],
            }
        )


__all__ = ["PhaseExecutor"]
