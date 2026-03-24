"""
StrategyOrchestrator - Facade for Multi-Phase LLM Training.

This is the main entry point that coordinates all orchestrator components:
- DatasetLoader: Load datasets
- MetricsCollector: Extract metrics
- ResumeManager: Handle resume logic
- PhaseExecutor: Execute phases
- ChainRunner: Run phase sequence

Architecture (Facade Pattern):
    StrategyOrchestrator (facade)
           ↓
    ┌──────┴──────┐
    │             │
    ▼             ▼
ResumeManager  ChainRunner
    │             │
    └──────┬──────┘
           ↓
     PhaseExecutor
           ↓
    DatasetLoader + MetricsCollector
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.training.managers.data_buffer import DataBuffer, DataBufferEventCallbacks, PhaseStatus
from src.training.orchestrator.chain_runner import ChainRunner
from src.training.orchestrator.metrics_collector import MetricsCollector
from src.training.orchestrator.phase_executor import PhaseExecutor
from src.training.orchestrator.resume_manager import ResumeManager
from src.training.orchestrator.shutdown_handler import ShutdownHandler
from src.training.strategies.factory import StrategyFactory
from src.training.trainers.factory import TrainerFactory
from src.utils.logger import logger
from src.utils.memory_manager import MemoryManager, get_memory_manager
from src.utils.result import Err, Ok, Result, TrainingError

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer

    from src.utils.config import PipelineConfig, StrategyPhaseConfig
    from src.utils.container import IDatasetLoader, IMLflowManager, IStrategyFactory, ITrainerFactory


class StrategyOrchestrator:
    """
    Facade for multi-phase LLM training orchestration.

    Coordinates all training components with a simple interface.
    Original 550-line class → now ~120 lines (Facade pattern).

    Supports Dependency Injection for factories (testability).

    Example:
        orchestrator = StrategyOrchestrator(model, tokenizer, config)
        result = orchestrator.run_chain()
        if result.is_success():
            final_model = result.unwrap()

        # With DI (for testing)
        orchestrator = StrategyOrchestrator(
            model,
            tokenizer,
            config,
            strategy_factory=mock_sf,
            trainer_factory=mock_tf,
        )
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        config: PipelineConfig,
        *,
        memory_manager: MemoryManager | None = None,
        strategy_factory: IStrategyFactory | None = None,
        trainer_factory: ITrainerFactory | None = None,
        dataset_loader: IDatasetLoader | None = None,
        shutdown_handler: ShutdownHandler | None = None,
        mlflow_manager: IMLflowManager | None = None,
        graceful_shutdown: bool = True,
    ):
        """
        Initialize StrategyOrchestrator.

        Args:
            model: Pre-trained model to fine-tune
            tokenizer: Tokenizer for the model
            config: Pipeline configuration
            memory_manager: Optional custom MemoryManager
            strategy_factory: Optional StrategyFactory instance (for DI/testing)
            trainer_factory: Optional TrainerFactory instance (for DI/testing)
            dataset_loader: Optional IDatasetLoader instance (for DI/testing)
            shutdown_handler: Optional ShutdownHandler for graceful shutdown
            mlflow_manager: Optional MLflowManager for experiment tracking
            graceful_shutdown: If True (default), create ShutdownHandler if not provided
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.memory_manager = memory_manager or get_memory_manager(auto_configure=True)
        self.buffer: DataBuffer | None = None

        # Factories: use injected or create default instances
        self._strategy_factory: IStrategyFactory = (
            strategy_factory if strategy_factory is not None else StrategyFactory()
        )
        self._trainer_factory: ITrainerFactory = trainer_factory if trainer_factory is not None else TrainerFactory()

        # Dataset loader: use injected or create default JsonDatasetLoader
        if dataset_loader is not None:
            self._dataset_loader: IDatasetLoader = dataset_loader
        else:
            from src.data.loaders import JsonDatasetLoader

            self._dataset_loader = JsonDatasetLoader(config)

        # Shutdown handler: for graceful SIGINT/SIGTERM handling
        if shutdown_handler is not None:
            self._shutdown_handler: ShutdownHandler | None = shutdown_handler
        elif graceful_shutdown:
            self._shutdown_handler = ShutdownHandler()
        else:
            self._shutdown_handler = None

        # MLflow manager: for experiment tracking and event logging (optional)
        self._mlflow_manager: IMLflowManager | None = mlflow_manager

        # Create DataBuffer callbacks for MLflow integration
        data_buffer_callbacks = self._create_data_buffer_callbacks()

        # Initialize other components
        self._metrics_collector = MetricsCollector()
        self._resume_manager = ResumeManager(config, data_buffer_callbacks=data_buffer_callbacks)

        self._phase_executor = PhaseExecutor(
            tokenizer=tokenizer,
            config=config,
            memory_manager=self.memory_manager,
            dataset_loader=self._dataset_loader,
            metrics_collector=self._metrics_collector,
            shutdown_handler=self._shutdown_handler,
            strategy_factory=self._strategy_factory,
            trainer_factory=self._trainer_factory,
            mlflow_manager=self._mlflow_manager,
        )

        self._chain_runner = ChainRunner(
            self._phase_executor,
            mlflow_manager=self._mlflow_manager,
        )

        logger.debug(
            f"[SO:INIT] StrategyOrchestrator initialized (Facade) "
            f"(sf_injected={strategy_factory is not None}, tf_injected={trainer_factory is not None}, "
            f"dl_injected={dataset_loader is not None}, dl_type={type(self._dataset_loader).__name__}, "
            f"shutdown_handler={self._shutdown_handler is not None}, "
            f"mlflow_enabled={self._mlflow_manager is not None})"
        )
        logger.info("StrategyOrchestrator initialized")

    def _create_data_buffer_callbacks(self) -> DataBufferEventCallbacks | None:
        """Create DataBuffer callbacks for MLflow integration."""
        if self._mlflow_manager is None:
            return None

        def _on_pipeline_initialized(run_id: str, total_phases: int, strategy_chain: list[str]) -> None:
            if self._mlflow_manager:
                self._mlflow_manager.log_pipeline_initialized(run_id, total_phases, strategy_chain)

        def _on_state_saved(run_id: str, path: str) -> None:
            if self._mlflow_manager:
                self._mlflow_manager.log_state_saved(run_id, path)

        def _on_phase_started(idx: int, strategy: str) -> None:
            if self._mlflow_manager:
                self._mlflow_manager.log_event_start(
                    f"Phase {idx} ({strategy}) started",
                    category="training",
                    source="DataBuffer",
                    phase_idx=idx,
                    strategy=strategy,
                )

        def _on_phase_completed(idx: int, strategy: str, status: str) -> None:
            if self._mlflow_manager:
                self._mlflow_manager.log_event_complete(
                    f"Phase {idx} ({strategy}) {status}",
                    category="training",
                    source="DataBuffer",
                    phase_idx=idx,
                    strategy=strategy,
                    status=status,
                )

        def _on_checkpoint_cleanup(cleaned_count: int, freed_mb: int) -> None:
            if self._mlflow_manager:
                self._mlflow_manager.log_checkpoint_cleanup(cleaned_count, freed_mb)

        return DataBufferEventCallbacks(
            on_pipeline_initialized=_on_pipeline_initialized,
            on_state_saved=_on_state_saved,
            on_phase_started=_on_phase_started,
            on_phase_completed=_on_phase_completed,
            on_checkpoint_cleanup=_on_checkpoint_cleanup,
        )

    def run_chain(
        self,
        strategies: list[StrategyPhaseConfig] | None = None,
        *,
        resume: bool = False,
        run_id: str | None = None,
    ) -> Result[PreTrainedModel, TrainingError]:
        """
        Run the complete training strategy chain.

        Automatically handles:
        - Graceful shutdown on SIGINT/SIGTERM (saves checkpoint)
        - Resume from interrupted/failed phases
        - Strategy chain validation

        Args:
            strategies: List of strategy phases (None = use config)
            resume: If True, resume from last incomplete phase
            run_id: Optional run ID for reproducibility

        Returns:
            Result[PreTrainedModel, TrainingError]: Final trained model or error
        """
        # 1. GET STRATEGIES
        strategies = strategies or self.config.training.get_strategy_chain()
        if not strategies:
            return Err(
                TrainingError(
                    message="No strategies configured. Add training.strategies to config.",
                    code="TRAINING_NO_STRATEGIES",
                )
            )

        # NOTE: Strategy chain validation is done in PipelineOrchestrator (early fail-fast)
        # No need to re-validate here - chain is already validated before deployment

        chain_str = " → ".join(s.strategy_type.upper() for s in strategies)
        logger.debug(f"[SO:CHAIN_START] chain={chain_str}")
        logger.info(f"🚀 Starting training chain: {chain_str}")

        # 3. SETUP RESUME STATE
        self.buffer, start_phase, should_load = self._resume_manager.setup_buffer(
            strategies, resume=resume, run_id=run_id
        )

        # 3.1 CHECK IF PREVIOUS RUN WAS INTERRUPTED
        if resume and self._resume_manager.was_interrupted(self.buffer):
            interrupt_info = self._resume_manager.get_interrupt_info(self.buffer)
            if interrupt_info:
                logger.info(f"⚠️ Previous run was interrupted: {interrupt_info.get('reason')}")
                logger.info(f"   Resuming from phase {interrupt_info.get('phase_idx')}")

        # 4. CHECK IF ALL COMPLETE
        if resume and self._resume_manager.is_all_complete(self.buffer):
            return Ok(self.model)

        # 5. LOAD CHECKPOINT IF RESUMING
        current_model = self.model
        if should_load:
            checkpoint_path = self._resume_manager.get_checkpoint_path_for_phase(self.buffer, start_phase)
            if checkpoint_path:
                load_result = self._resume_manager.load_model_from_checkpoint(checkpoint_path, self.model)
                if load_result.is_failure():
                    return load_result
                loaded_model = load_result.unwrap()
                if loaded_model is None:
                    return Err(
                        TrainingError(  # type: ignore[unreachable]
                            message="Model is None after checkpoint loading",
                            code="TRAINING_CHECKPOINT_LOAD_NULL",
                        )
                    )
                current_model = loaded_model

        # 6. RUN CHAIN WITH GRACEFUL SHUTDOWN SUPPORT
        if self._shutdown_handler is not None:
            # Register signal handlers for graceful shutdown
            self._shutdown_handler.register()
            logger.debug("[SO:SHUTDOWN_HANDLER_REGISTERED]")

        try:
            result = self._chain_runner.run(
                strategies=strategies,
                model=current_model,
                buffer=self.buffer,
                start_phase=start_phase,
            )

            # Check if training was interrupted
            if self._shutdown_handler is not None and self._shutdown_handler.should_stop():
                logger.warning("⚠️ Training was interrupted by user signal")
                # The PhaseExecutor already saved the checkpoint and marked phase as INTERRUPTED

            return result

        finally:
            # Always unregister signal handlers
            if self._shutdown_handler is not None:
                self._shutdown_handler.unregister()
                logger.debug("[SO:SHUTDOWN_HANDLER_UNREGISTERED]")

    def run_single_phase(
        self,
        phase_idx: int,
        phase: StrategyPhaseConfig,
        model: PreTrainedModel | None = None,
    ) -> Result[PreTrainedModel, TrainingError]:
        """
        Run a single training phase (for testing/debugging).

        Args:
            phase_idx: Phase index
            phase: Strategy phase configuration
            model: Model to train (None = use self.model)

        Returns:
            Result[PreTrainedModel, TrainingError]: Trained model or error
        """
        model_to_train = model or self.model

        if self.buffer is None:
            self.buffer = DataBuffer(
                base_output_dir="output",
                base_model_path=self.config.model.name,
            )
            self.buffer.init_pipeline([phase], global_hyperparams=self.config.training.hyperparams)

        return self._phase_executor.execute(
            phase_idx=phase_idx,
            phase=phase,
            model=model_to_train,
            buffer=self.buffer,
        )

    # =========================================================================
    # UTILITY METHODS (kept in Facade for convenience)
    # =========================================================================

    def get_summary(self) -> dict[str, Any]:
        """Get orchestrator status summary."""
        if self.buffer is None:
            return {"status": "not_initialized"}
        return self.buffer.get_summary()

    def can_resume(self) -> bool:
        """Check if there's a previous run that can be resumed."""
        return self._resume_manager.can_resume(self.buffer)

    def get_completed_phases(self) -> list[int]:
        """Get list of completed phase indices."""
        if self.buffer is None or not self.buffer.is_initialized:
            return []
        return [p.phase_idx for p in self.buffer.state.phases if p.status == PhaseStatus.COMPLETED]

    def was_interrupted(self) -> bool:
        """Check if training was interrupted by user signal."""
        if self._shutdown_handler is not None:
            return self._shutdown_handler.should_stop()
        return False

    def get_interrupted_phases(self) -> list[int]:
        """Get list of interrupted phase indices."""
        if self.buffer is None or not self.buffer.is_initialized:
            return []
        return [p.phase_idx for p in self.buffer.state.phases if p.status == PhaseStatus.INTERRUPTED]

    def __repr__(self) -> str:
        """String representation."""
        if self.buffer:
            return f"StrategyOrchestrator({self.buffer})"
        return "StrategyOrchestrator(not_initialized)"


__all__ = ["StrategyOrchestrator"]
