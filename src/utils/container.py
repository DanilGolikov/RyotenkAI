"""
Dependency Injection Container for Training Pipeline.

Provides centralized dependency management for:
- MemoryManager: GPU memory and OOM protection
- StrategyOrchestrator: Multi-phase training coordination
- TrainerFactory: TRL trainer creation
- DatasetLoader: Dataset loading

Key Features:
- Lazy initialization: dependencies created on first access
- Easy override: replace any dependency for testing
- Singleton support: MemoryManager uses single instance
- No external libraries: pure Python implementation

Design Pattern: Service Locator + Dependency Injection

Example:
    # Production usage
    container = TrainingContainer(config)
    model, tokenizer = container.load_model_and_tokenizer()
    orchestrator = container.create_orchestrator(model, tokenizer)

    # Testing with mocks
    container = TrainingContainer.for_testing(
        config,
        memory_manager=MockMemoryManager(),
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, cast, runtime_checkable

from src.utils.logger import logger

if TYPE_CHECKING:
    from datasets import Dataset
    from transformers import PreTrainedModel, PreTrainedTokenizer

    from src.training.mlflow import IMLflowManager  # re-export; canonical home is src/training/mlflow
    from src.training.orchestrator import StrategyOrchestrator
    from src.utils.config import PipelineConfig, StrategyPhaseConfig
    from src.utils.memory_manager import GPUInfo, GPUPreset, MemoryManager, MemoryStats


# =============================================================================
# INTERFACES (Protocols for duck typing)
# =============================================================================


@runtime_checkable
class IMemoryManager(Protocol):
    """
    Interface for memory management.

    Allows mocking MemoryManager in tests without GPU.
    """

    def get_memory_stats(self) -> MemoryStats | None:
        """Get current GPU memory statistics."""
        ...

    def is_memory_critical(self) -> bool:
        """Check if memory is at critical level."""
        ...

    def clear_cache(self) -> int:
        """Clear CUDA cache and return freed MB."""
        ...

    def aggressive_cleanup(self) -> int:
        """Aggressive cleanup for OOM recovery."""
        ...

    def safe_operation(
        self,
        operation_name: str,
        context: dict[str, Any] | None = None,
    ) -> Any:
        """Context manager for memory-safe operations."""
        ...

    @property
    def gpu_info(self) -> GPUInfo | None:
        """Detected GPU info (if CUDA available)."""
        ...

    @property
    def preset(self) -> GPUPreset | None:
        """Active GPU preset (if any)."""
        ...


@runtime_checkable
class ICompletionNotifier(Protocol):
    """
    Interface for training completion notification.

    Allows pluggable notification strategies (marker files, logging, webhook).
    """

    def notify_complete(self, data: dict[str, Any]) -> None:
        """Notify successful completion."""
        ...

    def notify_failed(self, error: str, data: dict[str, Any]) -> None:
        """Notify failure."""
        ...


@runtime_checkable
class IStrategyFactory(Protocol):
    """
    Interface for StrategyFactory.

    Methods:
        create: Create a training strategy by type
        create_from_phase: Create strategy from phase config
        list_available: List all registered strategies
    """

    def create(
        self,
        strategy_type: str,
        config: PipelineConfig,
    ) -> Any:
        """Create a training strategy instance."""
        ...

    def create_from_phase(
        self,
        phase: Any,  # StrategyPhaseConfig
        config: PipelineConfig,
    ) -> Any:
        """Create a strategy from phase configuration."""
        ...

    def list_available(self) -> dict[str, str]:
        """List all available strategies."""
        ...

    def is_registered(self, strategy_type: str) -> bool:
        """Check if strategy type is registered."""
        ...


@runtime_checkable
class IDatasetLoader(Protocol):
    """
    Interface for dataset loading.

    Allows pluggable dataset loading strategies:
    - JSON files (local)
    - HuggingFace Hub datasets
    - Custom dataset formats
    """

    def load(
        self,
        source: str,
        split: str = "train",
        max_samples: int | None = None,
    ) -> Any:  # Dataset
        """Load dataset from source."""
        ...

    def load_for_phase(
        self,
        phase: Any,  # StrategyPhaseConfig
    ) -> Any:  # Result[Dataset, str]
        """Load dataset for a training phase."""
        ...

    def validate_source(self, source: str) -> bool:
        """Validate that dataset source exists/is accessible."""
        ...


@runtime_checkable
class ITrainerFactory(Protocol):
    """
    Interface for TrainerFactory.

    Methods:
        create: Create TRL trainer for a strategy
        get_trainer_class: Get trainer class for strategy
        list_supported_strategies: List supported strategy types
    """

    def create(
        self,
        strategy_type: str,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        train_dataset: Dataset,
        config: PipelineConfig,
        *,
        output_dir: str,
        eval_dataset: Dataset | None = None,
        phase_config: StrategyPhaseConfig | None = None,
        ref_model: PreTrainedModel | None = None,
        mlflow_manager: IMLflowManager | None = None,
        **kwargs: Any,
    ) -> Any:
        """Create a TRL trainer instance."""
        ...

    def create_from_phase(
        self,
        phase: StrategyPhaseConfig,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        train_dataset: Dataset,
        config: PipelineConfig,
        *,
        output_dir: str,
        mlflow_manager: IMLflowManager | None = None,
        **kwargs: Any,
    ) -> Any:
        """Create trainer from a phase config (supports extra kwargs like eval_dataset)."""
        ...

    def get_trainer_class(self, strategy_type: str) -> type:
        """Get the trainer class for a strategy type."""
        ...

    def list_supported_strategies(self) -> list[str]:
        """List all supported strategy types."""
        ...


# =============================================================================
# CONTAINER
# =============================================================================


@dataclass
class TrainingContainer:
    """
    Dependency Injection Container for Training Pipeline.

    Centralizes creation and management of all training dependencies.
    Supports lazy initialization and easy override for testing.

    Attributes:
        config: Pipeline configuration (required)
        _memory_manager: Optional pre-configured MemoryManager
        _completion_notifier: Optional completion notifier

    Usage:
        # Production
        container = TrainingContainer(config)
        model, tokenizer = container.load_model_and_tokenizer()

        # Testing with mock
        mock_mm = MockMemoryManager()
        container = TrainingContainer(config, _memory_manager=mock_mm)
    """

    config: PipelineConfig

    # Private fields for dependency injection
    _memory_manager: IMemoryManager | None = field(default=None, repr=False)
    _completion_notifier: ICompletionNotifier | None = field(default=None, repr=False)
    _strategy_factory: IStrategyFactory | None = field(default=None, repr=False)
    _trainer_factory: ITrainerFactory | None = field(default=None, repr=False)
    _dataset_loader: IDatasetLoader | None = field(default=None, repr=False)
    _mlflow_manager: IMLflowManager | None = field(default=None, repr=False)

    # Cache for lazy-initialized dependencies
    _lazy_memory_manager: MemoryManager | None = field(default=None, repr=False)
    _lazy_strategy_factory: IStrategyFactory | None = field(default=None, repr=False)
    _lazy_trainer_factory: ITrainerFactory | None = field(default=None, repr=False)
    _lazy_dataset_loader: IDatasetLoader | None = field(default=None, repr=False)
    _lazy_mlflow_manager: IMLflowManager | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Log container initialization."""
        logger.debug(
            f"[CONTAINER:INIT] config={self.config.model.name}, "
            f"mm_injected={self._memory_manager is not None}, "
            f"sf_injected={self._strategy_factory is not None}, "
            f"tf_injected={self._trainer_factory is not None}, "
            f"dl_injected={self._dataset_loader is not None}, "
            f"mlflow_injected={self._mlflow_manager is not None}"
        )

    # =========================================================================
    # MEMORY MANAGER
    # =========================================================================

    @property
    def memory_manager(self) -> IMemoryManager:
        """
        Get MemoryManager instance.

        Returns injected mock or creates real auto-configured instance.

        Returns:
            IMemoryManager: Memory manager instance
        """
        # Return injected instance if provided
        if self._memory_manager is not None:
            return self._memory_manager

        # Lazy initialization of real MemoryManager
        if self._lazy_memory_manager is None:
            from src.utils.memory_manager import MemoryManager

            self._lazy_memory_manager = MemoryManager.auto_configure()
            logger.debug("[CONTAINER:MM_CREATED] MemoryManager auto-configured")

        return self._lazy_memory_manager

    def create_memory_manager_with_callbacks(
        self,
        mlflow_manager: IMLflowManager | None = None,
    ) -> IMemoryManager:
        """
        Create MemoryManager with MLflow event callbacks.

        This method creates a new MemoryManager instance with callbacks
        that log memory events to MLflow.

        Args:
            mlflow_manager: MLflowManager for event logging

        Returns:
            IMemoryManager with callbacks configured
        """
        from src.utils.memory_manager import MemoryEventCallbacks, MemoryManager

        # If no MLflow manager, return regular memory manager
        if mlflow_manager is None:
            return self.memory_manager

        # Create callbacks that delegate to MLflowManager
        callbacks = MemoryEventCallbacks(
            on_gpu_detected=lambda name, vram, tier: mlflow_manager.log_gpu_detection(name, vram, tier),
            on_cache_cleared=lambda freed: mlflow_manager.log_cache_cleared(freed),
            on_memory_warning=lambda util, used, total, is_crit: mlflow_manager.log_memory_warning(
                util, used, total, is_critical=is_crit
            ),
            on_oom=lambda op, free: mlflow_manager.log_oom(op, free),
            on_oom_retry=lambda op, attempt, max_att: mlflow_manager.log_oom_recovery(op, attempt, max_att),
        )

        mm = MemoryManager.auto_configure(callbacks=callbacks)
        logger.debug("[CONTAINER:MM_CREATED] MemoryManager with MLflow callbacks")
        return mm

    # =========================================================================
    # STRATEGY FACTORY
    # =========================================================================

    @property
    def strategy_factory(self) -> IStrategyFactory:
        """
        Get StrategyFactory instance.

        Returns injected mock or creates real StrategyFactory instance.

        Returns:
            IStrategyFactory: Strategy factory instance
        """
        if self._strategy_factory is not None:
            return self._strategy_factory

        # Lazy initialization of real StrategyFactory
        if self._lazy_strategy_factory is None:
            from src.training.strategies.factory import StrategyFactory

            self._lazy_strategy_factory = StrategyFactory()
            logger.debug("[CONTAINER:SF_CREATED] StrategyFactory instance created")

        return self._lazy_strategy_factory

    # =========================================================================
    # TRAINER FACTORY
    # =========================================================================

    @property
    def trainer_factory(self) -> ITrainerFactory:
        """
        Get TrainerFactory instance.

        Returns injected mock or creates real TrainerFactory instance.

        Returns:
            ITrainerFactory: Trainer factory instance
        """
        if self._trainer_factory is not None:
            return self._trainer_factory

        # Lazy initialization of real TrainerFactory
        if self._lazy_trainer_factory is None:
            from src.training.trainers.factory import TrainerFactory

            # PyCharm: may not reliably treat Protocols as structural types here (false positive).
            self._lazy_trainer_factory = cast("ITrainerFactory", cast("object", TrainerFactory()))
            logger.debug("[CONTAINER:TF_CREATED] TrainerFactory instance created")

        return self._lazy_trainer_factory

    # =========================================================================
    # DATASET LOADER
    # =========================================================================

    @property
    def dataset_loader(self) -> IDatasetLoader:
        """
        Get DatasetLoader instance.

        Returns injected mock or creates default loader via DatasetLoaderFactory.

        Returns:
            IDatasetLoader: Dataset loader instance
        """
        if self._dataset_loader is not None:
            return self._dataset_loader

        # Lazy initialization: runtime training loader (uses source_local.training_paths / source_hf.*)
        if self._lazy_dataset_loader is None:
            from src.training.orchestrator.dataset_loader import DatasetLoader

            self._lazy_dataset_loader = DatasetLoader(config=self.config)
            logger.debug("[CONTAINER:DL_CREATED] DatasetLoader (training runtime)")

        return self._lazy_dataset_loader

    def get_loader_for_dataset(self, dataset_name: str) -> IDatasetLoader:
        """
        Get loader for specific dataset by name.

        Creates appropriate loader based on dataset's source_type.

        Args:
            dataset_name: Name of dataset in config (e.g., "default", "alpaca")

        Returns:
            IDatasetLoader: Loader for the dataset's source type
        """
        from src.data.loaders import DatasetLoaderFactory

        # Get dataset config
        dataset_config = self.config.datasets.get(dataset_name)
        if dataset_config is None:
            logger.warning(f"[CONTAINER:DL] Dataset '{dataset_name}' not found, using default loader")
            return self.dataset_loader

        # Create loader via factory
        factory = DatasetLoaderFactory(config=self.config)
        loader = factory.create_for_dataset(dataset_config)
        logger.debug(f"[CONTAINER:DL] Created loader for '{dataset_name}' (source={dataset_config.get_source_type()})")
        return loader

    # =========================================================================
    # MLFLOW MANAGER
    # =========================================================================

    @property
    def mlflow_manager(self) -> IMLflowManager | None:
        """
        Get MLflowManager instance.

        Returns injected mock or creates real MLflowManager instance.
        Returns None if MLflow is disabled in config.

        Returns:
            IMLflowManager | None: MLflow manager or None if disabled
        """
        if self._mlflow_manager is not None:
            return self._mlflow_manager

        # Lazy initialization of real MLflowManager
        if self._lazy_mlflow_manager is None:
            # Check if MLflow is enabled in config
            mlflow_config = getattr(self.config.experiment_tracking, "mlflow", None)
            if mlflow_config is None or not mlflow_config.enabled:
                logger.debug("[CONTAINER:MLFLOW_DISABLED] MLflow disabled in config")
                return None

            from src.training.managers.mlflow_manager import MLflowManager

            # PyCharm: may not reliably treat Protocols as structural types here (false positive).
            self._lazy_mlflow_manager = cast("IMLflowManager", cast("object", MLflowManager(self.config)))
            logger.debug("[CONTAINER:MLFLOW_CREATED] MLflowManager instance created")

        return self._lazy_mlflow_manager

    # =========================================================================
    # COMPLETION NOTIFIER
    # =========================================================================

    @property
    def completion_notifier(self) -> ICompletionNotifier:
        """
        Get completion notifier.

        Returns injected mock or creates default MarkerFileNotifier.

        Workspace path resolution (priority):
            1. config.training.marker_path (explicit config)
            2. HELIX_WORKSPACE env var (set by deployment_manager)
            3. cwd() fallback (for local testing)

        Returns:
            ICompletionNotifier: Notifier instance
        """
        if self._completion_notifier is not None:
            return self._completion_notifier

        import os
        from pathlib import Path

        from src.training.notifiers.marker_file import MarkerFileNotifier

        # Priority: config > HELIX_WORKSPACE env > cwd
        marker_path = getattr(self.config.training, "marker_path", None)
        if not marker_path:
            marker_path = os.environ.get("HELIX_WORKSPACE")
        if not marker_path:
            marker_path = str(Path.cwd())

        return MarkerFileNotifier(base_path=marker_path)

    # =========================================================================
    # ORCHESTRATOR FACTORY
    # =========================================================================

    def create_orchestrator(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        mlflow_manager: IMLflowManager | None = None,
    ) -> StrategyOrchestrator:
        """
        Create StrategyOrchestrator with all dependencies injected.

        Args:
            model: Pre-trained model
            tokenizer: Tokenizer for the model
            mlflow_manager: Optional MLflowManager for experiment tracking

        Returns:
            StrategyOrchestrator: Configured orchestrator
        """
        from src.training.orchestrator import StrategyOrchestrator

        # Use provided mlflow_manager or get from container
        mlflow = mlflow_manager or self.mlflow_manager

        orchestrator = StrategyOrchestrator(
            model=model,
            tokenizer=tokenizer,
            config=self.config,
            memory_manager=self.memory_manager,  # type: ignore
            strategy_factory=self.strategy_factory,
            trainer_factory=self.trainer_factory,
            dataset_loader=self.dataset_loader,
            mlflow_manager=mlflow,
        )

        logger.debug(f"[CONTAINER:ORCHESTRATOR_CREATED] StrategyOrchestrator created (mlflow={mlflow is not None})")
        return orchestrator

    # =========================================================================
    # MODEL LOADING
    # =========================================================================

    def load_model_and_tokenizer(
        self,
    ) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
        """
        Load model and tokenizer with memory protection.

        Uses unified loader:
        - HuggingFace AutoModel for any model architecture
        - Quantization (4-bit/8-bit) based on config
        - Memory-safe loading with MemoryManager

        Note: PEFT (LoRA/QLoRA) is NOT applied here!
        PEFT is applied by TRL trainer via peft_config parameter.

        Returns:
            Tuple of (base model, tokenizer)
        """
        logger.info(f"📦 Loading model: {self.config.model.name}")
        logger.debug(
            f"[CONTAINER:MODEL_LOADING] model={self.config.model.name}, "
            f"training_type={self.config.training.type}, "
            f"4bit={self.config.training.get_effective_load_in_4bit()}"
        )

        from src.training.models.loader import load_model_and_tokenizer as _load_model_and_tokenizer

        with self.memory_manager.safe_operation("model_loading"):
            model, tokenizer = _load_model_and_tokenizer(config=self.config)

        logger.info("✅ Model and tokenizer loaded (PEFT will be applied by TRL)")
        logger.debug(f"[CONTAINER:MODEL_LOADED] model_class={type(model).__name__}")
        return model, tokenizer

    # =========================================================================
    # FACTORY METHODS
    # =========================================================================

    @classmethod
    def for_testing(
        cls,
        config: PipelineConfig,
        memory_manager: IMemoryManager | None = None,
        completion_notifier: ICompletionNotifier | None = None,
        strategy_factory: IStrategyFactory | None = None,
        trainer_factory: ITrainerFactory | None = None,
        dataset_loader: IDatasetLoader | None = None,
    ) -> TrainingContainer:
        """
        Create container with testing dependencies.

        If dependencies not provided, creates lightweight mocks.

        Args:
            config: Pipeline configuration
            memory_manager: Mock memory manager (optional)
            completion_notifier: Mock notifier (optional)
            strategy_factory: Mock strategy factory (optional)
            trainer_factory: Mock trainer factory (optional)
            dataset_loader: Mock dataset loader (optional)

        Returns:
            TrainingContainer: Container with test dependencies
        """
        mm = memory_manager or _create_noop_memory_manager()
        notifier = completion_notifier or _create_noop_notifier()

        container = cls(
            config=config,
            _memory_manager=mm,
            _completion_notifier=notifier,
            _strategy_factory=strategy_factory,
            _trainer_factory=trainer_factory,
            _dataset_loader=dataset_loader,
        )

        logger.debug("[CONTAINER:FOR_TESTING] Created container with test dependencies")
        return container

    @classmethod
    def from_config_path(cls, config_path: str) -> TrainingContainer:
        """
        Create container from config file path.

        Args:
            config_path: Path to pipeline config YAML

        Returns:
            TrainingContainer: Initialized container
        """
        from pathlib import Path

        from src.utils.config import load_config

        config = load_config(Path(config_path))
        return cls(config=config)

    # =========================================================================
    # UTILITIES
    # =========================================================================

    def override(
        self,
        memory_manager: IMemoryManager | None = None,
        completion_notifier: ICompletionNotifier | None = None,
        strategy_factory: IStrategyFactory | None = None,
        trainer_factory: ITrainerFactory | None = None,
        dataset_loader: IDatasetLoader | None = None,
    ) -> TrainingContainer:
        """
        Create new container with some dependencies overridden.

        Non-destructive: returns new container, doesn't modify self.

        Args:
            memory_manager: Override memory manager
            completion_notifier: Override notifier
            strategy_factory: Override strategy factory
            trainer_factory: Override trainer factory
            dataset_loader: Override dataset loader

        Returns:
            New TrainingContainer with overrides applied
        """
        return TrainingContainer(
            config=self.config,
            _memory_manager=memory_manager or self._memory_manager,
            _completion_notifier=completion_notifier or self._completion_notifier,
            _strategy_factory=strategy_factory or self._strategy_factory,
            _trainer_factory=trainer_factory or self._trainer_factory,
            _dataset_loader=dataset_loader or self._dataset_loader,
        )


# =============================================================================
# HELPER FACTORIES FOR TESTING
# =============================================================================


def _create_noop_memory_manager() -> IMemoryManager:
    """Create a no-op MemoryManager for testing without GPU."""

    class NoOpMemoryManager:
        """Memory manager that does nothing (for CPU testing)."""

        @property
        def gpu_info(self) -> GPUInfo | None:
            return None

        @property
        def preset(self) -> GPUPreset | None:
            return None

        @staticmethod
        def get_memory_stats() -> MemoryStats | None:
            return None

        @staticmethod
        def is_memory_critical() -> bool:
            return False

        @staticmethod
        def clear_cache() -> int:
            return 0

        @staticmethod
        def aggressive_cleanup() -> int:
            return 0

        @staticmethod
        def safe_operation(
            operation_name: str,
            context: dict[str, Any] | None = None,
        ):
            """Context manager that does nothing."""
            _ = operation_name  # Mark as unused
            _ = context  # Mark as unused
            from contextlib import nullcontext

            return nullcontext()

    return NoOpMemoryManager()


def _create_noop_notifier() -> ICompletionNotifier:
    """Create a no-op notifier for testing."""

    class NoOpNotifier:
        """Notifier that just logs (no file creation)."""

        @staticmethod
        def notify_complete(data: dict[str, Any]) -> None:
            logger.info(f"[TEST] Training complete: {data.get('output_path', 'unknown')}")

        @staticmethod
        def notify_failed(error: str, data: dict[str, Any]) -> None:
            _ = data  # Unused parameter (NoOp notifier)
            logger.error(f"[TEST] Training failed: {error}")

    return NoOpNotifier()


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "ICompletionNotifier",
    "IDatasetLoader",
    "IMemoryManager",
    "IStrategyFactory",
    "ITrainerFactory",
    "TrainingContainer",
]
