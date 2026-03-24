"""
Training Strategy Base Classes

Implements Strategy Pattern for different training methods:
- CPT (Continual Pre-Training): Domain adaptation
- SFT (Supervised Fine-Tuning): Instruction following
- CoT (Chain-of-Thought): Reasoning abilities
- DPO (Direct Preference Optimization): Alignment

Training Adapters (QLoRA, LoRA, Full FT) handle HOW to train.
Training Strategies handle WHAT to train.

Combined: Adapter + Strategy = Complete training approach
Example: QLoRAAdapter + CoTStrategy = QLoRA training with CoT data
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from src.training.constants import (
    BATCH_SIZE_DEFAULT_FALLBACK,
    EPOCHS_DEFAULT_FALLBACK,
    LEARNING_RATE_DEFAULT_FALLBACK,
)
from src.utils.logger import logger

if TYPE_CHECKING:
    from datasets import Dataset
    from transformers import PreTrainedTokenizer

    from src.utils.config import PipelineConfig
    from src.utils.result import Result, StrategyError


@dataclass
class StrategyMetadata:
    """Metadata about a training strategy."""

    name: str
    version: str
    description: str
    strategy_type: str  # "cpt", "sft", "cot", "dpo"
    data_format: str  # Expected data format
    objective: str  # Training objective
    recommended_use: str
    dependencies: dict[str, str] | None = None

    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = {}


class TrainingStrategy(ABC):
    """
    Abstract base class for training strategies.

    Each strategy implements a specific training METHOD (CPT, SFT, CoT, DPO).
    Works in combination with Training Adapters (which handle HOW to train).

    Responsibility:
    - Data preparation and formatting
    - Training objective configuration
    - Loss function setup
    - Method-specific logic

    Example:
        # Use with any adapter
        adapter = QLoRAAdapter(model, tokenizer, config)
        strategy = CoTStrategy(config)

        # Strategy prepares data
        prepared_data = strategy.prepare_dataset(raw_data)

        # Adapter handles training
        result = adapter.train(prepared_data)
    """

    def __init__(self, config: PipelineConfig):
        """
        Initialize training strategy.

        Args:
            config: Pipeline configuration
        """
        self.config = config
        logger.info(f"Initialized {self.__class__.__name__}")

    @abstractmethod
    def prepare_dataset(self, dataset: Dataset, tokenizer: PreTrainedTokenizer) -> Result[Dataset, StrategyError]:
        """
        Prepare dataset for this training strategy.

        Args:
            dataset: Raw dataset
            tokenizer: Tokenizer for formatting

        Returns:
            Result[Dataset, StrategyError]: Prepared dataset or error
        """
        pass

    @abstractmethod
    def validate_dataset(self, dataset: Dataset) -> Result[bool, StrategyError]:
        """
        Validate dataset structure for this strategy.

        Args:
            dataset: Dataset to validate

        Returns:
            Result[bool, StrategyError]: True if valid, error otherwise
        """
        pass

    @abstractmethod
    def get_training_objective(self) -> str:
        """
        Get the training objective for this strategy.

        Returns:
            str: Objective description (e.g., "language_modeling", "preference_optimization")
        """
        pass

    @abstractmethod
    def get_metadata(self) -> StrategyMetadata:
        """
        Get metadata about this strategy.

        Returns:
            StrategyMetadata: Strategy metadata
        """
        pass

    @abstractmethod
    def get_trainer_type(self) -> str:
        """
        Get TRL trainer type for this strategy.

        Returns:
            str: Trainer type ("sft", "dpo", "orpo")
        """
        pass

    @abstractmethod
    def get_trainer_class(self) -> Any:
        """
        Get TRL Trainer class for this strategy.

        Returns:
            Type[Trainer]: TRL Trainer class (e.g. SFTTrainer, DPOTrainer)
        """
        pass

    @abstractmethod
    def get_config_class(self) -> Any:
        """
        Get TRL Config class for this strategy.

        Returns:
            Type[TrainingArguments]: TRL Config class (e.g. SFTConfig, DPOConfig)
        """
        pass

    @property
    def requires_reward_plugin(self) -> bool:
        """Declare whether this strategy requires an explicit reward plugin in phase config.

        When True, TrainerFactory will call build_reward_plugin_kwargs before creating
        the trainer. The reward plugin name must be present in phase_config.params.
        """
        return False

    @property
    def requires_reference_dataset(self) -> bool:
        """Declare whether this strategy needs train_dataset passed to build_trainer_kwargs.

        When True, TrainerFactory passes train_dataset into the strategy kwargs so that
        reward plugins can inspect dataset features at plugin-build time.
        """
        return False

    def build_config_kwargs(self, hp: Any, **kwargs: Any) -> dict[str, Any]:  # noqa: ARG002
        """
        Build strategy-specific config arguments.

        Maps HyperparametersConfig fields to TRL config arguments.

        Args:
            hp: HyperparametersConfig instance (merged)
            **kwargs: Additional context if needed

        Returns:
            dict: Arguments for Config constructor
        """
        return {}

    def build_trainer_kwargs(self, config: Any, **kwargs: Any) -> dict[str, Any]:  # noqa: ARG002
        """
        Build strategy-specific trainer arguments.

        Args:
            config: Instantiated TRL Config object
            **kwargs: Additional context

        Returns:
            dict: Additional arguments for Trainer constructor
        """
        return {}

    def post_build_config_hook(self, config: Any, **context: Any) -> None:  # noqa: B027
        """Apply post-creation mutations to the TRL config object if needed.

        Called by TrainerFactory after creating the config instance.
        Override in subclasses that need to configure fields on the config
        object that depend on runtime context (e.g. model type at build time).

        Args:
            config: Already-created TRL config instance (e.g. DPOConfig)
            **context: Runtime context, e.g. ``model``, ``ref_model``
        """

    def get_recommended_hyperparameters(self) -> dict[str, Any]:
        """
        Get recommended hyperparameters for this strategy.

        Override in subclass for strategy-specific recommendations.

        Returns:
            Dict with recommended hyperparameters
        """
        # Get current strategy info
        strategies = self.config.training.get_strategy_chain()
        current = strategies[0] if strategies else None

        current_lr = (
            (current.hyperparams.learning_rate if current else None)
            or self.config.training.hyperparams.learning_rate
            or LEARNING_RATE_DEFAULT_FALLBACK
        )
        current_epochs = (
            (current.hyperparams.epochs if current else None)
            or self.config.training.hyperparams.epochs
            or EPOCHS_DEFAULT_FALLBACK
        )
        current_bs = self.config.training.hyperparams.per_device_train_batch_size or BATCH_SIZE_DEFAULT_FALLBACK

        return {
            "learning_rate": current_lr,
            "num_epochs": current_epochs,
            "batch_size": current_bs,
        }

    def __repr__(self) -> str:
        """String representation."""
        metadata = self.get_metadata()
        return f"{self.__class__.__name__}(type={metadata.strategy_type}, version={metadata.version})"


__all__ = [
    "StrategyMetadata",
    "TrainingStrategy",
]
