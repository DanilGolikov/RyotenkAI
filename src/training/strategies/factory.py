"""
Strategy Factory for creating Training Strategies.

Factory Pattern implementation for training strategies (CPT, SFT, CoT, DPO, ORPO).
Separates WHAT to train (strategy) from HOW to train (adapter).

Key Features:
- Registry pattern for strategy lookup
- Default hyperparameters per strategy
- Integration with PluginRegistry for metadata
- Type-safe strategy creation

Architecture:
    StrategyFactory → TrainingStrategy (data prep, hyperparameters)
    TrainerFactory  → TRL Trainers (actual training execution)
    TrainingFactory → TrainingAdapter (QLoRA, LoRA, Full FT)

Example:
    from src.training.strategies.factory import StrategyFactory

    # List available strategies
    strategies = StrategyFactory.list_available()
    # {'sft': StrategyMetadata(...), 'dpo': StrategyMetadata(...)}

    # Create strategy instance
    strategy = StrategyFactory.create("sft", config)

    # Get default hyperparameters
    defaults = StrategyFactory.get_default_hyperparameters("dpo")
    # {'learning_rate': 5e-6, 'epochs': 1, 'batch_size': 4}
"""

from typing import ClassVar

from src.constants import (
    DEFAULT_BATCH_SIZES,
    DEFAULT_EPOCHS,
    DEFAULT_LEARNING_RATES,
    STRATEGY_COT,
    STRATEGY_CPT,
    STRATEGY_DPO,
    STRATEGY_GRPO,
    STRATEGY_ORPO,
    STRATEGY_SAPO,
    STRATEGY_SFT,
)
from src.training.constants import (
    BATCH_SIZE_DEFAULT_FALLBACK,
    EPOCHS_DEFAULT_FALLBACK,
    LEARNING_RATE_DEFAULT_FALLBACK,
    STRATEGY_VERSION_DEFAULT,
)
from src.training.strategies.base import StrategyMetadata, TrainingStrategy
from src.utils.config import PipelineConfig, StrategyPhaseConfig
from src.utils.logger import logger

_DEP_TRL = "trl"


class StrategyFactory:
    """
    Factory for creating Training Strategy instances.

    Implements Factory Pattern with Registry for strategy lookup.
    Supports both class-level registration and instance-level operations for DI.

    Available strategies:
    - "cpt": Continual Pre-Training (domain adaptation)
    - "sft": Supervised Fine-Tuning (instruction following)
    - "cot": Chain-of-Thought (reasoning)
    - "dpo": Direct Preference Optimization (alignment)
    - "orpo": ORPO (combined SFT+alignment)

    Example:
        # Instance-based usage (recommended for DI)
        factory = StrategyFactory()
        strategy = factory.create("sft", config)

        # With custom registry (for testing)
        mock_registry = {"sft": MockSFTStrategy}
        factory = StrategyFactory(registry=mock_registry)

        # Class-level registration still works
        StrategyFactory.register("custom", CustomStrategy)
    """

    # Global registry: strategy_type -> class (shared across all instances)
    _strategies: ClassVar[dict[str, type[TrainingStrategy]]] = {}

    # Optional metadata storage (for introspection/debugging)
    _metadata: ClassVar[dict[str, StrategyMetadata]] = {}

    def __init__(
        self,
        registry: dict[str, type[TrainingStrategy]] | None = None,
    ) -> None:
        """
        Initialize StrategyFactory.

        Args:
            registry: Optional custom registry (for testing).
                      If None, uses global _strategies registry.

        Example:
            # Production: use global registry
            factory = StrategyFactory()

            # Testing: use mock registry
            factory = StrategyFactory(registry={"sft": MockStrategy})
        """
        # Use custom registry or fall back to global
        self._registry = registry if registry is not None else self.__class__._strategies
        logger.debug(f"[SF:INIT] StrategyFactory initialized with {len(self._registry)} strategies")

    @classmethod
    def register(
        cls,
        strategy_type: str,
        strategy_class: type[TrainingStrategy],
        *,
        metadata: StrategyMetadata | None = None,
    ) -> None:
        """
        Register a training strategy (class method for global registration).

        Args:
            strategy_type: Type identifier (e.g., "sft", "dpo")
            strategy_class: Strategy class to register
            metadata: Optional strategy metadata

        Example:
            from src.training.strategies.sft import SFTStrategy
            StrategyFactory.register("sft", SFTStrategy)
        """
        strategy_type = strategy_type.lower()

        if strategy_type in cls._strategies:
            logger.warning(f"Strategy '{strategy_type}' already registered. Overwriting.")

        cls._strategies[strategy_type] = strategy_class
        logger.debug(f"[SF:STRATEGY_REGISTERED] type={strategy_type}, class={strategy_class.__name__}")

        # Store metadata for introspection/debugging (optional)
        if metadata:
            cls._metadata[strategy_type] = metadata
            logger.debug(
                f"[SF:METADATA_STORED] type={strategy_type}, version={metadata.version}, format={metadata.data_format}"
            )

    def create(
        self,
        strategy_type: str,
        config: PipelineConfig,
    ) -> TrainingStrategy:
        """
        Create a strategy instance.

        Args:
            strategy_type: Type of strategy ("sft", "dpo", etc.)
            config: Pipeline configuration

        Returns:
            TrainingStrategy instance

        Raises:
            ValueError: If strategy_type is not registered

        Example:
            factory = StrategyFactory()
            strategy = factory.create("sft", config)
            prepared = strategy.prepare_dataset(dataset, tokenizer)
        """
        strategy_type = strategy_type.lower()

        if strategy_type not in self._registry:
            available = ", ".join(self._registry.keys())
            logger.debug(f"[SF:STRATEGY_NOT_FOUND] type={strategy_type}, available={available}")
            raise ValueError(f"Unknown strategy type: '{strategy_type}'. Available: {available}")

        strategy_class = self._registry[strategy_type]
        defaults = self.get_default_hyperparameters(strategy_type)
        logger.debug(
            f"[SF:STRATEGY_CREATING] type={strategy_type}, "
            f"class={strategy_class.__name__}, "
            f"default_lr={defaults['learning_rate']}, "
            f"default_epochs={defaults['epochs']}"
        )
        logger.info(f"Creating {strategy_type} strategy...")
        return strategy_class(config)

    def create_from_phase(
        self,
        phase: StrategyPhaseConfig,
        config: PipelineConfig,
    ) -> TrainingStrategy:
        """
        Create a strategy instance from a phase config.

        Convenience method for StrategyOrchestrator.

        Args:
            phase: Strategy phase configuration
            config: Pipeline configuration

        Returns:
            TrainingStrategy instance
        """
        # NOTE: StrategyPhaseConfig stores hyperparams overrides inside `phase.hyperparams`.
        # Do NOT access legacy flat fields like `phase.epochs` or `phase.learning_rate`.
        epochs = getattr(phase.hyperparams, "epochs", None)
        lr = getattr(phase.hyperparams, "learning_rate", None)
        logger.debug(
            f"[SF:CREATE_FROM_PHASE] strategy={phase.strategy_type}, epochs={epochs}, lr={lr}, dataset={phase.dataset}"
        )
        return self.create(phase.strategy_type, config)

    @staticmethod
    def get_default_hyperparameters(strategy_type: str) -> dict:
        """
        Get default hyperparameters for a strategy.

        Returns research-backed defaults for each strategy type.

        Args:
            strategy_type: Type of strategy

        Returns:
            Dict with learning_rate, epochs, batch_size

        Example:
            factory = StrategyFactory()
            defaults = factory.get_default_hyperparameters("dpo")
            # {'learning_rate': 5e-6, 'epochs': 1, 'batch_size': 4}
        """
        strategy_type = strategy_type.lower()

        return {
            "learning_rate": DEFAULT_LEARNING_RATES.get(strategy_type, LEARNING_RATE_DEFAULT_FALLBACK),
            "epochs": DEFAULT_EPOCHS.get(strategy_type, EPOCHS_DEFAULT_FALLBACK),
            "batch_size": DEFAULT_BATCH_SIZES.get(strategy_type, BATCH_SIZE_DEFAULT_FALLBACK),
        }

    @staticmethod
    def get_learning_rate(strategy_type: str) -> float:
        """Get default learning rate for a strategy."""
        return DEFAULT_LEARNING_RATES.get(strategy_type.lower(), LEARNING_RATE_DEFAULT_FALLBACK)

    @staticmethod
    def get_epochs(strategy_type: str) -> int:
        """Get default epochs for a strategy."""
        return DEFAULT_EPOCHS.get(strategy_type.lower(), EPOCHS_DEFAULT_FALLBACK)

    def list_available(self) -> dict[str, str]:
        """
        List all available strategies with their descriptions.

        Returns:
            Dict mapping strategy name to description (sourced from StrategyMetadata).
        """
        result = {}
        for name in self._registry:
            if name in self._metadata:
                result[name] = self._metadata[name].description
            else:
                result[name] = self._registry[name].__doc__ or "No description"
        return result

    def list_with_metadata(self) -> dict[str, StrategyMetadata]:
        """
        List all strategies with full metadata.

        Returns:
            Dict mapping strategy name to StrategyMetadata
        """
        result = {}
        for name in self._registry:
            if name in self._metadata:
                result[name] = self._metadata[name]
            else:
                result[name] = StrategyMetadata(
                    name=name,
                    version=STRATEGY_VERSION_DEFAULT,
                    description=self._registry[name].__doc__ or "No description",
                    strategy_type=name,
                    data_format="varies",
                    objective="varies",
                    recommended_use="",
                )
        return result

    def is_registered(self, strategy_type: str) -> bool:
        """Check if a strategy type is registered."""
        return strategy_type.lower() in self._registry

    def validate_strategy_type(self, strategy_type: str) -> tuple[bool, str]:
        """
        Validate a strategy type.

        Args:
            strategy_type: Type to validate

        Returns:
            (is_valid, error_message)
        """
        if not strategy_type:
            return False, "Strategy type cannot be empty"

        if strategy_type.lower() not in self._registry:
            available = ", ".join(self._registry.keys())
            return False, f"Unknown strategy '{strategy_type}'. Available: {available}"

        return True, ""


def register_builtin_strategies() -> None:
    """
    Register all built-in strategies.

    Called automatically when module is imported.
    """
    # Import here to avoid circular imports
    from src.training.strategies.cot import CoTStrategy
    from src.training.strategies.cpt import CPTStrategy
    from src.training.strategies.dpo import DPOStrategy
    from src.training.strategies.grpo import GRPOStrategy
    from src.training.strategies.orpo import ORPOStrategy
    from src.training.strategies.sapo import SAPOStrategy
    from src.training.strategies.sft import SFTStrategy

    # Register with metadata
    StrategyFactory.register(
        STRATEGY_CPT,
        CPTStrategy,
        metadata=StrategyMetadata(
            name=STRATEGY_CPT,
            version=STRATEGY_VERSION_DEFAULT,
            description="Continual Pre-Training for domain adaptation",
            strategy_type=STRATEGY_CPT,
            data_format="plain_text",
            objective="language_modeling",
            recommended_use="Domain adaptation with large unlabeled corpus",
        ),
    )

    StrategyFactory.register(
        STRATEGY_SFT,
        SFTStrategy,
        metadata=StrategyMetadata(
            name=STRATEGY_SFT,
            version=STRATEGY_VERSION_DEFAULT,
            description="Supervised Fine-Tuning for instruction following",
            strategy_type=STRATEGY_SFT,
            data_format="instruction_response",
            objective="supervised_learning",
            recommended_use="Standard instruction fine-tuning",
        ),
    )

    StrategyFactory.register(
        STRATEGY_COT,
        CoTStrategy,
        metadata=StrategyMetadata(
            name=STRATEGY_COT,
            version=STRATEGY_VERSION_DEFAULT,
            description="Chain-of-Thought for reasoning abilities",
            strategy_type=STRATEGY_COT,
            data_format="instruction_reasoning_answer",
            objective="reasoning_with_steps",
            recommended_use="Complex reasoning tasks",
            dependencies={"performance_gain": "+8-15% on reasoning tasks"},
        ),
    )

    StrategyFactory.register(
        STRATEGY_DPO,
        DPOStrategy,
        metadata=StrategyMetadata(
            name=STRATEGY_DPO,
            version=STRATEGY_VERSION_DEFAULT,
            description="Direct Preference Optimization for alignment",
            strategy_type=STRATEGY_DPO,
            data_format="chosen/rejected pairs",
            objective="preference_optimization",
            recommended_use="Alignment after SFT, reducing harmful outputs",
            dependencies={_DEP_TRL: ">=0.8.0", "critical": "LR must be 10-100x lower than SFT!"},
        ),
    )

    StrategyFactory.register(
        STRATEGY_ORPO,
        ORPOStrategy,
        metadata=StrategyMetadata(
            name=STRATEGY_ORPO,
            version=STRATEGY_VERSION_DEFAULT,
            description="Odds Ratio Preference Optimization (combined SFT+alignment)",
            strategy_type=STRATEGY_ORPO,
            data_format="chosen/rejected pairs",
            objective="combined_sft_preference",
            recommended_use="Single-pass instruction tuning + alignment",
            dependencies={_DEP_TRL: ">=0.8.0"},
        ),
    )

    StrategyFactory.register(
        STRATEGY_GRPO,
        GRPOStrategy,
        metadata=StrategyMetadata(
            name=STRATEGY_GRPO,
            version=STRATEGY_VERSION_DEFAULT,
            description="Group Relative Policy Optimization (GRPO)",
            strategy_type=STRATEGY_GRPO,
            data_format="prompt_only_with_reference_answer",
            objective="compiler_backed_reinforcement_learning",
            recommended_use="Online RL baseline after SFT",
            dependencies={_DEP_TRL: ">=0.26.0"},
        ),
    )

    StrategyFactory.register(
        STRATEGY_SAPO,
        SAPOStrategy,
        metadata=StrategyMetadata(
            name=STRATEGY_SAPO,
            version=STRATEGY_VERSION_DEFAULT,
            description="Soft Adaptive Policy Optimization (SAPO)",
            strategy_type=STRATEGY_SAPO,
            data_format="prompt_completion",
            objective="reinforcement_learning",
            recommended_use="Alignment phase after SFT",
            dependencies={_DEP_TRL: ">=0.26.0"},
        ),
    )

    registered = list(StrategyFactory().list_available().keys())
    logger.info(f"Registered {len(registered)} built-in strategies: {registered}")


# Auto-register built-in strategies on module import
register_builtin_strategies()


__all__ = [
    "DEFAULT_BATCH_SIZES",
    "DEFAULT_EPOCHS",
    "DEFAULT_LEARNING_RATES",
    "StrategyFactory",
    "register_builtin_strategies",
]
