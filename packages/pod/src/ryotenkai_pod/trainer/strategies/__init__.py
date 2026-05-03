"""
Training Strategies

Strategy Pattern for different training methods.

Available strategies:
- CPTStrategy: Continual Pre-Training
- SFTStrategy: Supervised Fine-Tuning
- CoTStrategy: Chain-of-Thought
- DPOStrategy: Direct Preference Optimization
- ORPOStrategy: Odds Ratio Preference Optimization

Factory:
- StrategyFactory: Creates strategy instances by type

Example:
    from src.training.strategies import StrategyFactory
    strategy = StrategyFactory.create("sft", config)
    prepared = strategy.prepare_dataset(dataset, tokenizer)
"""

from src.constants import STRATEGY_DESCRIPTIONS  # backward-compat re-export
from src.training.strategies.base import StrategyMetadata, TrainingStrategy
from src.training.strategies.base_rl import BaseRLStrategy
from src.training.strategies.cot import CoTStrategy
from src.training.strategies.cpt import CPTStrategy
from src.training.strategies.dpo import DPOStrategy
from src.training.strategies.factory import (
    DEFAULT_BATCH_SIZES,
    DEFAULT_EPOCHS,
    DEFAULT_LEARNING_RATES,
    StrategyFactory,
)
from src.training.strategies.grpo import GRPOStrategy
from src.training.strategies.orpo import ORPOStrategy
from src.training.strategies.sapo import SAPOStrategy
from src.training.strategies.sft import SFTStrategy

__all__ = [
    "DEFAULT_BATCH_SIZES",
    "DEFAULT_EPOCHS",
    # Constants
    "DEFAULT_LEARNING_RATES",
    "STRATEGY_DESCRIPTIONS",
    # Strategies
    "BaseRLStrategy",
    "CPTStrategy",
    "CoTStrategy",
    "DPOStrategy",
    "GRPOStrategy",
    "ORPOStrategy",
    "SAPOStrategy",
    "SFTStrategy",
    # Factory
    "StrategyFactory",
    "StrategyMetadata",
    # Base classes
    "TrainingStrategy",
]
