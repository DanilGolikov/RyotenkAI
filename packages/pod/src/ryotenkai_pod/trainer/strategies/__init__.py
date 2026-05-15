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
    from ryotenkai_pod.trainer.strategies import StrategyFactory
    strategy = StrategyFactory.create("sft", config)
    prepared = strategy.prepare_dataset(dataset, tokenizer)
"""

from ryotenkai_pod.trainer.strategies.base import StrategyMetadata, TrainingStrategy
from ryotenkai_pod.trainer.strategies.base_rl import BaseRLStrategy
from ryotenkai_pod.trainer.strategies.cot import CoTStrategy
from ryotenkai_pod.trainer.strategies.cpt import CPTStrategy
from ryotenkai_pod.trainer.strategies.dpo import DPOStrategy
from ryotenkai_pod.trainer.strategies.factory import (
    DEFAULT_BATCH_SIZES,
    DEFAULT_EPOCHS,
    DEFAULT_LEARNING_RATES,
    StrategyFactory,
)
from ryotenkai_pod.trainer.strategies.grpo import GRPOStrategy
from ryotenkai_pod.trainer.strategies.orpo import ORPOStrategy
from ryotenkai_pod.trainer.strategies.sapo import SAPOStrategy
from ryotenkai_pod.trainer.strategies.sft import SFTStrategy
from ryotenkai_shared.constants import STRATEGY_DESCRIPTIONS  # backward-compat re-export

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
