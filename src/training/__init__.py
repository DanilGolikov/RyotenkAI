"""
Training module for RyotenkAI.

Key Components:
- StrategyOrchestrator: Multi-phase training (CPT → SFT → DPO)
- run_training(): Main entry point for training
- notifiers: Completion notification (MarkerFileNotifier, LogNotifier)
- trainer_builder: Unified trainer creation
"""

from .notifiers import LogNotifier, MarkerFileNotifier
from .orchestrator import StrategyOrchestrator
from .run_training import run_training, train_v2  # train_v2 is alias for backward compat
from .trainer_builder import create_peft_config, create_trainer, create_training_args

__all__ = [
    "LogNotifier",
    "MarkerFileNotifier",
    "StrategyOrchestrator",
    "create_peft_config",
    "create_trainer",
    "create_training_args",
    "run_training",
    "train_v2",  # backward compatibility
]
