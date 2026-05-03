"""
Training module for RyotenkAI.

Key Components:
- StrategyOrchestrator: Multi-phase training (CPT → SFT → DPO)
- run_training(): Main entry point for training
- trainer_builder: Unified trainer creation

Note: completion notifiers (LogNotifier, MarkerFileNotifier) were removed
in Phase 6.3b. Trainer-side completion signalling now flows through
:class:`src.training.callbacks.runner_event_callback.RunnerEventCallback`,
which pushes structured events to the in-pod runner over loopback HTTP.
The runner's FSM emits ``trainer_exited`` + terminal-state transitions
the Mac client subscribes to over WebSocket.
"""

from .orchestrator import StrategyOrchestrator
from .run_training import run_training, train_v2  # train_v2 is alias for backward compat
from .trainer_builder import create_peft_config, create_trainer, create_training_args

__all__ = [
    "StrategyOrchestrator",
    "create_peft_config",
    "create_trainer",
    "create_training_args",
    "run_training",
    "train_v2",  # backward compatibility
]
