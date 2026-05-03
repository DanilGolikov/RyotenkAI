"""
Orchestrator Package - Modular Multi-Phase Training Orchestration.

Components:
- DatasetLoader: Load and validate datasets for phases
- MetricsCollector: Extract training metrics
- ResumeManager: Handle checkpoint resume logic
- PhaseExecutor: Execute single training phase
- ChainRunner: Run sequence of phases
- ShutdownHandler: Graceful shutdown on SIGINT/SIGTERM
- StrategyOrchestrator: Facade coordinating all components
"""

from ryotenkai_pod.trainer.orchestrator.chain_runner import ChainRunner
from ryotenkai_pod.trainer.orchestrator.dataset_loader import DatasetLoader
from ryotenkai_pod.trainer.orchestrator.metrics_collector import MetricsCollector
from ryotenkai_pod.trainer.orchestrator.phase_executor import PhaseExecutor
from ryotenkai_pod.trainer.orchestrator.resume_manager import ResumeManager
from ryotenkai_pod.trainer.orchestrator.shutdown_handler import (
    ShutdownHandler,
    ShutdownReason,
    ShutdownState,
    get_shutdown_handler,
    reset_shutdown_handler,
)
from ryotenkai_pod.trainer.orchestrator.strategy_orchestrator import StrategyOrchestrator

__all__ = [
    "ChainRunner",
    "DatasetLoader",
    "MetricsCollector",
    "PhaseExecutor",
    "ResumeManager",
    "ShutdownHandler",
    "ShutdownReason",
    "ShutdownState",
    "StrategyOrchestrator",
    "get_shutdown_handler",
    "reset_shutdown_handler",
]
