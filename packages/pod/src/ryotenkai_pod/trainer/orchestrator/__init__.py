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

from src.training.orchestrator.chain_runner import ChainRunner
from src.training.orchestrator.dataset_loader import DatasetLoader
from src.training.orchestrator.metrics_collector import MetricsCollector
from src.training.orchestrator.phase_executor import PhaseExecutor
from src.training.orchestrator.resume_manager import ResumeManager
from src.training.orchestrator.shutdown_handler import (
    ShutdownHandler,
    ShutdownReason,
    ShutdownState,
    get_shutdown_handler,
    reset_shutdown_handler,
)
from src.training.orchestrator.strategy_orchestrator import StrategyOrchestrator

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
