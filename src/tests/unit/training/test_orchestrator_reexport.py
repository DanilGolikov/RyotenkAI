from __future__ import annotations


def test_training_orchestrator_reexport() -> None:
    from src.training.orchestrator import StrategyOrchestrator

    assert StrategyOrchestrator is not None
