"""
phase_executor — backward-compatible package facade.

`from src.training.orchestrator.phase_executor import PhaseExecutor`
continues to work unchanged.
"""

from __future__ import annotations

from src.training.orchestrator.phase_executor.executor import PhaseExecutor

__all__ = ["PhaseExecutor"]
