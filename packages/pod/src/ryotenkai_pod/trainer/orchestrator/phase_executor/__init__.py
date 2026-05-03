"""
phase_executor — backward-compatible package facade.

`from ryotenkai_pod.trainer.orchestrator.phase_executor import PhaseExecutor`
continues to work unchanged.
"""

from __future__ import annotations

from ryotenkai_pod.trainer.orchestrator.phase_executor.executor import PhaseExecutor

__all__ = ["PhaseExecutor"]
