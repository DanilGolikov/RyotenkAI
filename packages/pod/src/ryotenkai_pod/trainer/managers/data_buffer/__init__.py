"""
data_buffer — backward-compatible package facade.

All names that were previously importable from
`src.training.managers.data_buffer` remain importable unchanged.

Internal structure:
    manager.py          ← DataBuffer
    state_models.py     ← PhaseStatus, PhaseState, PipelineState
    events.py           ← DataBufferEventCallbacks
    fault_simulator.py  ← FaultSimulator, SimulatedFaultError
    checkpoint_utils.py ← _sanitize_metrics, _extract_checkpoint_step,
                          _get_sorted_checkpoints, list_available_runs
"""

from __future__ import annotations

from src.training.managers.data_buffer.checkpoint_utils import (
    _extract_checkpoint_step,
    _get_sorted_checkpoints,
    _sanitize_metrics,
    list_available_runs,
)
from src.training.managers.data_buffer.events import DataBufferEventCallbacks
from src.training.managers.data_buffer.fault_simulator import (
    FaultSimulator,
    SimulatedFaultError,
)
from src.training.managers.data_buffer.manager import DataBuffer
from src.training.managers.data_buffer.state_models import (
    PhaseState,
    PhaseStatus,
    PipelineState,
)

__all__ = [
    "DataBuffer",
    "DataBufferEventCallbacks",
    "FaultSimulator",
    "PhaseState",
    "PhaseStatus",
    "PipelineState",
    "SimulatedFaultError",
    "list_available_runs",
    # private helpers kept accessible for tests that import them directly
    "_get_sorted_checkpoints",
    "_extract_checkpoint_step",
    "_sanitize_metrics",
]
