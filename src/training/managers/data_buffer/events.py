"""
Event callbacks for DataBuffer.

Provides a SOLID-compliant observer interface that decouples DataBuffer
from consumers (MLflow, UI, tests) without direct dependencies.
"""

from __future__ import annotations

from collections.abc import Callable  # noqa: TC003
from dataclasses import dataclass


@dataclass
class DataBufferEventCallbacks:
    """
    Callbacks for DataBuffer events (SOLID-compliant event collection).

    Used to integrate DataBuffer with MLflow or other logging systems
    without creating direct dependencies.

    Example:
        callbacks = DataBufferEventCallbacks(
            on_pipeline_initialized=lambda run_id, phases, chain: print(f"Pipeline: {run_id}"),
            on_phase_started=lambda idx, strategy: print(f"Phase {idx}: {strategy}"),
        )
        buffer = DataBuffer(..., callbacks=callbacks)
    """

    # Pipeline initialized event
    on_pipeline_initialized: Callable[[str, int, list[str]], None] | None = None
    # Args: run_id, total_phases, strategy_chain

    # State saved event
    on_state_saved: Callable[[str, str], None] | None = None
    # Args: run_id, state_file_path

    # Phase started event
    on_phase_started: Callable[[int, str], None] | None = None
    # Args: phase_idx, strategy_type

    # Phase completed event
    on_phase_completed: Callable[[int, str, str], None] | None = None
    # Args: phase_idx, strategy_type, status ("completed", "failed", "interrupted")

    # Checkpoint cleanup event
    on_checkpoint_cleanup: Callable[[int, int], None] | None = None
    # Args: removed_count, freed_mb (approximate)


__all__ = ["DataBufferEventCallbacks"]
