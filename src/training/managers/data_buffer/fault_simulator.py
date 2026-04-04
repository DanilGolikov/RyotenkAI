"""
Fault injection support for DataBuffer (testing infrastructure).

Used to simulate I/O failures, corrupted state, and missing checkpoints
without modifying production code.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FaultSimulator:
    """
    Configuration for fault injection in DataBuffer (for testing).

    Allows simulating various failure scenarios without modifying production code.
    Pass instance to DataBuffer constructor to enable simulation.

    Example:
        simulator = FaultSimulator(fail_on_save=True)
        buffer = DataBuffer(..., _fault_simulator=simulator)
        buffer.save_state()  # Will raise SimulatedFaultError

    Simulation points (marked with # SIMULATION_POINT in code):
        - save_state(): fail_on_save, slow_io_delay_ms, fail_after_n_saves
        - load_state(): fail_on_load, corrupt_state, slow_io_delay_ms
        - get_model_path_for_phase(): missing_checkpoint
        - cleanup_old_checkpoints(): fail_on_cleanup
    """

    # Save operation failures
    fail_on_save: bool = False
    fail_on_save_error: str = "Simulated save failure"

    # Load operation failures
    fail_on_load: bool = False
    fail_on_load_error: str = "Simulated load failure"

    # Simulate corrupted JSON in state file
    corrupt_state: bool = False

    # Simulate slow IO (milliseconds delay)
    slow_io_delay_ms: int = 0

    # Simulate missing checkpoints
    missing_checkpoint: bool = False

    # Fail after N successful saves (for testing mid-pipeline failures)
    fail_after_n_saves: int | None = None

    # Fail on cleanup operation
    fail_on_cleanup: bool = False

    # Internal counter for fail_after_n_saves
    _save_count: int = field(default=0, repr=False)

    def should_fail_save(self) -> bool:
        """Check if save should fail based on simulation config."""
        if self.fail_on_save:
            return True
        if self.fail_after_n_saves is not None:
            self._save_count += 1
            if self._save_count > self.fail_after_n_saves:
                return True
        return False

    def get_io_delay(self) -> float:
        """Get IO delay in seconds."""
        return self.slow_io_delay_ms / 1000.0 if self.slow_io_delay_ms > 0 else 0.0


class SimulatedFaultError(Exception):
    """Exception raised when a simulated fault is triggered."""

    pass


__all__ = ["FaultSimulator", "SimulatedFaultError"]
