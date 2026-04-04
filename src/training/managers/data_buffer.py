"""
DataBuffer - Checkpoint Management for Multi-Phase Training.

Manages state and checkpoints between training phases in multi-phase pipelines.
Enables resume capability after failures and tracks training progress.

Key Features:
- Per-phase output directories (phase_0_sft/, phase_1_dpo/)
- Pipeline state persistence (pipeline_state.json)
- Resume from any failed/incomplete phase
- Automatic checkpoint cleanup to save disk space
- Integration with TRL trainers via output_dir/resume_from_checkpoint

Architecture:
    StrategyOrchestrator
           │
           ▼
       DataBuffer ──┬─── PhaseState (per-phase)
           │        └─── PipelineState (overall)
           ▼
    TrainerFactory (uses output_dir from DataBuffer)

Example:
    from src.training.managers.data_buffer import DataBuffer

    # Initialize for new training run
    buffer = DataBuffer(
        base_output_dir="output",
        base_model_path="Qwen/Qwen2.5-7B-Instruct",
    )

    # Initialize pipeline with strategies
    buffer.init_pipeline(
        strategies=[
            StrategyPhaseConfig(strategy_type="sft"),
            StrategyPhaseConfig(strategy_type="dpo"),
        ]
    )

    # Get paths for each phase
    phase_0_dir = buffer.get_phase_output_dir(0)  # output/phase_0_sft/
    model_for_phase_1 = buffer.get_model_path_for_phase(1)  # previous checkpoint

    # Track progress
    buffer.mark_phase_started(0)
    buffer.mark_phase_completed(0, checkpoint_path, metrics)
    buffer.save_state()

    # Resume after failure (within the same run workspace)
    buffer = DataBuffer.load_existing(base_output_dir="output")
    resume_from = buffer.get_resume_phase()  # Returns first incomplete phase
"""

from __future__ import annotations

import dataclasses
import json
import os
import shutil
import tempfile
import time
from collections.abc import Callable  # noqa: TC003
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.training.managers.constants import (
    CHECKPOINT_FINAL_DIR,
    CHECKPOINT_SIZE_ESTIMATE_MB,
    KEY_COMPLETED_AT,
    KEY_PHASES,
    KEY_STARTED_AT,
    KEY_STATUS,
    RUN_ID_TIMESTAMP_LEN,
)
from src.utils.logger import logger

if TYPE_CHECKING:
    from src.utils.config import StrategyPhaseConfig


# =============================================================================
# SERIALIZATION HELPERS
# =============================================================================


def _sanitize_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    """
    Convert metrics to JSON-serializable types.

    Handles numpy/torch scalars and arrays.

    Args:
        metrics: Dict with potentially non-serializable values

    Returns:
        Dict with JSON-serializable values only
    """
    result: dict[str, Any] = {}
    for key, value in metrics.items():
        if hasattr(value, "item"):  # numpy/torch scalar
            result[key] = value.item()
        elif hasattr(value, "tolist"):  # numpy array
            result[key] = value.tolist()
        elif isinstance(value, int | float | str | bool | type(None)):
            result[key] = value
        elif isinstance(value, dict):
            result[key] = _sanitize_metrics(value)  # Recurse for nested dicts
        elif isinstance(value, list):
            result[key] = [v.item() if hasattr(v, "item") else v for v in value]
        else:
            result[key] = str(value)  # Fallback to string
    return result


# =============================================================================
# FAULT SIMULATION (for testing)
# =============================================================================


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


# =============================================================================
# EVENT CALLBACKS
# =============================================================================


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


# =============================================================================
# ENUMS AND DATA CLASSES
# =============================================================================


class PhaseStatus(Enum):
    """Status of a training phase."""

    PENDING = "pending"  # Not started yet
    RUNNING = "running"  # Currently executing
    COMPLETED = "completed"  # Successfully finished
    FAILED = "failed"  # Failed with error
    SKIPPED = "skipped"  # Skipped (e.g., resume from later phase)
    INTERRUPTED = "interrupted"  # Interrupted by SIGINT/SIGTERM


@dataclass
class PhaseState:
    """
    State of a single training phase.

    Tracks execution status, paths, and metrics for one phase
    in the multi-phase training pipeline.
    """

    phase_idx: int
    strategy_type: str
    status: PhaseStatus = PhaseStatus.PENDING
    output_dir: str | None = None
    checkpoint_path: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error_message: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)

    # Config that was used for this phase
    epochs: int = 1
    learning_rate: float | None = None
    dataset_name: str | None = None

    # Adapter cache state (populated when adapter_cache.enabled=true)
    adapter_cache_hit: bool = False
    adapter_cache_tag: str | None = None
    adapter_cache_upload_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data[KEY_STATUS] = self.status.value
        # Sanitize metrics to ensure JSON-serializable types
        data["metrics"] = _sanitize_metrics(self.metrics)
        # Convert datetime to ISO string
        if self.started_at:
            data[KEY_STARTED_AT] = self.started_at.isoformat()
        if self.completed_at:
            data[KEY_COMPLETED_AT] = self.completed_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PhaseState:
        """Create from dictionary (JSON deserialization).

        Handles forward/backward compatibility by filtering unknown fields.
        """
        # FIX BUG-001: Filter only known fields to handle version migrations
        known_fields = {f.name for f in dataclasses.fields(cls)}
        filtered_data = {k: v for k, v in data.items() if k in known_fields}

        # Warn about unknown fields (helps debugging version mismatches)
        unknown_fields = set(data.keys()) - known_fields
        if unknown_fields:
            logger.debug(f"[DB:COMPAT] Ignoring unknown fields in PhaseState: {unknown_fields}")

        # Convert status string to enum
        if KEY_STATUS in filtered_data:
            try:
                filtered_data[KEY_STATUS] = PhaseStatus(filtered_data[KEY_STATUS])
            except ValueError:
                logger.warning(f"Unknown status '{filtered_data[KEY_STATUS]}', defaulting to PENDING")
                filtered_data[KEY_STATUS] = PhaseStatus.PENDING

        # Convert ISO strings to datetime
        if filtered_data.get(KEY_STARTED_AT):
            filtered_data[KEY_STARTED_AT] = datetime.fromisoformat(filtered_data[KEY_STARTED_AT])
        if filtered_data.get(KEY_COMPLETED_AT):
            filtered_data[KEY_COMPLETED_AT] = datetime.fromisoformat(filtered_data[KEY_COMPLETED_AT])

        return cls(**filtered_data)

    @property
    def is_complete(self) -> bool:
        """Check if phase is successfully completed."""
        return self.status == PhaseStatus.COMPLETED

    @property
    def is_running(self) -> bool:
        """Check if phase is currently running."""
        return self.status == PhaseStatus.RUNNING

    @property
    def duration_seconds(self) -> float | None:
        """Get phase duration in seconds (if completed)."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None


@dataclass
class PipelineState:
    """
    Overall state of the multi-phase training pipeline.

    Tracks all phases and provides resume/recovery capabilities.
    """

    run_id: str
    base_output_dir: str
    base_model_path: str
    total_phases: int
    current_phase: int = 0
    phases: list[PhaseState] = field(default_factory=list)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    status: str = "pending"  # pending, running, completed, failed

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "run_id": self.run_id,
            "base_output_dir": self.base_output_dir,
            "base_model_path": self.base_model_path,
            "total_phases": self.total_phases,
            "current_phase": self.current_phase,
            KEY_PHASES: [p.to_dict() for p in self.phases],
            KEY_STARTED_AT: self.started_at.isoformat() if self.started_at else None,
            KEY_COMPLETED_AT: (self.completed_at.isoformat() if self.completed_at else None),
            KEY_STATUS: self.status,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PipelineState:
        """Create from dictionary (JSON deserialization).

        Handles forward/backward compatibility and corrupted data.
        """
        # FIX BUG-003: Handle null/None phases gracefully
        phases_data = data.pop(KEY_PHASES, [])
        if phases_data is None:
            logger.warning("[DB:COMPAT] 'phases' is null in state file, using empty list")
            phases_data = []

        phases = [PhaseState.from_dict(p) for p in phases_data]

        # FIX BUG-001: Filter only known fields
        known_fields = {f.name for f in dataclasses.fields(cls)} - {KEY_PHASES}
        filtered_data = {k: v for k, v in data.items() if k in known_fields}

        unknown_fields = set(data.keys()) - known_fields
        if unknown_fields:
            logger.debug(f"[DB:COMPAT] Ignoring unknown fields in PipelineState: {unknown_fields}")

        if filtered_data.get(KEY_STARTED_AT):
            filtered_data[KEY_STARTED_AT] = datetime.fromisoformat(filtered_data[KEY_STARTED_AT])
        if filtered_data.get(KEY_COMPLETED_AT):
            filtered_data[KEY_COMPLETED_AT] = datetime.fromisoformat(filtered_data[KEY_COMPLETED_AT])

        return cls(**filtered_data, phases=phases)

    @property
    def completed_phases(self) -> list[PhaseState]:
        """Get list of completed phases."""
        return [p for p in self.phases if p.is_complete]

    @property
    def failed_phases(self) -> list[PhaseState]:
        """Get list of failed phases."""
        return [p for p in self.phases if p.status == PhaseStatus.FAILED]

    @property
    def progress_percent(self) -> float:
        """Get pipeline progress as percentage."""
        if self.total_phases == 0:
            return 0.0
        return (len(self.completed_phases) / self.total_phases) * 100


# =============================================================================
# CHECKPOINT HELPERS
# =============================================================================


def _extract_checkpoint_step(checkpoint_path: Path) -> int | float:
    """
    Extract step number from checkpoint-N directory name.

    Args:
        checkpoint_path: Path to checkpoint directory

    Returns:
        Step number, inf for checkpoint-final, -1 for unknown format
    """
    name = checkpoint_path.name
    if name == CHECKPOINT_FINAL_DIR:
        return float("inf")  # Always last
    try:
        # checkpoint-100 -> 100
        return int(name.split("-")[-1])
    except ValueError:
        return -1  # Unknown format — sort first


def _get_sorted_checkpoints(output_dir: Path) -> list[Path]:
    """
    Get checkpoint directories sorted by step number.

    Filters directories only (excludes files) and sorts numerically.

    Args:
        output_dir: Directory containing checkpoints

    Returns:
        List of checkpoint Paths sorted by step number
    """
    checkpoints = [p for p in output_dir.glob("checkpoint-*") if p.is_dir()]
    return sorted(checkpoints, key=_extract_checkpoint_step)


# =============================================================================
# DATA BUFFER CLASS
# =============================================================================


class DataBuffer:
    """
    Checkpoint and state manager for multi-phase training.

    Manages output directories, tracks progress, enables resume,
    and handles checkpoint cleanup.

    Responsibilities:
    - Create unique run directories for each training run
    - Generate per-phase output directories
    - Track phase execution status
    - Persist state to JSON for resume capability
    - Provide model paths for sequential phase loading
    - Clean up old checkpoints to save disk space

    Example:
        buffer = DataBuffer(
            base_output_dir="output/checkpoints",
            base_model_path="Qwen/Qwen2.5-7B-Instruct",
        )

        # Initialize new run
        buffer.init_pipeline(strategies)

        # Execute phases
        for phase_idx in range(buffer.total_phases):
            output_dir = buffer.get_phase_output_dir(phase_idx)
            model_path = buffer.get_model_path_for_phase(phase_idx)

            buffer.mark_phase_started(phase_idx)
            # ... run training ...
            buffer.mark_phase_completed(phase_idx, checkpoint, metrics)
            buffer.save_state()
    """

    # State file name within run directory
    STATE_FILENAME = "pipeline_state.json"

    def __init__(
        self,
        base_output_dir: str | Path,
        base_model_path: str,
        *,
        run_id: str | None = None,
        callbacks: DataBufferEventCallbacks | None = None,
        _fault_simulator: FaultSimulator | None = None,
    ):
        """
        Initialize DataBuffer.

        Args:
            base_output_dir: Output root directory for this run (recommended: "output" inside run workspace)
            base_model_path: Path/name of base model (HuggingFace or local)
            run_id: Optional run ID (auto-generated if None)
            callbacks: Optional event callbacks for MLflow integration
            _fault_simulator: Optional fault simulator for testing (internal use)
        """
        self.base_output_dir = Path(base_output_dir)
        self.base_model_path = base_model_path
        self.run_id = run_id or self._generate_run_id()
        self._state: PipelineState | None = None
        self._callbacks = callbacks or DataBufferEventCallbacks()
        self._fault_simulator = _fault_simulator

        # Ensure base directory exists
        self.base_output_dir.mkdir(parents=True, exist_ok=True)

        # Log initialization with simulation status
        sim_status = "ENABLED" if _fault_simulator else "disabled"
        logger.debug(
            f"[DB:INIT] base_dir={self.base_output_dir}, model={self.base_model_path}, "
            f"run_id={self.run_id}, simulation={sim_status}"
        )
        logger.info(f"DataBuffer initialized: run_id={self.run_id}")
        if _fault_simulator:
            logger.warning(f"[DB:SIMULATION] Fault simulation ENABLED: {_fault_simulator}")

    @staticmethod
    def _generate_run_id() -> str:
        """Generate unique run ID: timestamp with milliseconds + pid."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:RUN_ID_TIMESTAMP_LEN]
        pid = os.getpid()
        return f"run_{timestamp}_pid{pid}"

    @property
    def run_dir(self) -> Path:
        """
        Get the run output directory.

        IMPORTANT (path layout v2):
        - Provider/deployer already runs training in an isolated run workspace.
        - We store outputs directly under `base_output_dir` (no nested `run_id/` folder).
        """
        return self.base_output_dir

    @property
    def state_file(self) -> Path:
        """Get path to state file."""
        return self.run_dir / self.STATE_FILENAME

    @property
    def state(self) -> PipelineState:
        """Get current pipeline state (raises if not initialized)."""
        if self._state is None:
            raise RuntimeError("Pipeline not initialized. Call init_pipeline() first.")
        return self._state

    @property
    def is_initialized(self) -> bool:
        """Whether the pipeline state is initialized."""
        return self._state is not None

    @property
    def total_phases(self) -> int:
        """Get total number of phases."""
        return self.state.total_phases if self._state else 0

    @property
    def current_phase(self) -> int:
        """Get current phase index."""
        return self.state.current_phase if self._state else 0

    # =========================================================================
    # INITIALIZATION
    # =========================================================================

    def init_pipeline(
        self,
        strategies: list[StrategyPhaseConfig],
        *,
        global_hyperparams: Any | None = None,
        force: bool = False,
    ) -> None:
        """
        Initialize pipeline state for a new training run.

        Creates run directory and initializes phase states.

        Args:
            strategies: List of strategy phase configurations
            global_hyperparams: Optional global hyperparameters (used as baseline for all phases)
            force: If True, overwrite existing state

        Raises:
            ValueError: If strategies list is empty or contains None
            RuntimeError: If run already exists and force=False
        """
        if not strategies:
            raise ValueError("Strategies list cannot be empty")

        # FIX BUG-005: Check for None elements in strategies list
        if any(s is None for s in strategies):
            raise ValueError("Strategies list cannot contain None values")

        # Check for existing run
        if self.state_file.exists() and not force:
            raise RuntimeError(
                f"Run '{self.run_id}' already exists. Use force=True to overwrite or load_existing() to resume."
            )

        # Create run directory
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Create phase states
        phases = []
        for idx, strategy in enumerate(strategies):
            phase_output = self.run_dir / f"phase_{idx}_{strategy.strategy_type}"

            # Strict config: phase hyperparams are the only supported source for per-phase overrides.
            # For display/debugging we also allow fallback to global hyperparams (if provided by caller).
            epochs = strategy.hyperparams.epochs
            if epochs is None and global_hyperparams is not None:
                epochs = getattr(global_hyperparams, "epochs", None)
            if epochs is None:
                epochs = 1

            learning_rate = strategy.hyperparams.learning_rate
            if learning_rate is None and global_hyperparams is not None:
                learning_rate = getattr(global_hyperparams, "learning_rate", None)

            phase = PhaseState(
                phase_idx=idx,
                strategy_type=strategy.strategy_type,
                output_dir=str(phase_output),
                epochs=epochs,
                learning_rate=learning_rate,
                dataset_name=strategy.dataset,
            )
            phases.append(phase)

        # Create pipeline state
        self._state = PipelineState(
            run_id=self.run_id,
            base_output_dir=str(self.base_output_dir),
            base_model_path=self.base_model_path,
            total_phases=len(strategies),
            current_phase=0,
            phases=phases,
            started_at=datetime.now(),
            status="running",
        )

        # Save initial state
        self.save_state()

        phase_types = [s.strategy_type for s in strategies]
        logger.debug(
            f"[DB:RUN_INITIALIZED] run_id={self.run_id}, "
            f"phases={len(strategies)}, types={phase_types}, "
            f"run_dir={self.run_dir}"
        )
        logger.info(f"Pipeline initialized: {len(strategies)} phases, run_id={self.run_id}")

        # Fire callback
        if self._callbacks.on_pipeline_initialized:
            strategy_chain = [s.strategy_type for s in strategies]
            self._callbacks.on_pipeline_initialized(self.run_id, len(strategies), strategy_chain)

    @classmethod
    def load_existing(
        cls,
        base_output_dir: str | Path,
        run_id: str | None = None,
    ) -> DataBuffer:
        """
        Load existing DataBuffer from saved state.

        Args:
            base_output_dir: Base directory for checkpoints
            run_id: Run ID to load

        Returns:
            DataBuffer with loaded state

        Raises:
            FileNotFoundError: If state file doesn't exist
        """
        base_path = Path(base_output_dir)
        state_file = base_path / cls.STATE_FILENAME

        if not state_file.exists():
            raise FileNotFoundError(f"State file not found: {state_file}")

        # FIX BUG-002: Handle corrupted JSON gracefully
        try:
            with state_file.open() as f:
                state_data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Corrupted state file: {state_file}. "
                f"JSON error at line {e.lineno}, column {e.colno}: {e.msg}. "
                f"Consider starting fresh with force=True or manually fix the file."
            ) from e

        # Validate basic structure
        if not isinstance(state_data, dict):
            raise ValueError(f"Invalid state file format: {state_file}. Expected dict, got {type(state_data).__name__}")

        state = PipelineState.from_dict(state_data)

        effective_run_id = run_id or state.run_id
        buffer = cls(
            base_output_dir=base_output_dir,
            base_model_path=state.base_model_path,
            run_id=effective_run_id,
        )
        buffer._state = state

        logger.debug(
            f"[DB:STATE_LOADED] run_id={buffer.run_id}, "
            f"completed={len(state.completed_phases)}/{state.total_phases}, "
            f"status={state.status}"
        )
        logger.info(
            f"Loaded existing run: {buffer.run_id}, progress: {len(state.completed_phases)}/{state.total_phases}"
        )

        return buffer

    # =========================================================================
    # PATH MANAGEMENT
    # =========================================================================

    def get_phase_output_dir(self, phase_idx: int) -> str:
        """
        Get output directory for a specific phase.

        Args:
            phase_idx: Phase index (0-based)

        Returns:
            Path to phase output directory

        Example:
            buffer.get_phase_output_dir(0)
            # 'output/run_xxx/phase_0_sft/'
        """
        if phase_idx < 0 or phase_idx >= self.total_phases:
            raise IndexError(f"Phase index {phase_idx} out of range [0, {self.total_phases})")

        phase = self.state.phases[phase_idx]
        if phase.output_dir:
            # Ensure directory exists
            Path(phase.output_dir).mkdir(parents=True, exist_ok=True)
            return phase.output_dir

        # Fallback: generate from strategy type
        output_dir = self.run_dir / f"phase_{phase_idx}_{phase.strategy_type}"
        output_dir.mkdir(parents=True, exist_ok=True)
        return str(output_dir)

    def get_model_path_for_phase(self, phase_idx: int) -> str:
        """
        Get model path to load for a specific phase.

        - Phase 0: Returns base_model_path
        - Phase N (N>0): Returns checkpoint from phase N-1

        Args:
            phase_idx: Phase index (0-based)

        Returns:
            Path to model (HuggingFace name or local checkpoint)

        Example:
            buffer.get_model_path_for_phase(0)
            # 'Qwen/Qwen2.5-7B-Instruct'
            buffer.get_model_path_for_phase(1)
            # 'output/run_xxx/phase_0_sft/checkpoint-final/'

        SIMULATION_POINT: missing_checkpoint
        """
        if phase_idx < 0 or phase_idx >= self.total_phases:
            raise IndexError(f"Phase index {phase_idx} out of range [0, {self.total_phases})")

        # Phase 0 uses base model
        if phase_idx == 0:
            logger.debug(f"[DB:MODEL_PATH_RESOLVED] phase={phase_idx}, source=base_model, path={self.base_model_path}")
            return self.base_model_path

        # ===== SIMULATION_POINT: missing_checkpoint =====
        # TODO[SIMULATION]: Hook to simulate a missing checkpoint.
        # Used to test fallback to base_model when a checkpoint is lost.
        if self._fault_simulator and self._fault_simulator.missing_checkpoint:
            logger.warning(
                f"[DB:SIMULATION] Simulated missing checkpoint for phase {phase_idx - 1}. "
                f"Falling back to base model: {self.base_model_path}"
            )
            return self.base_model_path

        # Later phases use checkpoint from previous phase
        prev_phase = self.state.phases[phase_idx - 1]

        if prev_phase.checkpoint_path:
            logger.debug(
                f"[DB:MODEL_PATH_RESOLVED] phase={phase_idx}, source=prev_checkpoint, path={prev_phase.checkpoint_path}"
            )
            return prev_phase.checkpoint_path

        # Fallback: try to find checkpoint in previous phase output
        prev_output = prev_phase.output_dir
        if prev_output:
            checkpoint_final = Path(prev_output) / CHECKPOINT_FINAL_DIR
            if checkpoint_final.exists():
                logger.debug(
                    f"[DB:MODEL_PATH_RESOLVED] phase={phase_idx}, source=checkpoint-final, path={checkpoint_final}"
                )
                return str(checkpoint_final)

            # Look for latest checkpoint-N directory (sorted by step number)
            checkpoints = _get_sorted_checkpoints(Path(prev_output))
            checkpoints = [c for c in checkpoints if c.name != CHECKPOINT_FINAL_DIR]
            if checkpoints:
                logger.debug(
                    f"[DB:MODEL_PATH_RESOLVED] phase={phase_idx}, source=latest_checkpoint, path={checkpoints[-1]}"
                )
                return str(checkpoints[-1])

        # No checkpoint found - warning and use base model
        logger.warning(f"No checkpoint found for phase {phase_idx - 1}. Using base model: {self.base_model_path}")
        return self.base_model_path

    # =========================================================================
    # PHASE STATUS TRACKING
    # =========================================================================

    def mark_phase_started(self, phase_idx: int) -> None:
        """
        Mark a phase as started.

        Args:
            phase_idx: Phase index to mark
        """
        if phase_idx < 0 or phase_idx >= self.total_phases:
            raise IndexError(f"Phase index {phase_idx} out of range")

        phase = self.state.phases[phase_idx]
        phase.status = PhaseStatus.RUNNING
        phase.started_at = datetime.now()
        self.state.current_phase = phase_idx

        logger.debug(
            f"[DB:PHASE_STARTED] phase={phase_idx}, strategy={phase.strategy_type}, output_dir={phase.output_dir}"
        )
        logger.info(f"Phase {phase_idx} ({phase.strategy_type}) started")
        self.save_state()

        # Fire callback
        if self._callbacks.on_phase_started:
            self._callbacks.on_phase_started(phase_idx, phase.strategy_type)

    def mark_phase_completed(
        self,
        phase_idx: int,
        checkpoint_path: str | None = None,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        """
        Mark a phase as successfully completed.

        Args:
            phase_idx: Phase index to mark
            checkpoint_path: Path to saved checkpoint
            metrics: Training metrics (loss, etc.)
        """
        if phase_idx < 0 or phase_idx >= self.total_phases:
            raise IndexError(f"Phase index {phase_idx} out of range")

        phase = self.state.phases[phase_idx]
        phase.status = PhaseStatus.COMPLETED
        phase.completed_at = datetime.now()
        phase.checkpoint_path = checkpoint_path
        if metrics:
            phase.metrics = metrics

        # Update pipeline status if all phases complete
        if all(p.is_complete for p in self.state.phases):
            self.state.status = "completed"
            self.state.completed_at = datetime.now()

        duration = phase.duration_seconds
        duration_str = f"{duration / 60:.1f} min" if duration else "unknown"

        logger.debug(
            f"[DB:PHASE_COMPLETED] phase={phase_idx}, strategy={phase.strategy_type}, "
            f"duration={duration_str}, checkpoint={checkpoint_path}, metrics={metrics}"
        )
        logger.info(f"Phase {phase_idx} ({phase.strategy_type}) completed in {duration_str}")
        self.save_state()

        # Fire callback
        if self._callbacks.on_phase_completed:
            self._callbacks.on_phase_completed(phase_idx, phase.strategy_type, "completed")

    def mark_phase_failed(
        self,
        phase_idx: int,
        error_message: str,
    ) -> None:
        """
        Mark a phase as failed.

        Args:
            phase_idx: Phase index to mark
            error_message: Error description
        """
        if phase_idx < 0 or phase_idx >= self.total_phases:
            raise IndexError(f"Phase index {phase_idx} out of range")

        phase = self.state.phases[phase_idx]
        phase.status = PhaseStatus.FAILED
        phase.error_message = error_message
        phase.completed_at = datetime.now()

        self.state.status = "failed"

        logger.debug(f"[DB:PHASE_FAILED] phase={phase_idx}, strategy={phase.strategy_type}, error={error_message}")
        logger.error(f"Phase {phase_idx} ({phase.strategy_type}) failed: {error_message}")
        self.save_state()

        # Fire callback
        if self._callbacks.on_phase_completed:
            self._callbacks.on_phase_completed(phase_idx, phase.strategy_type, "failed")

    def mark_phase_interrupted(
        self,
        phase_idx: int,
        reason: str,
        checkpoint_path: str | None = None,
    ) -> None:
        """
        Mark a phase as interrupted (SIGINT/SIGTERM).

        Interrupted phases can be resumed from their checkpoint.

        Args:
            phase_idx: Phase index to mark
            reason: Reason for interruption (sigint, sigterm, timeout)
            checkpoint_path: Path to emergency checkpoint if saved
        """
        if phase_idx < 0 or phase_idx >= self.total_phases:
            raise IndexError(f"Phase index {phase_idx} out of range")

        phase = self.state.phases[phase_idx]
        phase.status = PhaseStatus.INTERRUPTED
        phase.error_message = f"Interrupted: {reason}"
        phase.completed_at = datetime.now()

        # Save checkpoint path if provided
        if checkpoint_path:
            phase.checkpoint_path = checkpoint_path

        self.state.status = "interrupted"

        logger.debug(
            f"[DB:PHASE_INTERRUPTED] phase={phase_idx}, strategy={phase.strategy_type}, "
            f"reason={reason}, checkpoint={checkpoint_path}"
        )
        logger.warning(f"Phase {phase_idx} ({phase.strategy_type}) interrupted: {reason}")
        self.save_state()

        # Fire callback
        if self._callbacks.on_phase_completed:
            self._callbacks.on_phase_completed(phase_idx, phase.strategy_type, "interrupted")

    def mark_phase_skipped(
        self,
        phase_idx: int,
        reason: str,
        checkpoint_path: str | None = None,
    ) -> None:
        """
        Mark a phase as skipped (adapter cache hit).

        Skipped phases loaded a pre-trained adapter from HF Hub instead of training.

        Args:
            phase_idx: Phase index to mark
            reason: Reason for skipping (e.g., "adapter_cache_hit: phase-0-sft-dsABC123")
            checkpoint_path: Optional path if a local checkpoint was also used
        """
        if phase_idx < 0 or phase_idx >= self.total_phases:
            raise IndexError(f"Phase index {phase_idx} out of range")

        phase = self.state.phases[phase_idx]
        phase.status = PhaseStatus.SKIPPED
        phase.completed_at = datetime.now()
        if checkpoint_path:
            phase.checkpoint_path = checkpoint_path

        logger.debug(
            f"[DB:PHASE_SKIPPED] phase={phase_idx}, strategy={phase.strategy_type}, reason={reason}"
        )
        logger.info(f"Phase {phase_idx} ({phase.strategy_type}) skipped: {reason}")
        self.save_state()

        # Fire callback
        if self._callbacks.on_phase_completed:
            self._callbacks.on_phase_completed(phase_idx, phase.strategy_type, "skipped")

    # =========================================================================
    # STATE PERSISTENCE
    # =========================================================================

    def save_state(self) -> None:
        """
        Save current state to JSON file (atomic write).

        Uses tmp file + atomic rename to prevent corruption on crash.
        Called automatically after status changes.

        SIMULATION_POINT: fail_on_save, slow_io_delay_ms, fail_after_n_saves
        """
        start_time = time.perf_counter()

        if self._state is None:
            logger.warning("[DB:SAVE_SKIP] No state to save (_state is None)")
            return

        # ===== SIMULATION_POINT: fail_on_save, fail_after_n_saves =====
        # TODO[SIMULATION]: Hook to simulate a save failure.
        # Used to test graceful handling when writes fail.
        if self._fault_simulator and self._fault_simulator.should_fail_save():
            logger.error("[DB:SIMULATION] Simulated save failure triggered")
            raise SimulatedFaultError(self._fault_simulator.fail_on_save_error)

        # ===== SIMULATION_POINT: slow_io_delay_ms =====
        # TODO[SIMULATION]: Hook to simulate slow I/O.
        # Used to test timeouts and performance behavior.
        if self._fault_simulator and self._fault_simulator.slow_io_delay_ms > 0:
            delay = self._fault_simulator.get_io_delay()
            logger.debug(f"[DB:SIMULATION] Simulating slow IO: {delay:.3f}s delay")
            time.sleep(delay)

        # Ensure run directory exists
        self.run_dir.mkdir(parents=True, exist_ok=True)

        state_data = self._state.to_dict()

        # Log state details before save
        completed_count = len(self._state.completed_phases)
        logger.debug(
            f"[DB:SAVE_START] run_id={self.run_id}, status={self._state.status}, "
            f"phases={completed_count}/{self._state.total_phases}, current={self._state.current_phase}"
        )

        # Atomic write: tmp file -> fsync -> rename
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=self.run_dir,
            prefix=".pipeline_state_",
            suffix=".tmp",
        )
        try:
            with os.fdopen(tmp_fd, "w") as f:
                json.dump(state_data, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            Path(tmp_path).replace(self.state_file)
        except Exception:
            # Cleanup tmp file on error
            tmp_path_obj = Path(tmp_path)
            if tmp_path_obj.exists():
                tmp_path_obj.unlink()
            raise

        # Calculate file size and duration
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        file_size_kb = self.state_file.stat().st_size / 1024

        logger.debug(
            f"[DB:STATE_SAVED] file={self.state_file}, status={self._state.status}, "
            f"size={file_size_kb:.1f}KB, duration={elapsed_ms:.1f}ms"
        )

        # Fire callback
        if self._callbacks.on_state_saved:
            self._callbacks.on_state_saved(self.run_id, str(self.state_file))

    def load_state(self) -> bool:
        """
        Load state from JSON file.

        Returns:
            True if state was loaded, False if file doesn't exist or is corrupted

        SIMULATION_POINT: fail_on_load, corrupt_state, slow_io_delay_ms
        """
        start_time = time.perf_counter()

        logger.debug(f"[DB:LOAD_START] Attempting to load state from {self.state_file}")

        if not self.state_file.exists():
            logger.debug(f"[DB:LOAD_SKIP] State file does not exist: {self.state_file}")
            return False

        # ===== SIMULATION_POINT: fail_on_load =====
        # TODO[SIMULATION]: Hook to simulate a load failure.
        # Used to test graceful handling when reads fail.
        if self._fault_simulator and self._fault_simulator.fail_on_load:
            logger.error("[DB:SIMULATION] Simulated load failure triggered")
            raise SimulatedFaultError(self._fault_simulator.fail_on_load_error)

        # ===== SIMULATION_POINT: slow_io_delay_ms =====
        # TODO[SIMULATION]: Hook to simulate slow I/O during load.
        if self._fault_simulator and self._fault_simulator.slow_io_delay_ms > 0:
            delay = self._fault_simulator.get_io_delay()
            logger.debug(f"[DB:SIMULATION] Simulating slow IO: {delay:.3f}s delay")
            time.sleep(delay)

        # ===== SIMULATION_POINT: corrupt_state =====
        # TODO[SIMULATION]: Hook to simulate corrupted JSON.
        # Used to test handling of corrupted files.
        if self._fault_simulator and self._fault_simulator.corrupt_state:
            logger.error("[DB:SIMULATION] Simulated corrupted state triggered")
            logger.error(
                f"[DB:CORRUPTED_STATE] {self.state_file}: Simulated corruption. "
                f"Consider starting fresh with force=True."
            )
            return False

        try:
            file_size_kb = self.state_file.stat().st_size / 1024
            with self.state_file.open() as f:
                state_data = json.load(f)
        except json.JSONDecodeError as e:
            logger.error(
                f"[DB:CORRUPTED_STATE] {self.state_file}: line {e.lineno}, col {e.colno}: {e.msg}. "
                f"Consider starting fresh with force=True or manually fix the file."
            )
            return False

        # Treat unexpected JSON shapes as corrupted (fault-tolerant resume path).
        if not isinstance(state_data, dict):
            logger.error(
                f"[DB:CORRUPTED_STATE] {self.state_file}: invalid format. "
                f"Expected JSON object (dict), got {type(state_data).__name__}. "
                "Consider starting fresh with force=True."
            )
            return False

        self._state = PipelineState.from_dict(state_data)

        # Calculate duration and log details
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        completed_count = len(self._state.completed_phases)

        logger.debug(
            f"[DB:STATE_LOADED] run_id={self._state.run_id}, status={self._state.status}, "
            f"phases={completed_count}/{self._state.total_phases}, "
            f"size={file_size_kb:.1f}KB, duration={elapsed_ms:.1f}ms"
        )
        logger.info(f"State loaded from: {self.state_file}")
        return True

    # =========================================================================
    # RESUME LOGIC
    # =========================================================================

    def get_resume_phase(self) -> int | None:
        """
        Get the phase index to resume from.

        Returns:
            - Index of first incomplete phase
            - None if all phases are complete

        Logic:
            1. If a phase is RUNNING → resume from that phase
            2. If a phase is INTERRUPTED → resume from that phase (has checkpoint)
            3. If a phase is FAILED → resume from that phase (retry)
            4. If a phase is PENDING and all previous are COMPLETED → resume from that
            5. If all phases are COMPLETED → return None
        """
        if self._state is None:
            return None

        for idx, phase in enumerate(self.state.phases):
            reason = {
                PhaseStatus.RUNNING: "was_running",
                PhaseStatus.INTERRUPTED: "was_interrupted",
                PhaseStatus.FAILED: "was_failed",
                PhaseStatus.PENDING: "is_pending",
            }.get(phase.status)
            if reason:
                logger.debug(f"[DB:RESUME_DETECTED] phase={idx}, reason={reason}")
                logger.info(f"Resume from {reason.replace('_', ' ')} phase {idx} ({phase.strategy_type})")
                return idx

        # All complete
        logger.debug("[DB:NO_RESUME] all_phases_complete=True")
        logger.info("All phases completed, nothing to resume")
        return None

    def can_resume(self) -> bool:
        """Check if there's anything to resume."""
        return self.get_resume_phase() is not None

    def get_resume_checkpoint(self, phase_idx: int) -> str | None:
        """
        Get checkpoint path to resume from for a specific phase.

        Used when a phase was interrupted mid-training.

        Args:
            phase_idx: Phase index

        Returns:
            Path to last checkpoint in phase output_dir, or None
        """
        if phase_idx < 0 or phase_idx >= self.total_phases:
            return None

        phase = self.state.phases[phase_idx]

        # If phase has explicit checkpoint path, use it
        if phase.checkpoint_path:
            checkpoint_path = Path(phase.checkpoint_path)
            if checkpoint_path.exists():
                return str(checkpoint_path)

        # Look for checkpoints in output dir (sorted by step number)
        if phase.output_dir:
            output_path = Path(phase.output_dir)
            if output_path.exists():
                checkpoints = _get_sorted_checkpoints(output_path)
                if checkpoints:
                    return str(checkpoints[-1])

        return None

    # =========================================================================
    # CLEANUP
    # =========================================================================

    def cleanup_old_checkpoints(
        self,
        keep_last: int = 2,
        *,
        dry_run: bool = False,
    ) -> list[str]:
        """
        Clean up old checkpoints to save disk space.

        Keeps the last N checkpoints per phase, plus the final checkpoint.

        Args:
            keep_last: Number of intermediate checkpoints to keep
            dry_run: If True, only report what would be deleted

        Returns:
            List of deleted (or would-be-deleted) directories

        SIMULATION_POINT: fail_on_cleanup
        """
        logger.debug(f"[DB:CLEANUP_START] keep_last={keep_last}, dry_run={dry_run}")
        deleted: list[str] = []

        if self._state is None:
            logger.debug("[DB:CLEANUP_SKIP] No state initialized")
            return deleted

        # ===== SIMULATION_POINT: fail_on_cleanup =====
        # TODO[SIMULATION]: Hook to simulate a cleanup failure.
        # Used to test graceful handling when deletion fails.
        if self._fault_simulator and self._fault_simulator.fail_on_cleanup:
            logger.error("[DB:SIMULATION] Simulated cleanup failure triggered")
            raise SimulatedFaultError("Simulated cleanup failure")

        for phase in self.state.phases:
            if not phase.output_dir:
                continue

            output_path = Path(phase.output_dir)
            if not output_path.exists():
                continue

            # Find all checkpoint directories (sorted by step number, dirs only)
            checkpoints = _get_sorted_checkpoints(output_path)

            # Never delete checkpoint-final
            checkpoints = [c for c in checkpoints if c.name != CHECKPOINT_FINAL_DIR]

            # Keep last N checkpoints (handle keep_last=0 explicitly)
            if keep_last <= 0:
                to_delete = checkpoints  # Delete all intermediate checkpoints
            elif len(checkpoints) > keep_last:
                to_delete = checkpoints[:-keep_last]
            else:
                to_delete = []

            for checkpoint_dir in to_delete:
                if dry_run:
                    logger.info(f"Would delete: {checkpoint_dir}")
                    deleted.append(str(checkpoint_dir))
                else:
                    # FIX BUG-006: Handle PermissionError gracefully
                    try:
                        shutil.rmtree(checkpoint_dir)
                        logger.info(f"Deleted: {checkpoint_dir}")
                        deleted.append(str(checkpoint_dir))
                    except PermissionError as e:
                        logger.warning(f"Cannot delete {checkpoint_dir}: {e} (skipping)")
                    except OSError as e:
                        logger.warning(f"Error deleting {checkpoint_dir}: {e} (skipping)")

        if deleted:
            size_saved = len(deleted) * CHECKPOINT_SIZE_ESTIMATE_MB
            logger.info(f"Cleanup: {len(deleted)} checkpoints deleted, ~{size_saved}MB freed")

            # Fire callback
            if not dry_run:
                cb = self._callbacks.on_checkpoint_cleanup
                if cb is not None:
                    cb(len(deleted), size_saved)

        return deleted

    # =========================================================================
    # UTILITIES
    # =========================================================================

    def get_summary(self) -> dict[str, Any]:
        """
        Get pipeline summary for logging/display.

        Returns:
            Dict with status, progress, phases info
        """
        if self._state is None:
            return {KEY_STATUS: "not_initialized"}

        return {
            "run_id": self.run_id,
            KEY_STATUS: self.state.status,
            "progress": f"{len(self.state.completed_phases)}/{self.state.total_phases}",
            "progress_percent": f"{self.state.progress_percent:.1f}%",
            "current_phase": self.state.current_phase,
            KEY_PHASES: [
                {
                    "idx": p.phase_idx,
                    "type": p.strategy_type,
                    KEY_STATUS: p.status.value,
                    "duration": f"{p.duration_seconds / 60:.1f}min" if p.duration_seconds else None,
                }
                for p in self.state.phases
            ],
        }

    def __repr__(self) -> str:
        if self._state is None:
            return f"DataBuffer(run_id={self.run_id}, not_initialized)"

        return f"DataBuffer(run_id={self.run_id}, phases={len(self.state.completed_phases)}/{self.state.total_phases})"


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def list_available_runs(base_output_dir: str | Path) -> list[dict[str, Any]]:
    """
    List all available training runs in output directory.

    Args:
        base_output_dir: Base directory for checkpoints

    Returns:
        List of dicts with run_id, status, created_at
    """
    base_path = Path(base_output_dir)
    runs: list[dict[str, Any]] = []

    if not base_path.exists():
        return runs

    for run_dir in base_path.iterdir():
        if not run_dir.is_dir():
            continue

        # Support both layouts:
        # - v2: <run_dir>/output/pipeline_state.json (provider run workspace layout)
        # - legacy: <run_dir>/pipeline_state.json
        state_file = run_dir / "output" / DataBuffer.STATE_FILENAME
        if not state_file.exists():
            state_file = run_dir / DataBuffer.STATE_FILENAME
        if state_file.exists():
            try:
                with state_file.open() as f:
                    state = json.load(f)
                runs.append(
                    {
                        "run_id": run_dir.name,
                        KEY_STATUS: state.get(KEY_STATUS, "unknown"),
                        "progress": f"{len([p for p in state.get(KEY_PHASES, []) if p.get(KEY_STATUS) == 'completed'])}/{state.get('total_phases', 0)}",
                        KEY_STARTED_AT: state.get(KEY_STARTED_AT),
                    }
                )
            except Exception as e:
                logger.warning(f"Could not read state for {run_dir.name}: {e}")

    return sorted(runs, key=lambda x: x.get(KEY_STARTED_AT, ""), reverse=True)


__all__ = [
    "DataBuffer",
    "DataBufferEventCallbacks",
    "FaultSimulator",
    "PhaseState",
    "PhaseStatus",
    "PipelineState",
    "SimulatedFaultError",
    "list_available_runs",
]
