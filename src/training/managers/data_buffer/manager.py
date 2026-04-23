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

    buffer = DataBuffer(
        base_output_dir="output",
        base_model_path="Qwen/Qwen2.5-7B-Instruct",
    )
    buffer.init_pipeline(
        strategies=[
            StrategyPhaseConfig(strategy_type="sft"),
            StrategyPhaseConfig(strategy_type="dpo"),
        ]
    )
    phase_0_dir = buffer.get_phase_output_dir(0)
    buffer.mark_phase_started(0)
    buffer.mark_phase_completed(0, checkpoint_path, metrics)
    buffer.save_state()

    # Resume after failure
    buffer = DataBuffer.load_existing(base_output_dir="output")
    resume_from = buffer.get_resume_phase()
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.training.managers.constants import (
    CHECKPOINT_FINAL_DIR,
    CHECKPOINT_SIZE_ESTIMATE_MB,
    KEY_STATUS,
    KEY_PHASES,
    RUN_ID_TIMESTAMP_LEN,
)
from src.utils.logger import logger

from src.training.managers.data_buffer.checkpoint_utils import _get_sorted_checkpoints
from src.training.managers.data_buffer.events import DataBufferEventCallbacks
from src.training.managers.data_buffer.fault_simulator import FaultSimulator, SimulatedFaultError
from src.training.managers.data_buffer.state_models import PhaseState, PhaseStatus, PipelineState
from src.training.metrics_models import TrainingMetricsSnapshot

if TYPE_CHECKING:
    from src.utils.config import StrategyPhaseConfig


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
            base_output_dir: Output root directory for this run
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

        Args:
            strategies: List of strategy phase configurations
            global_hyperparams: Optional global hyperparameters
            force: If True, overwrite existing state

        Raises:
            ValueError: If strategies list is empty or contains None
            RuntimeError: If run already exists and force=False
        """
        if not strategies:
            raise ValueError("Strategies list cannot be empty")

        if any(s is None for s in strategies):
            raise ValueError("Strategies list cannot contain None values")

        if self.state_file.exists() and not force:
            raise RuntimeError(
                f"Run '{self.run_id}' already exists. Use force=True to overwrite or load_existing() to resume."
            )

        self.run_dir.mkdir(parents=True, exist_ok=True)

        phases = []
        for idx, strategy in enumerate(strategies):
            phase_output = self.run_dir / f"phase_{idx}_{strategy.strategy_type}"

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

        self.save_state()

        phase_types = [s.strategy_type for s in strategies]
        logger.debug(
            f"[DB:RUN_INITIALIZED] run_id={self.run_id}, "
            f"phases={len(strategies)}, types={phase_types}, "
            f"run_dir={self.run_dir}"
        )
        logger.info(f"Pipeline initialized: {len(strategies)} phases, run_id={self.run_id}")

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

        try:
            with state_file.open() as f:
                state_data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Corrupted state file: {state_file}. "
                f"JSON error at line {e.lineno}, column {e.colno}: {e.msg}. "
                f"Consider starting fresh with force=True or manually fix the file."
            ) from e

        if not isinstance(state_data, dict):
            raise ValueError(
                f"Invalid state file format: {state_file}. Expected dict, got {type(state_data).__name__}"
            )

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
        """Get output directory for a specific phase."""
        if phase_idx < 0 or phase_idx >= self.total_phases:
            raise IndexError(f"Phase index {phase_idx} out of range [0, {self.total_phases})")

        phase = self.state.phases[phase_idx]
        if phase.output_dir:
            Path(phase.output_dir).mkdir(parents=True, exist_ok=True)
            return phase.output_dir

        output_dir = self.run_dir / f"phase_{phase_idx}_{phase.strategy_type}"
        output_dir.mkdir(parents=True, exist_ok=True)
        return str(output_dir)

    def get_model_path_for_phase(self, phase_idx: int) -> str:
        """
        Get model path to load for a specific phase.

        - Phase 0: Returns base_model_path
        - Phase N (N>0): Returns checkpoint from phase N-1

        SIMULATION_POINT: missing_checkpoint
        """
        if phase_idx < 0 or phase_idx >= self.total_phases:
            raise IndexError(f"Phase index {phase_idx} out of range [0, {self.total_phases})")

        if phase_idx == 0:
            logger.debug(
                f"[DB:MODEL_PATH_RESOLVED] phase={phase_idx}, source=base_model, path={self.base_model_path}"
            )
            return self.base_model_path

        # ===== SIMULATION_POINT: missing_checkpoint =====
        if self._fault_simulator and self._fault_simulator.missing_checkpoint:
            logger.warning(
                f"[DB:SIMULATION] Simulated missing checkpoint for phase {phase_idx - 1}. "
                f"Falling back to base model: {self.base_model_path}"
            )
            return self.base_model_path

        prev_phase = self.state.phases[phase_idx - 1]

        if prev_phase.checkpoint_path:
            logger.debug(
                f"[DB:MODEL_PATH_RESOLVED] phase={phase_idx}, source=prev_checkpoint, "
                f"path={prev_phase.checkpoint_path}"
            )
            return prev_phase.checkpoint_path

        prev_output = prev_phase.output_dir
        if prev_output:
            checkpoint_final = Path(prev_output) / CHECKPOINT_FINAL_DIR
            if checkpoint_final.exists():
                logger.debug(
                    f"[DB:MODEL_PATH_RESOLVED] phase={phase_idx}, source=checkpoint-final, "
                    f"path={checkpoint_final}"
                )
                return str(checkpoint_final)

            checkpoints = _get_sorted_checkpoints(Path(prev_output))
            checkpoints = [c for c in checkpoints if c.name != CHECKPOINT_FINAL_DIR]
            if checkpoints:
                logger.debug(
                    f"[DB:MODEL_PATH_RESOLVED] phase={phase_idx}, source=latest_checkpoint, "
                    f"path={checkpoints[-1]}"
                )
                return str(checkpoints[-1])

        logger.warning(
            f"No checkpoint found for phase {phase_idx - 1}. Using base model: {self.base_model_path}"
        )
        return self.base_model_path

    # =========================================================================
    # PHASE STATUS TRACKING
    # =========================================================================

    def mark_phase_started(self, phase_idx: int) -> None:
        """Mark a phase as started."""
        if phase_idx < 0 or phase_idx >= self.total_phases:
            raise IndexError(f"Phase index {phase_idx} out of range")

        phase = self.state.phases[phase_idx]
        phase.status = PhaseStatus.RUNNING
        phase.started_at = datetime.now()
        self.state.current_phase = phase_idx

        logger.debug(
            f"[DB:PHASE_STARTED] phase={phase_idx}, strategy={phase.strategy_type}, "
            f"output_dir={phase.output_dir}"
        )
        logger.info(f"Phase {phase_idx} ({phase.strategy_type}) started")
        self.save_state()

        if self._callbacks.on_phase_started:
            self._callbacks.on_phase_started(phase_idx, phase.strategy_type)

    def mark_phase_completed(
        self,
        phase_idx: int,
        checkpoint_path: str | None = None,
        metrics: TrainingMetricsSnapshot | dict[str, Any] | None = None,
    ) -> None:
        """Mark a phase as successfully completed.

        ``metrics`` may be provided as a :class:`TrainingMetricsSnapshot` (preferred)
        or a legacy ``dict`` for backward compatibility — the dict is coerced.
        """
        if phase_idx < 0 or phase_idx >= self.total_phases:
            raise IndexError(f"Phase index {phase_idx} out of range")

        phase = self.state.phases[phase_idx]
        phase.status = PhaseStatus.COMPLETED
        phase.completed_at = datetime.now()
        phase.checkpoint_path = checkpoint_path
        if metrics is not None:
            if isinstance(metrics, TrainingMetricsSnapshot):
                phase.metrics = metrics
            else:
                phase.metrics = TrainingMetricsSnapshot.from_dict(metrics)

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

        if self._callbacks.on_phase_completed:
            self._callbacks.on_phase_completed(phase_idx, phase.strategy_type, "completed")

    def mark_phase_failed(self, phase_idx: int, error_message: str) -> None:
        """Mark a phase as failed."""
        if phase_idx < 0 or phase_idx >= self.total_phases:
            raise IndexError(f"Phase index {phase_idx} out of range")

        phase = self.state.phases[phase_idx]
        phase.status = PhaseStatus.FAILED
        phase.error_message = error_message
        phase.completed_at = datetime.now()
        self.state.status = "failed"

        logger.debug(
            f"[DB:PHASE_FAILED] phase={phase_idx}, strategy={phase.strategy_type}, error={error_message}"
        )
        logger.error(f"Phase {phase_idx} ({phase.strategy_type}) failed: {error_message}")
        self.save_state()

        if self._callbacks.on_phase_completed:
            self._callbacks.on_phase_completed(phase_idx, phase.strategy_type, "failed")

    def mark_phase_interrupted(
        self,
        phase_idx: int,
        reason: str,
        checkpoint_path: str | None = None,
    ) -> None:
        """Mark a phase as interrupted (SIGINT/SIGTERM)."""
        if phase_idx < 0 or phase_idx >= self.total_phases:
            raise IndexError(f"Phase index {phase_idx} out of range")

        phase = self.state.phases[phase_idx]
        phase.status = PhaseStatus.INTERRUPTED
        phase.error_message = f"Interrupted: {reason}"
        phase.completed_at = datetime.now()

        if checkpoint_path:
            phase.checkpoint_path = checkpoint_path

        self.state.status = "interrupted"

        logger.debug(
            f"[DB:PHASE_INTERRUPTED] phase={phase_idx}, strategy={phase.strategy_type}, "
            f"reason={reason}, checkpoint={checkpoint_path}"
        )
        logger.warning(f"Phase {phase_idx} ({phase.strategy_type}) interrupted: {reason}")
        self.save_state()

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

        if self._callbacks.on_phase_completed:
            self._callbacks.on_phase_completed(phase_idx, phase.strategy_type, "skipped")

    # =========================================================================
    # STATE PERSISTENCE
    # =========================================================================

    def save_state(self) -> None:
        """
        Save current state to JSON file (atomic write).

        Uses tmp file + atomic rename to prevent corruption on crash.

        SIMULATION_POINT: fail_on_save, slow_io_delay_ms, fail_after_n_saves
        """
        start_time = time.perf_counter()

        if self._state is None:
            logger.warning("[DB:SAVE_SKIP] No state to save (_state is None)")
            return

        # ===== SIMULATION_POINT: fail_on_save, fail_after_n_saves =====
        if self._fault_simulator and self._fault_simulator.should_fail_save():
            logger.error("[DB:SIMULATION] Simulated save failure triggered")
            raise SimulatedFaultError(self._fault_simulator.fail_on_save_error)

        # ===== SIMULATION_POINT: slow_io_delay_ms =====
        if self._fault_simulator and self._fault_simulator.slow_io_delay_ms > 0:
            delay = self._fault_simulator.get_io_delay()
            logger.debug(f"[DB:SIMULATION] Simulating slow IO: {delay:.3f}s delay")
            time.sleep(delay)

        self.run_dir.mkdir(parents=True, exist_ok=True)
        state_data = self._state.to_dict()

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
            tmp_path_obj = Path(tmp_path)
            if tmp_path_obj.exists():
                tmp_path_obj.unlink()
            raise

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        file_size_kb = self.state_file.stat().st_size / 1024

        logger.debug(
            f"[DB:STATE_SAVED] file={self.state_file}, status={self._state.status}, "
            f"size={file_size_kb:.1f}KB, duration={elapsed_ms:.1f}ms"
        )

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
        if self._fault_simulator and self._fault_simulator.fail_on_load:
            logger.error("[DB:SIMULATION] Simulated load failure triggered")
            raise SimulatedFaultError(self._fault_simulator.fail_on_load_error)

        # ===== SIMULATION_POINT: slow_io_delay_ms =====
        if self._fault_simulator and self._fault_simulator.slow_io_delay_ms > 0:
            delay = self._fault_simulator.get_io_delay()
            logger.debug(f"[DB:SIMULATION] Simulating slow IO: {delay:.3f}s delay")
            time.sleep(delay)

        # ===== SIMULATION_POINT: corrupt_state =====
        if self._fault_simulator and self._fault_simulator.corrupt_state:
            logger.error("[DB:SIMULATION] Simulated corrupted state triggered")
            logger.error(
                f"[DB:CORRUPTED_STATE] {self.state_file}: Simulated corruption. "
                "Consider starting fresh with force=True."
            )
            return False

        try:
            file_size_kb = self.state_file.stat().st_size / 1024
            with self.state_file.open() as f:
                state_data = json.load(f)
        except json.JSONDecodeError as e:
            logger.error(
                f"[DB:CORRUPTED_STATE] {self.state_file}: line {e.lineno}, col {e.colno}: {e.msg}. "
                "Consider starting fresh with force=True or manually fix the file."
            )
            return False

        if not isinstance(state_data, dict):
            logger.error(
                f"[DB:CORRUPTED_STATE] {self.state_file}: invalid format. "
                f"Expected JSON object (dict), got {type(state_data).__name__}. "
                "Consider starting fresh with force=True."
            )
            return False

        self._state = PipelineState.from_dict(state_data)

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
                logger.info(
                    f"Resume from {reason.replace('_', ' ')} phase {idx} ({phase.strategy_type})"
                )
                return idx

        logger.debug("[DB:NO_RESUME] all_phases_complete=True")
        logger.info("All phases completed, nothing to resume")
        return None

    def can_resume(self) -> bool:
        """Check if there's anything to resume."""
        return self.get_resume_phase() is not None

    def get_resume_checkpoint(self, phase_idx: int) -> str | None:
        """Get checkpoint path to resume from for a specific phase."""
        if phase_idx < 0 or phase_idx >= self.total_phases:
            return None

        phase = self.state.phases[phase_idx]

        if phase.checkpoint_path:
            checkpoint_path = Path(phase.checkpoint_path)
            if checkpoint_path.exists():
                return str(checkpoint_path)

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

        SIMULATION_POINT: fail_on_cleanup
        """
        from src.training.managers.constants import CHECKPOINT_FINAL_DIR as _FINAL

        logger.debug(f"[DB:CLEANUP_START] keep_last={keep_last}, dry_run={dry_run}")
        deleted: list[str] = []

        if self._state is None:
            logger.debug("[DB:CLEANUP_SKIP] No state initialized")
            return deleted

        # ===== SIMULATION_POINT: fail_on_cleanup =====
        if self._fault_simulator and self._fault_simulator.fail_on_cleanup:
            logger.error("[DB:SIMULATION] Simulated cleanup failure triggered")
            raise SimulatedFaultError("Simulated cleanup failure")

        for phase in self.state.phases:
            if not phase.output_dir:
                continue

            output_path = Path(phase.output_dir)
            if not output_path.exists():
                continue

            checkpoints = _get_sorted_checkpoints(output_path)
            checkpoints = [c for c in checkpoints if c.name != _FINAL]

            if keep_last <= 0:
                to_delete = checkpoints
            elif len(checkpoints) > keep_last:
                to_delete = checkpoints[:-keep_last]
            else:
                to_delete = []

            for checkpoint_dir in to_delete:
                if dry_run:
                    logger.info(f"Would delete: {checkpoint_dir}")
                    deleted.append(str(checkpoint_dir))
                else:
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

            if not dry_run:
                cb = self._callbacks.on_checkpoint_cleanup
                if cb is not None:
                    cb(len(deleted), size_saved)

        return deleted

    # =========================================================================
    # UTILITIES
    # =========================================================================

    def get_summary(self) -> dict[str, Any]:
        """Get pipeline summary for logging/display."""
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
        return (
            f"DataBuffer(run_id={self.run_id}, "
            f"phases={len(self.state.completed_phases)}/{self.state.total_phases})"
        )


__all__ = ["DataBuffer"]
