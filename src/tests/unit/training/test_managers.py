"""
Unit tests for Training Managers (DataBuffer, etc.).

Tests state management, checkpoint handling, resume logic,
and cleanup functionality.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.training.managers.data_buffer import (
    DataBuffer,
    PhaseState,
    PhaseStatus,
    PipelineState,
    list_available_runs,
)
from src.utils.config import PhaseHyperparametersConfig, StrategyPhaseConfig

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def mock_strategies():
    """Create mock StrategyPhaseConfig list."""
    return [
        StrategyPhaseConfig(
            strategy_type="sft",
            dataset="default",
            hyperparams=PhaseHyperparametersConfig(epochs=1, learning_rate=2e-4),
        ),
        StrategyPhaseConfig(
            strategy_type="dpo",
            dataset="default",
            hyperparams=PhaseHyperparametersConfig(epochs=1, learning_rate=2e-4, beta=0.1),
        ),
    ]


@pytest.fixture
def data_buffer(tmp_path):
    """Create DataBuffer with temp directory."""
    return DataBuffer(
        base_output_dir=tmp_path / "checkpoints",
        base_model_path="test-model/test-llm",
    )


@pytest.fixture
def initialized_buffer(data_buffer, mock_strategies):
    """Create initialized DataBuffer with pipeline."""
    data_buffer.init_pipeline(mock_strategies)
    return data_buffer


# =============================================================================
# TEST CLASS: PhaseState
# =============================================================================


class TestPhaseState:
    """Unit tests for PhaseState dataclass."""

    def test_create_phase_state(self):
        """
        Given: Phase configuration
        When: PhaseState is created
        Then: Has correct default values
        """
        phase = PhaseState(
            phase_idx=0,
            strategy_type="sft",
        )

        assert phase.phase_idx == 0
        assert phase.strategy_type == "sft"
        assert phase.status == PhaseStatus.PENDING
        assert phase.output_dir is None
        assert phase.checkpoint_path is None

    def test_is_complete_property(self):
        """
        Given: PhaseState with COMPLETED status
        When: is_complete is checked
        Then: Returns True
        """
        phase = PhaseState(phase_idx=0, strategy_type="sft")
        phase.status = PhaseStatus.COMPLETED

        assert phase.is_complete is True

    def test_is_running_property(self):
        """
        Given: PhaseState with RUNNING status
        When: is_running is checked
        Then: Returns True
        """
        phase = PhaseState(phase_idx=0, strategy_type="sft")
        phase.status = PhaseStatus.RUNNING

        assert phase.is_running is True

    def test_to_dict_serialization(self):
        """
        Given: PhaseState
        When: to_dict is called
        Then: Returns JSON-serializable dict
        """
        phase = PhaseState(
            phase_idx=0,
            strategy_type="sft",
            output_dir="/tmp/test",
        )

        data = phase.to_dict()

        assert isinstance(data, dict)
        assert data["phase_idx"] == 0
        assert data["strategy_type"] == "sft"
        assert data["status"] == "pending"  # Enum converted to string

    def test_from_dict_deserialization(self):
        """
        Given: Dict with phase data
        When: from_dict is called
        Then: Returns PhaseState instance
        """
        data = {
            "phase_idx": 0,
            "strategy_type": "sft",
            "status": "completed",
            "output_dir": "/tmp/test",
            "checkpoint_path": "/tmp/checkpoint",
            "started_at": None,
            "completed_at": None,
            "error_message": None,
            "metrics": {},
            "epochs": 1,
            "learning_rate": 2e-4,
            "dataset_name": "default",
        }

        phase = PhaseState.from_dict(data)

        assert isinstance(phase, PhaseState)
        assert phase.phase_idx == 0
        assert phase.status == PhaseStatus.COMPLETED


# =============================================================================
# TEST CLASS: PipelineState
# =============================================================================


class TestPipelineState:
    """Unit tests for PipelineState dataclass."""

    def test_create_pipeline_state(self):
        """
        Given: Pipeline configuration
        When: PipelineState is created
        Then: Has correct values
        """
        state = PipelineState(
            run_id="test_run_123",
            base_output_dir="/tmp/output",
            base_model_path="test-model",
            total_phases=2,
        )

        assert state.run_id == "test_run_123"
        assert state.total_phases == 2
        assert state.current_phase == 0
        assert state.status == "pending"

    def test_completed_phases_property(self):
        """
        Given: PipelineState with some completed phases
        When: completed_phases is accessed
        Then: Returns only completed phases
        """
        phase1 = PhaseState(phase_idx=0, strategy_type="sft")
        phase1.status = PhaseStatus.COMPLETED

        phase2 = PhaseState(phase_idx=1, strategy_type="dpo")
        phase2.status = PhaseStatus.PENDING

        state = PipelineState(
            run_id="test",
            base_output_dir="/tmp",
            base_model_path="model",
            total_phases=2,
            phases=[phase1, phase2],
        )

        completed = state.completed_phases

        assert len(completed) == 1
        assert completed[0].phase_idx == 0

    def test_progress_percent(self):
        """
        Given: PipelineState with 1/2 phases completed
        When: progress_percent is accessed
        Then: Returns 50.0
        """
        phase1 = PhaseState(phase_idx=0, strategy_type="sft")
        phase1.status = PhaseStatus.COMPLETED

        phase2 = PhaseState(phase_idx=1, strategy_type="dpo")
        phase2.status = PhaseStatus.PENDING

        state = PipelineState(
            run_id="test",
            base_output_dir="/tmp",
            base_model_path="model",
            total_phases=2,
            phases=[phase1, phase2],
        )

        assert state.progress_percent == 50.0

    def test_to_dict_and_from_dict_roundtrip(self):
        """
        Given: PipelineState
        When: Serialized and deserialized
        Then: Data is preserved
        """
        original = PipelineState(
            run_id="test_run",
            base_output_dir="/tmp/output",
            base_model_path="test-model",
            total_phases=2,
            phases=[
                PhaseState(phase_idx=0, strategy_type="sft"),
                PhaseState(phase_idx=1, strategy_type="dpo"),
            ],
        )

        data = original.to_dict()
        restored = PipelineState.from_dict(data)

        assert restored.run_id == original.run_id
        assert restored.total_phases == original.total_phases
        assert len(restored.phases) == len(original.phases)


# =============================================================================
# TEST CLASS: DataBuffer Initialization
# =============================================================================


class TestDataBufferInit:
    """Tests for DataBuffer initialization."""

    def test_init_creates_base_dir(self, tmp_path):
        """
        Given: Non-existent base directory
        When: DataBuffer is created
        Then: Directory is created
        """
        base_dir = tmp_path / "new_checkpoints"

        DataBuffer(
            base_output_dir=base_dir,
            base_model_path="test-model",
        )

        assert base_dir.exists()

    def test_init_generates_run_id(self, tmp_path):
        """
        Given: No run_id provided
        When: DataBuffer is created
        Then: run_id is auto-generated
        """
        buffer = DataBuffer(
            base_output_dir=tmp_path,
            base_model_path="test-model",
        )

        assert buffer.run_id is not None
        assert buffer.run_id.startswith("run_")

    def test_init_with_custom_run_id(self, tmp_path):
        """
        Given: Custom run_id
        When: DataBuffer is created
        Then: Uses provided run_id
        """
        buffer = DataBuffer(
            base_output_dir=tmp_path,
            base_model_path="test-model",
            run_id="my_custom_run",
        )

        assert buffer.run_id == "my_custom_run"

    def test_init_pipeline_creates_state(self, data_buffer, mock_strategies):
        """
        Given: Uninitialized DataBuffer
        When: init_pipeline is called
        Then: State is created with correct phases
        """
        data_buffer.init_pipeline(mock_strategies)

        assert data_buffer.total_phases == 2
        assert data_buffer.state.status == "running"
        assert len(data_buffer.state.phases) == 2

    def test_init_pipeline_creates_state_file(self, data_buffer, mock_strategies):
        """
        Given: Initialized DataBuffer
        When: init_pipeline completes
        Then: State file is created
        """
        data_buffer.init_pipeline(mock_strategies)

        assert data_buffer.state_file.exists()

    def test_init_pipeline_empty_strategies_raises(self, data_buffer):
        """
        Given: Empty strategies list
        When: init_pipeline is called
        Then: Raises ValueError
        """
        with pytest.raises(ValueError, match="empty"):
            data_buffer.init_pipeline([])

    def test_init_pipeline_existing_run_raises(self, data_buffer, mock_strategies):
        """
        Given: Already initialized run
        When: init_pipeline is called again
        Then: Raises RuntimeError
        """
        data_buffer.init_pipeline(mock_strategies)

        with pytest.raises(RuntimeError, match="already exists"):
            data_buffer.init_pipeline(mock_strategies)

    def test_init_pipeline_force_overwrites(self, data_buffer, mock_strategies):
        """
        Given: Already initialized run
        When: init_pipeline is called with force=True
        Then: State is overwritten
        """
        data_buffer.init_pipeline(mock_strategies)
        data_buffer.init_pipeline(mock_strategies, force=True)

        assert data_buffer.total_phases == 2


# =============================================================================
# TEST CLASS: DataBuffer Path Management
# =============================================================================


class TestDataBufferPaths:
    """Tests for DataBuffer path management."""

    def test_get_phase_output_dir(self, initialized_buffer):
        """
        Given: Initialized DataBuffer
        When: get_phase_output_dir is called
        Then: Returns correct path
        """
        output_dir = initialized_buffer.get_phase_output_dir(0)

        assert "phase_0_sft" in output_dir
        assert Path(output_dir).exists()

    def test_get_phase_output_dir_invalid_index(self, initialized_buffer):
        """
        Given: Invalid phase index
        When: get_phase_output_dir is called
        Then: Raises IndexError
        """
        with pytest.raises(IndexError):
            initialized_buffer.get_phase_output_dir(10)

    def test_get_model_path_phase_0(self, initialized_buffer):
        """
        Given: Phase 0
        When: get_model_path_for_phase is called
        Then: Returns base model path
        """
        model_path = initialized_buffer.get_model_path_for_phase(0)

        assert model_path == "test-model/test-llm"

    def test_get_model_path_phase_1_uses_checkpoint(self, initialized_buffer):
        """
        Given: Phase 0 completed with checkpoint
        When: get_model_path_for_phase(1) is called
        Then: Returns checkpoint path
        """
        # Mark phase 0 as completed with checkpoint
        phase_0_dir = initialized_buffer.get_phase_output_dir(0)
        checkpoint_path = Path(phase_0_dir) / "checkpoint-final"
        checkpoint_path.mkdir(parents=True)

        initialized_buffer.mark_phase_started(0)
        initialized_buffer.mark_phase_completed(0, str(checkpoint_path))

        model_path = initialized_buffer.get_model_path_for_phase(1)

        assert "checkpoint-final" in model_path


# =============================================================================
# TEST CLASS: DataBuffer Phase Tracking
# =============================================================================


class TestDataBufferPhaseTracking:
    """Tests for phase status tracking."""

    def test_mark_phase_started(self, initialized_buffer):
        """
        Given: Initialized DataBuffer
        When: mark_phase_started is called
        Then: Phase status is RUNNING
        """
        initialized_buffer.mark_phase_started(0)

        phase = initialized_buffer.state.phases[0]
        assert phase.status == PhaseStatus.RUNNING
        assert phase.started_at is not None

    def test_mark_phase_completed(self, initialized_buffer):
        """
        Given: Running phase
        When: mark_phase_completed is called
        Then: Phase status is COMPLETED
        """
        initialized_buffer.mark_phase_started(0)
        initialized_buffer.mark_phase_completed(0, "/tmp/checkpoint", {"loss": 1.5})

        phase = initialized_buffer.state.phases[0]
        assert phase.status == PhaseStatus.COMPLETED
        assert phase.checkpoint_path == "/tmp/checkpoint"
        assert phase.metrics["loss"] == 1.5

    def test_mark_phase_failed(self, initialized_buffer):
        """
        Given: Running phase
        When: mark_phase_failed is called
        Then: Phase status is FAILED
        """
        initialized_buffer.mark_phase_started(0)
        initialized_buffer.mark_phase_failed(0, "OOM error")

        phase = initialized_buffer.state.phases[0]
        assert phase.status == PhaseStatus.FAILED
        assert phase.error_message == "OOM error"
        assert initialized_buffer.state.status == "failed"

    def test_mark_phase_interrupted(self, initialized_buffer):
        """
        Given: Running phase
        When: mark_phase_interrupted is called
        Then: Phase status is INTERRUPTED
        """
        initialized_buffer.mark_phase_started(0)
        initialized_buffer.mark_phase_interrupted(0, "sigint", "/tmp/emergency")

        phase = initialized_buffer.state.phases[0]
        assert phase.status == PhaseStatus.INTERRUPTED
        assert phase.checkpoint_path == "/tmp/emergency"
        assert initialized_buffer.state.status == "interrupted"

    def test_all_phases_completed_updates_pipeline_status(self, initialized_buffer):
        """
        Given: All phases completed
        When: Last phase is marked completed
        Then: Pipeline status is 'completed'
        """
        initialized_buffer.mark_phase_started(0)
        initialized_buffer.mark_phase_completed(0)
        initialized_buffer.mark_phase_started(1)
        initialized_buffer.mark_phase_completed(1)

        assert initialized_buffer.state.status == "completed"


# =============================================================================
# TEST CLASS: DataBuffer Resume Logic
# =============================================================================


class TestDataBufferResume:
    """Tests for resume functionality."""

    def test_get_resume_phase_running(self, initialized_buffer):
        """
        Given: Phase 0 is RUNNING
        When: get_resume_phase is called
        Then: Returns 0
        """
        initialized_buffer.mark_phase_started(0)

        resume_phase = initialized_buffer.get_resume_phase()

        assert resume_phase == 0

    def test_get_resume_phase_failed(self, initialized_buffer):
        """
        Given: Phase 0 FAILED
        When: get_resume_phase is called
        Then: Returns 0 (retry failed phase)
        """
        initialized_buffer.mark_phase_started(0)
        initialized_buffer.mark_phase_failed(0, "error")

        resume_phase = initialized_buffer.get_resume_phase()

        assert resume_phase == 0

    def test_get_resume_phase_pending(self, initialized_buffer):
        """
        Given: All phases PENDING
        When: get_resume_phase is called
        Then: Returns 0 (first pending)
        """
        resume_phase = initialized_buffer.get_resume_phase()

        assert resume_phase == 0

    def test_get_resume_phase_all_completed(self, initialized_buffer):
        """
        Given: All phases COMPLETED
        When: get_resume_phase is called
        Then: Returns None
        """
        initialized_buffer.mark_phase_started(0)
        initialized_buffer.mark_phase_completed(0)
        initialized_buffer.mark_phase_started(1)
        initialized_buffer.mark_phase_completed(1)

        resume_phase = initialized_buffer.get_resume_phase()

        assert resume_phase is None

    def test_can_resume(self, initialized_buffer):
        """
        Given: Pending phases
        When: can_resume is called
        Then: Returns True
        """
        assert initialized_buffer.can_resume() is True

    def test_load_existing_run(self, initialized_buffer, tmp_path):
        """
        Given: Saved state file
        When: load_existing is called
        Then: State is restored
        """
        initialized_buffer.mark_phase_started(0)
        initialized_buffer.save_state()

        # Load in new buffer instance
        loaded = DataBuffer.load_existing(
            base_output_dir=tmp_path / "checkpoints",
            run_id=initialized_buffer.run_id,
        )

        assert loaded.run_id == initialized_buffer.run_id
        assert loaded.state.phases[0].status == PhaseStatus.RUNNING


# =============================================================================
# TEST CLASS: DataBuffer Cleanup
# =============================================================================


class TestDataBufferCleanup:
    """Tests for checkpoint cleanup."""

    def test_cleanup_old_checkpoints(self, initialized_buffer):
        """
        Given: Multiple checkpoints in phase dir
        When: cleanup_old_checkpoints is called
        Then: Keeps only last N checkpoints
        """
        # Create multiple checkpoints
        phase_dir = Path(initialized_buffer.get_phase_output_dir(0))
        for i in range(5):
            (phase_dir / f"checkpoint-{i * 100}").mkdir()
        (phase_dir / "checkpoint-final").mkdir()

        # Cleanup keeping last 2
        deleted = initialized_buffer.cleanup_old_checkpoints(keep_last=2)

        # Should delete checkpoint-0, checkpoint-100, checkpoint-200
        assert len(deleted) == 3
        assert (phase_dir / "checkpoint-final").exists()  # Never deleted

    def test_cleanup_dry_run(self, initialized_buffer):
        """
        Given: Checkpoints to delete
        When: cleanup_old_checkpoints with dry_run=True
        Then: Nothing is actually deleted
        """
        phase_dir = Path(initialized_buffer.get_phase_output_dir(0))
        for i in range(5):
            (phase_dir / f"checkpoint-{i * 100}").mkdir()

        deleted = initialized_buffer.cleanup_old_checkpoints(keep_last=2, dry_run=True)

        assert len(deleted) == 3
        # All directories should still exist
        assert len(list(phase_dir.glob("checkpoint-*"))) == 5


# =============================================================================
# TEST CLASS: Utility Functions
# =============================================================================


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_list_available_runs(self, tmp_path):
        """
        Given: Multiple runs in directory
        When: list_available_runs is called
        Then: Returns list of runs
        """
        # Create some provider-like run workspaces:
        # <base>/<run_id>/output/pipeline_state.json
        for i in range(3):
            run_id = f"run_{i}"
            run_workspace = tmp_path / run_id
            buffer = DataBuffer(
                base_output_dir=run_workspace / "output",
                base_model_path="model",
                run_id=run_id,
            )
            strategy = StrategyPhaseConfig(
                strategy_type="sft",
                dataset="default",
                hyperparams=PhaseHyperparametersConfig(epochs=1, learning_rate=2e-4),
            )
            buffer.init_pipeline([strategy])

        runs = list_available_runs(tmp_path)

        assert len(runs) == 3
        assert all("run_id" in r for r in runs)
        assert all("status" in r for r in runs)

    def test_list_available_runs_empty_dir(self, tmp_path):
        """
        Given: Empty directory
        When: list_available_runs is called
        Then: Returns empty list
        """
        runs = list_available_runs(tmp_path)

        assert runs == []

    def test_get_summary(self, initialized_buffer):
        """
        Given: Initialized DataBuffer
        When: get_summary is called
        Then: Returns summary dict
        """
        summary = initialized_buffer.get_summary()

        assert "run_id" in summary
        assert "status" in summary
        assert "progress" in summary
        assert "phases" in summary


# =============================================================================
# TEST CLASS: DataBuffer Hardening (P0/P1 fixes)
# =============================================================================


class TestDataBufferHardening:
    """Tests for DataBuffer hardening fixes."""

    def test_load_state_returns_false_when_file_missing(self, tmp_path):
        """
        Given: DataBuffer with no state file
        When: load_state is called
        Then: Returns False (not raises exception)
        """
        buffer = DataBuffer(
            base_output_dir=tmp_path / "checkpoints",
            base_model_path="test-model",
        )

        result = buffer.load_state()

        assert result is False
        assert buffer._state is None

    def test_load_state_returns_false_on_corrupted_json(self, tmp_path):
        """
        Given: Corrupted JSON state file
        When: load_state is called
        Then: Returns False (handles JSONDecodeError gracefully)
        """
        buffer = DataBuffer(
            base_output_dir=tmp_path / "checkpoints",
            base_model_path="test-model",
        )

        # Create corrupted state file
        buffer.run_dir.mkdir(parents=True, exist_ok=True)
        with buffer.state_file.open("w") as f:
            f.write("{ invalid json }")

        result = buffer.load_state()

        assert result is False
        assert buffer._state is None

    def test_save_state_atomic_creates_no_tmp_on_success(self, initialized_buffer):
        """
        Given: Initialized DataBuffer
        When: save_state completes successfully
        Then: No tmp files remain in run_dir
        """
        initialized_buffer.save_state()

        # Check no tmp files remain
        tmp_files = list(initialized_buffer.run_dir.glob(".pipeline_state_*.tmp"))
        assert len(tmp_files) == 0

        # State file should exist
        assert initialized_buffer.state_file.exists()

    def test_checkpoint_sorting_by_step(self, tmp_path):
        """
        Given: Checkpoints with various step numbers
        When: _get_sorted_checkpoints is called
        Then: Returns checkpoints sorted by step number (not lexicographically)
        """
        from src.training.managers.data_buffer import _get_sorted_checkpoints

        # Create checkpoints that would sort incorrectly lexicographically
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()

        # Create: checkpoint-2, checkpoint-10, checkpoint-100
        (checkpoint_dir / "checkpoint-2").mkdir()
        (checkpoint_dir / "checkpoint-10").mkdir()
        (checkpoint_dir / "checkpoint-100").mkdir()
        (checkpoint_dir / "checkpoint-final").mkdir()

        sorted_checkpoints = _get_sorted_checkpoints(checkpoint_dir)
        names = [c.name for c in sorted_checkpoints]

        # Should be sorted numerically: 2, 10, 100, final(inf)
        assert names == ["checkpoint-2", "checkpoint-10", "checkpoint-100", "checkpoint-final"]

    def test_checkpoint_sorting_ignores_files(self, tmp_path):
        """
        Given: Checkpoints mixed with files
        When: _get_sorted_checkpoints is called
        Then: Only directories are returned
        """
        from src.training.managers.data_buffer import _get_sorted_checkpoints

        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()

        # Create directory and file
        (checkpoint_dir / "checkpoint-100").mkdir()
        (checkpoint_dir / "checkpoint-200.txt").touch()  # File, not directory

        sorted_checkpoints = _get_sorted_checkpoints(checkpoint_dir)
        names = [c.name for c in sorted_checkpoints]

        # Should only include the directory
        assert names == ["checkpoint-100"]

    def test_metrics_sanitization_numpy(self, initialized_buffer):
        """
        Given: Metrics with numpy types
        When: mark_phase_completed with numpy metrics
        Then: State saves successfully (no JSON serialization error)
        """
        import numpy as np

        initialized_buffer.mark_phase_started(0)

        # Create metrics with numpy types
        numpy_metrics = {
            "train_loss": np.float32(0.5),
            "global_step": np.int64(100),
            "learning_rate": np.float64(2e-4),
        }

        # This should not raise JSONDecodeError
        initialized_buffer.mark_phase_completed(0, "/tmp/checkpoint", numpy_metrics)

        # Verify state was saved
        assert initialized_buffer.state_file.exists()

        # Load and verify metrics are serializable
        import json

        with initialized_buffer.state_file.open() as f:
            data = json.load(f)

        assert data["phases"][0]["metrics"]["train_loss"] == 0.5
        assert data["phases"][0]["metrics"]["global_step"] == 100

    def test_cleanup_keep_last_zero_deletes_all(self, initialized_buffer):
        """
        Given: Multiple checkpoints
        When: cleanup_old_checkpoints with keep_last=0
        Then: All intermediate checkpoints are deleted (only checkpoint-final remains)
        """
        phase_dir = Path(initialized_buffer.get_phase_output_dir(0))
        for i in range(3):
            (phase_dir / f"checkpoint-{i * 100}").mkdir()
        (phase_dir / "checkpoint-final").mkdir()

        deleted = initialized_buffer.cleanup_old_checkpoints(keep_last=0)

        # All intermediate checkpoints should be deleted
        assert len(deleted) == 3
        # Only checkpoint-final should remain
        remaining = list(phase_dir.glob("checkpoint-*"))
        assert len(remaining) == 1
        assert remaining[0].name == "checkpoint-final"

    def test_run_id_includes_pid(self, tmp_path):
        """
        Given: DataBuffer with auto-generated run_id
        When: run_id is generated
        Then: Contains pid for uniqueness
        """
        import os

        buffer = DataBuffer(
            base_output_dir=tmp_path / "checkpoints",
            base_model_path="test-model",
        )

        assert f"_pid{os.getpid()}" in buffer.run_id


# =============================================================================
# TEST CLASS: FaultSimulator (Simulation Testing)
# =============================================================================


class TestFaultSimulator:
    """Tests for FaultSimulator and simulation scenarios."""

    def test_fault_simulator_save_failure(self, tmp_path, mock_strategies):
        """
        Given: DataBuffer with FaultSimulator(fail_on_save=True)
        When: save_state is called
        Then: SimulatedFaultError is raised
        """
        from src.training.managers.data_buffer import FaultSimulator, SimulatedFaultError

        # Initialize without simulator (init_pipeline calls save_state)
        buffer = DataBuffer(
            base_output_dir=tmp_path / "checkpoints",
            base_model_path="test-model",
        )
        buffer.init_pipeline(mock_strategies)

        # Enable simulator after initialization
        buffer._fault_simulator = FaultSimulator(fail_on_save=True)

        with pytest.raises(SimulatedFaultError, match="Simulated save failure"):
            buffer.save_state()

    def test_fault_simulator_load_failure(self, tmp_path, mock_strategies):
        """
        Given: DataBuffer with FaultSimulator(fail_on_load=True) and existing state
        When: load_state is called
        Then: SimulatedFaultError is raised
        """
        from src.training.managers.data_buffer import FaultSimulator, SimulatedFaultError

        # First create a valid state file
        buffer1 = DataBuffer(
            base_output_dir=tmp_path / "checkpoints",
            base_model_path="test-model",
        )
        buffer1.init_pipeline(mock_strategies)

        # Now try to load with fault simulator
        simulator = FaultSimulator(fail_on_load=True)
        buffer2 = DataBuffer(
            base_output_dir=tmp_path / "checkpoints",
            base_model_path="test-model",
            run_id=buffer1.run_id,
            _fault_simulator=simulator,
        )

        with pytest.raises(SimulatedFaultError, match="Simulated load failure"):
            buffer2.load_state()

    def test_fault_simulator_corrupt_state(self, tmp_path, mock_strategies):
        """
        Given: DataBuffer with FaultSimulator(corrupt_state=True) and existing state
        When: load_state is called
        Then: Returns False (simulated corruption)
        """
        from src.training.managers.data_buffer import FaultSimulator

        # First create a valid state file
        buffer1 = DataBuffer(
            base_output_dir=tmp_path / "checkpoints",
            base_model_path="test-model",
        )
        buffer1.init_pipeline(mock_strategies)

        # Now try to load with corrupt simulation
        simulator = FaultSimulator(corrupt_state=True)
        buffer2 = DataBuffer(
            base_output_dir=tmp_path / "checkpoints",
            base_model_path="test-model",
            run_id=buffer1.run_id,
            _fault_simulator=simulator,
        )

        result = buffer2.load_state()
        assert result is False

    def test_fault_simulator_missing_checkpoint(self, initialized_buffer):
        """
        Given: DataBuffer with FaultSimulator(missing_checkpoint=True)
        When: get_model_path_for_phase(1) is called
        Then: Returns base_model_path (simulated missing checkpoint)
        """
        from src.training.managers.data_buffer import FaultSimulator

        # Complete phase 0 to have a "checkpoint"
        initialized_buffer.mark_phase_started(0)
        initialized_buffer.mark_phase_completed(0, "/tmp/checkpoint")

        # Enable missing checkpoint simulation
        initialized_buffer._fault_simulator = FaultSimulator(missing_checkpoint=True)

        # Should return base model instead of checkpoint
        model_path = initialized_buffer.get_model_path_for_phase(1)
        assert model_path == initialized_buffer.base_model_path

    def test_fault_simulator_cleanup_failure(self, initialized_buffer):
        """
        Given: DataBuffer with FaultSimulator(fail_on_cleanup=True)
        When: cleanup_old_checkpoints is called
        Then: SimulatedFaultError is raised
        """
        from src.training.managers.data_buffer import FaultSimulator, SimulatedFaultError

        # Create some checkpoints
        phase_dir = Path(initialized_buffer.get_phase_output_dir(0))
        for i in range(3):
            (phase_dir / f"checkpoint-{i * 100}").mkdir()

        # Enable cleanup failure simulation
        initialized_buffer._fault_simulator = FaultSimulator(fail_on_cleanup=True)

        with pytest.raises(SimulatedFaultError, match="Simulated cleanup failure"):
            initialized_buffer.cleanup_old_checkpoints(keep_last=1)

    def test_fault_simulator_fail_after_n_saves(self, tmp_path, mock_strategies):
        """
        Given: DataBuffer with FaultSimulator(fail_after_n_saves=2)
        When: save_state is called 3 times
        Then: First 2 succeed, 3rd raises SimulatedFaultError
        """
        from src.training.managers.data_buffer import FaultSimulator, SimulatedFaultError

        simulator = FaultSimulator(fail_after_n_saves=2)
        buffer = DataBuffer(
            base_output_dir=tmp_path / "checkpoints",
            base_model_path="test-model",
            _fault_simulator=simulator,
        )
        buffer.init_pipeline(mock_strategies)

        # First 2 saves should succeed (init_pipeline already did one save)
        buffer.save_state()  # Save #2

        # Third save should fail
        with pytest.raises(SimulatedFaultError):
            buffer.save_state()  # Save #3

    def test_fault_simulator_slow_io(self, tmp_path, mock_strategies):
        """
        Given: DataBuffer with FaultSimulator(slow_io_delay_ms=100)
        When: save_state is called
        Then: Operation takes at least 100ms
        """
        import time

        from src.training.managers.data_buffer import FaultSimulator

        simulator = FaultSimulator(slow_io_delay_ms=100)
        buffer = DataBuffer(
            base_output_dir=tmp_path / "checkpoints",
            base_model_path="test-model",
            _fault_simulator=simulator,
        )
        buffer.init_pipeline(mock_strategies)

        start = time.perf_counter()
        buffer.save_state()
        elapsed_ms = (time.perf_counter() - start) * 1000

        # Should take at least 100ms due to simulated delay
        assert elapsed_ms >= 100

    def test_simulation_is_stored_in_buffer(self, tmp_path):
        """
        Given: DataBuffer with FaultSimulator enabled
        When: Buffer is created
        Then: _fault_simulator is properly stored and accessible
        """
        from src.training.managers.data_buffer import FaultSimulator

        simulator = FaultSimulator(
            fail_on_save=True,
            slow_io_delay_ms=100,
            missing_checkpoint=True,
        )
        buffer = DataBuffer(
            base_output_dir=tmp_path / "checkpoints",
            base_model_path="test-model",
            _fault_simulator=simulator,
        )

        # Verify simulator is stored
        assert buffer._fault_simulator is simulator
        assert buffer._fault_simulator.fail_on_save is True
        assert buffer._fault_simulator.slow_io_delay_ms == 100
        assert buffer._fault_simulator.missing_checkpoint is True
