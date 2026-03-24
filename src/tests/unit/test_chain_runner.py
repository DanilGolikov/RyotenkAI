"""
ChainRunner Tests - Strategy Chain Execution.

Tests for ChainRunner which iterates over phases and coordinates execution.

Focus on:
- Single-phase execution
- Multi-phase execution (model passed between phases)
- Resume from specific phase
- Phase failure handling
- Edge cases (empty strategies)
"""

from unittest.mock import MagicMock

import pytest

from src.utils.config import PhaseHyperparametersConfig, StrategyPhaseConfig
from src.utils.result import Err, Ok

# =============================================================================
# HELPER: Create test strategies
# =============================================================================


def make_strategy(strategy_type: str, **kwargs) -> StrategyPhaseConfig:
    """Create a StrategyPhaseConfig for testing."""
    hp_kwargs: dict[str, object] = {}
    if "epochs" in kwargs and kwargs["epochs"] is not None:
        hp_kwargs["epochs"] = kwargs["epochs"]
    if "learning_rate" in kwargs and kwargs["learning_rate"] is not None:
        hp_kwargs["learning_rate"] = kwargs["learning_rate"]
    if "beta" in kwargs and kwargs["beta"] is not None:
        hp_kwargs["beta"] = kwargs["beta"]

    return StrategyPhaseConfig(
        strategy_type=strategy_type,
        dataset=kwargs.get("dataset", "default"),
        hyperparams=PhaseHyperparametersConfig(**hp_kwargs),  # type: ignore[arg-type]
    )


# =============================================================================
# TEST CLASS: Single Phase Execution
# =============================================================================


class TestSinglePhaseExecution:
    """Tests for single-phase chain execution."""

    def test_single_phase_success(self):
        """Single phase should execute and return model."""
        from src.training.orchestrator.chain_runner import ChainRunner

        # Mock PhaseExecutor
        mock_executor = MagicMock()
        mock_model = MagicMock()
        mock_executor.execute.return_value = Ok(mock_model)

        # Mock DataBuffer
        mock_buffer = MagicMock()
        mock_buffer.run_id = "test_run_001"

        runner = ChainRunner(mock_executor)

        strategies = [make_strategy("sft")]
        initial_model = MagicMock()

        result = runner.run(strategies, initial_model, mock_buffer)

        assert result.is_success()
        assert result.unwrap() == mock_model

        # Verify executor was called once
        mock_executor.execute.assert_called_once()
        call_kwargs = mock_executor.execute.call_args[1]
        assert call_kwargs["phase_idx"] == 0
        assert call_kwargs["phase"].strategy_type == "sft"

    def test_single_phase_failure(self):
        """Single phase failure should propagate error."""
        from src.training.orchestrator.chain_runner import ChainRunner

        mock_executor = MagicMock()
        mock_executor.execute.return_value = Err("Training failed: CUDA OOM")

        mock_buffer = MagicMock()
        mock_buffer.run_id = "test_run_002"

        runner = ChainRunner(mock_executor)

        strategies = [make_strategy("sft")]
        initial_model = MagicMock()

        result = runner.run(strategies, initial_model, mock_buffer)

        assert result.is_failure()
        assert "OOM" in str(result.error)


# =============================================================================
# TEST CLASS: Multi-Phase Execution
# =============================================================================


class TestMultiPhaseExecution:
    """Tests for multi-phase chain execution."""

    def test_two_phase_execution(self):
        """Two phases should execute sequentially."""
        from src.training.orchestrator.chain_runner import ChainRunner

        mock_executor = MagicMock()

        model_after_phase_0 = MagicMock(name="model_sft")
        model_after_phase_1 = MagicMock(name="model_dpo")

        # Return different models for each phase
        mock_executor.execute.side_effect = [
            Ok(model_after_phase_0),
            Ok(model_after_phase_1),
        ]

        mock_buffer = MagicMock()
        mock_buffer.run_id = "test_run_003"

        runner = ChainRunner(mock_executor)

        strategies = [make_strategy("sft"), make_strategy("dpo")]
        initial_model = MagicMock(name="base_model")

        result = runner.run(strategies, initial_model, mock_buffer)

        assert result.is_success()
        assert result.unwrap() == model_after_phase_1

        # Verify both phases were executed
        assert mock_executor.execute.call_count == 2

        # Verify phase 0 used initial model
        call_0 = mock_executor.execute.call_args_list[0][1]
        assert call_0["phase_idx"] == 0
        assert call_0["model"] == initial_model

        # Verify phase 1 used model from phase 0
        call_1 = mock_executor.execute.call_args_list[1][1]
        assert call_1["phase_idx"] == 1
        assert call_1["model"] == model_after_phase_0

    def test_three_phase_execution(self):
        """Three phases should execute sequentially, passing model."""
        from src.training.orchestrator.chain_runner import ChainRunner

        mock_executor = MagicMock()

        models = [MagicMock(name=f"model_{i}") for i in range(3)]
        mock_executor.execute.side_effect = [Ok(m) for m in models]

        mock_buffer = MagicMock()
        mock_buffer.run_id = "test_run_004"

        runner = ChainRunner(mock_executor)

        strategies = [
            make_strategy("cpt"),
            make_strategy("sft"),
            make_strategy("dpo"),
        ]
        initial_model = MagicMock(name="base")

        result = runner.run(strategies, initial_model, mock_buffer)

        assert result.is_success()
        assert result.unwrap() == models[2]  # Final model
        assert mock_executor.execute.call_count == 3

    def test_phase_failure_stops_chain(self):
        """Phase failure should stop chain execution."""
        from src.training.orchestrator.chain_runner import ChainRunner

        mock_executor = MagicMock()

        # Phase 0 succeeds, phase 1 fails
        mock_executor.execute.side_effect = [
            Ok(MagicMock()),
            Err("Phase 1 failed: Gradient explosion"),
        ]

        mock_buffer = MagicMock()
        mock_buffer.run_id = "test_run_005"

        runner = ChainRunner(mock_executor)

        strategies = [
            make_strategy("sft"),
            make_strategy("cot"),
            make_strategy("dpo"),  # Should never execute
        ]

        result = runner.run(strategies, MagicMock(), mock_buffer)

        assert result.is_failure()
        assert "Phase 1" in str(result.error) or "Gradient" in str(result.error)

        # Phase 2 should NOT be executed
        assert mock_executor.execute.call_count == 2


# =============================================================================
# TEST CLASS: Resume from Phase
# =============================================================================


class TestResumeFromPhase:
    """Tests for resuming from a specific phase."""

    def test_resume_from_phase_1(self):
        """start_phase=1 should skip phase 0."""
        from src.training.orchestrator.chain_runner import ChainRunner

        mock_executor = MagicMock()
        mock_executor.execute.return_value = Ok(MagicMock())

        mock_buffer = MagicMock()
        mock_buffer.run_id = "test_run_006"

        runner = ChainRunner(mock_executor)

        strategies = [
            make_strategy("sft"),  # Should be skipped
            make_strategy("dpo"),  # Should execute
        ]

        # Resume from phase 1 (checkpoint model)
        checkpoint_model = MagicMock(name="checkpoint_model")

        result = runner.run(strategies, checkpoint_model, mock_buffer, start_phase=1)

        assert result.is_success()

        # Only phase 1 should be executed
        assert mock_executor.execute.call_count == 1

        call_kwargs = mock_executor.execute.call_args[1]
        assert call_kwargs["phase_idx"] == 1
        assert call_kwargs["phase"].strategy_type == "dpo"

    def test_resume_from_phase_2_of_4(self):
        """Resume from phase 2 should execute phases 2 and 3."""
        from src.training.orchestrator.chain_runner import ChainRunner

        mock_executor = MagicMock()
        mock_executor.execute.return_value = Ok(MagicMock())

        mock_buffer = MagicMock()
        mock_buffer.run_id = "test_run_007"

        runner = ChainRunner(mock_executor)

        strategies = [
            make_strategy("cpt"),  # 0 - skip
            make_strategy("sft"),  # 1 - skip
            make_strategy("cot"),  # 2 - execute
            make_strategy("dpo"),  # 3 - execute
        ]

        result = runner.run(strategies, MagicMock(), mock_buffer, start_phase=2)

        assert result.is_success()
        assert mock_executor.execute.call_count == 2

        # Verify phases 2 and 3 were executed
        calls = mock_executor.execute.call_args_list
        assert calls[0][1]["phase_idx"] == 2
        assert calls[1][1]["phase_idx"] == 3

    def test_resume_from_last_phase(self):
        """Resume from last phase should execute only that phase."""
        from src.training.orchestrator.chain_runner import ChainRunner

        mock_executor = MagicMock()
        mock_executor.execute.return_value = Ok(MagicMock())

        mock_buffer = MagicMock()
        mock_buffer.run_id = "test_run_008"

        runner = ChainRunner(mock_executor)

        strategies = [
            make_strategy("sft"),
            make_strategy("cot"),
            make_strategy("dpo"),  # Only this
        ]

        result = runner.run(strategies, MagicMock(), mock_buffer, start_phase=2)

        assert result.is_success()
        assert mock_executor.execute.call_count == 1


# =============================================================================
# TEST CLASS: Edge Cases
# =============================================================================


class TestEdgeCases:
    """Edge cases for ChainRunner."""

    def test_empty_strategies_no_execution(self):
        """Empty strategies should not call executor."""
        from src.training.orchestrator.chain_runner import ChainRunner

        mock_executor = MagicMock()
        mock_buffer = MagicMock()
        mock_buffer.run_id = "test_run_009"

        runner = ChainRunner(mock_executor)

        runner.run([], MagicMock(), mock_buffer)

        # Empty strategies, should return immediately
        mock_executor.execute.assert_not_called()
        # Result depends on implementation - might be Ok(initial_model) or need validation

    def test_start_phase_beyond_strategies(self):
        """start_phase >= len(strategies) should not execute anything."""
        from src.training.orchestrator.chain_runner import ChainRunner

        mock_executor = MagicMock()
        mock_buffer = MagicMock()
        mock_buffer.run_id = "test_run_010"

        runner = ChainRunner(mock_executor)

        strategies = [make_strategy("sft")]

        runner.run(strategies, MagicMock(), mock_buffer, start_phase=5)

        # No phases to execute
        mock_executor.execute.assert_not_called()


# =============================================================================
# TEST CLASS: get_remaining_phases
# =============================================================================


class TestGetRemainingPhases:
    """Tests for get_remaining_phases utility method."""

    def test_remaining_from_start(self):
        """get_remaining_phases(0) should return all strategies."""
        from src.training.orchestrator.chain_runner import ChainRunner

        mock_executor = MagicMock()
        runner = ChainRunner(mock_executor)

        strategies = [
            make_strategy("sft"),
            make_strategy("cot"),
            make_strategy("dpo"),
        ]

        remaining = runner.get_remaining_phases(strategies, 0)

        assert len(remaining) == 3
        assert remaining[0].strategy_type == "sft"

    def test_remaining_from_middle(self):
        """get_remaining_phases(1) should return strategies[1:]."""
        from src.training.orchestrator.chain_runner import ChainRunner

        mock_executor = MagicMock()
        runner = ChainRunner(mock_executor)

        strategies = [
            make_strategy("sft"),
            make_strategy("cot"),
            make_strategy("dpo"),
        ]

        remaining = runner.get_remaining_phases(strategies, 1)

        assert len(remaining) == 2
        assert remaining[0].strategy_type == "cot"
        assert remaining[1].strategy_type == "dpo"

    def test_remaining_from_last(self):
        """get_remaining_phases(len-1) should return only last strategy."""
        from src.training.orchestrator.chain_runner import ChainRunner

        mock_executor = MagicMock()
        runner = ChainRunner(mock_executor)

        strategies = [
            make_strategy("sft"),
            make_strategy("dpo"),
        ]

        remaining = runner.get_remaining_phases(strategies, 1)

        assert len(remaining) == 1
        assert remaining[0].strategy_type == "dpo"

    def test_remaining_beyond_end(self):
        """get_remaining_phases(>len) should return empty list."""
        from src.training.orchestrator.chain_runner import ChainRunner

        mock_executor = MagicMock()
        runner = ChainRunner(mock_executor)

        strategies = [make_strategy("sft")]

        remaining = runner.get_remaining_phases(strategies, 5)

        assert len(remaining) == 0


# =============================================================================
# TEST CLASS: Phase Header Logging
# =============================================================================


class TestPhaseHeaderLogging:
    """Tests for phase header logging."""

    def test_log_phase_header_format(self):
        """_log_phase_header should log phase info (test via mock)."""
        from src.training.orchestrator.chain_runner import ChainRunner

        mock_executor = MagicMock()
        runner = ChainRunner(mock_executor)

        phase = make_strategy("sft", epochs=5, learning_rate=1e-4)

        # Just verify it doesn't raise
        # Actual logging is hard to test with our logger setup
        runner._log_phase_header(0, 3, phase)

        # Method exists and runs without error
        assert True


# =============================================================================
# TEST CLASS: Model Propagation
# =============================================================================


class TestModelPropagation:
    """Tests verifying model is correctly passed between phases."""

    def test_model_from_phase_n_goes_to_phase_n_plus_1(self):
        """Model returned by phase N should be input to phase N+1."""
        from src.training.orchestrator.chain_runner import ChainRunner

        mock_executor = MagicMock()

        # Create distinct models for tracking
        model_0 = MagicMock(name="model_after_sft")
        model_1 = MagicMock(name="model_after_cot")
        model_2 = MagicMock(name="model_after_dpo")

        mock_executor.execute.side_effect = [
            Ok(model_0),
            Ok(model_1),
            Ok(model_2),
        ]

        mock_buffer = MagicMock()
        mock_buffer.run_id = "test_run"

        runner = ChainRunner(mock_executor)

        strategies = [
            make_strategy("sft"),
            make_strategy("cot"),
            make_strategy("dpo"),
        ]

        base_model = MagicMock(name="base_model")

        runner.run(strategies, base_model, mock_buffer)

        # Verify model propagation
        calls = mock_executor.execute.call_args_list

        # Phase 0 gets base_model
        assert calls[0][1]["model"] == base_model

        # Phase 1 gets model_0 (output of phase 0)
        assert calls[1][1]["model"] == model_0

        # Phase 2 gets model_1 (output of phase 1)
        assert calls[2][1]["model"] == model_1


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
