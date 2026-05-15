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

from ryotenkai_shared.config import PhaseHyperparametersConfig, StrategyPhaseConfig
from ryotenkai_shared.errors import TrainingFailedError, TrainingOOMError

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
        from ryotenkai_pod.trainer.orchestrator.chain_runner import ChainRunner

        # Mock PhaseExecutor
        mock_executor = MagicMock()
        mock_model = MagicMock()
        mock_executor.execute.return_value = mock_model

        # Mock DataBuffer
        mock_buffer = MagicMock()
        mock_buffer.run_id = "test_run_001"

        runner = ChainRunner(mock_executor)

        strategies = [make_strategy("sft")]
        initial_model = MagicMock()

        out = runner.run(strategies, initial_model, mock_buffer)

        assert out is mock_model

        # Verify executor was called once
        mock_executor.execute.assert_called_once()
        call_kwargs = mock_executor.execute.call_args[1]
        assert call_kwargs["phase_idx"] == 0
        assert call_kwargs["phase"].strategy_type == "sft"

    def test_single_phase_failure_raises(self):
        """Single phase failure raises a typed exception (post-Batch-14)."""
        from ryotenkai_pod.trainer.orchestrator.chain_runner import ChainRunner

        mock_executor = MagicMock()
        mock_executor.execute.side_effect = TrainingOOMError(detail="Training failed: CUDA OOM")

        mock_buffer = MagicMock()
        mock_buffer.run_id = "test_run_002"

        runner = ChainRunner(mock_executor)

        strategies = [make_strategy("sft")]
        initial_model = MagicMock()

        with pytest.raises(TrainingOOMError) as exc_info:
            runner.run(strategies, initial_model, mock_buffer)
        assert "OOM" in (exc_info.value.detail or "")


# =============================================================================
# TEST CLASS: Multi-Phase Execution
# =============================================================================


class TestMultiPhaseExecution:
    """Tests for multi-phase chain execution."""

    def test_two_phase_execution(self):
        """Two phases should execute sequentially."""
        from ryotenkai_pod.trainer.orchestrator.chain_runner import ChainRunner

        mock_executor = MagicMock()

        model_after_phase_0 = MagicMock(name="model_sft")
        model_after_phase_1 = MagicMock(name="model_dpo")

        # Return different models for each phase
        mock_executor.execute.side_effect = [
            model_after_phase_0,
            model_after_phase_1,
        ]

        mock_buffer = MagicMock()
        mock_buffer.run_id = "test_run_003"
        # Buffer state ladder: phase 0 not skipped, phase 1 not skipped.
        from ryotenkai_pod.trainer.managers.data_buffer import PhaseStatus
        mock_buffer.state.phases = [
            MagicMock(status=PhaseStatus.COMPLETED),
            MagicMock(status=PhaseStatus.COMPLETED),
        ]

        runner = ChainRunner(mock_executor)

        strategies = [make_strategy("sft"), make_strategy("dpo")]
        initial_model = MagicMock(name="base_model")

        out = runner.run(strategies, initial_model, mock_buffer)

        assert out is model_after_phase_1

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
        from ryotenkai_pod.trainer.orchestrator.chain_runner import ChainRunner

        mock_executor = MagicMock()

        models = [MagicMock(name=f"model_{i}") for i in range(3)]
        mock_executor.execute.side_effect = list(models)

        mock_buffer = MagicMock()
        mock_buffer.run_id = "test_run_004"
        from ryotenkai_pod.trainer.managers.data_buffer import PhaseStatus
        mock_buffer.state.phases = [MagicMock(status=PhaseStatus.COMPLETED) for _ in range(3)]

        runner = ChainRunner(mock_executor)

        strategies = [
            make_strategy("cpt"),
            make_strategy("sft"),
            make_strategy("dpo"),
        ]
        initial_model = MagicMock(name="base")

        out = runner.run(strategies, initial_model, mock_buffer)

        assert out is models[2]  # Final model
        assert mock_executor.execute.call_count == 3

    def test_phase_failure_stops_chain(self):
        """Phase failure should stop chain execution (raises)."""
        from ryotenkai_pod.trainer.orchestrator.chain_runner import ChainRunner

        mock_executor = MagicMock()

        # Phase 0 succeeds, phase 1 fails
        mock_executor.execute.side_effect = [
            MagicMock(name="m0"),
            TrainingFailedError(detail="Phase 1 failed: Gradient explosion"),
        ]

        mock_buffer = MagicMock()
        mock_buffer.run_id = "test_run_005"
        from ryotenkai_pod.trainer.managers.data_buffer import PhaseStatus
        mock_buffer.state.phases = [
            MagicMock(status=PhaseStatus.COMPLETED),
            MagicMock(status=PhaseStatus.COMPLETED),
            MagicMock(status=PhaseStatus.COMPLETED),
        ]

        runner = ChainRunner(mock_executor)

        strategies = [
            make_strategy("sft"),
            make_strategy("cot"),
            make_strategy("dpo"),  # Should never execute
        ]

        with pytest.raises(TrainingFailedError) as exc:
            runner.run(strategies, MagicMock(), mock_buffer)
        assert "Phase 1" in (exc.value.detail or "") or "Gradient" in (exc.value.detail or "")

        # Phase 2 should NOT be executed
        assert mock_executor.execute.call_count == 2


# =============================================================================
# TEST CLASS: Resume from Phase
# =============================================================================


class TestResumeFromPhase:
    """Tests for resuming from a specific phase."""

    def _buf_with_phases(self, n: int):
        from ryotenkai_pod.trainer.managers.data_buffer import PhaseStatus
        mock_buffer = MagicMock()
        mock_buffer.state.phases = [MagicMock(status=PhaseStatus.COMPLETED) for _ in range(n)]
        return mock_buffer

    def test_resume_from_phase_1(self):
        """start_phase=1 should skip phase 0."""
        from ryotenkai_pod.trainer.orchestrator.chain_runner import ChainRunner

        mock_executor = MagicMock()
        mock_executor.execute.return_value = MagicMock()

        mock_buffer = self._buf_with_phases(2)
        mock_buffer.run_id = "test_run_006"

        runner = ChainRunner(mock_executor)

        strategies = [
            make_strategy("sft"),  # Should be skipped
            make_strategy("dpo"),  # Should execute
        ]

        # Resume from phase 1 (checkpoint model)
        checkpoint_model = MagicMock(name="checkpoint_model")

        runner.run(strategies, checkpoint_model, mock_buffer, start_phase=1)

        # Only phase 1 should be executed
        assert mock_executor.execute.call_count == 1

        call_kwargs = mock_executor.execute.call_args[1]
        assert call_kwargs["phase_idx"] == 1
        assert call_kwargs["phase"].strategy_type == "dpo"

    def test_resume_from_phase_2_of_4(self):
        """Resume from phase 2 should execute phases 2 and 3."""
        from ryotenkai_pod.trainer.orchestrator.chain_runner import ChainRunner

        mock_executor = MagicMock()
        mock_executor.execute.return_value = MagicMock()

        mock_buffer = self._buf_with_phases(4)
        mock_buffer.run_id = "test_run_007"

        runner = ChainRunner(mock_executor)

        strategies = [
            make_strategy("cpt"),  # 0 - skip
            make_strategy("sft"),  # 1 - skip
            make_strategy("cot"),  # 2 - execute
            make_strategy("dpo"),  # 3 - execute
        ]

        runner.run(strategies, MagicMock(), mock_buffer, start_phase=2)

        assert mock_executor.execute.call_count == 2

        # Verify phases 2 and 3 were executed
        calls = mock_executor.execute.call_args_list
        assert calls[0][1]["phase_idx"] == 2
        assert calls[1][1]["phase_idx"] == 3

    def test_resume_from_last_phase(self):
        """Resume from last phase should execute only that phase."""
        from ryotenkai_pod.trainer.orchestrator.chain_runner import ChainRunner

        mock_executor = MagicMock()
        mock_executor.execute.return_value = MagicMock()

        mock_buffer = self._buf_with_phases(3)
        mock_buffer.run_id = "test_run_008"

        runner = ChainRunner(mock_executor)

        strategies = [
            make_strategy("sft"),
            make_strategy("cot"),
            make_strategy("dpo"),  # Only this
        ]

        runner.run(strategies, MagicMock(), mock_buffer, start_phase=2)

        assert mock_executor.execute.call_count == 1


# =============================================================================
# TEST CLASS: Edge Cases
# =============================================================================


class TestEdgeCases:
    """Edge cases for ChainRunner."""

    def test_empty_strategies_no_execution(self):
        """Empty strategies should not call executor."""
        from ryotenkai_pod.trainer.orchestrator.chain_runner import ChainRunner

        mock_executor = MagicMock()
        mock_buffer = MagicMock()
        mock_buffer.run_id = "test_run_009"

        runner = ChainRunner(mock_executor)

        runner.run([], MagicMock(), mock_buffer)

        # Empty strategies, should return immediately
        mock_executor.execute.assert_not_called()

    def test_start_phase_beyond_strategies(self):
        """start_phase >= len(strategies) should not execute anything."""
        from ryotenkai_pod.trainer.orchestrator.chain_runner import ChainRunner

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
        from ryotenkai_pod.trainer.orchestrator.chain_runner import ChainRunner

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
        from ryotenkai_pod.trainer.orchestrator.chain_runner import ChainRunner

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
        from ryotenkai_pod.trainer.orchestrator.chain_runner import ChainRunner

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
        from ryotenkai_pod.trainer.orchestrator.chain_runner import ChainRunner

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
        from ryotenkai_pod.trainer.orchestrator.chain_runner import ChainRunner

        mock_executor = MagicMock()
        runner = ChainRunner(mock_executor)

        phase = make_strategy("sft", epochs=5, learning_rate=1e-4)

        # Just verify it doesn't raise
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
        from ryotenkai_pod.trainer.orchestrator.chain_runner import ChainRunner

        mock_executor = MagicMock()

        # Create distinct models for tracking
        model_0 = MagicMock(name="model_after_sft")
        model_1 = MagicMock(name="model_after_cot")
        model_2 = MagicMock(name="model_after_dpo")

        mock_executor.execute.side_effect = [model_0, model_1, model_2]

        mock_buffer = MagicMock()
        mock_buffer.run_id = "test_run"
        from ryotenkai_pod.trainer.managers.data_buffer import PhaseStatus
        mock_buffer.state.phases = [MagicMock(status=PhaseStatus.COMPLETED) for _ in range(3)]

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


class TestChainRunnerRaiseContract:
    """7-class coverage for the new raise-based ChainRunner.run."""

    def _buf(self, n: int):
        from ryotenkai_pod.trainer.managers.data_buffer import PhaseStatus
        buf = MagicMock()
        buf.run_id = "rid"
        buf.state.phases = [MagicMock(status=PhaseStatus.COMPLETED) for _ in range(n)]
        return buf

    def test_positive_returns_model_not_result(self):
        from ryotenkai_pod.trainer.orchestrator.chain_runner import ChainRunner

        class _Model:  # plain object — no MagicMock auto-attrs
            pass

        executor = MagicMock()
        final = _Model()
        executor.execute.return_value = final
        out = ChainRunner(executor).run([make_strategy("sft")], MagicMock(), self._buf(1))
        assert out is final
        assert not hasattr(out, "is_failure")
        assert not hasattr(out, "unwrap_err")

    def test_negative_first_phase_raises_propagates(self):
        from ryotenkai_pod.trainer.orchestrator.chain_runner import ChainRunner
        executor = MagicMock()
        executor.execute.side_effect = TrainingFailedError(detail="x")
        with pytest.raises(TrainingFailedError):
            ChainRunner(executor).run([make_strategy("sft")], MagicMock(), self._buf(1))

    def test_boundary_empty_strategies_returns_initial_model(self):
        """Edge case: empty list returns the initial model untouched."""
        from ryotenkai_pod.trainer.orchestrator.chain_runner import ChainRunner
        executor = MagicMock()
        initial = MagicMock()
        out = ChainRunner(executor).run([], initial, self._buf(0))
        assert out is initial
        executor.execute.assert_not_called()

    def test_invariant_mlflow_failed_tag_logged_before_raise(self):
        """The MLflow status tag is set before propagating, so external
        observers see "failed" rather than just the missing run."""
        from ryotenkai_pod.trainer.orchestrator.chain_runner import ChainRunner
        executor = MagicMock()
        executor.execute.side_effect = TrainingFailedError(detail="x")
        mlflow = MagicMock()
        mlflow.is_active = True
        with pytest.raises(TrainingFailedError):
            ChainRunner(executor, mlflow_manager=mlflow).run(
                [make_strategy("sft")], MagicMock(), self._buf(1),
            )
        # Failed status should be tagged via set_tags.
        set_tags_calls = [c for c in mlflow.set_tags.call_args_list]
        assert any(
            "status" in c.args[0] and c.args[0]["status"] == "failed"
            for c in set_tags_calls
        )

    def test_dependency_error_oom_propagates(self):
        from ryotenkai_pod.trainer.orchestrator.chain_runner import ChainRunner
        executor = MagicMock()
        executor.execute.side_effect = TrainingOOMError(detail="oom")
        with pytest.raises(TrainingOOMError):
            ChainRunner(executor).run([make_strategy("sft")], MagicMock(), self._buf(1))

    def test_regression_no_result_imports(self):
        """Pin: the chain_runner module must not depend on the legacy
        ``Result`` machinery after Batch 14."""
        import ryotenkai_pod.trainer.orchestrator.chain_runner as cr_mod

        src = cr_mod.__file__
        with open(src, "r", encoding="utf-8") as f:
            content = f.read()
        assert "from ryotenkai_shared.utils.result" not in content
        assert "is_failure" not in content
        assert "unwrap_err" not in content

    def test_combinatorial_success_then_failure(self):
        from ryotenkai_pod.trainer.orchestrator.chain_runner import ChainRunner
        executor = MagicMock()
        ok_model = MagicMock()
        executor.execute.side_effect = [ok_model, TrainingFailedError(detail="phase 1 down")]
        with pytest.raises(TrainingFailedError):
            ChainRunner(executor).run(
                [make_strategy("sft"), make_strategy("dpo")], MagicMock(), self._buf(2),
            )
        # Second phase received the model from phase 0.
        assert executor.execute.call_args_list[1].kwargs["model"] is ok_model


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
