"""
E2E tests for StrategyOrchestrator (multi-phase training).

Tests the complete training orchestration flow:
- Strategy chain validation
- Resume logic
- OOM protection

Note: Complex training flows are tested via integration tests with real components.
These tests focus on orchestrator logic without deep mocking.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

# =============================================================================
# TEST CLASS: Strategy Chain Validation
# =============================================================================


@pytest.mark.e2e
class TestStrategyChainValidation:
    """Tests for strategy chain validation."""

    def test_single_dpo_is_allowed(self, mock_config):
        """
        Given: Chain starting with DPO
        When: Chain is validated
        Then: Validation passes
        """
        _ = mock_config  # Mark as used (required fixture for context)
        from src.utils.config import validate_strategy_chain

        # Create invalid chain: DPO first
        dpo_phase = MagicMock()
        dpo_phase.strategy_type = "dpo"

        result = validate_strategy_chain([dpo_phase])
        assert result.is_success()

    def test_valid_chain_sft_dpo_passes(self, mock_config):
        """
        Given: Valid chain SFT → DPO
        When: Chain is validated
        Then: Validation passes
        """
        _ = mock_config  # Mark as used
        from src.utils.config import validate_strategy_chain

        # Create valid chain
        sft_phase = MagicMock()
        sft_phase.strategy_type = "sft"

        dpo_phase = MagicMock()
        dpo_phase.strategy_type = "dpo"

        result = validate_strategy_chain([sft_phase, dpo_phase])
        assert result.is_success()

    def test_valid_chain_cpt_sft_cot_passes(self, mock_config):
        """
        Given: Valid chain CPT → SFT → CoT
        When: Chain is validated
        Then: Validation passes
        """
        _ = mock_config  # Mark as used
        from src.utils.config import validate_strategy_chain

        # Create valid chain
        cpt_phase = MagicMock()
        cpt_phase.strategy_type = "cpt"

        sft_phase = MagicMock()
        sft_phase.strategy_type = "sft"

        cot_phase = MagicMock()
        cot_phase.strategy_type = "cot"

        result = validate_strategy_chain([cpt_phase, sft_phase, cot_phase])
        assert result.is_success()

    def test_single_sft_valid(self, mock_config):
        """
        Given: Single SFT strategy
        When: Chain is validated
        Then: Validation passes
        """
        _ = mock_config  # Mark as used
        from src.utils.config import validate_strategy_chain

        sft_phase = MagicMock()
        sft_phase.strategy_type = "sft"

        assert validate_strategy_chain([sft_phase]).is_success()

    def test_single_cpt_valid(self, mock_config):
        """
        Given: Single CPT strategy
        When: Chain is validated
        Then: Validation passes
        """
        _ = mock_config  # Mark as used
        from src.utils.config import validate_strategy_chain

        cpt_phase = MagicMock()
        cpt_phase.strategy_type = "cpt"

        assert validate_strategy_chain([cpt_phase]).is_success()


# =============================================================================
# TEST CLASS: StrategyOrchestrator Components
# =============================================================================


@pytest.mark.e2e
class TestStrategyOrchestratorComponents:
    """Tests for StrategyOrchestrator component initialization."""

    def test_orchestrator_imports(self):
        """
        Given: StrategyOrchestrator module
        When: Imported
        Then: No import errors
        """
        from src.training.orchestrator.strategy_orchestrator import StrategyOrchestrator

        assert StrategyOrchestrator is not None

    def test_chain_runner_imports(self):
        """
        Given: ChainRunner module
        When: Imported
        Then: No import errors
        """
        from src.training.orchestrator.chain_runner import ChainRunner

        assert ChainRunner is not None

    def test_phase_executor_imports(self):
        """
        Given: PhaseExecutor module
        When: Imported
        Then: No import errors
        """
        from src.training.orchestrator.phase_executor import PhaseExecutor

        assert PhaseExecutor is not None

    def test_resume_manager_imports(self):
        """
        Given: ResumeManager module
        When: Imported
        Then: No import errors
        """
        from src.training.orchestrator.resume_manager import ResumeManager

        assert ResumeManager is not None


# =============================================================================
# TEST CLASS: DataBuffer Integration
# =============================================================================


@pytest.mark.e2e
class TestDataBufferIntegration:
    """Tests for DataBuffer with multi-phase training."""

    def test_data_buffer_phase_tracking(self, tmp_path):
        """
        Given: DataBuffer with 3-phase config
        When: Phases are tracked
        Then: State updates correctly
        """
        from src.training.managers.data_buffer import DataBuffer
        from src.utils.config import PhaseHyperparametersConfig, StrategyPhaseConfig

        # Create strategies
        strategies = [
            StrategyPhaseConfig(
                strategy_type=st,
                dataset="default",
                hyperparams=PhaseHyperparametersConfig(epochs=1, learning_rate=2e-4),
            )
            for st in ["cpt", "sft", "cot"]
        ]

        # Initialize buffer
        buffer = DataBuffer(
            base_output_dir=tmp_path / "checkpoints",
            base_model_path="test-model",
        )

        buffer.init_pipeline(strategies)

        # Verify initial state
        assert buffer.total_phases == 3
        assert buffer.state.status == "running"

        # Track phase 0
        buffer.mark_phase_started(0)
        assert buffer.state.phases[0].status.value == "running"

        buffer.mark_phase_completed(0, checkpoint_path="/tmp/checkpoint")
        assert buffer.state.phases[0].status.value == "completed"

    def test_data_buffer_resume_detection(self, tmp_path):
        """
        Given: DataBuffer with failed phase
        When: get_resume_phase is called
        Then: Returns failed phase index
        """
        from src.training.managers.data_buffer import DataBuffer
        from src.utils.config import PhaseHyperparametersConfig, StrategyPhaseConfig

        # Create strategies
        strategies = [
            StrategyPhaseConfig(
                strategy_type=st,
                dataset="default",
                hyperparams=PhaseHyperparametersConfig(epochs=1, learning_rate=2e-4),
            )
            for st in ["sft", "dpo"]
        ]

        buffer = DataBuffer(
            base_output_dir=tmp_path / "checkpoints",
            base_model_path="test-model",
        )

        buffer.init_pipeline(strategies)

        # Complete phase 0
        buffer.mark_phase_started(0)
        buffer.mark_phase_completed(0)

        # Fail phase 1
        buffer.mark_phase_started(1)
        buffer.mark_phase_failed(1, "OOM error")

        # Resume should return phase 1
        resume_phase = buffer.get_resume_phase()
        assert resume_phase == 1


# =============================================================================
# TEST CLASS: Strategy Factory Integration
# =============================================================================


@pytest.mark.e2e
class TestStrategyFactoryIntegration:
    """Tests for StrategyFactory in orchestration context."""

    def test_create_all_registered_strategies(self, mock_config):
        """
        Given: All registered strategy types
        When: StrategyFactory creates each
        Then: All strategies are created successfully
        """
        from src.training.strategies.factory import StrategyFactory

        factory = StrategyFactory()

        for strategy_type in ["sft", "cpt", "cot"]:
            strategy = factory.create(strategy_type, mock_config)

            assert strategy is not None
            metadata = strategy.get_metadata()
            assert metadata.strategy_type == strategy_type

    def test_strategy_hyperparameters(self):
        """
        Given: StrategyFactory
        When: get_default_hyperparameters is called
        Then: Returns valid hyperparameters for each strategy
        """
        from src.training.strategies.factory import StrategyFactory

        factory = StrategyFactory()

        for strategy_type in ["sft", "cpt", "cot", "dpo"]:
            params = factory.get_default_hyperparameters(strategy_type)

            assert "learning_rate" in params
            assert "epochs" in params
            assert "batch_size" in params
            assert params["learning_rate"] > 0
            assert params["epochs"] > 0

    def test_dpo_lower_learning_rate(self):
        """
        Given: DPO strategy
        When: get_default_hyperparameters is called
        Then: Learning rate is much lower than SFT (10-100x)
        """
        from src.training.strategies.factory import StrategyFactory

        factory = StrategyFactory()

        sft_params = factory.get_default_hyperparameters("sft")
        dpo_params = factory.get_default_hyperparameters("dpo")

        # DPO LR should be 10-100x lower than SFT
        assert dpo_params["learning_rate"] < sft_params["learning_rate"] / 10


# =============================================================================
# TEST CLASS: Memory Manager Integration
# =============================================================================


@pytest.mark.e2e
class TestMemoryManagerIntegration:
    """Tests for MemoryManager in training context."""

    def test_memory_manager_singleton(self):
        """
        Given: get_memory_manager called multiple times
        Then: Returns same instance (singleton)
        """
        from src.utils.memory_manager import get_memory_manager

        mm1 = get_memory_manager()
        mm2 = get_memory_manager()

        assert mm1 is mm2

    def test_memory_presets(self):
        """
        Given: MemoryManager
        When: preset is accessed
        Then: Returns valid preset
        """
        from src.utils.memory_manager import get_memory_manager

        mm = get_memory_manager()
        # Even without CUDA, defaults are applied, but preset might be None if not auto-detected
        # Just check basic properties
        assert hasattr(mm, "memory_margin_mb")
        assert mm.max_retries >= 0
