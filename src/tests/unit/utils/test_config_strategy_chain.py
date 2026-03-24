"""
Comprehensive tests for strategy chain validation logic.

Tests cover:
- Valid single-phase chains
- Valid multi-phase chains (2-4 phases)
- Invalid start strategies
- Invalid transitions
- Empty chains
- None values in chain (BUG-010 fix)
- Boundary cases
- Integration with TrainingOnlyConfig

Functions/Constants tested:
- VALID_STRATEGY_TRANSITIONS
- VALID_START_STRATEGIES
- validate_strategy_chain()
- TrainingOnlyConfig.validate_chain()
"""

import pytest
from pydantic import ValidationError

from src.utils.config import (
    VALID_START_STRATEGIES,
    VALID_STRATEGY_TRANSITIONS,
    StrategyPhaseConfig,
    TrainingOnlyConfig,
    validate_strategy_chain,
)

# =============================================================================
# TEST HELPERS
# =============================================================================

# Minimal required blocks to construct TrainingOnlyConfig in tests.
# We keep them explicit to match "no magic" config policy.
_MIN_GLOBAL_HYPERPARAMS = {
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 1,
    "learning_rate": 2e-4,
    "warmup_ratio": 0.0,
    "epochs": 1,
}

_MIN_LORA = {
    "r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "bias": "none",
    "target_modules": "all-linear",
    "use_dora": False,
    "use_rslora": False,
    "init_lora_weights": "gaussian",
}


def _mk_phase(strategy_type: str) -> StrategyPhaseConfig:
    """
    Build StrategyPhaseConfig with required strategy-specific params.

    NOTE: Some strategies (e.g. SAPO) require additional hyperparams.
    """
    if strategy_type == "sapo":
        return StrategyPhaseConfig(
            strategy_type=strategy_type,
            hyperparams={
                "max_prompt_length": 1024,
                "max_completion_length": 512,
            },
            params={"reward_plugin": "helixql_compiler_semantic"},
        )
    if strategy_type == "grpo":
        return StrategyPhaseConfig(
            strategy_type=strategy_type,
            hyperparams={
                "max_prompt_length": 1024,
                "max_completion_length": 512,
            },
            params={"reward_plugin": "helixql_compiler_semantic"},
        )
    return StrategyPhaseConfig(strategy_type=strategy_type)


# =============================================================================
# TEST: Constants
# =============================================================================


class TestStrategyTransitionConstants:
    """Test strategy transition constants and rules."""

    def test_valid_transitions_structure(self):
        """VALID_STRATEGY_TRANSITIONS should have expected structure."""
        from collections.abc import Mapping

        assert isinstance(VALID_STRATEGY_TRANSITIONS, Mapping)

        # Check all strategy types are present
        assert "cpt" in VALID_STRATEGY_TRANSITIONS
        assert "sft" in VALID_STRATEGY_TRANSITIONS
        assert "cot" in VALID_STRATEGY_TRANSITIONS
        assert "dpo" in VALID_STRATEGY_TRANSITIONS
        assert "orpo" in VALID_STRATEGY_TRANSITIONS
        assert "grpo" in VALID_STRATEGY_TRANSITIONS
        assert "sapo" in VALID_STRATEGY_TRANSITIONS

    def test_terminal_strategies_have_empty_transitions(self):
        """Terminal strategies (DPO, ORPO, SAPO) should have no transitions."""
        assert VALID_STRATEGY_TRANSITIONS["dpo"] == ()
        assert VALID_STRATEGY_TRANSITIONS["orpo"] == ()
        assert VALID_STRATEGY_TRANSITIONS["grpo"] == ()
        assert VALID_STRATEGY_TRANSITIONS["sapo"] == ()

    def test_cpt_transitions(self):
        """CPT can transition to SFT or CoT."""
        assert set(VALID_STRATEGY_TRANSITIONS["cpt"]) == {"sft", "cot"}

    def test_sft_transitions(self):
        """SFT can transition to CoT, DPO, ORPO, SAPO."""
        assert set(VALID_STRATEGY_TRANSITIONS["sft"]) == {"cot", "dpo", "orpo", "grpo", "sapo"}

    def test_cot_transitions(self):
        """CoT can transition to DPO, ORPO, SAPO."""
        assert set(VALID_STRATEGY_TRANSITIONS["cot"]) == {"dpo", "orpo", "grpo", "sapo"}

    def test_valid_start_strategies(self):
        """VALID_START_STRATEGIES should contain expected strategies."""
        assert isinstance(VALID_START_STRATEGIES, (list, tuple))

        # Must contain these
        assert "cpt" in VALID_START_STRATEGIES
        assert "sft" in VALID_START_STRATEGIES
        assert "orpo" in VALID_START_STRATEGIES
        assert "grpo" in VALID_START_STRATEGIES
        assert "sapo" in VALID_START_STRATEGIES

        # Must NOT contain these (not valid start)
        assert "dpo" not in VALID_START_STRATEGIES
        assert "cot" not in VALID_START_STRATEGIES


# =============================================================================
# TEST: Valid Chains
# =============================================================================


class TestValidSinglePhaseChains:
    """Test valid single-phase strategy chains."""

    @pytest.mark.parametrize("strategy_type", ["cpt", "sft", "orpo", "grpo", "sapo"])
    def test_single_phase_valid(self, strategy_type):
        """Single valid start strategy should pass."""
        strategies = [_mk_phase(strategy_type)]

        is_valid, error_msg = validate_strategy_chain(strategies)

        assert is_valid is True
        assert error_msg == ""

    def test_single_phase_dpo_invalid(self):
        """Single DPO (not a start strategy) should fail."""
        strategies = [_mk_phase("dpo")]

        is_valid, error_msg = validate_strategy_chain(strategies)

        assert is_valid is False
        assert "must start with" in error_msg
        assert "dpo" in error_msg

    def test_single_phase_cot_invalid(self):
        """Single CoT (not a start strategy) should fail."""
        strategies = [_mk_phase("cot")]

        is_valid, error_msg = validate_strategy_chain(strategies)

        assert is_valid is False
        assert "must start with" in error_msg


class TestValidTwoPhaseChains:
    """Test valid two-phase strategy chains."""

    @pytest.mark.parametrize(
        "chain",
        [
            ["cpt", "sft"],
            ["cpt", "cot"],
            ["sft", "dpo"],
            ["sft", "orpo"],
            ["sft", "sapo"],
            ["sft", "cot"],
        ],
    )
    def test_two_phase_valid(self, chain):
        """Valid two-phase chains should pass."""
        strategies = [_mk_phase(t) for t in chain]

        is_valid, error_msg = validate_strategy_chain(strategies)

        assert is_valid is True, f"Chain {chain} should be valid but got: {error_msg}"
        assert error_msg == ""


class TestValidThreePhaseChains:
    """Test valid three-phase strategy chains."""

    @pytest.mark.parametrize(
        "chain",
        [
            ["cpt", "sft", "dpo"],
            ["cpt", "sft", "orpo"],
            ["cpt", "sft", "sapo"],
            ["cpt", "cot", "dpo"],
            ["cpt", "cot", "orpo"],
            ["cpt", "cot", "sapo"],
            ["sft", "cot", "dpo"],
            ["sft", "cot", "orpo"],
            ["sft", "cot", "sapo"],
            ["cpt", "sft", "cot"],
        ],
    )
    def test_three_phase_valid(self, chain):
        """Valid three-phase chains should pass."""
        strategies = [_mk_phase(t) for t in chain]

        is_valid, error_msg = validate_strategy_chain(strategies)

        assert is_valid is True, f"Chain {chain} should be valid but got: {error_msg}"
        assert error_msg == ""


class TestValidFourPhaseChains:
    """Test valid four-phase strategy chains (maximum length)."""

    @pytest.mark.parametrize(
        "chain",
        [
            ["cpt", "sft", "cot", "dpo"],
            ["cpt", "sft", "cot", "orpo"],
            ["cpt", "sft", "cot", "sapo"],
        ],
    )
    def test_four_phase_valid(self, chain):
        """Maximum length chains should pass."""
        strategies = [_mk_phase(t) for t in chain]

        is_valid, error_msg = validate_strategy_chain(strategies)

        assert is_valid is True, f"Chain {chain} should be valid but got: {error_msg}"
        assert error_msg == ""


# =============================================================================
# TEST: Invalid Chains
# =============================================================================


class TestInvalidChains:
    """Test invalid strategy chains and error messages."""

    def test_empty_chain(self):
        """Empty chain should fail with appropriate error."""
        strategies = []

        is_valid, error_msg = validate_strategy_chain(strategies)

        assert is_valid is False
        assert "cannot be empty" in error_msg

    def test_none_in_chain(self):
        """Chain with None should fail (BUG-010 fix)."""
        strategies = [None]  # type: ignore

        is_valid, error_msg = validate_strategy_chain(strategies)

        assert is_valid is False
        assert "cannot contain None" in error_msg

    def test_none_in_middle_of_chain(self):
        """None in middle of chain should fail."""
        strategies = [
            _mk_phase("sft"),
            None,  # type: ignore
            _mk_phase("dpo"),
        ]

        is_valid, error_msg = validate_strategy_chain(strategies)

        assert is_valid is False
        assert "cannot contain None" in error_msg

    def test_none_at_start(self):
        """None at start should fail."""
        strategies = [None, _mk_phase("sft")]  # type: ignore

        is_valid, error_msg = validate_strategy_chain(strategies)

        assert is_valid is False
        assert "cannot contain None" in error_msg


class TestInvalidStartStrategies:
    """Test chains with invalid start strategies."""

    @pytest.mark.parametrize("invalid_start", ["dpo", "cot"])
    def test_invalid_start_strategy(self, invalid_start):
        """Invalid start strategy should fail."""
        strategies = [_mk_phase(invalid_start)]

        is_valid, error_msg = validate_strategy_chain(strategies)

        assert is_valid is False
        assert "must start with" in error_msg
        assert invalid_start in error_msg
        # Check that error includes list of valid starts
        assert str(VALID_START_STRATEGIES) in error_msg or "cpt" in error_msg


class TestInvalidTransitions:
    """Test chains with invalid transitions."""

    @pytest.mark.parametrize(
        "chain,error_contains",
        [
            (["sft", "dpo", "sft"], "Invalid transition"),
            (["orpo", "dpo"], "Invalid transition"),
            (["cpt", "dpo"], "Invalid transition"),
            (["sft", "sft"], "Invalid transition"),
            (["sft", "dpo", "orpo"], "Invalid transition"),
            (["sapo", "sft"], "Invalid transition"),
        ],
    )
    def test_invalid_transition(self, chain, error_contains):
        """Invalid transitions should fail with correct error."""
        strategies = [_mk_phase(t) for t in chain]

        is_valid, error_msg = validate_strategy_chain(strategies)

        assert is_valid is False
        assert error_contains in error_msg

    def test_dpo_is_terminal_no_next(self):
        """DPO cannot transition to anything."""
        strategies = [
            _mk_phase("sft"),
            _mk_phase("dpo"),
            _mk_phase("sft"),  # Invalid
        ]

        is_valid, error_msg = validate_strategy_chain(strategies)

        assert is_valid is False
        assert "dpo" in error_msg
        assert "[]" in error_msg or "()" in error_msg or "Valid transitions from 'dpo':" in error_msg

    def test_orpo_is_terminal_no_next(self):
        """ORPO cannot transition to anything."""
        strategies = [
            _mk_phase("orpo"),
            _mk_phase("dpo"),  # Invalid
        ]

        is_valid, error_msg = validate_strategy_chain(strategies)

        assert is_valid is False
        assert "orpo" in error_msg

    def test_sapo_is_terminal_no_next(self):
        """SAPO cannot transition to anything."""
        strategies = [
            _mk_phase("sapo"),
            _mk_phase("sft"),  # Invalid
        ]

        is_valid, error_msg = validate_strategy_chain(strategies)

        assert is_valid is False
        assert "sapo" in error_msg


# =============================================================================
# TEST: Integration with TrainingOnlyConfig
# =============================================================================


class TestTrainingConfigIntegration:
    """Test integration with TrainingOnlyConfig."""

    def test_training_config_valid_chain(self):
        """TrainingOnlyConfig with valid chain should work."""
        config = TrainingOnlyConfig(
            type="qlora",
            lora=_MIN_LORA,
            hyperparams=_MIN_GLOBAL_HYPERPARAMS,
            strategies=[_mk_phase("sft"), _mk_phase("dpo")],
        )

        is_valid, error_msg = config.validate_chain()

        assert is_valid is True
        assert error_msg == ""

    def test_training_config_invalid_chain_detected(self):
        """TrainingOnlyConfig with invalid chain should fail-fast at load time."""
        with pytest.raises(ValidationError) as exc_info:
            _ = TrainingOnlyConfig(
                type="qlora",
                lora=_MIN_LORA,
                hyperparams=_MIN_GLOBAL_HYPERPARAMS,
                strategies=[
                    _mk_phase("dpo"),  # Invalid start
                    _mk_phase("sft"),
                ],
            )

        assert "Chain must start" in str(exc_info.value)

    def test_training_config_terminal_then_more(self):
        """TrainingOnlyConfig with terminal then more should fail-fast at load time."""
        with pytest.raises(ValidationError) as exc_info:
            _ = TrainingOnlyConfig(
                type="qlora",
                lora=_MIN_LORA,
                hyperparams=_MIN_GLOBAL_HYPERPARAMS,
                strategies=[
                    _mk_phase("sft"),
                    _mk_phase("dpo"),
                    _mk_phase("orpo"),  # Invalid: after terminal
                ],
            )

        assert "Invalid transition" in str(exc_info.value)

    def test_training_config_single_strategy(self):
        """TrainingOnlyConfig with single strategy should validate."""
        config = TrainingOnlyConfig(
            type="qlora",
            lora=_MIN_LORA,
            hyperparams=_MIN_GLOBAL_HYPERPARAMS,
            strategies=[_mk_phase("sft")],
        )

        is_valid, error_msg = config.validate_chain()

        assert is_valid is True

    def test_training_config_orpo_standalone(self):
        """ORPO as standalone (valid start + terminal) should work."""
        config = TrainingOnlyConfig(
            type="qlora",
            lora=_MIN_LORA,
            hyperparams=_MIN_GLOBAL_HYPERPARAMS,
            strategies=[_mk_phase("orpo")],
        )

        is_valid, error_msg = config.validate_chain()

        assert is_valid is True

    def test_training_config_sapo_standalone(self):
        """SAPO as standalone should work."""
        config = TrainingOnlyConfig(
            type="qlora",
            lora=_MIN_LORA,
            hyperparams=_MIN_GLOBAL_HYPERPARAMS,
            strategies=[_mk_phase("sapo")],
        )

        is_valid, error_msg = config.validate_chain()

        assert is_valid is True


# =============================================================================
# TEST: Boundary Cases
# =============================================================================


class TestBoundaryCases:
    """Test boundary and edge cases."""

    def test_very_long_valid_chain(self):
        """Maximum valid chain (4 phases) should work."""
        strategies = [
            _mk_phase("cpt"),
            _mk_phase("sft"),
            _mk_phase("cot"),
            _mk_phase("dpo"),
        ]

        is_valid, error_msg = validate_strategy_chain(strategies)

        assert is_valid is True

    def test_all_terminal_strategies_can_be_standalone(self):
        """All terminal strategies should work as standalone."""
        for terminal in ["dpo", "orpo", "sapo"]:
            # They can't start on their own except ORPO and SAPO
            if terminal in ["orpo", "sapo"]:
                strategies = [_mk_phase(terminal)]
                is_valid, _ = validate_strategy_chain(strategies)
                assert is_valid is True
            else:
                # DPO cannot start
                strategies = [_mk_phase(terminal)]
                is_valid, _ = validate_strategy_chain(strategies)
                assert is_valid is False

    def test_cpt_then_sft_then_cot_then_dpo(self):
        """Full pipeline CPT → SFT → CoT → DPO should work."""
        strategies = [
            _mk_phase("cpt"),
            _mk_phase("sft"),
            _mk_phase("cot"),
            _mk_phase("dpo"),
        ]

        is_valid, error_msg = validate_strategy_chain(strategies)

        assert is_valid is True
        assert error_msg == ""

    def test_alternative_terminal_orpo(self):
        """SFT → CoT → ORPO should work."""
        strategies = [
            _mk_phase("sft"),
            _mk_phase("cot"),
            _mk_phase("orpo"),
        ]

        is_valid, error_msg = validate_strategy_chain(strategies)

        assert is_valid is True

    def test_alternative_terminal_sapo(self):
        """SFT → CoT → SAPO should work."""
        strategies = [
            _mk_phase("sft"),
            _mk_phase("cot"),
            _mk_phase("sapo"),
        ]

        is_valid, error_msg = validate_strategy_chain(strategies)

        assert is_valid is True


# =============================================================================
# TEST: Error Message Quality
# =============================================================================


class TestErrorMessages:
    """Test that error messages are helpful and informative."""

    def test_error_message_includes_valid_transitions(self):
        """Error should list valid transitions."""
        strategies = [
            _mk_phase("cpt"),
            _mk_phase("dpo"),  # Invalid: CPT can't go to DPO
        ]

        is_valid, error_msg = validate_strategy_chain(strategies)

        assert is_valid is False
        # Should mention what CPT CAN transition to
        assert "sft" in error_msg or "cot" in error_msg

    def test_error_message_includes_strategy_names(self):
        """Error should include the strategy names involved."""
        strategies = [
            _mk_phase("sft"),
            _mk_phase("sft"),  # Invalid: can't repeat
        ]

        is_valid, error_msg = validate_strategy_chain(strategies)

        assert is_valid is False
        assert "sft" in error_msg

    def test_empty_chain_error_is_clear(self):
        """Empty chain error should be clear."""
        is_valid, error_msg = validate_strategy_chain([])

        assert is_valid is False
        assert "empty" in error_msg.lower()

    def test_none_chain_error_is_clear(self):
        """None in chain error should be clear."""
        is_valid, error_msg = validate_strategy_chain([None])  # type: ignore

        assert is_valid is False
        assert "none" in error_msg.lower()
