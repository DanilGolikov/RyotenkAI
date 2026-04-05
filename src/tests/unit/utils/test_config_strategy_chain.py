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

from unittest.mock import patch

import pytest

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


def _mk_phase(strategy_type: str, dataset: str | None = None) -> StrategyPhaseConfig:
    """
    Build StrategyPhaseConfig with required strategy-specific params.

    NOTE: Some strategies (e.g. SAPO) require additional hyperparams.
    """
    extra: dict = {}
    if dataset is not None:
        extra["dataset"] = dataset

    if strategy_type == "sapo":
        return StrategyPhaseConfig(
            strategy_type=strategy_type,
            hyperparams={
                "max_prompt_length": 1024,
                "max_completion_length": 512,
            },
            params={"reward_plugin": "helixql_compiler_semantic"},
            **extra,
        )
    if strategy_type == "grpo":
        return StrategyPhaseConfig(
            strategy_type=strategy_type,
            hyperparams={
                "max_prompt_length": 1024,
                "max_completion_length": 512,
            },
            params={"reward_plugin": "helixql_compiler_semantic"},
            **extra,
        )
    return StrategyPhaseConfig(strategy_type=strategy_type, **extra)


def _warning_text(mock_warning) -> str:
    return "\n".join(str(call.args[0]) for call in mock_warning.call_args_list)


def _assert_ok(result) -> None:
    assert result.is_success()
    assert result.unwrap() is None


def _assert_err(result, *, code: str | None = None) -> str:
    assert result.is_failure()
    err = result.unwrap_err()
    if code is not None:
        assert err.code == code
    return str(err)


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
        assert "dpo" in VALID_START_STRATEGIES

        # Must NOT contain these (not valid start)
        assert "cot" not in VALID_START_STRATEGIES


# =============================================================================
# TEST: Valid Chains
# =============================================================================


class TestValidSinglePhaseChains:
    """Test valid single-phase strategy chains."""

    @pytest.mark.parametrize("strategy_type", ["cpt", "sft", "orpo", "grpo", "sapo", "dpo"])
    def test_single_phase_valid(self, strategy_type):
        """Single valid start strategy should pass."""
        strategies = [_mk_phase(strategy_type)]
        _assert_ok(validate_strategy_chain(strategies))

    def test_single_phase_cot_warns_but_passes(self):
        """Single CoT should emit warning but still pass validation."""
        strategies = [_mk_phase("cot")]
        with patch("src.utils.logger.logger.warning") as mock_warning:
            result = validate_strategy_chain(strategies)
        warning_text = _warning_text(mock_warning)
        _assert_ok(result)
        assert "reason=invalid_start" in warning_text


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
        strategies = [_mk_phase(t, dataset=f"ds_{t}") for t in chain]
        _assert_ok(validate_strategy_chain(strategies))


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
        strategies = [_mk_phase(t, dataset=f"ds_{t}") for t in chain]
        _assert_ok(validate_strategy_chain(strategies))


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
        strategies = [_mk_phase(t, dataset=f"ds_{t}") for t in chain]
        _assert_ok(validate_strategy_chain(strategies))


# =============================================================================
# TEST: Invalid Chains
# =============================================================================


class TestInvalidChains:
    """Test invalid strategy chains and error messages."""

    def test_empty_chain(self):
        """Empty chain should fail with appropriate error."""
        strategies = []
        error_msg = _assert_err(validate_strategy_chain(strategies), code="STRATEGY_CHAIN_EMPTY")
        assert "cannot be empty" in error_msg

    def test_none_in_chain(self):
        """Chain with None should fail (BUG-010 fix)."""
        strategies = [None]  # type: ignore
        error_msg = _assert_err(validate_strategy_chain(strategies), code="STRATEGY_CHAIN_CONTAINS_NONE")  # type: ignore[arg-type]
        assert "cannot contain None" in error_msg

    def test_none_in_middle_of_chain(self):
        """None in middle of chain should fail."""
        strategies = [
            _mk_phase("sft", dataset="ds_sft"),
            None,  # type: ignore
            _mk_phase("dpo", dataset="ds_dpo"),
        ]
        error_msg = _assert_err(validate_strategy_chain(strategies), code="STRATEGY_CHAIN_CONTAINS_NONE")  # type: ignore[arg-type]
        assert "cannot contain None" in error_msg

    def test_none_at_start(self):
        """None at start should fail."""
        strategies = [None, _mk_phase("sft")]  # type: ignore
        error_msg = _assert_err(validate_strategy_chain(strategies), code="STRATEGY_CHAIN_CONTAINS_NONE")  # type: ignore[arg-type]
        assert "cannot contain None" in error_msg


class TestInvalidStartStrategies:
    """Test chains with invalid start strategies (warning-only)."""

    def test_invalid_start_strategy(self):
        """Invalid start strategy should warn but still validate."""
        invalid_start = "cot"
        strategies = [_mk_phase(invalid_start)]

        with patch("src.utils.logger.logger.warning") as mock_warning:
            result = validate_strategy_chain(strategies)
        warning_text = _warning_text(mock_warning)

        _assert_ok(result)
        assert "reason=invalid_start" in warning_text
        assert "got=cot" in warning_text
        assert "cpt" in warning_text


class TestInvalidTransitions:
    """Test chains with invalid transitions (warning-only)."""

    @pytest.mark.parametrize(
        "chain,from_strategy,to_strategy",
        [
            (["sft", "dpo", "sft"], "dpo", "sft"),
            (["orpo", "dpo"], "orpo", "dpo"),
            (["cpt", "dpo"], "cpt", "dpo"),
            (["sft", "sft"], "sft", "sft"),
            (["sft", "dpo", "orpo"], "dpo", "orpo"),
            (["sapo", "sft"], "sapo", "sft"),
        ],
    )
    def test_invalid_transition(self, chain, from_strategy, to_strategy):
        """Invalid transitions should warn but still pass validation."""
        strategies = [_mk_phase(t, dataset=f"ds_{idx}_{t}") for idx, t in enumerate(chain)]

        with patch("src.utils.logger.logger.warning") as mock_warning:
            result = validate_strategy_chain(strategies)
        warning_text = _warning_text(mock_warning)

        _assert_ok(result)
        assert "reason=invalid_transition" in warning_text
        assert f"from={from_strategy}" in warning_text
        assert f"to={to_strategy}" in warning_text

    def test_dpo_is_terminal_no_next(self):
        """DPO cannot transition to anything without warning."""
        strategies = [
            _mk_phase("sft", dataset="ds_sft"),
            _mk_phase("dpo", dataset="ds_dpo"),
            _mk_phase("sft", dataset="ds_sft2"),  # Invalid
        ]

        with patch("src.utils.logger.logger.warning") as mock_warning:
            result = validate_strategy_chain(strategies)
        warning_text = _warning_text(mock_warning)

        _assert_ok(result)
        assert "from=dpo" in warning_text
        assert "valid=()" in warning_text

    def test_orpo_is_terminal_no_next(self):
        """ORPO cannot transition to anything without warning."""
        strategies = [
            _mk_phase("orpo", dataset="ds_orpo"),
            _mk_phase("dpo", dataset="ds_dpo"),  # Invalid
        ]

        with patch("src.utils.logger.logger.warning") as mock_warning:
            result = validate_strategy_chain(strategies)
        warning_text = _warning_text(mock_warning)

        _assert_ok(result)
        assert "from=orpo" in warning_text

    def test_sapo_is_terminal_no_next(self):
        """SAPO cannot transition to anything without warning."""
        strategies = [
            _mk_phase("sapo", dataset="ds_sapo"),
            _mk_phase("sft", dataset="ds_sft"),  # Invalid
        ]

        with patch("src.utils.logger.logger.warning") as mock_warning:
            result = validate_strategy_chain(strategies)
        warning_text = _warning_text(mock_warning)

        _assert_ok(result)
        assert "from=sapo" in warning_text


# =============================================================================
# TEST: Integration with TrainingOnlyConfig
# =============================================================================


class TestTrainingConfigIntegration:
    """Test integration with TrainingOnlyConfig."""

    def test_training_config_valid_chain(self):
        """TrainingOnlyConfig with valid chain should work."""
        config = TrainingOnlyConfig(
            type="qlora",
            qlora=_MIN_LORA,
            hyperparams=_MIN_GLOBAL_HYPERPARAMS,
            strategies=[_mk_phase("sft", dataset="ds_sft"), _mk_phase("dpo", dataset="ds_dpo")],
        )

        _assert_ok(config.validate_chain())

    def test_training_config_invalid_start_warns_but_builds(self):
        """TrainingOnlyConfig should still build when ordering is only semantically invalid."""
        with patch("src.utils.logger.logger.warning") as mock_warning:
            config = TrainingOnlyConfig(
                type="qlora",
                qlora=_MIN_LORA,
                hyperparams=_MIN_GLOBAL_HYPERPARAMS,
                strategies=[
                    _mk_phase("cot", dataset="ds_cot"),
                    _mk_phase("sft", dataset="ds_sft"),
                ],
            )
        warning_text = _warning_text(mock_warning)

        assert config.strategies[0].strategy_type == "cot"
        assert "reason=invalid_start" in warning_text
        assert "reason=invalid_transition" in warning_text

    def test_training_config_terminal_then_more_warns_but_builds(self):
        """Terminal strategy followed by another phase should only warn."""
        with patch("src.utils.logger.logger.warning") as mock_warning:
            config = TrainingOnlyConfig(
                type="qlora",
                qlora=_MIN_LORA,
                hyperparams=_MIN_GLOBAL_HYPERPARAMS,
                strategies=[
                    _mk_phase("sft", dataset="ds_sft"),
                    _mk_phase("dpo", dataset="ds_dpo"),
                    _mk_phase("orpo", dataset="ds_orpo"),
                ],
            )
        warning_text = _warning_text(mock_warning)

        assert len(config.strategies) == 3
        assert "reason=invalid_transition" in warning_text

    def test_training_config_single_strategy(self):
        """TrainingOnlyConfig with single strategy should validate."""
        config = TrainingOnlyConfig(
            type="qlora",
            qlora=_MIN_LORA,
            hyperparams=_MIN_GLOBAL_HYPERPARAMS,
            strategies=[_mk_phase("sft")],
        )

        _assert_ok(config.validate_chain())

    def test_training_config_orpo_standalone(self):
        """ORPO as standalone (valid start + terminal) should work."""
        config = TrainingOnlyConfig(
            type="qlora",
            qlora=_MIN_LORA,
            hyperparams=_MIN_GLOBAL_HYPERPARAMS,
            strategies=[_mk_phase("orpo")],
        )

        _assert_ok(config.validate_chain())

    def test_training_config_sapo_standalone(self):
        """SAPO as standalone should work."""
        config = TrainingOnlyConfig(
            type="qlora",
            qlora=_MIN_LORA,
            hyperparams=_MIN_GLOBAL_HYPERPARAMS,
            strategies=[_mk_phase("sapo")],
        )

        _assert_ok(config.validate_chain())


# =============================================================================
# TEST: Boundary Cases
# =============================================================================


class TestBoundaryCases:
    """Test boundary and edge cases."""

    def test_very_long_valid_chain(self):
        """Maximum valid chain (4 phases) should work."""
        strategies = [
            _mk_phase("cpt", dataset="ds_cpt"),
            _mk_phase("sft", dataset="ds_sft"),
            _mk_phase("cot", dataset="ds_cot"),
            _mk_phase("dpo", dataset="ds_dpo"),
        ]
        _assert_ok(validate_strategy_chain(strategies))

    def test_all_terminal_strategies_can_be_standalone(self):
        """All terminal strategies should work as standalone."""
        for terminal in ["dpo", "orpo", "sapo"]:
            strategies = [_mk_phase(terminal)]
            _assert_ok(validate_strategy_chain(strategies))

    def test_cpt_then_sft_then_cot_then_dpo(self):
        """Full pipeline CPT → SFT → CoT → DPO should work."""
        strategies = [
            _mk_phase("cpt", dataset="ds_cpt"),
            _mk_phase("sft", dataset="ds_sft"),
            _mk_phase("cot", dataset="ds_cot"),
            _mk_phase("dpo", dataset="ds_dpo"),
        ]

        _assert_ok(validate_strategy_chain(strategies))

    def test_alternative_terminal_orpo(self):
        """SFT → CoT → ORPO should work."""
        strategies = [
            _mk_phase("sft", dataset="ds_sft"),
            _mk_phase("cot", dataset="ds_cot"),
            _mk_phase("orpo", dataset="ds_orpo"),
        ]

        _assert_ok(validate_strategy_chain(strategies))

    def test_alternative_terminal_sapo(self):
        """SFT → CoT → SAPO should work."""
        strategies = [
            _mk_phase("sft", dataset="ds_sft"),
            _mk_phase("cot", dataset="ds_cot"),
            _mk_phase("sapo", dataset="ds_sapo"),
        ]

        _assert_ok(validate_strategy_chain(strategies))


# =============================================================================
# TEST: Error Message Quality
# =============================================================================


class TestErrorMessages:
    """Test that error messages are helpful and informative."""

    def test_warning_message_includes_valid_transitions(self):
        """Warning should list valid transitions."""
        strategies = [
            _mk_phase("cpt", dataset="ds_cpt"),
            _mk_phase("dpo", dataset="ds_dpo"),  # Invalid: CPT can't go to DPO
        ]

        with patch("src.utils.logger.logger.warning") as mock_warning:
            result = validate_strategy_chain(strategies)
        warning_text = _warning_text(mock_warning)

        _assert_ok(result)
        assert "valid=('sft', 'cot')" in warning_text

    def test_warning_message_includes_strategy_names(self):
        """Warning should include the strategy names involved."""
        strategies = [
            _mk_phase("sft", dataset="ds_sft1"),
            _mk_phase("sft", dataset="ds_sft2"),  # Invalid: can't repeat
        ]

        with patch("src.utils.logger.logger.warning") as mock_warning:
            result = validate_strategy_chain(strategies)
        warning_text = _warning_text(mock_warning)

        _assert_ok(result)
        assert "from=sft" in warning_text
        assert "to=sft" in warning_text

    def test_empty_chain_error_is_clear(self):
        """Empty chain error should be clear."""
        error_msg = _assert_err(validate_strategy_chain([]), code="STRATEGY_CHAIN_EMPTY")
        assert "empty" in error_msg.lower()

    def test_none_chain_error_is_clear(self):
        """None in chain error should be clear."""
        error_msg = _assert_err(validate_strategy_chain([None]), code="STRATEGY_CHAIN_CONTAINS_NONE")  # type: ignore[arg-type]
        assert "none" in error_msg.lower()


# =============================================================================
# TEST: Duplicate Dataset Validation
# =============================================================================


class TestDuplicateDatasetValidation:
    """Test that strategies sharing the same dataset are rejected."""

    def test_two_strategies_same_explicit_dataset(self):
        """Two strategies with same dataset name should fail."""
        strategies = [
            _mk_phase("sft", dataset="shared"),
            _mk_phase("dpo", dataset="shared"),
        ]
        error_msg = _assert_err(validate_strategy_chain(strategies), code="STRATEGY_CHAIN_DUPLICATE_DATASET")
        assert "Duplicate dataset" in error_msg
        assert "shared" in error_msg

    def test_two_strategies_both_default_none(self):
        """Two strategies with dataset=None (both resolve to 'default') should fail."""
        strategies = [
            _mk_phase("sft"),  # dataset=None → "default"
            _mk_phase("dpo"),  # dataset=None → "default"
        ]
        error_msg = _assert_err(validate_strategy_chain(strategies), code="STRATEGY_CHAIN_DUPLICATE_DATASET")
        assert "Duplicate dataset" in error_msg
        assert "default" in error_msg

    def test_single_strategy_no_duplicate_check(self):
        """Single strategy never triggers duplicate check."""
        strategies = [_mk_phase("sft")]
        _assert_ok(validate_strategy_chain(strategies))

    def test_two_strategies_different_datasets_ok(self):
        """Two strategies with different datasets should pass."""
        strategies = [
            _mk_phase("sft", dataset="sft_data"),
            _mk_phase("dpo", dataset="pref_data"),
        ]
        _assert_ok(validate_strategy_chain(strategies))

    def test_error_mentions_both_strategy_types(self):
        """Error should mention which strategies collide."""
        strategies = [
            _mk_phase("sft", dataset="same"),
            _mk_phase("cot", dataset="same"),
        ]
        error_msg = _assert_err(validate_strategy_chain(strategies), code="STRATEGY_CHAIN_DUPLICATE_DATASET")
        assert "sft" in error_msg
        assert "cot" in error_msg
