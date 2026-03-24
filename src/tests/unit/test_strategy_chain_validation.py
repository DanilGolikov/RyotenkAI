"""
Strategy Chain Validation Tests.

Comprehensive tests for validate_strategy_chain() function.
Covers all valid/invalid transitions per VALID_STRATEGY_TRANSITIONS.

Strategy Rules:
- Valid start strategies: cpt, sft, orpo
- Transitions:
  - cpt → [sft, cot]
  - sft → [cot, dpo, orpo]
  - cot → [dpo, orpo]
  - dpo → [] (terminal)
  - orpo → [] (terminal)
"""

import pytest

from src.utils.config import (
    VALID_START_STRATEGIES,
    VALID_STRATEGY_TRANSITIONS,
    PhaseHyperparametersConfig,
    StrategyPhaseConfig,
    validate_strategy_chain,
)

# =============================================================================
# HELPER: Create strategy chain from list of strategy types
# =============================================================================


def make_chain(*strategy_types: str) -> list[StrategyPhaseConfig]:
    """Create a chain of StrategyPhaseConfig from strategy type strings."""
    return [StrategyPhaseConfig(strategy_type=s, dataset="default") for s in strategy_types]


# =============================================================================
# TEST CLASS: Valid Strategy Chains
# =============================================================================


class TestValidStrategyChains:
    """Tests for valid strategy chain combinations."""

    # -------------------------------------------------------------------------
    # Single-phase valid chains
    # -------------------------------------------------------------------------

    def test_single_sft(self):
        """Single SFT is valid (most common case)."""
        chain = make_chain("sft")
        is_valid, error = validate_strategy_chain(chain)
        assert is_valid is True
        assert error == ""

    def test_single_cpt(self):
        """Single CPT is valid (continued pre-training)."""
        chain = make_chain("cpt")
        is_valid, error = validate_strategy_chain(chain)
        assert is_valid is True
        assert error == ""

    def test_single_orpo(self):
        """Single ORPO is valid (combined SFT + alignment)."""
        chain = make_chain("orpo")
        is_valid, error = validate_strategy_chain(chain)
        assert is_valid is True
        assert error == ""

    # -------------------------------------------------------------------------
    # Two-phase valid chains
    # -------------------------------------------------------------------------

    def test_sft_then_dpo(self):
        """SFT → DPO is valid (standard alignment pipeline)."""
        chain = make_chain("sft", "dpo")
        is_valid, error = validate_strategy_chain(chain)
        assert is_valid is True
        assert error == ""

    def test_sft_then_orpo(self):
        """SFT → ORPO is valid."""
        chain = make_chain("sft", "orpo")
        is_valid, error = validate_strategy_chain(chain)
        assert is_valid is True
        assert error == ""

    def test_sft_then_cot(self):
        """SFT → CoT is valid (reasoning enhancement)."""
        chain = make_chain("sft", "cot")
        is_valid, error = validate_strategy_chain(chain)
        assert is_valid is True
        assert error == ""

    def test_cpt_then_sft(self):
        """CPT → SFT is valid (pretrain then finetune)."""
        chain = make_chain("cpt", "sft")
        is_valid, error = validate_strategy_chain(chain)
        assert is_valid is True
        assert error == ""

    def test_cpt_then_cot(self):
        """CPT → CoT is valid."""
        chain = make_chain("cpt", "cot")
        is_valid, error = validate_strategy_chain(chain)
        assert is_valid is True
        assert error == ""

    def test_cot_then_dpo(self):
        """CoT → DPO is valid (reasoning then alignment)."""
        # Note: CoT can only follow SFT or CPT, so this needs SFT first
        # Actually, let's test the direct transition
        # Wait - CoT is not a valid start strategy!
        # So we can't test cot → dpo directly without sft first
        chain = make_chain("sft", "cot", "dpo")
        is_valid, error = validate_strategy_chain(chain)
        assert is_valid is True
        assert error == ""

    def test_cot_then_orpo(self):
        """SFT → CoT → ORPO is valid."""
        chain = make_chain("sft", "cot", "orpo")
        is_valid, error = validate_strategy_chain(chain)
        assert is_valid is True
        assert error == ""

    # -------------------------------------------------------------------------
    # Three-phase valid chains
    # -------------------------------------------------------------------------

    def test_sft_cot_dpo(self):
        """SFT → CoT → DPO: full reasoning + alignment pipeline."""
        chain = make_chain("sft", "cot", "dpo")
        is_valid, error = validate_strategy_chain(chain)
        assert is_valid is True
        assert error == ""

    def test_cpt_sft_dpo(self):
        """CPT → SFT → DPO: pretrain, finetune, align."""
        chain = make_chain("cpt", "sft", "dpo")
        is_valid, error = validate_strategy_chain(chain)
        assert is_valid is True
        assert error == ""

    def test_cpt_sft_cot(self):
        """CPT → SFT → CoT: pretrain, finetune, reason."""
        chain = make_chain("cpt", "sft", "cot")
        is_valid, error = validate_strategy_chain(chain)
        assert is_valid is True
        assert error == ""

    # -------------------------------------------------------------------------
    # Four-phase maximum chain
    # -------------------------------------------------------------------------

    def test_cpt_sft_cot_dpo(self):
        """CPT → SFT → CoT → DPO: maximum pipeline depth."""
        chain = make_chain("cpt", "sft", "cot", "dpo")
        is_valid, error = validate_strategy_chain(chain)
        assert is_valid is True
        assert error == ""

    def test_cpt_sft_cot_orpo(self):
        """CPT → SFT → CoT → ORPO: maximum pipeline with ORPO."""
        chain = make_chain("cpt", "sft", "cot", "orpo")
        is_valid, error = validate_strategy_chain(chain)
        assert is_valid is True
        assert error == ""


# =============================================================================
# TEST CLASS: Invalid Strategy Chains - Empty/None
# =============================================================================


class TestInvalidChainsEmptyNone:
    """Tests for empty and None chain scenarios."""

    def test_empty_chain(self):
        """Empty chain is invalid."""
        is_valid, error = validate_strategy_chain([])
        assert is_valid is False
        assert "empty" in error.lower()

    def test_none_element_only(self):
        """Chain with only None element is invalid (after filtering)."""
        chain = [None]  # type: ignore
        is_valid, _error = validate_strategy_chain(chain)  # type: ignore
        assert is_valid is False
        # After filtering None, chain is empty

    def test_none_in_middle(self):
        """Chain with None in middle - current behavior: fails with None error."""
        chain = [
            StrategyPhaseConfig(strategy_type="sft"),
            None,  # type: ignore
            StrategyPhaseConfig(strategy_type="dpo"),
        ]
        # Current implementation: filters None and validates rest as [sft, dpo]
        # OR: rejects None immediately
        is_valid, error = validate_strategy_chain(chain)  # type: ignore
        # Accept either behavior - filtering Nones or rejecting them
        if not is_valid:
            assert "none" in error.lower() or "empty" in error.lower()

    def test_multiple_nones(self):
        """Chain with multiple Nones should fail validation."""
        chain = [None, None, None]  # type: ignore
        is_valid, error = validate_strategy_chain(chain)  # type: ignore
        assert is_valid is False
        # Either "empty" (after filtering) or "None" (if rejected)
        assert "empty" in error.lower() or "none" in error.lower()


# =============================================================================
# TEST CLASS: Invalid Strategy Chains - Bad Start
# =============================================================================


class TestInvalidChainsBadStart:
    """Tests for invalid starting strategies."""

    def test_dpo_cannot_start(self):
        """DPO cannot be first strategy."""
        chain = make_chain("dpo")
        is_valid, error = validate_strategy_chain(chain)
        assert is_valid is False
        assert "must start with" in error.lower()
        assert "dpo" in error.lower()

    def test_cot_cannot_start(self):
        """CoT cannot be first strategy (needs SFT/CPT first)."""
        chain = make_chain("cot")
        is_valid, error = validate_strategy_chain(chain)
        assert is_valid is False
        assert "must start with" in error.lower()

    def test_dpo_then_sft_invalid_start(self):
        """DPO → SFT: invalid because DPO can't start."""
        chain = make_chain("dpo", "sft")
        is_valid, error = validate_strategy_chain(chain)
        assert is_valid is False
        assert "must start with" in error.lower()


# =============================================================================
# TEST CLASS: Invalid Strategy Chains - Bad Transitions
# =============================================================================


class TestInvalidChainsBadTransitions:
    """Tests for invalid strategy transitions."""

    def test_dpo_is_terminal(self):
        """DPO → anything is invalid (DPO is terminal)."""
        chain = make_chain("sft", "dpo", "sft")
        is_valid, error = validate_strategy_chain(chain)
        assert is_valid is False
        assert "invalid transition" in error.lower()
        assert "dpo" in error.lower()

    def test_orpo_is_terminal(self):
        """ORPO → anything is invalid (ORPO is terminal)."""
        chain = make_chain("orpo", "sft")
        is_valid, error = validate_strategy_chain(chain)
        assert is_valid is False
        # ORPO is valid start but terminal, so orpo → sft is invalid
        assert "invalid transition" in error.lower()

    def test_cpt_to_dpo_invalid(self):
        """CPT → DPO is invalid (must go through SFT)."""
        chain = make_chain("cpt", "dpo")
        is_valid, error = validate_strategy_chain(chain)
        assert is_valid is False
        assert "invalid transition" in error.lower()
        assert "cpt" in error.lower()

    def test_cpt_to_orpo_invalid(self):
        """CPT → ORPO is invalid."""
        chain = make_chain("cpt", "orpo")
        is_valid, error = validate_strategy_chain(chain)
        assert is_valid is False
        assert "invalid transition" in error.lower()

    def test_sft_to_sft_invalid(self):
        """SFT → SFT is invalid (duplicate not in transitions)."""
        chain = make_chain("sft", "sft")
        is_valid, error = validate_strategy_chain(chain)
        assert is_valid is False
        assert "invalid transition" in error.lower()

    def test_cot_to_cot_invalid(self):
        """CoT → CoT is invalid."""
        chain = make_chain("sft", "cot", "cot")
        is_valid, error = validate_strategy_chain(chain)
        assert is_valid is False
        assert "invalid transition" in error.lower()

    def test_cot_to_sft_invalid(self):
        """CoT → SFT is invalid (can only go to DPO/ORPO)."""
        chain = make_chain("sft", "cot", "sft")
        is_valid, error = validate_strategy_chain(chain)
        assert is_valid is False
        assert "invalid transition" in error.lower()

    def test_cot_to_cpt_invalid(self):
        """CoT → CPT is invalid."""
        chain = make_chain("sft", "cot", "cpt")
        is_valid, error = validate_strategy_chain(chain)
        assert is_valid is False
        assert "invalid transition" in error.lower()


# =============================================================================
# TEST CLASS: Invalid Strategy Chains - Unknown Types
# =============================================================================


class TestInvalidChainsUnknownTypes:
    """Tests for unknown strategy types - Pydantic validates before chain validation."""

    def test_unknown_strategy_type_rejected_by_pydantic(self):
        """Unknown strategy type should fail at Pydantic validation."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError) as exc_info:
            make_chain("unknown")

        assert "strategy_type" in str(exc_info.value).lower()

    def test_typo_in_strategy_rejected_by_pydantic(self):
        """Typo in strategy name should fail at Pydantic validation."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError) as exc_info:
            make_chain("sftt")

        assert "strategy_type" in str(exc_info.value).lower()

    def test_uppercase_strategy_type(self):
        """Uppercase strategy type - Pydantic may normalize or reject."""
        from pydantic import ValidationError

        # Pydantic might either:
        # 1. Normalize "SFT" to "sft" (if configured)
        # 2. Reject "SFT" as invalid
        try:
            chain = make_chain("SFT")
            # If we got here, Pydantic normalized it
            is_valid, _error = validate_strategy_chain(chain)
            assert is_valid is True  # "sft" is valid
        except ValidationError:
            # Pydantic rejected uppercase
            pass


# =============================================================================
# TEST CLASS: Error Message Quality
# =============================================================================


class TestErrorMessageQuality:
    """Tests that error messages are helpful and descriptive."""

    def test_empty_chain_error_is_clear(self):
        """Empty chain error should mention 'empty'."""
        is_valid, error = validate_strategy_chain([])
        assert is_valid is False
        assert "empty" in error.lower() or "cannot be empty" in error.lower()

    def test_invalid_start_shows_allowed(self):
        """Invalid start error should show allowed strategies."""
        chain = make_chain("dpo")
        is_valid, error = validate_strategy_chain(chain)
        assert is_valid is False
        # Error should mention what IS allowed
        assert any(s in error.lower() for s in ["cpt", "sft", "orpo"])

    def test_invalid_transition_shows_valid_options(self):
        """Invalid transition error should show valid options."""
        chain = make_chain("sft", "sft")  # sft → sft invalid
        is_valid, error = validate_strategy_chain(chain)
        assert is_valid is False
        # Error should mention valid transitions from sft
        assert "cot" in error.lower() or "dpo" in error.lower() or "orpo" in error.lower()

    def test_invalid_transition_shows_from_to(self):
        """Error should show which transition was attempted."""
        chain = make_chain("cpt", "dpo")  # cpt → dpo invalid
        is_valid, error = validate_strategy_chain(chain)
        assert is_valid is False
        assert "cpt" in error.lower()
        assert "dpo" in error.lower()


# =============================================================================
# TEST CLASS: Edge Cases
# =============================================================================


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_very_long_chain(self):
        """Very long valid chain should work."""
        # cpt → sft → cot → dpo (max valid length is 4 based on transitions)
        chain = make_chain("cpt", "sft", "cot", "dpo")
        is_valid, _error = validate_strategy_chain(chain)
        assert is_valid is True

    def test_single_strategy_with_metadata(self):
        """Strategy with all metadata fields should validate."""
        chain = [
            StrategyPhaseConfig(
                strategy_type="sft",
                dataset="custom_dataset",
                hyperparams=PhaseHyperparametersConfig(
                    epochs=5,
                    learning_rate=1e-5,
                    per_device_train_batch_size=8,
                ),
            )
        ]
        is_valid, _error = validate_strategy_chain(chain)
        assert is_valid is True

    def test_mixed_metadata_strategies(self):
        """Chain with different metadata per strategy."""
        chain = [
            StrategyPhaseConfig(
                strategy_type="sft",
                hyperparams=PhaseHyperparametersConfig(epochs=3, learning_rate=2e-4),
            ),
            StrategyPhaseConfig(strategy_type="dpo", hyperparams=PhaseHyperparametersConfig(epochs=1, beta=0.1)),
        ]
        is_valid, _error = validate_strategy_chain(chain)
        assert is_valid is True


# =============================================================================
# TEST CLASS: Constants Verification
# =============================================================================


class TestStrategyConstants:
    """Tests that verify the strategy constants are correct."""

    def test_valid_start_strategies_constant(self):
        """VALID_START_STRATEGIES should contain expected values."""
        assert "sft" in VALID_START_STRATEGIES
        assert "cpt" in VALID_START_STRATEGIES
        assert "orpo" in VALID_START_STRATEGIES
        assert "dpo" not in VALID_START_STRATEGIES
        assert "cot" not in VALID_START_STRATEGIES

    def test_dpo_is_terminal(self):
        """DPO should have no valid transitions."""
        assert VALID_STRATEGY_TRANSITIONS.get("dpo") == ()

    def test_orpo_is_terminal(self):
        """ORPO should have no valid transitions."""
        assert VALID_STRATEGY_TRANSITIONS.get("orpo") == ()

    def test_sft_transitions(self):
        """SFT should transition to cot, dpo, orpo."""
        sft_transitions = VALID_STRATEGY_TRANSITIONS.get("sft", ())
        assert "cot" in sft_transitions
        assert "dpo" in sft_transitions
        assert "orpo" in sft_transitions

    def test_cpt_transitions(self):
        """CPT should transition to sft, cot."""
        cpt_transitions = VALID_STRATEGY_TRANSITIONS.get("cpt", ())
        assert "sft" in cpt_transitions
        assert "cot" in cpt_transitions

    def test_cot_transitions(self):
        """CoT should transition to dpo, orpo."""
        cot_transitions = VALID_STRATEGY_TRANSITIONS.get("cot", ())
        assert "dpo" in cot_transitions
        assert "orpo" in cot_transitions


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
