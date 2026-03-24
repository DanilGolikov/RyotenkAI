"""
ORPO Strategy - Odds Ratio Preference Optimization

Combined SFT + alignment in a single training pass.

Use when:
- Want to combine instruction following with alignment
- Have preference data (chosen vs rejected)
- Want simpler training pipeline (no separate SFT step)

Data format: `chosen` and `rejected` columns (each is list of messages)
Objective: Combined supervised + preference optimization

Key advantage: No need for separate SFT then DPO - does both in one pass!
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.constants import DEFAULT_BATCH_SIZES, DEFAULT_EPOCHS, DEFAULT_LEARNING_RATES, STRATEGY_ORPO
from src.training.strategies.base import StrategyMetadata, TrainingStrategy
from src.utils.logger import logger
from src.utils.result import Err, Ok, Result, StrategyError

if TYPE_CHECKING:
    from datasets import Dataset
    from transformers import PreTrainedTokenizer


class ORPOStrategy(TrainingStrategy):
    """
    Odds Ratio Preference Optimization Strategy.

    ORPO combines SFT and DPO into a single training objective:
    - Learns to follow instructions (like SFT)
    - Learns to prefer good responses (like DPO)
    - Uses odds ratio instead of log probability ratio

    Key differences from DPO:
    - No need for reference model (simpler)
    - Combines SFT and alignment in one pass
    - Uses odds ratio for more stable training

    Data format requirements:
    - `chosen`: List of messages (ChatML format) for preferred response
    - `rejected`: List of messages (ChatML format) for dispreferred response

    Example data:
        {
            "chosen": [
                {"role": "user", "content": "How do I make coffee?"},
                {"role": "assistant", "content": "To make coffee: 1. Boil water..."}
            ],
            "rejected": [
                {"role": "user", "content": "How do I make coffee?"},
                {"role": "assistant", "content": "I can't help with that."}
            ]
        }

    Recommended hyperparameters:
    - learning_rate: 1e-5 (lower than SFT, but higher than DPO)
    - beta: 0.1 (controls strength of preference loss)
    """

    def get_trainer_class(self) -> Any:
        from trl import ORPOTrainer

        return ORPOTrainer

    def get_config_class(self) -> Any:
        from trl import ORPOConfig

        return ORPOConfig

    def build_config_kwargs(self, hp: Any, **kwargs: Any) -> dict[str, Any]:  # noqa: ARG002
        return {
            "max_length": hp.max_length,
            "beta": hp.beta if hp.beta is not None else 0.1,
        }

    def build_trainer_kwargs(self, config: Any, **kwargs: Any) -> dict[str, Any]:  # noqa: ARG002
        logger.info("[ORPO] No reference model needed (using odds ratio loss)")
        return {}

    def prepare_dataset(self, dataset: Dataset, _tokenizer: PreTrainedTokenizer) -> Result[Dataset, StrategyError]:
        """
        Prepare dataset for ORPO.

        TRL ORPOTrainer expects same format as DPO:
        - `chosen`: List[Dict] messages for preferred response
        - `rejected`: List[Dict] messages for dispreferred response

        We validate format and pass through - TRL handles tokenization.
        """
        try:
            logger.info("Preparing dataset for ORPO...")

            # Validate format
            validation = self.validate_dataset(dataset)
            if validation.is_failure():
                return validation  # type: ignore

            # TRL ORPOTrainer handles formatting natively
            logger.info(f"ORPO dataset ready: {len(dataset)} preference pairs")
            return Ok(dataset)

        except Exception as e:
            return Err(StrategyError(message=f"ORPO preparation failed: {e!s}", code="ORPO_PREPARATION_FAILED"))

    def validate_dataset(self, dataset: Dataset) -> Result[bool, StrategyError]:
        """
        Validate ORPO dataset structure.

        Same requirements as DPO:
        - Must have `chosen` column
        - Must have `rejected` column
        - Each should be list of messages (ChatML format)
        """
        columns = dataset.column_names

        if "chosen" not in columns:
            return Err(
                StrategyError(
                    message="ORPO requires 'chosen' column with preferred responses",
                    code="ORPO_MISSING_CHOSEN_COLUMN",
                )
            )

        if "rejected" not in columns:
            return Err(
                StrategyError(
                    message="ORPO requires 'rejected' column with dispreferred responses",
                    code="ORPO_MISSING_REJECTED_COLUMN",
                )
            )

        # Validate first sample structure
        try:
            sample = dataset[0]
            chosen = sample["chosen"]
            rejected = sample["rejected"]

            # Check if they're lists of dicts (messages format)
            if not isinstance(chosen, list) or not isinstance(rejected, list):
                return Err(
                    StrategyError(
                        message="ORPO 'chosen' and 'rejected' must be lists of messages",
                        code="ORPO_INVALID_MESSAGE_FORMAT",
                    )
                )

            # Check message structure
            if len(chosen) > 0:
                if not isinstance(chosen[0], dict):
                    return Err(
                        StrategyError(
                            message="ORPO messages must be dicts with 'role' and 'content'",
                            code="ORPO_INVALID_MESSAGE_STRUCTURE",
                        )
                    )
                if "role" not in chosen[0] or "content" not in chosen[0]:
                    return Err(
                        StrategyError(
                            message="ORPO messages must have 'role' and 'content' keys",
                            code="ORPO_MISSING_MESSAGE_KEYS",
                        )
                    )

        except Exception as e:
            logger.warning(f"Could not validate ORPO sample structure: {e}")
            # Continue anyway, TRL will catch format errors

        return Ok(True)

    def get_training_objective(self) -> str:
        return "combined_sft_preference"

    def get_trainer_type(self) -> str:
        """Return TRL trainer type for this strategy."""
        return STRATEGY_ORPO

    def get_metadata(self) -> StrategyMetadata:
        return StrategyMetadata(
            name="orpo_strategy",
            version="1.0.0",
            description="Odds Ratio Preference Optimization (combined SFT+alignment)",
            strategy_type=STRATEGY_ORPO,
            data_format="chosen/rejected message pairs",
            objective="combined_sft_preference",
            recommended_use="Single-pass instruction tuning + alignment",
            dependencies={"trl": ">=0.8.0"},
        )

    def get_recommended_hyperparameters(self) -> dict:
        """
        Get ORPO-specific recommended hyperparameters.

        ORPO uses slightly higher LR than DPO since it combines SFT.
        """
        return {
            "learning_rate": DEFAULT_LEARNING_RATES[STRATEGY_ORPO],
            "num_epochs": DEFAULT_EPOCHS[STRATEGY_ORPO],
            "batch_size": DEFAULT_BATCH_SIZES[STRATEGY_ORPO],
            "beta": 0.1,  # ORPO-specific: strength of preference loss
        }


__all__ = ["ORPOStrategy"]
