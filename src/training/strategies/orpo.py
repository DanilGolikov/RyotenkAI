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

from src.constants import STRATEGY_ORPO
from src.training.strategies.base import StrategyMetadata, TrainingStrategy
from src.utils.logger import logger
from src.utils.result import Err, Ok, Result, StrategyError

if TYPE_CHECKING:
    from datasets import Dataset


class ORPOStrategy(TrainingStrategy):
    """
    Odds Ratio Preference Optimization Strategy.

    ORPO combines SFT and DPO into a single training objective:
    - Learns to follow instructions (like SFT)
    - Learns to prefer good responses (like DPO)
    - Uses odds ratio instead of log probability ratio

    Accepts canonical TRL format:
    - `chosen`: preferred response (list of messages or plain text)
    - `rejected`: dispreferred response (list of messages or plain text)

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

    def validate_dataset(self, dataset: Dataset) -> Result[bool, StrategyError]:
        """Validate ORPO dataset has required TRL columns."""
        columns = dataset.column_names or []

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

        return Ok(True)

    def get_training_objective(self) -> str:
        return "combined_sft_preference"

    def get_trainer_type(self) -> str:
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


__all__ = ["ORPOStrategy"]
