"""
SFT Strategy - Supervised Fine-Tuning

Instruction following through supervised learning on instruction-response pairs.

Use when:
- Need instruction-following capabilities
- Have labeled examples
- Want task-specific performance

Data format: `messages` column (ChatML) or `text` column
Objective: Supervised learning

Note: TRL SFTTrainer handles chat_template application automatically!
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ryotenkai_pod.trainer.strategies.base import StrategyMetadata, TrainingStrategy
from ryotenkai_shared.constants import STRATEGY_SFT
from ryotenkai_shared.errors import DatasetValidationFailedError

if TYPE_CHECKING:
    from datasets import Dataset


class SFTStrategy(TrainingStrategy):
    """
    Supervised Fine-Tuning Strategy.

    Accepts canonical TRL formats:
    - `messages` column (ChatML) — TRL applies chat_template automatically
    - `text` column — used directly
    """

    def get_trainer_class(self) -> Any:
        from trl import SFTTrainer

        return SFTTrainer

    def get_config_class(self) -> Any:
        from trl import SFTConfig

        return SFTConfig

    def build_config_kwargs(self, hp: Any, **kwargs: Any) -> dict[str, Any]:  # noqa: ARG002
        return {
            "max_length": hp.max_length,
            "packing": hp.packing,
        }

    def validate_dataset(self, dataset: Dataset) -> None:
        """Validate SFT dataset has required TRL columns.

        Raises:
            DatasetValidationFailedError: When required columns are missing.
        """
        columns = dataset.column_names or []
        if "messages" not in columns and "text" not in columns:
            raise DatasetValidationFailedError(
                detail="SFT requires 'messages' (ChatML) or 'text' column",
                context={
                    "legacy_code": "SFT_MISSING_REQUIRED_COLUMN",
                    "strategy": STRATEGY_SFT,
                    "expected_one_of": ["messages", "text"],
                    "available_columns": list(columns),
                },
            )

    def get_training_objective(self) -> str:
        return "supervised_learning"

    def get_trainer_type(self) -> str:
        return STRATEGY_SFT

    def get_metadata(self) -> StrategyMetadata:
        return StrategyMetadata(
            name="sft_strategy",
            version="2.0.0",
            description="Supervised Fine-Tuning for instruction following",
            strategy_type=STRATEGY_SFT,
            data_format="messages (ChatML) or text",
            objective="supervised_learning",
            recommended_use="Standard instruction fine-tuning",
        )


__all__ = ["SFTStrategy"]
