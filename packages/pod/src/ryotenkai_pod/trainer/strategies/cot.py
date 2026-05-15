"""
CoT Strategy - Chain-of-Thought Fine-Tuning

Multi-step reasoning through explicit reasoning traces.

Use when:
- Need reasoning capabilities
- Have CoT examples with reasoning steps
- Want interpretable reasoning

Data format: `messages` column (ChatML) or `text` column
Objective: Reasoning with explicit steps

Note: CoT uses SFTTrainer. Dataset must arrive with reasoning already
formatted (e.g. <think>...</think> tags in assistant content).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ryotenkai_pod.trainer.strategies.base import StrategyMetadata, TrainingStrategy
from ryotenkai_shared.constants import STRATEGY_COT, STRATEGY_SFT
from ryotenkai_shared.errors import DatasetValidationFailedError

if TYPE_CHECKING:
    from datasets import Dataset


class CoTStrategy(TrainingStrategy):
    """
    Chain-of-Thought Fine-Tuning Strategy.

    Accepts canonical TRL formats:
    - `messages` column (ChatML) — assistant content should include reasoning tags
    - `text` column — pre-formatted text with reasoning tags

    Dataset must arrive ready — no in-pipeline conversion is done.
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
        """Validate CoT dataset has required TRL columns.

        Raises:
            DatasetValidationFailedError: When required columns are missing.
        """
        columns = dataset.column_names or []
        if "messages" not in columns and "text" not in columns:
            raise DatasetValidationFailedError(
                detail="CoT requires 'messages' (ChatML) or 'text' column",
                context={
                    "legacy_code": "COT_MISSING_REQUIRED_COLUMNS",
                    "strategy": STRATEGY_COT,
                    "expected_one_of": ["messages", "text"],
                    "available_columns": list(columns),
                },
            )

    def get_training_objective(self) -> str:
        return "chain_of_thought_learning"

    def get_trainer_type(self) -> str:
        return STRATEGY_SFT  # SFTTrainer works for CoT

    def get_metadata(self) -> StrategyMetadata:
        return StrategyMetadata(
            name="cot_strategy",
            version="2.0.0",
            description="Chain-of-Thought training for reasoning",
            strategy_type=STRATEGY_COT,
            data_format="messages (ChatML) or text",
            objective="reasoning_with_steps",
            recommended_use="Complex reasoning tasks (math, logic, code)",
            dependencies={"performance_gain": "+8-15% on reasoning tasks"},
        )


__all__ = ["CoTStrategy"]
