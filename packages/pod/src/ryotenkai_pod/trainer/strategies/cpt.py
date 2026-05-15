"""
CPT Strategy - Continual Pre-Training

Domain adaptation through language modeling on domain-specific corpus.

Use when:
- Need domain-specific knowledge
- Have large unlabeled corpus
- Want to adapt base model to new domain

Data format: `text` column
Objective: Language modeling (predict next token)

Note: TRL SFTTrainer handles tokenization automatically!
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ryotenkai_pod.trainer.strategies.base import StrategyMetadata, TrainingStrategy
from ryotenkai_shared.constants import STRATEGY_CPT, STRATEGY_SFT
from ryotenkai_shared.errors import DatasetValidationFailedError

if TYPE_CHECKING:
    from datasets import Dataset


class CPTStrategy(TrainingStrategy):
    """
    Continual Pre-Training Strategy.

    Accepts canonical TRL format:
    - `text` column — used directly by SFTTrainer for language modeling
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
        """Validate CPT dataset has required TRL column.

        Raises:
            DatasetValidationFailedError: When the 'text' column is missing.
        """
        columns = dataset.column_names or []
        if "text" not in columns:
            raise DatasetValidationFailedError(
                detail="CPT requires 'text' column",
                context={
                    "legacy_code": "CPT_MISSING_TEXT_COLUMN",
                    "strategy": STRATEGY_CPT,
                    "missing_column": "text",
                    "available_columns": list(columns),
                },
            )

    def get_training_objective(self) -> str:
        return "language_modeling"

    def get_trainer_type(self) -> str:
        return STRATEGY_SFT  # SFTTrainer works for CPT with text data

    def get_metadata(self) -> StrategyMetadata:
        return StrategyMetadata(
            name="cpt_strategy",
            version="2.0.0",
            description="Continual Pre-Training for domain adaptation",
            strategy_type=STRATEGY_CPT,
            data_format="text",
            objective="language_modeling",
            recommended_use="Domain adaptation with large unlabeled corpus",
        )


__all__ = ["CPTStrategy"]
