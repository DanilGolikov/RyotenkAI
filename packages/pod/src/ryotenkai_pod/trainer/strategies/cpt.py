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

from src.constants import STRATEGY_CPT, STRATEGY_SFT
from src.training.strategies.base import StrategyMetadata, TrainingStrategy
from src.utils.result import Err, Ok, Result, StrategyError

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

    def validate_dataset(self, dataset: Dataset) -> Result[bool, StrategyError]:
        """Validate CPT dataset has required TRL column."""
        columns = dataset.column_names or []
        if "text" not in columns:
            return Err(
                StrategyError(
                    message="CPT requires 'text' column",
                    code="CPT_MISSING_TEXT_COLUMN",
                )
            )
        return Ok(True)

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
