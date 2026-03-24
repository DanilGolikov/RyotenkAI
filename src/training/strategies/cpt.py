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

from src.constants import DEFAULT_BATCH_SIZES, DEFAULT_EPOCHS, DEFAULT_LEARNING_RATES, STRATEGY_CPT, STRATEGY_SFT
from src.training.strategies.base import StrategyMetadata, TrainingStrategy
from src.utils.logger import logger
from src.utils.result import Err, Ok, Result, StrategyError

if TYPE_CHECKING:
    from datasets import Dataset
    from transformers import PreTrainedTokenizer


class CPTStrategy(TrainingStrategy):
    """
    Continual Pre-Training Strategy.

    Simplified for TRL integration:
    - TRL SFTTrainer accepts `text` column directly
    - No manual tokenization needed
    - Strategy only validates format
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

    def prepare_dataset(self, dataset: Dataset, tokenizer: PreTrainedTokenizer) -> Result[Dataset, StrategyError]:  # noqa: ARG002
        """
        Prepare dataset for CPT.

        TRL SFTTrainer handles tokenization automatically.
        We just validate `text` column exists.
        """
        try:
            logger.info("Preparing dataset for CPT...")

            # Validate format
            validation = self.validate_dataset(dataset)
            if validation.is_failure():
                return validation  # type: ignore

            # TRL handles `text` column natively
            logger.info(f"✅ CPT dataset ready: {len(dataset)} samples")
            return Ok(dataset)

        except Exception as e:
            return Err(StrategyError(message=f"CPT preparation failed: {e!s}", code="CPT_PREPARATION_FAILED"))

    def validate_dataset(self, dataset: Dataset) -> Result[bool, StrategyError]:
        """Validate CPT dataset has 'text' field."""
        if "text" not in dataset.column_names:
            return Err(StrategyError(message="CPT requires 'text' field in dataset", code="CPT_MISSING_TEXT_COLUMN"))
        return Ok(True)

    def get_training_objective(self) -> str:
        return "language_modeling"

    def get_trainer_type(self) -> str:
        """Return TRL trainer type for this strategy."""
        return STRATEGY_SFT  # SFTTrainer works for CPT with text data

    def get_metadata(self) -> StrategyMetadata:
        return StrategyMetadata(
            name="cpt_strategy",
            version="2.0.0",  # Updated for TRL native integration
            description="Continual Pre-Training for domain adaptation",
            strategy_type=STRATEGY_CPT,
            data_format="text",
            objective="language_modeling",
            recommended_use="Domain adaptation with large unlabeled corpus",
        )

    def get_recommended_hyperparameters(self) -> dict:
        return {
            "learning_rate": DEFAULT_LEARNING_RATES[STRATEGY_CPT],
            "num_epochs": DEFAULT_EPOCHS[STRATEGY_CPT],
            "batch_size": DEFAULT_BATCH_SIZES[STRATEGY_CPT],
            "gradient_accumulation_steps": 8,  # CPT-specific: larger effective batch
        }


__all__ = ["CPTStrategy"]
