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

from src.constants import DEFAULT_BATCH_SIZES, DEFAULT_EPOCHS, DEFAULT_LEARNING_RATES, STRATEGY_SFT
from src.training.constants import COL_INSTRUCTION
from src.training.strategies.base import StrategyMetadata, TrainingStrategy
from src.utils.logger import logger
from src.utils.result import Err, Ok, Result, StrategyError

if TYPE_CHECKING:
    from datasets import Dataset
    from transformers import PreTrainedTokenizer


class SFTStrategy(TrainingStrategy):
    """
    Supervised Fine-Tuning Strategy.

    Simplified for TRL integration:
    - TRL applies chat_template automatically for `messages` format
    - TRL uses `text` directly for pre-formatted data
    - Strategy only validates format, no preprocessing needed
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
        Prepare dataset for SFT.

        TRL SFTTrainer handles formatting automatically:
        - `messages` column → applies tokenizer.apply_chat_template()
        - `text` column → uses directly

        We just validate and pass through.
        """
        try:
            logger.info("Preparing dataset for SFT...")

            # Validate format
            validation = self.validate_dataset(dataset)
            if validation.is_failure():
                return validation  # type: ignore

            # TRL handles both formats natively!
            if "messages" in dataset.column_names:
                logger.info(f"✅ SFT dataset ready (messages format): {len(dataset)} samples")
                return Ok(dataset)

            if "text" in dataset.column_names:
                logger.info(f"✅ SFT dataset ready (text format): {len(dataset)} samples")
                return Ok(dataset)

            # Legacy: instruction/response format → convert to text
            if COL_INSTRUCTION in dataset.column_names:
                logger.info("Converting instruction format to text...")

                def format_instruction(examples):
                    texts = []
                    for i in range(len(examples[COL_INSTRUCTION])):
                        inst = examples[COL_INSTRUCTION][i]
                        resp = examples.get("response", examples.get("output", [""]))[i]
                        text = f"### Instruction:\n{inst}\n\n### Response:\n{resp}"
                        texts.append(text)
                    return {"text": texts}

                formatted = dataset.map(format_instruction, batched=True)
                logger.info(f"✅ SFT dataset prepared (instruction→text): {len(formatted)} samples")
                return Ok(formatted)

            return Err(
                StrategyError(
                    message="SFT requires 'messages', 'text', or 'instruction' field",
                    code="SFT_MISSING_REQUIRED_COLUMN",
                )
            )

        except Exception as e:
            return Err(StrategyError(message=f"SFT preparation failed: {e!s}", code="SFT_PREPARATION_FAILED"))

    def validate_dataset(self, dataset: Dataset) -> Result[bool, StrategyError]:
        """Validate SFT dataset structure."""
        has_messages = "messages" in dataset.column_names
        has_text = "text" in dataset.column_names
        has_instruction = COL_INSTRUCTION in dataset.column_names

        if not (has_messages or has_text or has_instruction):
            return Err(
                StrategyError(
                    message="SFT requires 'messages', 'text', or 'instruction' field",
                    code="SFT_MISSING_REQUIRED_COLUMN",
                )
            )

        return Ok(True)

    def get_training_objective(self) -> str:
        return "supervised_learning"

    def get_trainer_type(self) -> str:
        """Return TRL trainer type for this strategy."""
        return STRATEGY_SFT

    def get_recommended_hyperparameters(self) -> dict:
        return {
            "learning_rate": DEFAULT_LEARNING_RATES[STRATEGY_SFT],
            "num_epochs": DEFAULT_EPOCHS[STRATEGY_SFT],
            "batch_size": DEFAULT_BATCH_SIZES[STRATEGY_SFT],
        }

    def get_metadata(self) -> StrategyMetadata:
        return StrategyMetadata(
            name="sft_strategy",
            version="2.0.0",  # Updated for TRL native integration
            description="Supervised Fine-Tuning for instruction following",
            strategy_type=STRATEGY_SFT,
            data_format="messages (ChatML) or text",
            objective="supervised_learning",
            recommended_use="Standard instruction fine-tuning",
        )


__all__ = ["SFTStrategy"]
