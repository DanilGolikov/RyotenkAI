"""
CoT Strategy - Chain-of-Thought Fine-Tuning

Multi-step reasoning through explicit reasoning traces.

Use when:
- Need reasoning capabilities
- Have CoT examples with reasoning steps
- Want interpretable reasoning

Data format:
- `messages` with reasoning in content, OR
- `instruction`, `reasoning`/`chain_of_thought`, `answer` fields

Objective: Reasoning with explicit steps

Note: CoT has unique formatting logic (think/answer tags)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.constants import DEFAULT_BATCH_SIZES, DEFAULT_EPOCHS, DEFAULT_LEARNING_RATES, STRATEGY_COT, STRATEGY_SFT
from src.training.strategies.base import StrategyMetadata, TrainingStrategy
from src.utils.logger import logger
from src.utils.result import Err, Ok, Result, StrategyError

if TYPE_CHECKING:
    from datasets import Dataset
    from transformers import PreTrainedTokenizer


class CoTStrategy(TrainingStrategy):
    """
    Chain-of-Thought Fine-Tuning Strategy.

    Unlike SFT/CPT, CoT has unique formatting requirements:
    - Wraps reasoning in <think></think> tags
    - Wraps answer in <answer></answer> tags
    - This format teaches model to "think out loud"

    Accepts two formats:
    1. `messages` - ChatML with reasoning in assistant content (preferred)
    2. `instruction/reasoning/answer` - explicit fields (legacy)
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
        Prepare dataset for CoT.

        Formats data with think/answer tags for explicit reasoning.
        """
        try:
            logger.info("Preparing dataset for CoT...")

            # Validate format
            validation = self.validate_dataset(dataset)
            if validation.is_failure():
                return validation  # type: ignore

            # Format 1: ChatML messages (preferred - TRL handles it)
            if "messages" in dataset.column_names:
                logger.info(f"✅ CoT dataset ready (messages format): {len(dataset)} samples")
                return Ok(dataset)

            # Format 2: Explicit instruction/reasoning/answer fields
            # Need to convert to text with think/answer tags
            def format_cot(examples):
                texts = []
                for i in range(len(examples["instruction"])):
                    inst = examples["instruction"][i]
                    reasoning = examples.get("reasoning", examples.get("chain_of_thought", [""]))[i]
                    answer = examples["answer"][i]

                    # Format with explicit think/answer tags
                    text = (
                        f"### Instruction:\n{inst}\n\n<think>\n{reasoning}\n</think>\n\n<answer>\n{answer}\n</answer>"
                    )
                    texts.append(text)

                return {"text": texts}

            formatted = dataset.map(format_cot, batched=True)
            logger.info(f"✅ CoT dataset prepared: {len(formatted)} samples")
            return Ok(formatted)

        except Exception as e:
            return Err(StrategyError(message=f"CoT preparation failed: {e!s}", code="COT_PREPARATION_FAILED"))

    def validate_dataset(self, dataset: Dataset) -> Result[bool, StrategyError]:
        """Validate CoT dataset has required fields."""
        # Format 1: ChatML messages
        if "messages" in dataset.column_names:
            return Ok(True)

        # Format 2: Explicit fields
        required = ["instruction", "answer"]
        has_reasoning = any(f in dataset.column_names for f in ["reasoning", "chain_of_thought"])

        if not all(f in dataset.column_names for f in required):
            return Err(
                StrategyError(
                    message="CoT requires 'messages' OR ('instruction' + 'answer')",
                    code="COT_MISSING_REQUIRED_COLUMNS",
                )
            )

        if not has_reasoning:
            return Err(
                StrategyError(
                    message="CoT requires 'reasoning' or 'chain_of_thought' field",
                    code="COT_MISSING_REASONING_COLUMN",
                )
            )

        return Ok(True)

    def get_training_objective(self) -> str:
        return "chain_of_thought_learning"

    def get_trainer_type(self) -> str:
        """Return TRL trainer type for this strategy."""
        return STRATEGY_SFT  # SFTTrainer works for CoT

    def get_metadata(self) -> StrategyMetadata:
        return StrategyMetadata(
            name="cot_strategy",
            version="2.0.0",  # Updated for TRL integration
            description="Chain-of-Thought training for reasoning",
            strategy_type=STRATEGY_COT,
            data_format="messages OR instruction/reasoning/answer",
            objective="reasoning_with_steps",
            recommended_use="Complex reasoning tasks (math, logic, code)",
            dependencies={"performance_gain": "+8-15% on reasoning tasks"},
        )

    def get_recommended_hyperparameters(self) -> dict:
        return {
            "learning_rate": DEFAULT_LEARNING_RATES[STRATEGY_COT],
            "num_epochs": DEFAULT_EPOCHS[STRATEGY_COT],
            "batch_size": DEFAULT_BATCH_SIZES[STRATEGY_COT],
            "max_seq_length": 4096,  # CoT-specific: needs longer context
        }


__all__ = ["CoTStrategy"]
