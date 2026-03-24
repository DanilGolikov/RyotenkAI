"""
DPO Strategy - Direct Preference Optimization

Alignment through preference learning on chosen/rejected pairs.

Use when:
- Need to align model with human preferences
- Have preference data (chosen vs rejected)
- Want to reduce harmful outputs

Data format: `chosen` and `rejected` columns (each is list of messages)
Objective: Preference optimization (maximize log ratio of chosen over rejected)

Critical: Use learning rate 10-100x LOWER than SFT! (5e-6 recommended)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from peft import PeftModel

from src.constants import DEFAULT_BATCH_SIZES, DEFAULT_EPOCHS, DEFAULT_LEARNING_RATES, STRATEGY_DPO
from src.training.strategies.base import StrategyMetadata, TrainingStrategy
from src.utils.logger import logger
from src.utils.result import Err, Ok, Result, StrategyError

if TYPE_CHECKING:
    from datasets import Dataset
    from transformers import PreTrainedTokenizer


class DPOStrategy(TrainingStrategy):
    """
    Direct Preference Optimization Strategy.

    DPO trains the model to prefer "chosen" responses over "rejected" ones
    without needing a separate reward model (unlike RLHF).

    Data format requirements:
    - `chosen`: List of messages (ChatML format) for preferred response
    - `rejected`: List of messages (ChatML format) for dispreferred response

    Example data:
        {
            "chosen": [
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "2+2 equals 4."}
            ],
            "rejected": [
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "I don't know math."}
            ]
        }

    Critical hyperparameters:
    - learning_rate: 5e-6 (10-100x lower than SFT!)
    - beta: 0.1 (controls strength of preference, 0.1-0.5 typical)
    """

    def get_trainer_class(self) -> Any:
        from trl import DPOTrainer

        return DPOTrainer

    def get_config_class(self) -> Any:
        from trl import DPOConfig

        return DPOConfig

    def build_config_kwargs(self, hp: Any, **kwargs: Any) -> dict[str, Any]:  # noqa: ARG002
        return {
            "max_length": hp.max_length,
            "beta": hp.beta if hp.beta is not None else 0.1,
        }

    def build_trainer_kwargs(self, config: Any, **kwargs: Any) -> dict[str, Any]:  # noqa: ARG002
        """Return ref_model if provided. PEFT adapter config mutation is handled by post_build_config_hook."""
        trainer_kwargs: dict[str, Any] = {}
        ref_model = kwargs.get("ref_model")

        if ref_model is not None:
            trainer_kwargs["ref_model"] = ref_model
            logger.debug("[DPO] Using explicit reference model")
        else:
            model = kwargs.get("model")
            if not isinstance(model, PeftModel):
                logger.warning(
                    "[DPO] No reference model provided. DPO will compute reference logprobs internally (slower)."
                )

        return trainer_kwargs

    def post_build_config_hook(self, config: Any, **context: Any) -> None:
        """Configure PEFT adapter names on DPOConfig when a PEFT model is used.

        Called by TrainerFactory after creating the DPOConfig instance.
        Sets model_adapter_name / ref_adapter_name and loads the reference adapter
        if the active model is a PeftModel.
        """
        model = context.get("model")
        ref_model = context.get("ref_model")

        if ref_model is not None or not isinstance(model, PeftModel):
            return

        if not hasattr(model, "peft_config"):
            return

        active_adapter = getattr(model, "active_adapter", None)
        if not active_adapter:
            return

        logger.debug("[DPO] Using PEFT adapter for reference model")
        config.model_adapter_name = "train"
        config.ref_adapter_name = "reference"

        if "reference" not in model.peft_config:
            try:
                adapter_path = model.peft_config[active_adapter].base_model_name_or_path
                if adapter_path:
                    model.load_adapter(adapter_path, adapter_name="reference")
                    logger.info("[DPO] Loaded reference adapter from PEFT")
            except Exception as e:
                logger.warning("[DPO] Could not load reference adapter: %s", e)

    def prepare_dataset(self, dataset: Dataset, _tokenizer: PreTrainedTokenizer) -> Result[Dataset, StrategyError]:
        """
        Prepare dataset for DPO.

        TRL DPOTrainer expects:
        - `chosen`: List[Dict] messages for preferred response
        - `rejected`: List[Dict] messages for dispreferred response

        We validate format and pass through - TRL handles tokenization.
        """
        try:
            logger.info("Preparing dataset for DPO...")

            # Validate format
            validation = self.validate_dataset(dataset)
            if validation.is_failure():
                return validation  # type: ignore

            # TRL DPOTrainer handles formatting natively
            logger.info(f"DPO dataset ready: {len(dataset)} preference pairs")
            return Ok(dataset)

        except Exception as e:
            return Err(StrategyError(message=f"DPO preparation failed: {e!s}", code="DPO_PREPARATION_FAILED"))

    def validate_dataset(self, dataset: Dataset) -> Result[bool, StrategyError]:
        """
        Validate DPO dataset structure.

        Requirements:
        - Must have `chosen` column
        - Must have `rejected` column
        - Each should be list of messages (ChatML format)
        """
        columns = dataset.column_names

        if "chosen" not in columns:
            return Err(
                StrategyError(
                    message="DPO requires 'chosen' column with preferred responses",
                    code="DPO_MISSING_CHOSEN_COLUMN",
                )
            )

        if "rejected" not in columns:
            return Err(
                StrategyError(
                    message="DPO requires 'rejected' column with dispreferred responses",
                    code="DPO_MISSING_REJECTED_COLUMN",
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
                        message="DPO 'chosen' and 'rejected' must be lists of messages",
                        code="DPO_INVALID_MESSAGE_FORMAT",
                    )
                )

            # Check message structure
            if len(chosen) > 0:
                if not isinstance(chosen[0], dict):
                    return Err(
                        StrategyError(
                            message="DPO messages must be dicts with 'role' and 'content'",
                            code="DPO_INVALID_MESSAGE_STRUCTURE",
                        )
                    )
                if "role" not in chosen[0] or "content" not in chosen[0]:
                    return Err(
                        StrategyError(
                            message="DPO messages must have 'role' and 'content' keys",
                            code="DPO_MISSING_MESSAGE_KEYS",
                        )
                    )

        except Exception as e:
            logger.warning(f"Could not validate DPO sample structure: {e}")
            # Continue anyway, TRL will catch format errors

        return Ok(True)

    def get_training_objective(self) -> str:
        return "preference_optimization"

    def get_trainer_type(self) -> str:
        """Return TRL trainer type for this strategy."""
        return STRATEGY_DPO

    def get_metadata(self) -> StrategyMetadata:
        return StrategyMetadata(
            name="dpo_strategy",
            version="1.0.0",
            description="Direct Preference Optimization for alignment",
            strategy_type=STRATEGY_DPO,
            data_format="chosen/rejected message pairs",
            objective="preference_optimization",
            recommended_use="Alignment after SFT, reducing harmful outputs",
            dependencies={"trl": ">=0.8.0"},
        )

    def get_recommended_hyperparameters(self) -> dict:
        """
        Get DPO-specific recommended hyperparameters.

        Critical: DPO requires much lower learning rate than SFT!
        """
        return {
            "learning_rate": DEFAULT_LEARNING_RATES[STRATEGY_DPO],
            "num_epochs": DEFAULT_EPOCHS[STRATEGY_DPO],
            "batch_size": DEFAULT_BATCH_SIZES[STRATEGY_DPO],
            "beta": 0.1,  # DPO-specific: strength of preference
        }


__all__ = ["DPOStrategy"]
