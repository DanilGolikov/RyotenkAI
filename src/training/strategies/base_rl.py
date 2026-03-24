"""
BaseRLStrategy — shared foundation for GRPO / SAPO and any future online-RL strategies.

Responsibilities:
- Declares requires_reward_plugin = True and requires_reference_dataset = True
  so that TrainerFactory can resolve reward plugins without hardcoded strategy names.
- Provides get_trainer_class / get_config_class (both return GRPOTrainer / GRPOConfig).
- Provides _base_rl_config_kwargs() with the common GRPO-family hyperparams.
- Provides validate_dataset / prepare_dataset for the shared prompt+reference dataset
  contract used by all online-RL strategies.
- Accepts an optional schema_extractor callable (injected at construction time) so
  that domain-specific schema extraction (e.g. HelixQL) can be plugged in without
  importing domain code from within the core training package.
"""

from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING, Any

from src.training.constants import COL_MESSAGES, COL_PROMPT, COL_REFERENCE_ANSWER, COL_SCHEMA_CONTEXT
from src.training.strategies.base import TrainingStrategy
from src.utils.logger import logger
from src.utils.result import Err, Ok, Result, StrategyError
from src.utils.text_utils import extract_nested_text

if TYPE_CHECKING:
    from collections.abc import Callable

    from datasets import Dataset
    from transformers import PreTrainedTokenizer


def _extract_message_text(messages: list[Any], role: str) -> str:
    from collections.abc import Mapping

    for message in messages:
        if isinstance(message, Mapping) and message.get("role") == role:
            return str(message.get("content", "") or "")
    return ""


class BaseRLStrategy(TrainingStrategy, ABC):
    """
    Shared base for all online-RL strategies (GRPO, SAPO, and future variants).

    Subclasses must implement:
    - get_metadata()
    - get_trainer_type()
    - get_training_objective()
    - build_config_kwargs(hp) — must set loss_type and any strategy-specific fields

    Subclasses inherit:
    - requires_reward_plugin = True
    - requires_reference_dataset = True
    - get_trainer_class() → GRPOTrainer
    - get_config_class() → GRPOConfig
    - validate_dataset()
    - prepare_dataset()
    - build_trainer_kwargs()
    """

    @property
    def requires_reward_plugin(self) -> bool:
        return True

    @property
    def requires_reference_dataset(self) -> bool:
        return True

    def __init__(
        self,
        config: Any,
        *,
        schema_extractor: Callable[[str], str] | None = None,
    ) -> None:
        super().__init__(config)
        # Domain-specific schema extraction hook.
        # Defaults to a no-op so that the core strategy carries zero domain knowledge.
        # Inject e.g. ``src.utils.domains.helixql.extract_schema_block`` from outside when needed.
        self._schema_extractor: Callable[[str], str] = schema_extractor or (lambda _: "")

    # ------------------------------------------------------------------
    # Trainer / Config classes (shared by all RL strategies)
    # ------------------------------------------------------------------

    def get_trainer_class(self) -> Any:
        from trl import GRPOTrainer

        return GRPOTrainer

    def get_config_class(self) -> Any:
        from trl import GRPOConfig

        return GRPOConfig

    # ------------------------------------------------------------------
    # Shared config kwargs — common GRPO-family parameters
    # ------------------------------------------------------------------

    def _base_rl_config_kwargs(self, hp: Any) -> dict[str, Any]:
        """Build the common GRPOConfig kwargs shared by all RL strategies."""
        config_kwargs: dict[str, Any] = {}
        if hp.num_generations is not None:
            config_kwargs["num_generations"] = hp.num_generations
        if hp.max_prompt_length is not None:
            config_kwargs["max_prompt_length"] = hp.max_prompt_length
        if hp.max_completion_length is not None:
            config_kwargs["max_completion_length"] = hp.max_completion_length
        if hp.beta is not None:
            config_kwargs["beta"] = hp.beta
        return config_kwargs

    # ------------------------------------------------------------------
    # Trainer kwargs — reward funcs are injected via RewardPlugin, not here
    # ------------------------------------------------------------------

    def build_trainer_kwargs(self, config: Any, **kwargs: Any) -> dict[str, Any]:  # noqa: ARG002
        if "reward_funcs" in kwargs:
            return {"reward_funcs": kwargs["reward_funcs"]}
        return {}

    # ------------------------------------------------------------------
    # Dataset contract — prompt + reference_answer required
    # ------------------------------------------------------------------

    def validate_dataset(self, dataset: Dataset) -> Result[bool, StrategyError]:
        features = dataset.features
        has_prompt = COL_PROMPT in features
        has_messages = COL_MESSAGES in features
        has_reference = COL_REFERENCE_ANSWER in features or has_messages

        if not (has_prompt or has_messages):
            return Err(
                StrategyError(
                    message="Dataset must contain 'prompt' or 'messages' column for RL training",
                    code="RL_MISSING_PROMPT_COLUMN",
                )
            )
        if not has_reference:
            return Err(
                StrategyError(
                    message="Dataset must contain 'reference_answer' or 'messages' column for RL reward",
                    code="RL_MISSING_REFERENCE_ANSWER",
                )
            )
        return Ok(True)

    def prepare_dataset(self, dataset: Dataset, tokenizer: PreTrainedTokenizer) -> Result[Dataset, StrategyError]:
        strategy_name = self.get_trainer_type().upper()
        logger.info("[STRATEGY:%s] Preparing dataset...", strategy_name)

        if COL_PROMPT in dataset.features:
            logger.info("[STRATEGY:%s] Using existing prompt-based dataset", strategy_name)
            if COL_SCHEMA_CONTEXT in dataset.features and COL_REFERENCE_ANSWER in dataset.features:
                return Ok(dataset)
            try:
                prepared_dataset = dataset.map(
                    lambda example: {
                        COL_SCHEMA_CONTEXT: extract_nested_text(example.get(COL_SCHEMA_CONTEXT))
                        or self._schema_extractor(extract_nested_text(example.get(COL_PROMPT))),
                        COL_REFERENCE_ANSWER: extract_nested_text(example.get(COL_REFERENCE_ANSWER))
                        or extract_nested_text(example.get("expected_answer")),
                    }
                )
                return Ok(prepared_dataset)
            except Exception as e:
                return Err(
                    StrategyError(
                        message=f"Failed to enrich RL prompt dataset: {e}",
                        code="RL_PROMPT_ENRICH_FAILED",
                    )
                )

        if COL_MESSAGES in dataset.features:
            logger.info("[STRATEGY:%s] Converting chat samples to prompt-only RL dataset", strategy_name)
            try:
                prepared_dataset = dataset.map(
                    lambda example: self._extract_prompt_payload(example=example, tokenizer=tokenizer)
                )
                return Ok(prepared_dataset)
            except Exception as e:
                return Err(
                    StrategyError(
                        message=f"Failed to prepare RL dataset: {e}",
                        code="RL_PREPARATION_FAILED",
                    )
                )

        return Err(
            StrategyError(
                message="Could not find usable prompt in dataset",
                code="RL_NO_USABLE_PROMPT",
            )
        )

    def _extract_prompt_payload(
        self,
        *,
        example: Any,
        tokenizer: PreTrainedTokenizer,
    ) -> dict[str, str]:
        """Convert a ChatML messages example into a prompt-only RL training record."""
        from collections.abc import Mapping

        messages = example.get(COL_MESSAGES) if isinstance(example, Mapping) else None
        if not isinstance(messages, list):
            return {
                COL_PROMPT: extract_nested_text(example.get(COL_PROMPT) if isinstance(example, Mapping) else None),
                COL_SCHEMA_CONTEXT: extract_nested_text(
                    example.get(COL_SCHEMA_CONTEXT) if isinstance(example, Mapping) else None
                ),
                COL_REFERENCE_ANSWER: extract_nested_text(
                    example.get(COL_REFERENCE_ANSWER) if isinstance(example, Mapping) else None
                ),
            }

        user_text = _extract_message_text(messages, "user")
        assistant_text = _extract_message_text(messages, "assistant")

        prompt_text = user_text
        if getattr(tokenizer, "chat_template", None):
            try:
                prompt_text = str(
                    tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
                )
            except (AttributeError, IndexError, KeyError, TypeError, ValueError):
                prompt_text = user_text

        return {
            COL_PROMPT: prompt_text,
            COL_SCHEMA_CONTEXT: self._schema_extractor(user_text),
            COL_REFERENCE_ANSWER: assistant_text,
        }


__all__ = ["BaseRLStrategy"]
