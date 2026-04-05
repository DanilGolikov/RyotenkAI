"""
BaseRLStrategy — shared foundation for GRPO / SAPO and any future online-RL strategies.

Responsibilities:
- Declares requires_reward_plugin = True and requires_reference_dataset = True
  so that TrainerFactory can resolve reward plugins without hardcoded strategy names.
- Provides get_trainer_class / get_config_class (both return GRPOTrainer / GRPOConfig).
- Provides _base_rl_config_kwargs() with the common GRPO-family hyperparams.
- Provides validate_dataset for the canonical prompt-based dataset contract.
- Accepts an optional schema_extractor callable (injected at construction time) so
  that domain-specific schema extraction (e.g. HelixQL) can be plugged in without
  importing domain code from within the core training package.

Dataset contract: dataset must arrive with `prompt` column (canonical TRL format).
Conversion from other formats (e.g. messages) is the responsibility of the dataset owner.
"""

from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING, Any

from src.training.constants import COL_PROMPT
from src.training.strategies.base import TrainingStrategy
from src.utils.logger import logger
from src.utils.result import Err, Ok, Result, StrategyError

if TYPE_CHECKING:
    from collections.abc import Callable

    from datasets import Dataset


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
    # Trainer kwargs — reward_funcs are injected via RewardPlugin, not here
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Dataset contract — prompt column required (canonical TRL format)
    # ------------------------------------------------------------------

    def validate_dataset(self, dataset: Dataset) -> Result[bool, StrategyError]:
        """Validate RL dataset has required TRL column."""
        columns = dataset.column_names or []

        if COL_PROMPT not in columns:
            return Err(
                StrategyError(
                    message=f"Dataset must contain '{COL_PROMPT}' column for RL training (GRPOTrainer requirement)",
                    code="RL_MISSING_PROMPT_COLUMN",
                )
            )

        logger.debug("[BaseRL] Dataset format validated: '%s' column present", COL_PROMPT)
        return Ok(True)


__all__ = ["BaseRLStrategy"]
