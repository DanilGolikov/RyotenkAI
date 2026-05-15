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

from ryotenkai_pod.trainer.strategies.base import StrategyMetadata, TrainingStrategy
from ryotenkai_shared.constants import STRATEGY_DPO
from ryotenkai_shared.errors import DatasetValidationFailedError
from ryotenkai_shared.utils.logger import logger

if TYPE_CHECKING:
    from datasets import Dataset


class DPOStrategy(TrainingStrategy):
    """
    Direct Preference Optimization Strategy.

    DPO trains the model to prefer "chosen" responses over "rejected" ones
    without needing a separate reward model (unlike RLHF).

    Accepts canonical TRL format:
    - `chosen`: preferred response (list of messages or plain text)
    - `rejected`: dispreferred response (list of messages or plain text)

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
        """Return ref_model if explicitly provided; otherwise TRL handles it."""
        trainer_kwargs: dict[str, Any] = {}
        ref_model = kwargs.get("ref_model")

        if ref_model is not None:
            trainer_kwargs["ref_model"] = ref_model
            logger.debug("[DPO] Using explicit reference model")

        return trainer_kwargs

    def post_build_config_hook(self, config: Any, **context: Any) -> None:
        """No-op: TRL DPOTrainer handles PeftModel reference logprobs natively.

        When a PeftModel is passed without ref_model, TRL temporarily disables the
        adapter to compute reference logprobs. No manual adapter name wiring needed.
        """

    def validate_dataset(self, dataset: Dataset) -> None:
        """Validate DPO dataset has required TRL columns.

        Raises:
            DatasetValidationFailedError: When 'chosen' or 'rejected' column is missing.
        """
        columns = dataset.column_names or []

        if "chosen" not in columns:
            raise DatasetValidationFailedError(
                detail="DPO requires 'chosen' column with preferred responses",
                context={
                    "legacy_code": "DPO_MISSING_CHOSEN_COLUMN",
                    "strategy": STRATEGY_DPO,
                    "missing_column": "chosen",
                    "available_columns": list(columns),
                },
            )

        if "rejected" not in columns:
            raise DatasetValidationFailedError(
                detail="DPO requires 'rejected' column with dispreferred responses",
                context={
                    "legacy_code": "DPO_MISSING_REJECTED_COLUMN",
                    "strategy": STRATEGY_DPO,
                    "missing_column": "rejected",
                    "available_columns": list(columns),
                },
            )

    def get_training_objective(self) -> str:
        return "preference_optimization"

    def get_trainer_type(self) -> str:
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


__all__ = ["DPOStrategy"]
