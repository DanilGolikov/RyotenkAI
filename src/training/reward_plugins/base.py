from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from src.utils.plugin_base import BasePlugin

if TYPE_CHECKING:
    from datasets import Dataset

    from src.utils.config import PipelineConfig, StrategyPhaseConfig


class RewardPlugin(BasePlugin, ABC):
    """
    Generic reward plugin contract for GRPO/SAPO-like trainers.

    name / priority / version — inherited from BasePlugin.
    """

    def __init__(self, params: dict[str, Any]):
        self.params = params
        self._validate_params()

    def _validate_params(self) -> None:
        return None

    @abstractmethod
    def build_trainer_kwargs(
        self,
        *,
        train_dataset: Dataset,
        phase_config: StrategyPhaseConfig,
        pipeline_config: PipelineConfig,
    ) -> dict[str, Any]:
        """
        Return extra kwargs for the trainer, e.g. reward_funcs / reward_weights.
        """


__all__ = ["RewardPlugin"]
