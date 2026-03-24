"""
GRPO (Group Relative Policy Optimization) strategy.
"""

from __future__ import annotations

from typing import Any

from src.constants import STRATEGY_GRPO
from src.training.strategies.base import StrategyMetadata
from src.training.strategies.base_rl import BaseRLStrategy


class GRPOStrategy(BaseRLStrategy):
    """
    Plain GRPO strategy.

    Extends BaseRLStrategy directly — no SAPO inheritance, no fragile super() + key deletion.
    Only difference from SAPO: loss_type = "grpo", no SAPO temperature fields.
    """

    def build_config_kwargs(self, hp: Any, **kwargs: Any) -> dict[str, Any]:  # noqa: ARG002
        config_kwargs: dict[str, Any] = {"loss_type": STRATEGY_GRPO}
        config_kwargs.update(self._base_rl_config_kwargs(hp))
        return config_kwargs

    def get_metadata(self) -> StrategyMetadata:
        return StrategyMetadata(
            name="Group Relative Policy Optimization",
            version="1.0.0",
            description="Reinforcement Learning with Group Relative Policy Optimization (GRPO)",
            strategy_type=STRATEGY_GRPO,
            data_format="prompt_only_with_reference_answer",
            objective="compiler_backed_reinforcement_learning",
            recommended_use="Baseline online RL phase after SFT",
            dependencies={"trl": ">=0.26.0"},
        )

    def get_trainer_type(self) -> str:
        return STRATEGY_GRPO

    def get_training_objective(self) -> str:
        return "group_relative_policy_optimization"


__all__ = ["GRPOStrategy"]
