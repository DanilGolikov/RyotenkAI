"""
SAPO (Soft Adaptive Policy Optimization) strategy.
"""

from __future__ import annotations

from typing import Any

from src.constants import STRATEGY_SAPO
from src.training.strategies.base import StrategyMetadata
from src.training.strategies.base_rl import BaseRLStrategy


class SAPOStrategy(BaseRLStrategy):
    """
    SAPO (Soft Adaptive Policy Optimization) strategy implementation.

    Uses GRPOTrainer with loss_type="sapo".
    Extends BaseRLStrategy: inherits GRPOTrainer/GRPOConfig, shared dataset contract,
    and reward-plugin plumbing.

    Adds SAPO-specific temperature parameters (sapo_temperature_pos / neg).
    """

    def build_config_kwargs(self, hp: Any, **kwargs: Any) -> dict[str, Any]:  # noqa: ARG002
        config_kwargs: dict[str, Any] = {"loss_type": STRATEGY_SAPO}
        config_kwargs.update(self._base_rl_config_kwargs(hp))

        if hp.sapo_temperature_pos is not None:
            config_kwargs["temperature"] = hp.sapo_temperature_pos
            config_kwargs["sapo_temperature_pos"] = hp.sapo_temperature_pos
        if hp.sapo_temperature_neg is not None:
            config_kwargs["sapo_temperature_neg"] = hp.sapo_temperature_neg

        return config_kwargs

    def get_metadata(self) -> StrategyMetadata:
        return StrategyMetadata(
            name="Soft Adaptive Policy Optimization",
            version="1.0.0",
            description="Reinforcement Learning with Soft Adaptive Policy Optimization (SAPO)",
            strategy_type=STRATEGY_SAPO,
            data_format="prompt_only_with_reference_answer",
            objective="compiler_backed_reinforcement_learning",
            recommended_use="Alignment phase after SFT",
            dependencies={"trl": ">=0.26.0"},
        )

    def get_trainer_type(self) -> str:
        return STRATEGY_SAPO

    def get_training_objective(self) -> str:
        return "soft_adaptive_policy_optimization"


__all__ = ["SAPOStrategy"]
