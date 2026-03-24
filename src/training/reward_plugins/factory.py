from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.training.reward_plugins.discovery import ensure_reward_plugins_discovered
from src.training.reward_plugins.registry import RewardPluginRegistry

if TYPE_CHECKING:
    from datasets import Dataset

    from src.utils.config import PipelineConfig, StrategyPhaseConfig


def build_reward_plugin_kwargs(
    *,
    train_dataset: Dataset,
    phase_config: StrategyPhaseConfig,
    pipeline_config: PipelineConfig,
) -> dict[str, Any]:
    plugin_name = str(phase_config.params.get("reward_plugin") or "").strip()
    if not plugin_name:
        raise ValueError(
            f"{phase_config.strategy_type.upper()} requires explicit phase params.reward_plugin. "
            "Core training code does not embed domain-specific reward logic."
        )

    reward_params_raw = phase_config.params.get("reward_params", {})
    reward_params = reward_params_raw if isinstance(reward_params_raw, dict) else {}

    ensure_reward_plugins_discovered()

    plugin = RewardPluginRegistry.create(plugin_name, reward_params)
    return plugin.build_trainer_kwargs(
        train_dataset=train_dataset,
        phase_config=phase_config,
        pipeline_config=pipeline_config,
    )


__all__ = ["build_reward_plugin_kwargs"]
