from __future__ import annotations

from typing import TYPE_CHECKING, Any, NamedTuple

from src.training.reward_plugins.discovery import ensure_reward_plugins_discovered
from src.training.reward_plugins.registry import RewardPluginRegistry

if TYPE_CHECKING:
    from datasets import Dataset

    from src.utils.config import PipelineConfig, StrategyPhaseConfig


class RewardPluginResult(NamedTuple):
    """Structured result from a reward plugin split by routing destination."""

    config_kwargs: dict[str, Any]
    trainer_kwargs: dict[str, Any]


def build_reward_plugin_result(
    *,
    train_dataset: Dataset,
    phase_config: StrategyPhaseConfig,
    pipeline_config: PipelineConfig,
) -> RewardPluginResult:
    """Instantiate the reward plugin and return config-level and trainer-level kwargs separately.

    config_kwargs  → merged into the TRL *Config constructor (e.g. reward_weights).
    trainer_kwargs → merged into the TRL Trainer constructor (e.g. reward_funcs).
    """
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
    return RewardPluginResult(
        config_kwargs=plugin.build_config_kwargs(
            train_dataset=train_dataset,
            phase_config=phase_config,
            pipeline_config=pipeline_config,
        ),
        trainer_kwargs=plugin.build_trainer_kwargs(
            train_dataset=train_dataset,
            phase_config=phase_config,
            pipeline_config=pipeline_config,
        ),
    )


__all__ = ["RewardPluginResult", "build_reward_plugin_result"]
