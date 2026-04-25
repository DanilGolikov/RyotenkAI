from __future__ import annotations

from typing import TYPE_CHECKING, Any, NamedTuple

from src.community.catalog import catalog
from src.training.reward_plugins.registry import reward_registry
from src.training.reward_plugins.secrets import SecretsResolver as RewardSecretsResolver
from src.utils.logger import logger

if TYPE_CHECKING:
    from datasets import Dataset

    from src.config.secrets.model import Secrets
    from src.training.reward_plugins.base import RewardPlugin
    from src.utils.config import PipelineConfig, StrategyPhaseConfig


class RewardPluginResult(NamedTuple):
    """Structured result from a reward plugin split by routing destination."""

    config_kwargs: dict[str, Any]
    trainer_kwargs: dict[str, Any]
    plugin: RewardPlugin | None = None


def build_reward_plugin_result(
    *,
    train_dataset: Dataset,
    phase_config: StrategyPhaseConfig,
    pipeline_config: PipelineConfig,
    secrets: Secrets | None = None,
) -> RewardPluginResult:
    """Instantiate the reward plugin, run its setup, and return kwargs.

    Lifecycle: create → setup() → build_config_kwargs / build_trainer_kwargs.
    The caller is responsible for calling ``plugin.teardown()`` after training.

    config_kwargs  → merged into the TRL *Config constructor (e.g. reward_weights).
    trainer_kwargs → merged into the TRL Trainer constructor (e.g. reward_funcs).

    ``secrets`` — optional. When passed, a ``RWRD_*`` resolver is built and
    handed to the registry so reward plugins declaring required secrets in
    their manifest get them auto-injected. When ``None``, plugins that need
    secrets fail fast at ``registry.instantiate(...)`` with a clear error.
    """
    plugin_name = str(phase_config.params.get("reward_plugin") or "").strip()
    if not plugin_name:
        raise ValueError(
            f"{phase_config.strategy_type.upper()} requires explicit phase params.reward_plugin. "
            "Core training code does not embed domain-specific reward logic."
        )

    reward_params_raw = phase_config.params.get("reward_params", {})
    reward_params = reward_params_raw if isinstance(reward_params_raw, dict) else {}

    catalog.ensure_loaded()

    resolver = RewardSecretsResolver(secrets) if secrets is not None else None
    plugin = reward_registry.instantiate(
        plugin_name,
        resolver=resolver,
        params=reward_params,
    )

    # Reward broadcast visibility (PR13): the UI's Configure modal saves
    # reward plugin params *once* but the underlying YAML may apply them
    # across multiple training phases that share the same plugin id.
    # Logging the (strategy, plugin, params-keys) triple at each
    # instantiation gives the user a verifiable trail in the run log
    # so a surprising "params also affected the SAPO phase!" outcome is
    # auditable post-hoc. Param values themselves are intentionally NOT
    # logged — they may include secrets through ``reward_params``.
    logger.info(
        "[REWARD_PLUGIN] strategy=%s plugin=%r params=%s",
        phase_config.strategy_type,
        plugin_name,
        sorted(reward_params.keys()),
    )

    logger.info("[REWARD_PLUGIN] Running setup for %r ...", plugin_name)
    plugin.setup()
    logger.info("[REWARD_PLUGIN] Setup complete for %r", plugin_name)

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
        plugin=plugin,
    )


__all__ = ["RewardPluginResult", "build_reward_plugin_result"]
