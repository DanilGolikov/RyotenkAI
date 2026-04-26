"""Pure helpers around ``providers.<name>.training`` config blocks.

The provider configuration tree is queried by both
:class:`DependencyInstaller` (to find the docker image to verify) and
:class:`TrainingLauncher` (to choose between Docker and cloud spawn
paths). These helpers are pure functions taking ``config`` explicitly
so each component injects its own state — no shared mutable singleton.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.constants import PROVIDER_SINGLE_NODE

if TYPE_CHECKING:
    from src.utils.config import PipelineConfig


def get_active_provider_name(config: PipelineConfig) -> str:
    """Best-effort active provider name.

    PipelineConfig provides ``get_active_provider_name()`` in production;
    unit tests may pass MagicMock-like configs whose accessors raise.
    Falls back to ``training.provider`` and then to ``single_node``.
    """
    try:
        return config.get_active_provider_name()
    except Exception:
        training = getattr(config, "training", None)
        name = getattr(training, "provider", None) if training else None
        if isinstance(name, str) and name:
            return name
        return PROVIDER_SINGLE_NODE


def is_single_node_provider(config: PipelineConfig) -> bool:
    return get_active_provider_name(config) == PROVIDER_SINGLE_NODE


def get_provider_training_cfg(config: PipelineConfig, provider_name: str) -> dict[str, Any]:
    """Best-effort ``providers.<provider_name>.training`` as a raw dict.

    Defensive: unit tests inject MagicMock configs; provider config may
    be absent. Falls back to the default provider config when the named
    provider is not found.
    """
    provider_cfg_obj: Any
    try:
        provider_cfg_obj = config.get_provider_config(provider_name)
    except (AttributeError, KeyError, ValueError, TypeError):
        try:
            provider_cfg_obj = config.get_provider_config()
        except (AttributeError, KeyError, ValueError, TypeError):
            provider_cfg_obj = {}

    provider_cfg = provider_cfg_obj if isinstance(provider_cfg_obj, dict) else {}
    training_cfg = provider_cfg.get("training")
    return training_cfg if isinstance(training_cfg, dict) else {}


def get_single_node_training_cfg(config: PipelineConfig) -> dict[str, Any]:
    return get_provider_training_cfg(config, "single_node")


def get_cloud_training_cfg(config: PipelineConfig) -> dict[str, Any]:
    return get_provider_training_cfg(config, get_active_provider_name(config))


__all__ = [
    "get_active_provider_name",
    "get_cloud_training_cfg",
    "get_provider_training_cfg",
    "get_single_node_training_cfg",
    "is_single_node_provider",
]
