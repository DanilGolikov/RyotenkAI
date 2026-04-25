"""Registry for reward plugins loaded from the community catalogue.

Thin subclass over :class:`PluginRegistry`. Reward plugins take only
``params`` (no thresholds — pass/fail criteria don't apply during
training; the trainer composes the reward signal directly).

Module-level singleton :data:`reward_registry` is what the rest of
the codebase imports.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from src.community.registry_base import PluginRegistry

if TYPE_CHECKING:
    from src.training.reward_plugins.base import RewardPlugin


class RewardPluginRegistry(PluginRegistry["RewardPlugin"]):
    """Reward-kind registry. Plugin ctor expects ``(params)``."""

    _kind: ClassVar[str] = "reward"

    def _make_init_kwargs(self, init_kwargs: dict[str, Any]) -> dict[str, Any]:
        return {"params": dict(init_kwargs.get("params") or {})}


reward_registry = RewardPluginRegistry()


__all__ = ["RewardPluginRegistry", "reward_registry"]
