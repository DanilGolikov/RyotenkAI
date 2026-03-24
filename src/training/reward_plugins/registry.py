from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from src.training.reward_plugins.base import RewardPlugin


class RewardPluginRegistry:
    """
    Registry for reward plugins.

    Plugins self-register via the ``@RewardPluginRegistry.register`` decorator.
    The registry key is read from ``plugin_cls.name`` (inherited from BasePlugin) —
    no string duplication needed.

    Example:
        @RewardPluginRegistry.register
        class MyRewardPlugin(RewardPlugin):
            name = "my_reward"
            ...
    """

    _registry: ClassVar[dict[str, type[RewardPlugin]]] = {}

    @classmethod
    def register(cls, plugin_cls: type[RewardPlugin]) -> type[RewardPlugin]:
        """Decorator: register a plugin class by its ``name`` ClassVar."""
        name: str = plugin_cls.name
        if not name:
            raise ValueError(f"RewardPlugin subclass {plugin_cls.__name__!r} must define a non-empty 'name' ClassVar.")
        cls._registry[name] = plugin_cls
        return plugin_cls

    @classmethod
    def create(cls, name: str, params: dict[str, Any]) -> RewardPlugin:
        if name not in cls._registry:
            available = sorted(cls._registry.keys())
            raise KeyError(
                f"Reward plugin {name!r} is not registered. Available plugins: {available}. "
                "Run reward plugin discovery before lookup."
            )
        return cls._registry[name](params)

    @classmethod
    def get_all(cls) -> dict[str, type[RewardPlugin]]:
        return dict(cls._registry)

    @classmethod
    def clear(cls) -> None:
        cls._registry.clear()


__all__ = ["RewardPluginRegistry"]
