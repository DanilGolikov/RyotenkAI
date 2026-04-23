"""Registry for reward plugins loaded from the community catalogue."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from src.utils.logger import logger

if TYPE_CHECKING:
    from src.community.loader import LoadedPlugin
    from src.training.reward_plugins.base import RewardPlugin


class RewardPluginRegistry:
    """Name → (plugin_class, manifest) map populated by ``CommunityCatalog``."""

    _registry: ClassVar[dict[str, type[RewardPlugin]]] = {}
    _manifests: ClassVar[dict[str, dict[str, Any]]] = {}

    @classmethod
    def register_from_community(cls, loaded: LoadedPlugin) -> None:
        plugin_id = loaded.manifest.plugin.id
        cls._registry[plugin_id] = loaded.plugin_cls
        cls._manifests[plugin_id] = loaded.manifest.ui_manifest()
        logger.debug("[REWARD_REGISTRY] Registered plugin: %s", plugin_id)

    @classmethod
    def create(cls, name: str, params: dict[str, Any]) -> RewardPlugin:
        if name not in cls._registry:
            available = sorted(cls._registry.keys())
            raise KeyError(
                f"Reward plugin {name!r} is not registered. Available plugins: {available}. "
                "Ensure CommunityCatalog.ensure_loaded() was called."
            )
        return cls._registry[name](params)

    @classmethod
    def get_all(cls) -> dict[str, type[RewardPlugin]]:
        return dict(cls._registry)

    @classmethod
    def list_manifests(cls) -> list[dict[str, Any]]:
        return [dict(manifest) for manifest in cls._manifests.values()]

    @classmethod
    def clear(cls) -> None:
        cls._registry.clear()
        cls._manifests.clear()


__all__ = ["RewardPluginRegistry"]
