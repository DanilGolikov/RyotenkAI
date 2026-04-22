"""Registry for evaluation plugins loaded from the community catalogue."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from src.utils.logger import logger

if TYPE_CHECKING:
    from src.community.loader import LoadedPlugin

    from .base import EvaluatorPlugin


class EvaluatorPluginRegistry:
    """Name → (plugin_class, manifest) map populated by ``CommunityCatalog``."""

    _registry: ClassVar[dict[str, type[EvaluatorPlugin]]] = {}
    _manifests: ClassVar[dict[str, dict[str, Any]]] = {}

    @classmethod
    def register_from_community(cls, loaded: LoadedPlugin) -> None:
        plugin_id = loaded.manifest.plugin.id
        if plugin_id in cls._registry and cls._registry[plugin_id] is not loaded.plugin_cls:
            raise ValueError(
                f"Evaluator plugin id {plugin_id!r} is already registered by "
                f"{cls._registry[plugin_id].__name__!r}."
            )
        cls._registry[plugin_id] = loaded.plugin_cls
        cls._manifests[plugin_id] = loaded.manifest.ui_manifest()
        logger.debug("[EVALUATOR_REGISTRY] Registered plugin: %s", plugin_id)

    @classmethod
    def get(cls, name: str) -> type[EvaluatorPlugin]:
        if name not in cls._registry:
            available = sorted(cls._registry.keys())
            raise KeyError(
                f"Evaluator plugin {name!r} is not registered. "
                f"Available plugins: {available}. "
                "Ensure CommunityCatalog.ensure_loaded() was called."
            )
        return cls._registry[name]

    @classmethod
    def get_all(cls) -> dict[str, type[EvaluatorPlugin]]:
        return dict(cls._registry)

    @classmethod
    def list_manifests(cls) -> list[dict[str, Any]]:
        return [dict(manifest) for manifest in cls._manifests.values()]

    @classmethod
    def is_registered(cls, name: str) -> bool:
        return name in cls._registry

    @classmethod
    def clear(cls) -> None:
        cls._registry.clear()
        cls._manifests.clear()


__all__ = ["EvaluatorPluginRegistry"]
