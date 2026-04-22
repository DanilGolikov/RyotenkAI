"""Registry for validation plugins loaded from the community catalogue."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from src.utils.logger import logger

if TYPE_CHECKING:
    from src.community.loader import LoadedPlugin
    from src.data.validation.base import ValidationPlugin


class ValidationPluginRegistry:
    """Name → (plugin_class, manifest) map populated by ``CommunityCatalog``."""

    _plugins: ClassVar[dict[str, type[ValidationPlugin]]] = {}
    _manifests: ClassVar[dict[str, dict[str, Any]]] = {}

    @classmethod
    def register_from_community(cls, loaded: LoadedPlugin) -> None:
        plugin_id = loaded.manifest.plugin.id
        if plugin_id in cls._plugins:
            logger.warning(
                "[VALIDATION_REGISTRY] Plugin %r already registered, overwriting", plugin_id
            )
        cls._plugins[plugin_id] = loaded.plugin_cls
        cls._manifests[plugin_id] = loaded.manifest.ui_manifest()
        logger.debug("[VALIDATION_REGISTRY] Registered plugin: %s", plugin_id)

    @classmethod
    def get_plugin(
        cls,
        name: str,
        params: dict[str, Any] | None = None,
        thresholds: dict[str, Any] | None = None,
    ) -> ValidationPlugin:
        if name not in cls._plugins:
            available = ", ".join(cls._plugins.keys())
            raise KeyError(
                f"Plugin '{name}' not found. Available plugins: {available}. "
                "Ensure CommunityCatalog.ensure_loaded() was called."
            )
        plugin_class = cls._plugins[name]
        return plugin_class(params, thresholds)

    @classmethod
    def list_plugins(cls) -> list[str]:
        return list(cls._plugins.keys())

    @classmethod
    def list_manifests(cls) -> list[dict[str, Any]]:
        return [dict(manifest) for manifest in cls._manifests.values()]

    @classmethod
    def clear(cls) -> None:
        cls._plugins.clear()
        cls._manifests.clear()


__all__ = ["ValidationPluginRegistry"]
