"""
Plugin Registry for auto-discovery and instantiation.

Provides decorator-based registration and plugin retrieval.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from src.utils.logger import logger

if TYPE_CHECKING:
    from src.data.validation.base import ValidationPlugin


class ValidationPluginRegistry:
    """
    Registry for validation plugins.

    Provides:
    - Decorator-based plugin registration
    - Plugin lookup by name
    - Explicit discovery-driven lifecycle

    Example:
        @ValidationPluginRegistry.register
        class MyPlugin(ValidationPlugin):
            name = "my_plugin"
            # ...

        plugin = ValidationPluginRegistry.get_plugin("my_plugin", config={})
    """

    _plugins: ClassVar[dict[str, type[ValidationPlugin]]] = {}

    @classmethod
    def register(cls, plugin_class: type[ValidationPlugin]) -> type[ValidationPlugin]:
        """
        Decorator to register a plugin by its ``name`` ClassVar.

        The plugin class must set a non-empty ``name`` class variable
        (inherited from BasePlugin). This is the single source of truth
        for the registry key — no duplication needed.

        Example:
            @ValidationPluginRegistry.register
            class MinSamplesValidator(ValidationPlugin):
                name = "min_samples"
        """
        name: str = plugin_class.name
        if not name:
            raise ValueError(
                f"ValidationPlugin subclass {plugin_class.__name__!r} must define a non-empty 'name' ClassVar."
            )
        if name in cls._plugins:
            logger.warning("[VALIDATION_REGISTRY] Plugin %r already registered, overwriting", name)
        cls._plugins[name] = plugin_class
        logger.debug("[VALIDATION_REGISTRY] Registered plugin: %s", name)
        return plugin_class

    @classmethod
    def get_plugin(
        cls,
        name: str,
        params: dict[str, Any] | None = None,
        thresholds: dict[str, Any] | None = None,
    ) -> ValidationPlugin:
        """
        Get plugin instance by name.

        Args:
            name: Plugin name
            params: Plugin runtime and execution settings
            thresholds: Plugin pass/fail criteria

        Returns:
            Plugin instance

        Raises:
            KeyError: If plugin not found
        """
        if name not in cls._plugins:
            available = ", ".join(cls._plugins.keys())
            raise KeyError(
                f"Plugin '{name}' not found. Available plugins: {available}. "
                "Run validation plugin discovery before lookup."
            )

        plugin_class = cls._plugins[name]
        return plugin_class(params, thresholds)

    @classmethod
    def list_plugins(cls) -> list[str]:
        """
        List all registered plugin names.

        Returns:
            List of plugin names
        """
        return list(cls._plugins.keys())

    @classmethod
    def clear(cls) -> None:
        """Clear all registered plugins (for testing)."""
        cls._plugins.clear()


__all__ = ["ValidationPluginRegistry"]
