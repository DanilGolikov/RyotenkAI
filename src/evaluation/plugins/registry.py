"""
Evaluator plugin registry with decorator-based auto-registration.

Usage:
    # Register a plugin:
    @EvaluatorPluginRegistry.register
    class MyPlugin(EvaluatorPlugin):
        name = "my_plugin"
        ...

    # Look up a plugin by name:
    cls = EvaluatorPluginRegistry.get("my_plugin")
    plugin = cls(params={}, thresholds={})
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from .base import EvaluatorPlugin


class EvaluatorPluginRegistry:
    """
    Central registry for all evaluator plugins.

    Plugins self-register via the @EvaluatorPluginRegistry.register decorator.
    The registry is populated when plugin modules are imported by the explicit
    discovery lifecycle.
    """

    _registry: ClassVar[dict[str, type[EvaluatorPlugin]]] = {}

    @classmethod
    def register(cls, plugin_cls: type[EvaluatorPlugin]) -> type[EvaluatorPlugin]:
        """Decorator: register a plugin class by its .name attribute."""
        if not plugin_cls.name:
            raise ValueError(
                f"EvaluatorPlugin subclass {plugin_cls.__name__!r} must define a non-empty class variable 'name'."
            )
        if plugin_cls.name in cls._registry:
            existing = cls._registry[plugin_cls.name]
            if existing is not plugin_cls:
                raise ValueError(
                    f"Plugin name {plugin_cls.name!r} is already registered by {existing.__name__!r}. "
                    f"Each plugin must have a unique name."
                )
        cls._registry[plugin_cls.name] = plugin_cls
        return plugin_cls

    @classmethod
    def get(cls, name: str) -> type[EvaluatorPlugin]:
        """
        Get a plugin class by name.

        Raises:
            KeyError: if plugin is not registered.
        """
        if name not in cls._registry:
            available = sorted(cls._registry.keys())
            raise KeyError(
                f"Evaluator plugin {name!r} is not registered. "
                f"Available plugins: {available}. "
                "Run evaluation plugin discovery before lookup."
            )
        return cls._registry[name]

    @classmethod
    def get_all(cls) -> dict[str, type[EvaluatorPlugin]]:
        """Return a copy of the full registry."""
        return dict(cls._registry)

    @classmethod
    def list_manifests(cls) -> list[dict[str, Any]]:
        """Return normalised manifest dicts for every registered plugin."""
        return [plugin_cls.get_manifest() for plugin_cls in cls._registry.values()]

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Return True if a plugin with the given name is registered."""
        return name in cls._registry

    @classmethod
    def clear(cls) -> None:
        cls._registry.clear()


__all__ = [
    "EvaluatorPluginRegistry",
]
