from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from src.reports.plugins.interfaces import IReportBlockPlugin


def _validate_plugin_contract(plugin_cls: type[Any]) -> None:
    plugin_id = getattr(plugin_cls, "plugin_id", "")
    order = getattr(plugin_cls, "order", None)

    if not isinstance(plugin_id, str) or not plugin_id.strip():
        raise ValueError(f"Report plugin {plugin_cls.__name__!r} must define a non-empty 'plugin_id'")
    if not isinstance(order, int):
        raise ValueError(f"Report plugin {plugin_cls.__name__!r} must define integer 'order'")


def _validate_plugin_instances(plugins: list[IReportBlockPlugin]) -> None:
    plugin_ids = [plugin.plugin_id for plugin in plugins]
    if len(set(plugin_ids)) != len(plugin_ids):
        raise ValueError(f"Duplicate report plugin_id detected: {plugin_ids}")

    orders = [plugin.order for plugin in plugins]
    if len(set(orders)) != len(orders):
        raise ValueError(f"Duplicate report plugin order detected: {orders}")


class ReportPluginRegistry:
    _registry: ClassVar[dict[str, type[Any]]] = {}

    @classmethod
    def register(cls, plugin_cls: type[Any]) -> type[Any]:
        _validate_plugin_contract(plugin_cls)

        plugin_id = str(plugin_cls.plugin_id)
        existing = cls._registry.get(plugin_id)
        if existing is not None and existing is not plugin_cls:
            raise ValueError(
                f"Report plugin_id {plugin_id!r} is already registered by {existing.__name__!r}. "
                "Each report plugin must have a unique plugin_id."
            )

        cls._registry[plugin_id] = plugin_cls
        return plugin_cls

    @classmethod
    def get_all(cls) -> dict[str, type[Any]]:
        return dict(cls._registry)

    @classmethod
    def clear(cls) -> None:
        cls._registry.clear()


def build_report_plugins() -> list[IReportBlockPlugin]:
    plugins = [
        plugin_cls()
        for _, plugin_cls in sorted(
            ReportPluginRegistry.get_all().items(),
            key=lambda item: int(item[1].order),
        )
    ]
    _validate_plugin_instances(plugins)
    return plugins


__all__ = ["ReportPluginRegistry", "build_report_plugins"]
