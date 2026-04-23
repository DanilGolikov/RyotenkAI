"""Registry for report plugins loaded from the community catalogue.

Report plugins use a different metadata contract from BasePlugin-based
plugins: ``plugin_id`` (string) and ``order`` (int) are class-level
attributes that existed before the community rollout. The community
manifest mirrors them via ``plugin.id`` → ``plugin_id`` and the
``[reports]`` block's ``order`` field → ``order``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from src.community.loader import LoadedPlugin
    from src.reports.plugins.interfaces import IReportBlockPlugin


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
    def register_from_community(cls, loaded: LoadedPlugin) -> None:
        plugin_cls = loaded.plugin_cls
        plugin_id = loaded.manifest.plugin.id
        # Report plugins must declare [reports] order — enforced by
        # PluginManifest._reports_block_matches_kind. The assertion keeps
        # mypy/typing happy even though the validator guarantees truthiness.
        reports_spec = loaded.manifest.reports
        if reports_spec is None:  # pragma: no cover — defensive
            raise ValueError(
                f"report plugin {plugin_id!r} loaded without [reports] block"
            )
        order = reports_spec.order

        plugin_cls.plugin_id = plugin_id  # type: ignore[attr-defined]
        plugin_cls.order = order  # type: ignore[attr-defined]

        existing = cls._registry.get(plugin_id)
        if existing is not None and existing is not plugin_cls:
            raise ValueError(
                f"Report plugin_id {plugin_id!r} is already registered by {existing.__name__!r}."
            )
        cls._registry[plugin_id] = plugin_cls

    @classmethod
    def get_all(cls) -> dict[str, type[Any]]:
        return dict(cls._registry)

    @classmethod
    def clear(cls) -> None:
        cls._registry.clear()


def build_report_plugins() -> list[IReportBlockPlugin]:
    from src.community.catalog import catalog

    catalog.ensure_loaded()

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
