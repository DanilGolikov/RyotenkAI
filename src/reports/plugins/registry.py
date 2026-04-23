"""Registry for report plugins loaded from the community catalogue.

The registry itself is a plain id→class map. Section order does NOT live
in the manifest any more — it's decided by the pipeline config's
``reports.sections`` list (source of truth) or the built-in default in
``src/reports/plugins/defaults.py`` when that's absent.

``build_report_plugins`` is the single entry point: give it an ordered
list of plugin ids, it returns the corresponding instances with their
``order`` attribute assigned from the list position. Downstream rendering
(Composer + MarkdownBlockRenderer) still reads ``.order`` — this lets us
keep the Protocol stable while moving control upstream.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from src.reports.plugins.defaults import DEFAULT_REPORT_SECTIONS

if TYPE_CHECKING:
    from collections.abc import Sequence

    from src.community.loader import LoadedPlugin
    from src.reports.plugins.interfaces import IReportBlockPlugin


class ReportPluginRegistry:
    _registry: ClassVar[dict[str, type[Any]]] = {}

    @classmethod
    def register_from_community(cls, loaded: LoadedPlugin) -> None:
        """Register a community-loaded class under its manifest id.

        ``order`` is NOT assigned here — it's applied later from config
        by :func:`build_report_plugins`. This keeps registration
        idempotent across pipeline runs that use different section
        orders.
        """
        plugin_cls = loaded.plugin_cls
        plugin_id = loaded.manifest.plugin.id

        plugin_cls.plugin_id = plugin_id  # type: ignore[attr-defined]

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


def build_report_plugins(
    sections: Sequence[str] | None = None,
) -> list[IReportBlockPlugin]:
    """Return ordered plugin instances for the given section list.

    - ``sections=None`` → use :data:`DEFAULT_REPORT_SECTIONS`.
    - Unknown plugin id in ``sections`` → :class:`ValueError` listing
      both the unknown ids and the set of available ids.
    - Duplicate plugin id in ``sections`` → :class:`ValueError` (catches
      config typos like ``[a, b, a]``).
    - ``order`` is assigned from the list position (× 10 for visual
      breathing room in debug output) and written onto the class before
      instantiation, so downstream code that reads ``plugin.order`` keeps
      working unchanged.
    """
    from src.community.catalog import catalog

    catalog.ensure_loaded()
    registry = ReportPluginRegistry.get_all()

    ordered_ids: tuple[str, ...] = (
        tuple(sections) if sections is not None else DEFAULT_REPORT_SECTIONS
    )

    duplicates = [pid for pid in ordered_ids if ordered_ids.count(pid) > 1]
    if duplicates:
        raise ValueError(
            f"Duplicate report plugin ids in sections: {sorted(set(duplicates))!r}"
        )

    unknown = [pid for pid in ordered_ids if pid not in registry]
    if unknown:
        available = sorted(registry)
        raise ValueError(
            f"Unknown report plugin ids in reports.sections: {unknown!r}. "
            f"Available plugins: {available!r}"
        )

    plugins: list[IReportBlockPlugin] = []
    for idx, plugin_id in enumerate(ordered_ids):
        plugin_cls = registry[plugin_id]
        plugin_cls.order = idx * 10  # type: ignore[attr-defined]
        plugins.append(plugin_cls())
    return plugins


__all__ = ["ReportPluginRegistry", "build_report_plugins"]
