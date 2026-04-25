"""Registry for report plugins loaded from the community catalogue.

Thin subclass over :class:`PluginRegistry`. Report plugins take no
constructor kwargs — they're parameter-free units that the
:class:`ReportComposer` calls with a :class:`ReportPluginContext` at
render time.

Section ordering does NOT live in the manifest — it's decided by the
pipeline config's ``reports.sections`` list (source of truth) or
:data:`DEFAULT_REPORT_SECTIONS` when that's absent.
:func:`build_report_plugins` is the single entry point: give it an
ordered list of plugin ids, it returns the corresponding instances with
their ``order`` attribute assigned from list position.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from src.community.registry_base import PluginRegistry
from src.reports.plugins.defaults import DEFAULT_REPORT_SECTIONS

if TYPE_CHECKING:
    from collections.abc import Sequence

    from src.community.loader import LoadedPlugin
    from src.reports.plugins.interfaces import IReportBlockPlugin, ReportPlugin


class ReportPluginRegistry(PluginRegistry["ReportPlugin"]):
    """Report-kind registry. Plugin ctor takes no arguments.

    The legacy ``plugin_id = ...`` ClassVar pattern continues to work
    because :class:`ReportPlugin` still declares the slot. We additionally
    set it from the manifest at registration time (matching the old
    behaviour) so plugins that omit the ClassVar still get a populated id.
    """

    _kind: ClassVar[str] = "report"

    def register_from_community(self, loaded: LoadedPlugin) -> None:
        plugin_cls = loaded.plugin_cls
        plugin_id = loaded.manifest.plugin.id
        # Mirror manifest id onto the class so ``plugin.plugin_id`` works
        # even if the author omitted the ClassVar declaration.
        plugin_cls.plugin_id = plugin_id  # type: ignore[attr-defined]
        super().register_from_community(loaded)

    def _make_init_kwargs(self, init_kwargs: dict[str, Any]) -> dict[str, Any]:
        # Reports take no kwargs; ignore anything passed and warn-by-error
        # if the caller accidentally provided some, to surface bugs.
        if init_kwargs:
            raise TypeError(
                f"report plugin instantiation does not accept init_kwargs; "
                f"got {list(init_kwargs.keys())}"
            )
        return {}


report_registry = ReportPluginRegistry()


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
    available_ids = report_registry.list_ids()

    ordered_ids: tuple[str, ...] = (
        tuple(sections) if sections is not None else DEFAULT_REPORT_SECTIONS
    )

    duplicates = [pid for pid in ordered_ids if ordered_ids.count(pid) > 1]
    if duplicates:
        raise ValueError(
            f"Duplicate report plugin ids in sections: {sorted(set(duplicates))!r}"
        )

    unknown = [pid for pid in ordered_ids if not report_registry.is_registered(pid)]
    if unknown:
        raise ValueError(
            f"Unknown report plugin ids in reports.sections: {unknown!r}. "
            f"Available plugins: {sorted(available_ids)!r}"
        )

    plugins: list[IReportBlockPlugin] = []
    for idx, plugin_id in enumerate(ordered_ids):
        plugin_cls = report_registry.get_class(plugin_id)
        # ``order`` is part of the IReportBlockPlugin contract — mutate the
        # class so existing instances also see the new value (composer
        # reads ``.order`` during rendering).
        plugin_cls.order = idx * 10  # type: ignore[attr-defined]
        plugins.append(report_registry.instantiate(plugin_id))
    return plugins


__all__ = ["ReportPluginRegistry", "report_registry", "build_report_plugins"]
