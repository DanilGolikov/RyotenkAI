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
    from src.config.secrets.model import Secrets
    from src.reports.plugins.interfaces import IReportBlockPlugin, ReportPlugin


class ReportPluginRegistry(PluginRegistry["ReportPlugin"]):
    """Report-kind registry. Plugin ctor takes no arguments.

    ``plugin_id`` / ``title`` are manifest-backed properties on
    :class:`ReportPlugin`; the registry doesn't mirror anything onto the
    class — the loader's ``_attach_community_metadata`` already attaches
    ``_community_manifest`` and the properties read from it.
    """

    _kind: ClassVar[str] = "reports"

    def _make_init_kwargs(self, init_kwargs: dict[str, Any]) -> dict[str, Any]:
        # Reports take no kwargs; ignore anything passed and warn-by-error
        # if the caller accidentally provided some, to surface bugs.
        if init_kwargs:
            raise TypeError(
                f"reports plugin instantiation does not accept init_kwargs; "
                f"got {list(init_kwargs.keys())}"
            )
        return {}


report_registry = ReportPluginRegistry()


def build_report_plugins(
    sections: Sequence[str] | None = None,
    *,
    secrets: Secrets | None = None,
) -> list[IReportBlockPlugin]:
    """Return ordered plugin instances for the given section list.

    - ``sections=None`` → use :data:`DEFAULT_REPORT_SECTIONS`.
    - Unknown plugin id in ``sections`` → :class:`ValueError` listing
      both the unknown ids and the set of available ids.
    - Duplicate plugin id in ``sections`` → :class:`ValueError` (catches
      config typos like ``[a, b, a]``).
    - ``order`` is assigned from the list position (× 10 for visual
      breathing room in debug output) on each **instance** after
      construction, so concurrent uses of the same plugin class see
      independent orderings (no class-level mutation).

    ``secrets`` — optional. When provided, an ``RPRT_*`` resolver is
    built and threaded into ``registry.instantiate(...)`` so report
    plugins that declare required secrets get them auto-injected.
    None of the shipped community/reports/ plugins need this today, but
    the wiring matches the validation/evaluation/reward kinds for
    consistency and forward-compat.
    """
    from src.community.catalog import catalog
    from src.reports.plugins.secrets import SecretsResolver as ReportSecretsResolver

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

    resolver = ReportSecretsResolver(secrets) if secrets is not None else None

    plugins: list[IReportBlockPlugin] = []
    for idx, plugin_id in enumerate(ordered_ids):
        instance = report_registry.instantiate(plugin_id, resolver=resolver)
        # ``order`` is per-instance state on ``ReportPlugin`` — write to the
        # ``_order`` slot the property reads from, so the composer's
        # ``.order`` access returns the section-list position. Each
        # ``instantiate`` call returns a fresh object, so this never bleeds
        # across runs.
        instance._order = idx * 10  # type: ignore[attr-defined]
        plugins.append(instance)
    return plugins


__all__ = ["ReportPluginRegistry", "report_registry", "build_report_plugins"]
