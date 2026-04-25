"""
Plugin interfaces for experiment report generation.

Design goals:
- Explicit contracts with explicit discovery lifecycle.
- Deterministic ordering.
- Fail-open by default: a single plugin failure becomes an error block.

Two complementary abstractions live here:

- :class:`IReportBlockPlugin` — the structural contract every renderer
  consumes (Composer + MarkdownBlockRenderer). Kept as a runtime-checkable
  Protocol so external/test/duck-typed implementations stay usable.
- :class:`ReportPlugin` — the concrete ABC that **community-loaded
  plugins** must inherit from. It plugs into :class:`BasePlugin` (community
  metadata mirroring + secret injection), forces an abstract ``render``
  and exposes ``plugin_id`` / ``title`` as properties backed by the
  manifest (``[plugin].id`` / ``[plugin].name``). ``order`` is per-instance
  state assigned by :func:`build_report_plugins` from the user's
  ``reports.sections`` list — never declared on the class.

Authors no longer write ``plugin_id = "..."`` / ``title = "..."`` /
``order = ...`` on subclasses; the manifest is the single source of truth.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

from src.utils.plugin_base import BasePlugin

if TYPE_CHECKING:
    from logging import Logger

    from src.reports.document.nodes import DocBlock
    from src.reports.domain.entities import ExperimentData
    from src.reports.domain.interfaces import IExperimentDataProvider
    from src.reports.models.report import ExperimentReport

Clock = Callable[[], datetime]
PluginStatus = Literal["ok", "failed"]


@dataclass(frozen=True, slots=True)
class ReportPluginContext:
    """
    Immutable context passed to each report block plugin.
    """

    run_id: str
    data_provider: IExperimentDataProvider
    data: ExperimentData
    report: ExperimentReport
    logger: Logger
    clock: Clock = datetime.now


@dataclass(frozen=True, slots=True)
class ReportBlock:
    """
    One block of the final report.
    """

    block_id: str
    title: str
    order: int
    nodes: list[DocBlock]
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PluginExecutionRecord:
    """
    Observability record for one plugin execution.
    """

    plugin_id: str
    status: PluginStatus
    duration_ms: float
    error_type: str | None = None
    error_message: str | None = None


@runtime_checkable
class IReportBlockPlugin(Protocol):
    """
    Contract for a report block plugin.

    One plugin builds exactly one `ReportBlock`.
    """

    @property
    def plugin_id(self) -> str: ...

    @property
    def title(self) -> str: ...

    @property
    def order(self) -> int: ...

    def render(self, ctx: ReportPluginContext) -> ReportBlock: ...


class ReportPlugin(BasePlugin, ABC):
    """Abstract base for community-loaded report block plugins.

    Inherits :class:`BasePlugin`'s metadata mirroring (``name``, ``version``,
    ``_community_manifest`` etc. — populated by the community loader from
    ``manifest.toml``) and adds the :class:`IReportBlockPlugin` contract on
    top:

    - ``plugin_id`` / ``title`` are read-only properties backed by
      ``manifest.plugin.id`` / ``manifest.plugin.name``. Authors do **not**
      declare them on the subclass — the manifest is the single source of
      truth.
    - ``order`` is per-instance state, assigned by
      :func:`build_report_plugins` from the user's ``reports.sections``
      list position. Default ``0`` lets tests instantiate plugins outside
      the orchestration path without crashing; production paths overwrite
      it before render.

    Subclass example::

        # community/reports/header/plugin.py
        class HeaderBlockPlugin(ReportPlugin):
            def render(self, ctx: ReportPluginContext) -> ReportBlock:
                return ReportBlock(
                    block_id=self.plugin_id,   # → manifest.plugin.id
                    title=self.title,          # → manifest.plugin.name
                    order=self.order,          # → injected by build_report_plugins
                    nodes=[...],
                )
    """

    #: Per-instance render order. Re-assigned by
    #: :func:`build_report_plugins` based on the user's ``reports.sections``
    #: list position; default is fine for ad-hoc rendering and unit tests.
    _order: int = 0

    @property
    def plugin_id(self) -> str:
        """Manifest-backed plugin id. Raises if not loaded via the catalog."""
        manifest = type(self)._community_manifest
        if manifest is None:
            raise RuntimeError(
                f"{type(self).__name__}: plugin_id requires a community "
                "manifest. ReportPlugin subclasses must be loaded via the "
                "community catalog (CommunityCatalog._populate_registries)."
            )
        return manifest.plugin.id

    @property
    def title(self) -> str:
        """Manifest-backed display title (``[plugin].name``)."""
        manifest = type(self)._community_manifest
        if manifest is None:
            raise RuntimeError(
                f"{type(self).__name__}: title requires a community manifest."
            )
        return manifest.plugin.name

    @property
    def order(self) -> int:
        """Render order. Set by :func:`build_report_plugins`."""
        return self._order

    @abstractmethod
    def render(self, ctx: ReportPluginContext) -> ReportBlock:
        """Build the block this plugin owns. Pure function of ``ctx``."""


__all__ = [
    "Clock",
    "IReportBlockPlugin",
    "PluginExecutionRecord",
    "PluginStatus",
    "ReportBlock",
    "ReportPlugin",
    "ReportPluginContext",
]
