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
  metadata mirroring + secret injection) and forces the
  ``plugin_id`` / ``title`` / ``order`` ClassVars + abstract ``render``,
  giving authors a typed, lint-friendly surface.

Existing free-standing classes (``HeaderBlockPlugin`` etc.) migrate to
``ReportPlugin`` in the same PR that introduces the unified
``PluginRegistry[T]``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Protocol, runtime_checkable

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

    Carries the same metadata mirroring as the other plugin kinds (name,
    version, ``_community_manifest`` etc. — inherited from
    :class:`BasePlugin`) and pins the structural contract from
    :class:`IReportBlockPlugin`.

    ``plugin_id`` / ``title`` / ``order`` are ClassVars, not properties —
    matching the way community plugins have always declared them and
    avoiding boilerplate on every subclass. ``order`` is overwritten at
    composition time by :func:`build_report_plugins` based on the user's
    ``reports.sections`` ordering, so the value declared on the class is
    only a default for in-process use (tests, ad-hoc rendering).

    Subclass example::

        class HeaderBlockPlugin(ReportPlugin):
            plugin_id = "header"
            title = "Header"
            order = 10

            def render(self, ctx: ReportPluginContext) -> ReportBlock:
                ...
    """

    plugin_id: ClassVar[str] = ""
    title: ClassVar[str] = ""
    order: ClassVar[int] = 0

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
