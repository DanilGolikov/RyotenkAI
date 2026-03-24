"""
Plugin interfaces for experiment report generation.

Design goals:
- Explicit contracts with explicit discovery lifecycle.
- Deterministic ordering.
- Fail-open by default: a single plugin failure becomes an error block.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

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


__all__ = [
    "Clock",
    "IReportBlockPlugin",
    "PluginExecutionRecord",
    "PluginStatus",
    "ReportBlock",
    "ReportPluginContext",
]
