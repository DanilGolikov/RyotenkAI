"""
Report Models.

Typed dataclasses for report generation.
"""

from ryotenkai_control.reports.domain.entities import MetricHistory
from ryotenkai_control.reports.models.report import (
    ConfigInfo,
    ExperimentHealth,
    ExperimentReport,
    Issue,
    MemoryEvent,
    MemoryManagementInfo,
    MemoryPhaseStats,
    MetricAnalysis,
    MetricStatus,
    MetricTrend,
    ModelInfo,
    PercentileStats,
    PhaseInfo,
    ReportSummary,
    ResourcesInfo,
    RunStatus,
    TimelineEvent,
)

__all__ = [
    "ConfigInfo",
    "ExperimentHealth",
    "ExperimentReport",
    "Issue",
    "MemoryEvent",
    "MemoryManagementInfo",
    "MemoryPhaseStats",
    "MetricAnalysis",
    "MetricHistory",
    "MetricStatus",
    "MetricTrend",
    "ModelInfo",
    "PercentileStats",
    "PhaseInfo",
    "ReportSummary",
    "ResourcesInfo",
    "RunStatus",
    "TimelineEvent",
]
