"""
Report plugins package.

Provides a modular architecture for building `experiment_report.md` as an
ordered list of independent blocks (plugins).
"""

from .discovery import ensure_report_plugins_discovered
from .interfaces import IReportBlockPlugin, PluginExecutionRecord, ReportBlock, ReportPluginContext
from .registry import ReportPluginRegistry, build_report_plugins

__all__ = [
    "IReportBlockPlugin",
    "PluginExecutionRecord",
    "ReportBlock",
    "ReportPluginContext",
    "ReportPluginRegistry",
    "build_report_plugins",
    "ensure_report_plugins_discovered",
]
