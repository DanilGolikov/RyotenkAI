"""
Report plugins package.

Plugins live under ``community/reports/`` and are loaded by
``src.community.catalog``. This package provides only the base interfaces
and the registry populated by the loader.
"""

from .interfaces import (
    IReportBlockPlugin,
    PluginExecutionRecord,
    ReportBlock,
    ReportPlugin,
    ReportPluginContext,
)
from .registry import ReportPluginRegistry, build_report_plugins, report_registry

__all__ = [
    "IReportBlockPlugin",
    "PluginExecutionRecord",
    "ReportBlock",
    "ReportPlugin",
    "ReportPluginContext",
    "ReportPluginRegistry",
    "build_report_plugins",
    "report_registry",
]
