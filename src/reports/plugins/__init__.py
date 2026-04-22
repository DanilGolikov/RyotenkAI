"""
Report plugins package.

Plugins live under ``community/reports/`` and are loaded by
``src.community.catalog``. This package provides only the base interfaces
and the registry populated by the loader.
"""

from .interfaces import IReportBlockPlugin, PluginExecutionRecord, ReportBlock, ReportPluginContext
from .registry import ReportPluginRegistry, build_report_plugins

__all__ = [
    "IReportBlockPlugin",
    "PluginExecutionRecord",
    "ReportBlock",
    "ReportPluginContext",
    "ReportPluginRegistry",
    "build_report_plugins",
]
