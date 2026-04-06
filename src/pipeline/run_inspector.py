"""Compatibility wrapper for run inspection queries."""

from src.pipeline.run_queries import (
    ROOT_GROUP,
    RunInspectionData,
    RunInspector,
    RunSummaryRow,
    build_run_summary_row,
    diff_attempts,
    effective_pipeline_status,
    scan_runs_dir,
    scan_runs_dir_grouped,
    tail_lines,
)

__all__ = [
    "ROOT_GROUP",
    "RunInspectionData",
    "RunInspector",
    "RunSummaryRow",
    "build_run_summary_row",
    "diff_attempts",
    "effective_pipeline_status",
    "scan_runs_dir",
    "scan_runs_dir_grouped",
    "tail_lines",
]
