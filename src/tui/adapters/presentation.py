"""Backward-compatibility shim. Real module lives in src/pipeline/presentation."""

from __future__ import annotations

from src.pipeline.presentation import (
    STATUS_COLORS,
    STATUS_ICONS,
    effective_pipeline_status,
    format_duration,
    format_mode_label,
)

__all__ = [
    "STATUS_COLORS",
    "STATUS_ICONS",
    "effective_pipeline_status",
    "format_duration",
    "format_mode_label",
]
