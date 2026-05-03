"""
Constants for report generation.

Only truly shared types (used by 2+ files) live here.
Single-file constants are defined locally in their consumer modules.
"""

from __future__ import annotations


class RenderLimits:
    """Limits for rendering collections in reports."""

    MAX_CACHE_EVENTS_SHOWN = 10
    MAX_MEMORY_WARNINGS_SHOWN = 10
    MAX_OOM_EVENTS_SHOWN = 10
    MAX_TIMELINE_EVENTS_SHOWN = 50

    MAX_MESSAGE_LENGTH = 50
    MESSAGE_TRUNCATE_AT = 47
    MESSAGE_SUFFIX = "..."


class FormatSpec:
    """Numeric format specifications for consistent formatting."""

    LOSS_PRECISION = ".4f"
    LOSS_COMPACT = ".4g"
    PERCENT_1 = ".1f"
    PERCENT_2 = ".2f"
    SECONDS_1 = ".1f"
    MEGA_0 = ",.0f"
    INTEGER = ",d"


class MarkdownSymbols:
    """Common markdown formatting symbols."""

    DASH = "—"
    BULLET = "•"
    SEPARATOR = "---"
    CHECKMARK = "✅"
    WARNING = "⚠️"
    ERROR = "❌"
    INFO = "ℹ️"


class TrendIcons:
    """Icons for metric trends."""

    DECREASED = "↘️"
    INCREASED = "↗️"
    STABLE = "➡️"


class MetricThresholds:
    """Thresholds for metric analysis."""

    LOSS_SIGNIFICANT_DECREASE = -5.0
    LOSS_SIGNIFICANT_INCREASE = 5.0
    VOLATILITY_HIGH_THRESHOLD = 50.0
    GRAD_NORM_WARNING = 10.0
    GRAD_NORM_CRITICAL = 50.0
    ACCURACY_GOOD_THRESHOLD = 0.7
    ACCURACY_BAD_THRESHOLD = 0.3


# Config key constants shared between config_dump plugin and mlflow adapter
KEY_HYPERPARAMS = "hyperparams"
