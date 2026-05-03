"""
Value formatters for report rendering.

This module provides consistent formatting utilities to avoid code duplication
and ensure uniform presentation across all reports.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.reports.core.constants import FormatSpec, MarkdownSymbols

if TYPE_CHECKING:
    from datetime import datetime

    from src.reports.models.report import PercentileStats


class ValueFormatter:
    """Centralized value formatting utilities."""

    @staticmethod
    def format_or_dash(value: float | None, format_spec: str = FormatSpec.LOSS_PRECISION) -> str:
        """
        Format numeric value or return dash if None.

        Args:
            value: Numeric value to format
            format_spec: Python format specification (default: .4f for loss)

        Returns:
            Formatted string or "—" if value is None

        Examples:
            >>> ValueFormatter.format_or_dash(0.1234)
            '0.1234'
            >>> ValueFormatter.format_or_dash(None)
            '—'
            >>> ValueFormatter.format_or_dash(85.234, FormatSpec.PERCENT_1)
            '85.2'
        """
        return f"{value:{format_spec}}" if value is not None else MarkdownSymbols.DASH

    @staticmethod
    def format_seconds(seconds: float | None) -> str:
        """
        Format seconds with 1 decimal place or return dash.

        Examples:
            >>> ValueFormatter.format_seconds(123.456)
            '123.5s'
            >>> ValueFormatter.format_seconds(None)
            '—'
        """
        return f"{seconds:{FormatSpec.SECONDS_1}}s" if seconds is not None else MarkdownSymbols.DASH

    @staticmethod
    def format_mb(mb: float | None) -> str:
        """
        Format megabytes with thousands separator or return dash.

        Examples:
            >>> ValueFormatter.format_mb(1234567.89)
            '1,234,568 MB'
            >>> ValueFormatter.format_mb(None)
            '—'
        """
        return f"{mb:{FormatSpec.MEGA_0}} MB" if mb is not None else MarkdownSymbols.DASH

    @staticmethod
    def format_percent(value: float | None, precision: int = 1) -> str:
        """
        Format percentage with specified precision or return dash.

        Args:
            value: Percentage value (e.g., 85.234)
            precision: Decimal places (default: 1)

        Examples:
            >>> ValueFormatter.format_percent(85.234)
            '85.2%'
            >>> ValueFormatter.format_percent(85.234, precision=2)
            '85.23%'
            >>> ValueFormatter.format_percent(None)
            '—'
        """
        if value is None:
            return MarkdownSymbols.DASH
        format_spec = f".{precision}f"
        return f"{value:{format_spec}}%"

    @staticmethod
    def format_gb(gb: float | None, precision: int = 1) -> str:
        """
        Format gigabytes with specified precision.

        Examples:
            >>> ValueFormatter.format_gb(7.6)
            '7.6 GB'
            >>> ValueFormatter.format_gb(None)
            '—'
        """
        if gb is None:
            return MarkdownSymbols.DASH
        format_spec = f".{precision}f"
        return f"{gb:{format_spec}} GB"

    @staticmethod
    def format_integer(value: int | None, with_separator: bool = True) -> str:
        """
        Format integer with optional thousands separator.

        Examples:
            >>> ValueFormatter.format_integer(1234567)
            '1,234,567'
            >>> ValueFormatter.format_integer(1234567, with_separator=False)
            '1234567'
            >>> ValueFormatter.format_integer(None)
            '—'
        """
        if value is None:
            return MarkdownSymbols.DASH
        format_spec = FormatSpec.INTEGER if with_separator else "d"
        return f"{value:{format_spec}}"

    @staticmethod
    def format_timestamp(dt: datetime | None, format_str: str = "%H:%M:%S") -> str:
        """
        Format datetime or return dash.

        Args:
            dt: Datetime object
            format_str: strftime format string (default: time only)

        Examples:
            >>> ValueFormatter.format_timestamp(datetime(2025, 1, 1, 10, 30, 45))
            '10:30:45'
            >>> ValueFormatter.format_timestamp(None)
            '—'
        """
        return dt.strftime(format_str) if dt else MarkdownSymbols.DASH

    @staticmethod
    def format_loss_trend(
        first: float | None,
        last: float | None,
        direction: str | None = None,
        use_icon: bool = True,
    ) -> str:
        """
        Format loss trend with start→end and optional icon.

        Args:
            first: Starting loss value
            last: Ending loss value
            direction: Trend direction ("decreased", "increased", "stable")
            use_icon: Whether to append trend icon

        Examples:
            >>> ValueFormatter.format_loss_trend(2.5, 0.8, "decreased")
            '2.5000 → 0.8000 ↘️'
            >>> ValueFormatter.format_loss_trend(None, 0.8)
            '0.8000'
            >>> ValueFormatter.format_loss_trend(None, None)
            '—'
        """
        from src.reports.core.constants import TrendIcons

        if first is not None and last is not None:
            result = f"{first:{FormatSpec.LOSS_PRECISION}} → {last:{FormatSpec.LOSS_PRECISION}}"
            if use_icon and direction:
                icon = {
                    "decreased": TrendIcons.DECREASED,
                    "increased": TrendIcons.INCREASED,
                    "stable": TrendIcons.STABLE,
                }.get(direction, "")
                if icon:
                    result += f" {icon}"
            return result
        elif last is not None:
            return f"{last:{FormatSpec.LOSS_PRECISION}}"
        else:
            return MarkdownSymbols.DASH

    @staticmethod
    def format_percentile_stats(percentile_stats: PercentileStats | None) -> tuple[str, str, str]:
        """
        Format percentile stats as (avg, p95, max) strings.

        Returns:
            Tuple of (avg_str, p95_str, max_str)

        Examples:
            >>> stats = PercentileStats(avg=75.5, p95=92.3, max_val=98.1, ...)
            >>> ValueFormatter.format_percentile_stats(stats)
            ('75.5', '92.3', '98.1')
            >>> ValueFormatter.format_percentile_stats(None)
            ('—', '—', '—')
        """
        dash = MarkdownSymbols.DASH
        if not percentile_stats:
            return dash, dash, dash

        avg = f"{percentile_stats.avg:{FormatSpec.PERCENT_1}}" if percentile_stats.avg is not None else dash
        p95 = f"{percentile_stats.p95:{FormatSpec.PERCENT_1}}" if percentile_stats.p95 is not None else dash
        max_val = f"{percentile_stats.max_val:{FormatSpec.PERCENT_1}}" if percentile_stats.max_val is not None else dash

        return avg, p95, max_val

    @staticmethod
    def truncate_message(message: str, max_length: int | None = None) -> str:
        """
        Truncate message to max length with suffix.

        Args:
            message: Message string
            max_length: Maximum length (default: from RenderLimits)

        Examples:
            >>> ValueFormatter.truncate_message("Very long message here", max_length=10)
            'Very lo...'
            >>> ValueFormatter.truncate_message("Short")
            'Short'
        """
        from src.reports.core.constants import RenderLimits

        max_len = max_length or RenderLimits.MAX_MESSAGE_LENGTH
        # For custom max_length, truncate so that `len(result) <= max_len` (including suffix).
        # For default behavior, preserve the configured RenderLimits.* constants.
        suffix = RenderLimits.MESSAGE_SUFFIX
        truncate_at = RenderLimits.MESSAGE_TRUNCATE_AT if max_length is None else max(0, max_len - len(suffix))

        if len(message) > max_len:
            if max_len <= len(suffix):
                return suffix[:max_len]
            return message[:truncate_at] + suffix
        return message
