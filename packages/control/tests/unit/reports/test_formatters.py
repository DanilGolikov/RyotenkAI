from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

from src.reports.core.constants import MarkdownSymbols, RenderLimits, TrendIcons
from src.reports.core.formatters import ValueFormatter


def test_format_or_dash() -> None:
    assert ValueFormatter.format_or_dash(None) == MarkdownSymbols.DASH
    assert ValueFormatter.format_or_dash(0.1234) == "0.1234"
    assert ValueFormatter.format_or_dash(85.234, ".1f") == "85.2"


def test_format_seconds_handles_zero() -> None:
    assert ValueFormatter.format_seconds(None) == MarkdownSymbols.DASH
    assert ValueFormatter.format_seconds(0.0) == "0.0s"
    assert ValueFormatter.format_seconds(1.23) == "1.2s"


def test_format_mb() -> None:
    assert ValueFormatter.format_mb(None) == MarkdownSymbols.DASH
    assert ValueFormatter.format_mb(1234567.89) == "1,234,568 MB"


def test_format_percent_and_gb() -> None:
    assert ValueFormatter.format_percent(None) == MarkdownSymbols.DASH
    assert ValueFormatter.format_percent(85.234) == "85.2%"
    assert ValueFormatter.format_percent(85.234, precision=2) == "85.23%"

    assert ValueFormatter.format_gb(None) == MarkdownSymbols.DASH
    assert ValueFormatter.format_gb(7.6) == "7.6 GB"


def test_format_integer() -> None:
    assert ValueFormatter.format_integer(None) == MarkdownSymbols.DASH
    assert ValueFormatter.format_integer(1234567) == "1,234,567"
    assert ValueFormatter.format_integer(1234567, with_separator=False) == "1234567"


def test_format_timestamp() -> None:
    dt = datetime(2025, 1, 1, 10, 30, 45)
    assert ValueFormatter.format_timestamp(dt) == "10:30:45"
    assert ValueFormatter.format_timestamp(dt, "%Y-%m-%d") == "2025-01-01"
    assert ValueFormatter.format_timestamp(None) == MarkdownSymbols.DASH


def test_format_loss_trend() -> None:
    assert ValueFormatter.format_loss_trend(None, None) == MarkdownSymbols.DASH
    assert ValueFormatter.format_loss_trend(None, 0.8) == "0.8000"

    trend = ValueFormatter.format_loss_trend(2.5, 0.8, "decreased")
    assert trend == f"2.5000 → 0.8000 {TrendIcons.DECREASED}"

    # unknown direction should not add an icon
    assert ValueFormatter.format_loss_trend(2.5, 0.8, "weird") == "2.5000 → 0.8000"
    assert ValueFormatter.format_loss_trend(2.5, 0.8, "decreased", use_icon=False) == "2.5000 → 0.8000"


def test_format_percentile_stats() -> None:
    dash = MarkdownSymbols.DASH
    assert ValueFormatter.format_percentile_stats(None) == (dash, dash, dash)

    stats = SimpleNamespace(avg=75.5, p95=92.3, max_val=98.1)
    assert ValueFormatter.format_percentile_stats(stats) == ("75.5", "92.3", "98.1")

    stats2 = SimpleNamespace(avg=None, p95=92.3, max_val=None)
    assert ValueFormatter.format_percentile_stats(stats2) == (dash, "92.3", dash)


def test_truncate_message() -> None:
    assert ValueFormatter.truncate_message("Short") == "Short"
    # boundary: equal length -> no truncation
    msg_eq = "x" * RenderLimits.MAX_MESSAGE_LENGTH
    assert ValueFormatter.truncate_message(msg_eq) == msg_eq

    # explicit max_length: shortens and adds suffix at truncate_at
    long_msg = "Very long message here"
    assert ValueFormatter.truncate_message(long_msg, max_length=10) == "Very lo..."
