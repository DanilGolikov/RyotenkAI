from __future__ import annotations

import re
from typing import Any

from rich.text import Text

_DURATION_PART_RE = re.compile(r"(\d+)([hms])")


def plain_sort_key(value: Any) -> str:
    """Strip Rich markup from a cell value so sorting compares plain text."""
    if isinstance(value, Text):
        return value.plain
    return Text.from_markup(str(value)).plain


def duration_sort_seconds(value: Any) -> int:
    """Convert textual durations like `1h 2m 3s` into sortable seconds."""
    plain = plain_sort_key(value)
    total = 0
    matched = False
    for amount, unit in _DURATION_PART_RE.findall(plain):
        matched = True
        if unit == "h":
            total += int(amount) * 3600
        elif unit == "m":
            total += int(amount) * 60
        else:
            total += int(amount)
    return total if matched else -1


def created_timestamp_sort_key(value: Any) -> str:
    """Extract the sortable timestamp prefix from a rendered Created cell."""
    return plain_sort_key(value)[:16]


def integer_sort_key(value: Any) -> int:
    """Parse a rendered integer cell into a numeric sort key."""
    plain = plain_sort_key(value).strip()
    try:
        return int(plain)
    except ValueError:
        return -1
