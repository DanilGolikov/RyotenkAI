"""Helpers to normalise CLI vs API responses for parity comparison.

The two surfaces produce equivalent payloads but differ in cosmetic
ways: ``Path`` objects from the CLI become ``str`` already (renderer
handles that), while the API may emit fully-qualified ISO timestamps
with microseconds, dataclass field-order, and so on.

:func:`normalise` walks the structure once and applies every cosmetic
fix in one pass so contract assertions stay declarative
(``assert normalise(cli) == normalise(api)``).
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

#: Stable identifier keys used to sort lists of dicts. The first key
#: that's present in every element wins. Order matters — tuples are
#: tried left-to-right.
_SORT_KEYS: tuple[str, ...] = ("id", "run_id", "name", "stage")

_ISO_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}",
)


def normalise(value: Any) -> Any:
    """Return a deep copy of ``value`` with cosmetic differences flattened.

    Rules:
    - ``dict`` → recurse, keys remain as-is.
    - ``list`` of ``dict``: recurse into each element, then sort by the
      first key in :data:`_SORT_KEYS` that every dict has.
    - ``list`` of scalars: recurse into each element (preserves order —
      tests that need order-sensitive comparison stay correct).
    - ISO-8601 strings: truncated to seconds, normalised to UTC.
    - everything else: returned as-is.
    """
    if isinstance(value, dict):
        return {k: normalise(v) for k, v in value.items()}
    if isinstance(value, list):
        out = [normalise(v) for v in value]
        if out and all(isinstance(item, dict) for item in out):
            for key in _SORT_KEYS:
                if all(key in item for item in out):
                    out.sort(key=lambda item: str(item[key]))
                    break
        return out
    if isinstance(value, str) and _ISO_RE.match(value):
        return _normalise_iso(value)
    return value


def _normalise_iso(value: str) -> str:
    """Truncate to seconds + force UTC; fall through unchanged on parse fail."""
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return value
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    else:
        parsed = parsed.astimezone(timezone.utc)
    return parsed.replace(microsecond=0).isoformat()


__all__ = ["normalise"]
