"""Severity literal for event envelopes.

The unified event system uses a small, ordered severity ladder modelled
after syslog/OTel conventions. The literal is exported as a named alias
so concrete event classes can pin a single value (``severity:
Literal["info"] = "info"``) while the envelope retains the full union for
``UnknownEvent`` catch-all variants.

Ordering (low → high) is informational only — comparisons rely on string
equality and ``in`` checks. Consumers that need numeric thresholds (e.g.
"flush journal immediately on severity >= error") translate the literal
via the :data:`SEVERITY_ORDER` mapping below.
"""

from __future__ import annotations

from typing import Literal

Severity = Literal["debug", "info", "warning", "error", "critical"]

# Ordered low → high. Consumers wanting threshold comparisons should look
# events up in this mapping rather than comparing strings directly.
SEVERITY_ORDER: dict[Severity, int] = {
    "debug": 10,
    "info": 20,
    "warning": 30,
    "error": 40,
    "critical": 50,
}


__all__ = ["SEVERITY_ORDER", "Severity"]
