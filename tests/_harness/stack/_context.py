"""Per-task context handle for the active :class:`Stack`.

Lets the debug-bundle plugin (and any future cross-cutting consumer) discover
the live :class:`Stack` instance without import-cycle gymnastics. Each L5+
test sets ``current_stack`` on entry; the bundle writer reads it on failure.
"""

from __future__ import annotations

from contextvars import ContextVar
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tests._harness.stack.orchestrator import Stack


current_stack: ContextVar[Stack | None] = ContextVar("current_stack", default=None)


__all__ = ["current_stack"]
