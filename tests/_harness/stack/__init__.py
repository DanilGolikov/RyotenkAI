"""Hermetic-stack harness for L5+ tests.

Stack-level orchestrator that boots fake-* sidecars as Python subprocesses,
wires the real control plane / runner against them, and broadcasts
``/control/advance_clock`` so per-sidecar :class:`ManualClock` instances stay
in lock-step.
"""

from tests._harness.stack._context import current_stack
from tests._harness.stack.orchestrator import Stack

__all__ = ["Stack", "current_stack"]
