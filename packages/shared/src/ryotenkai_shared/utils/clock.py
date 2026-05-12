"""``Clock`` Protocol — production-side definition (Phase 4).

Phase 4 promotion of ``tests/_harness/clock.py``'s ``Clock`` Protocol to
the production tree. The Protocol is **additive-only**: no production
call-site is converted yet (that work belongs to Phase 5+ when chaos
scenarios need ManualClock injection at runtime).

What lives where:

* ``Clock`` — runtime-checkable Protocol, defined here. Two methods:
  :meth:`now` (monotonic-style seconds float) and async :meth:`sleep`.
* ``RealClock`` — production default implementation backed by
  :func:`time.monotonic` and :func:`asyncio.sleep`. Constructable
  with no args; safe for production use.
* ``ManualClock`` — **test-only**. Stays in :mod:`tests._harness.clock`
  so production never accidentally pulls a deterministic-time
  dependency. The test module re-exports ``Clock`` and ``RealClock``
  from here for backward compatibility with existing tests written
  before Phase 4.

Why a Protocol and not an abstract base class:

* Structural typing means existing classes that *happen* to expose
  ``now()`` + ``sleep()`` already satisfy the Protocol without
  inheritance, which keeps Phase 4 perfectly additive.
* ``@runtime_checkable`` lets compliance tests (and the fakes that
  consume the Protocol) assert membership at runtime —
  ``isinstance(clock, Clock)`` is the keystone fake-vs-real check.

Design constraint: time units.

* :meth:`now` returns ``float`` seconds and is monotonic-style
  (always non-decreasing across calls on a single instance). Tests
  use ``ManualClock`` whose origin is configurable (defaults to
  ``0.0``); production uses :func:`time.monotonic`.
* :meth:`sleep` accepts ``float`` seconds; passing zero or a
  negative value returns immediately without yielding to the loop.
"""

from __future__ import annotations

import asyncio
import time
from typing import Protocol, runtime_checkable


@runtime_checkable
class Clock(Protocol):
    """Time + sleep surface used by infrastructure clients.

    See module docstring for context and units. Implementations:

    * :class:`RealClock` — production default.
    * ``tests._harness.clock.ManualClock`` — deterministic test helper
      that advances time only via explicit ``advance(seconds)`` calls.
    """

    def now(self) -> float:
        """Return monotonic-style seconds since an unspecified origin."""
        ...

    async def sleep(self, seconds: float) -> None:
        """Sleep ``seconds`` seconds. Zero/negative is a no-op."""
        ...


class RealClock:
    """Production ``Clock`` implementation backed by :mod:`asyncio`/:mod:`time`."""

    def now(self) -> float:
        return time.monotonic()

    async def sleep(self, seconds: float) -> None:
        if seconds <= 0:
            return
        await asyncio.sleep(seconds)


__all__ = [
    "Clock",
    "RealClock",
]
