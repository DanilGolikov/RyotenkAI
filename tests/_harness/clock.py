"""Clock Protocol + RealClock + ManualClock — deterministic time for tests.

Per Phase 0 ADR (docs/adrs/2026-05-10-greenfield-testing-architecture.md):
Clock lives in tests/_harness/clock.py until Phase 1+ promotes it to
``ryotenkai_shared.utils.clock``. Until that promotion lands, the test side
defines the Protocol locally so nothing depends on the production import.

Invariants:

* :class:`Clock` is a runtime-checkable Protocol with ``now()`` and async
  ``sleep()``. Fakes accept ``Clock`` so tests can swap in :class:`ManualClock`.
* :class:`RealClock` uses :func:`time.monotonic` + :func:`asyncio.sleep`.
* :class:`ManualClock` time advances **only** via :meth:`advance` — never via
  :meth:`sleep`. ``sleep`` parks the caller in a min-heap of waiters keyed by
  deadline; ``advance`` pops every waiter whose deadline has elapsed, in
  deadline-then-FIFO order so concurrent sleeps wake deterministically.
* Negative or zero sleep is a no-op (no waiter registered).
"""

from __future__ import annotations

import asyncio
import heapq
import itertools
import time
from typing import Protocol, runtime_checkable


@runtime_checkable
class Clock(Protocol):
    """Minimal time-source Protocol used by all canonical fakes."""

    def now(self) -> float:
        """Return a monotonic seconds-since-epoch reading."""
        ...

    async def sleep(self, seconds: float) -> None:
        """Async sleep — semantics depend on the concrete implementation."""
        ...


class RealClock:
    """Production-style Clock — wraps :mod:`time` and :mod:`asyncio`."""

    def now(self) -> float:
        return time.monotonic()

    async def sleep(self, seconds: float) -> None:
        if seconds <= 0:
            return
        await asyncio.sleep(seconds)


class ManualClock:
    """Deterministic Clock: time only advances via :meth:`advance`.

    See module docstring for invariants.
    """

    def __init__(self, start: float = 0.0) -> None:
        self._now = float(start)
        self._counter = itertools.count()
        self._waiters: list[tuple[float, int, asyncio.Event]] = []

    def now(self) -> float:
        return self._now

    async def sleep(self, seconds: float) -> None:
        if seconds <= 0:
            return
        deadline = self._now + seconds
        event = asyncio.Event()
        heapq.heappush(self._waiters, (deadline, next(self._counter), event))
        await event.wait()

    def advance(self, seconds: float) -> None:
        if seconds < 0:
            raise ValueError("advance() requires non-negative seconds")
        self._now += seconds
        while self._waiters and self._waiters[0][0] <= self._now:
            _, _, event = heapq.heappop(self._waiters)
            event.set()


__all__ = ["Clock", "ManualClock", "RealClock"]
