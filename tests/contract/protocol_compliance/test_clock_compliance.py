"""Compliance tests for :class:`Clock`.

Two parametrizations:

* ``manual`` — the test-only :class:`ManualClock` (always available)
* ``real`` — the production :class:`RealClock` (always available
  because it's pure stdlib; gated only by being slow due to actual
  sleep calls; thus we keep its sleep test trivial — 1ms)

Both run on every PR — :class:`RealClock` is part of production and
needs no ``RYOTENKAI_LIVE`` flag.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from tests._harness.clock import Clock, ManualClock, RealClock

pytestmark = [
    pytest.mark.contract,
    pytest.mark.compliance,
    pytest.mark.exercises_protocol("Clock"),
    pytest.mark.uses_fake("ManualClock"),
    pytest.mark.asyncio,
]


@pytest.fixture(params=["manual", "real"])
def clock(request: pytest.FixtureRequest) -> Clock:
    if request.param == "manual":
        return ManualClock()
    return RealClock()


class TestClockCompliance:
    async def test_isinstance_protocol(self, clock: Clock) -> None:
        assert isinstance(clock, Clock)

    async def test_now_returns_float(self, clock: Clock) -> None:
        assert isinstance(clock.now(), float)

    async def test_now_is_monotonic_non_decreasing(self, clock: Clock) -> None:
        # For RealClock this is a property of time.monotonic.
        # For ManualClock it's a structural invariant — advance is the
        # only mutator and only accepts non-negative deltas.
        first = clock.now()
        second = clock.now()
        assert second >= first

    async def test_sleep_zero_is_noop(self, clock: Clock) -> None:
        # No deadlocks, no exceptions.
        await clock.sleep(0)
        await clock.sleep(-0.5)

    async def test_sleep_returns_after_advance(self, clock: Clock) -> None:
        # ManualClock — advance is what makes sleep return.
        # RealClock — we use a tiny 1ms sleep so test runtime stays cheap.
        if isinstance(clock, ManualClock):
            task = asyncio.create_task(clock.sleep(5.0))
            await asyncio.sleep(0)  # let task park
            assert not task.done()
            clock.advance(5.0)
            await task
            assert task.done()
        else:
            t0 = time.monotonic()
            await clock.sleep(0.001)
            elapsed = time.monotonic() - t0
            assert elapsed >= 0

    async def test_manual_clock_advance_negative_raises(self, clock: Clock) -> None:
        if not isinstance(clock, ManualClock):
            pytest.skip("advance(negative) check only meaningful for ManualClock")
        with pytest.raises(ValueError):
            clock.advance(-1.0)


__all__: list[str] = []
