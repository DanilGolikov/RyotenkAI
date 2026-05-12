"""Unit tests for Eventually/Consistently against ManualClock."""

from __future__ import annotations

import asyncio

import pytest

from tests._harness.clock import ManualClock
from tests._harness.wait import Consistently, Eventually


@pytest.mark.asyncio
async def test_eventually_succeeds_when_condition_flips() -> None:
    clock = ManualClock()
    flipped = False

    def cond() -> bool:
        return flipped

    async def driver() -> None:
        nonlocal flipped
        await asyncio.sleep(0)
        clock.advance(0.5)
        await asyncio.sleep(0)
        flipped = True
        clock.advance(0.5)

    await asyncio.gather(
        Eventually(cond, timeout=5.0, poll=0.5, clock=clock),
        driver(),
    )


@pytest.mark.asyncio
async def test_eventually_raises_on_timeout() -> None:
    clock = ManualClock()

    async def driver() -> None:
        for _ in range(20):
            await asyncio.sleep(0)
            clock.advance(0.5)

    with pytest.raises(TimeoutError, match="custom"):
        await asyncio.gather(
            Eventually(lambda: False, timeout=2.0, poll=0.5, clock=clock, message="custom timeout"),
            driver(),
        )


@pytest.mark.asyncio
async def test_consistently_passes_when_condition_holds() -> None:
    clock = ManualClock()

    async def driver() -> None:
        for _ in range(20):
            await asyncio.sleep(0)
            clock.advance(0.25)

    await asyncio.gather(
        Consistently(lambda: True, duration=2.0, poll=0.5, clock=clock),
        driver(),
    )


@pytest.mark.asyncio
async def test_consistently_fails_on_flip() -> None:
    clock = ManualClock()
    counter = {"n": 0}

    def cond() -> bool:
        counter["n"] += 1
        return counter["n"] < 3

    async def driver() -> None:
        for _ in range(20):
            await asyncio.sleep(0)
            clock.advance(0.5)

    with pytest.raises(AssertionError):
        await asyncio.gather(
            Consistently(cond, duration=5.0, poll=0.5, clock=clock),
            driver(),
        )


@pytest.mark.asyncio
async def test_eventually_supports_async_condition() -> None:
    clock = ManualClock()
    state = {"ready": False}

    async def cond() -> bool:
        return state["ready"]

    async def driver() -> None:
        await asyncio.sleep(0)
        state["ready"] = True
        clock.advance(0.5)

    await asyncio.gather(
        Eventually(cond, timeout=5.0, poll=0.5, clock=clock),
        driver(),
    )
