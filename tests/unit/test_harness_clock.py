"""Unit tests for ManualClock + RealClock."""

from __future__ import annotations

import asyncio

import pytest

from tests._harness.clock import Clock, ManualClock, RealClock


def test_manual_clock_advances_now() -> None:
    clock = ManualClock(start=10.0)
    assert clock.now() == 10.0
    clock.advance(2.5)
    assert clock.now() == 12.5


def test_manual_clock_rejects_negative_advance() -> None:
    clock = ManualClock()
    with pytest.raises(ValueError):
        clock.advance(-1)


def test_real_and_manual_satisfy_protocol() -> None:
    assert isinstance(RealClock(), Clock)
    assert isinstance(ManualClock(), Clock)


@pytest.mark.asyncio
async def test_manual_clock_sleep_blocks_until_advance() -> None:
    clock = ManualClock()
    woken = asyncio.Event()

    async def sleeper() -> None:
        await clock.sleep(5.0)
        woken.set()

    task = asyncio.create_task(sleeper())
    await asyncio.sleep(0)
    assert not woken.is_set()
    clock.advance(2.0)
    await asyncio.sleep(0)
    assert not woken.is_set()
    clock.advance(3.0)
    await asyncio.wait_for(woken.wait(), timeout=1.0)
    await task


@pytest.mark.asyncio
async def test_manual_clock_sleep_zero_returns_immediately() -> None:
    clock = ManualClock()
    await asyncio.wait_for(clock.sleep(0), timeout=1.0)


@pytest.mark.asyncio
async def test_manual_clock_releases_in_deadline_order() -> None:
    clock = ManualClock()
    order: list[str] = []

    async def named(name: str, seconds: float) -> None:
        await clock.sleep(seconds)
        order.append(name)

    tasks = [
        asyncio.create_task(named("a", 3.0)),
        asyncio.create_task(named("b", 1.0)),
        asyncio.create_task(named("c", 2.0)),
    ]
    await asyncio.sleep(0)
    clock.advance(3.0)
    await asyncio.gather(*tasks)
    assert order == ["b", "c", "a"]
