"""Phase 4.2 — :class:`HealthReporter` contract.

The reporter is shape-symmetric with :class:`IdleDetector`, so the
test scaffolding (TickClock, fake provider, ``_run_until_ticks``)
mirrors :mod:`test_idle_detector`. We assert:

- snapshots flow onto the bus at the configured interval;
- a provider exception is swallowed (the loop keeps running);
- start/stop is idempotent and re-startable.
"""

from __future__ import annotations

import asyncio

import pytest

from src.runner.event_bus import EventBus
from src.runner.health_reporter import (
    DEFAULT_HEALTH_INTERVAL,
    HealthReporter,
    HealthSnapshot,
)

pytestmark = pytest.mark.asyncio


class _TickSleep:
    """Counts ``await sleep(...)`` calls so tests can wait for N ticks."""

    def __init__(self) -> None:
        self.ticks = 0

    async def sleep(self, seconds: float) -> None:
        self.ticks += 1
        await asyncio.sleep(0)


async def _wait_ticks(t: _TickSleep, *, target: int, max_yields: int = 1000) -> None:
    yields = 0
    while t.ticks < target and yields < max_yields:
        await asyncio.sleep(0)
        yields += 1


async def _wait_events(
    bus: EventBus, kind: str, *, target: int, max_yields: int = 1000,
) -> None:
    """Yield until ``target`` events of ``kind`` are on the bus.

    Counting events directly is more robust than counting sleep ticks
    — there's a one-yield gap between waking from sleep and actually
    publishing, so a tick-based wait can race ahead of the publish.
    """
    yields = 0
    while (
        sum(1 for e in list(bus._buffer) if e.kind == kind) < target
        and yields < max_yields
    ):
        await asyncio.sleep(0)
        yields += 1


def _scripted_provider(snapshots: list[HealthSnapshot]):
    """Yield each snapshot in order, then stick on the last."""

    async def _read() -> HealthSnapshot:
        if not snapshots:
            return {}
        if len(snapshots) > 1:
            return snapshots.pop(0)
        return snapshots[0]

    return _read


@pytest.fixture
def bus() -> EventBus:
    return EventBus(capacity=100)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestEmitSnapshots:
    async def test_publishes_each_snapshot(self, bus: EventBus) -> None:
        ticker = _TickSleep()
        reporter = HealthReporter(
            bus,
            interval=0.001,
            snapshot_provider=_scripted_provider([
                {"gpu_util_percent": 30.0, "cpu_percent": 50.0},
                {"gpu_util_percent": 80.0, "cpu_percent": 75.0},
                {"gpu_util_percent": 90.0, "cpu_percent": 95.0},
            ]),
            sleep=ticker.sleep,
        )
        reporter.start()
        await _wait_events(bus, "health_snapshot", target=3)
        await reporter.stop()

        snapshots = [
            e.payload for e in list(bus._buffer) if e.kind == "health_snapshot"
        ]
        assert len(snapshots) >= 3
        assert snapshots[0]["gpu_util_percent"] == 30.0
        assert snapshots[1]["gpu_util_percent"] == 80.0


# ---------------------------------------------------------------------------
# Error tolerance
# ---------------------------------------------------------------------------


class TestProviderError:
    async def test_provider_exception_does_not_kill_loop(
        self, bus: EventBus,
    ) -> None:
        ticker = _TickSleep()
        invocations = {"count": 0}

        async def _flaky() -> HealthSnapshot:
            invocations["count"] += 1
            if invocations["count"] == 1:
                raise RuntimeError("transient nvidia-smi failure")
            return {"gpu_util_percent": 42.0}

        reporter = HealthReporter(
            bus,
            interval=0.001,
            snapshot_provider=_flaky,
            sleep=ticker.sleep,
        )
        reporter.start()
        await _wait_events(bus, "health_snapshot", target=2)
        await reporter.stop()

        kinds = [e.kind for e in list(bus._buffer) if e.kind == "health_snapshot"]
        # First poll skipped (provider raised); subsequent ones land normally.
        assert len(kinds) >= 2


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


class TestLifecycle:
    async def test_start_idempotent(self, bus: EventBus) -> None:
        ticker = _TickSleep()
        reporter = HealthReporter(
            bus, interval=0.001,
            snapshot_provider=_scripted_provider([{"x": 1}]),
            sleep=ticker.sleep,
        )
        reporter.start()
        first = reporter._task
        reporter.start()
        assert reporter._task is first
        await reporter.stop()

    async def test_stop_safe_when_never_started(self, bus: EventBus) -> None:
        reporter = HealthReporter(bus)
        await reporter.stop()  # no raise

    async def test_default_interval(self) -> None:
        # Production constant must stay reasonable (30 s — the same
        # cadence the watchdog used for GPU samples).
        assert DEFAULT_HEALTH_INTERVAL == 30.0
