"""Phase 11.E — :class:`ControlPlaneHeartbeat` service contract.

Pin the lifetime + transport contract of the Mac-side ping service:
* ``start()`` sends an immediate ping, then schedules periodic pings.
* ``stop()`` cancels the task; survives being called from a finally
  block while the task is still running.
* Transient transport errors logged, NEVER raised into orchestrator.
* Counters track success/failure for tests + dashboards.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from src.pipeline.heartbeat.heartbeat import ControlPlaneHeartbeat


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeJobClient:
    """Stand-in for :class:`JobClient.send_heartbeat`. Counts pings,
    optionally fails on the Nth call (or every call)."""

    def __init__(
        self,
        *,
        fail_after: int | None = None,
        fail_with: Exception | None = None,
        return_false_after: int | None = None,
    ) -> None:
        self.calls: list[float | None] = []
        self._fail_after = fail_after
        self._fail_with = fail_with
        self._false_after = return_false_after

    async def send_heartbeat(
        self, *, ttl_seconds: float | None = None,
    ) -> bool:
        self.calls.append(ttl_seconds)
        n = len(self.calls)
        if self._fail_after is not None and n > self._fail_after:
            raise self._fail_with or RuntimeError("simulated transport")
        if self._false_after is not None and n > self._false_after:
            return False
        return True


# ---------------------------------------------------------------------------
# 1. Positive — ping cadence
# ---------------------------------------------------------------------------


class TestPositive:
    @pytest.mark.asyncio
    async def test_start_sends_immediate_ping(self) -> None:
        client = _FakeJobClient()
        svc = ControlPlaneHeartbeat(client, ping_interval_seconds=10.0)
        await svc.start()
        # Immediately after start: at least 1 ping (the synchronous
        # initial one) must have happened.
        assert len(client.calls) >= 1
        assert svc.ping_success_count >= 1
        await svc.stop()

    @pytest.mark.asyncio
    async def test_periodic_pings_after_interval(self) -> None:
        client = _FakeJobClient()
        svc = ControlPlaneHeartbeat(client, ping_interval_seconds=1.0)
        await svc.start()
        # Sleep a bit longer than the interval to allow at least one
        # background ping to land.
        await asyncio.sleep(2.5)
        await svc.stop()
        # At least 2 pings (initial + 1 background).
        assert len(client.calls) >= 2

    @pytest.mark.asyncio
    async def test_ttl_seconds_passed_through(self) -> None:
        client = _FakeJobClient()
        svc = ControlPlaneHeartbeat(
            client, ping_interval_seconds=10.0, ttl_seconds=90.0,
        )
        await svc.start()
        await svc.stop()
        assert client.calls[0] == 90.0


# ---------------------------------------------------------------------------
# 2. Stop semantics
# ---------------------------------------------------------------------------


class TestStop:
    @pytest.mark.asyncio
    async def test_stop_idempotent(self) -> None:
        client = _FakeJobClient()
        svc = ControlPlaneHeartbeat(client, ping_interval_seconds=1.0)
        await svc.start()
        await svc.stop()
        await svc.stop()  # NO-OP — must not raise.
        assert svc.is_running is False

    @pytest.mark.asyncio
    async def test_stop_unblocks_immediately(self) -> None:
        # Service is sleeping inside the interval; stop() must
        # interrupt the sleep, not wait the full interval.
        client = _FakeJobClient()
        svc = ControlPlaneHeartbeat(client, ping_interval_seconds=60.0)
        await svc.start()
        # Issue stop and time how long it takes.
        loop = asyncio.get_event_loop()
        t0 = loop.time()
        await svc.stop()
        elapsed = loop.time() - t0
        # Stop should complete in well under the 60s interval.
        assert elapsed < 1.0

    @pytest.mark.asyncio
    async def test_start_idempotent(self) -> None:
        client = _FakeJobClient()
        svc = ControlPlaneHeartbeat(client, ping_interval_seconds=10.0)
        await svc.start()
        # Second start MUST be a no-op (no extra task spawned).
        await svc.start()
        assert svc.is_running
        await svc.stop()


# ---------------------------------------------------------------------------
# 3. Failure tolerance
# ---------------------------------------------------------------------------


class TestFailureTolerance:
    @pytest.mark.asyncio
    async def test_send_heartbeat_raise_does_not_crash_task(self) -> None:
        # JobClient raises → service catches + counts + continues.
        client = _FakeJobClient(fail_after=0, fail_with=RuntimeError("boom"))
        svc = ControlPlaneHeartbeat(client, ping_interval_seconds=0.5)
        await svc.start()
        await asyncio.sleep(1.5)
        # Service is still running despite repeated failures.
        assert svc.is_running
        assert svc.ping_failure_count >= 1
        await svc.stop()

    @pytest.mark.asyncio
    async def test_non_200_response_counts_as_failure(self) -> None:
        # send_heartbeat returns False (server replied non-200).
        client = _FakeJobClient(return_false_after=0)
        svc = ControlPlaneHeartbeat(client, ping_interval_seconds=10.0)
        await svc.start()
        # Initial ping returned False → failure count incremented.
        assert svc.ping_failure_count >= 1
        await svc.stop()

    @pytest.mark.asyncio
    async def test_on_error_callback_invoked(self) -> None:
        errors: list[Any] = []
        client = _FakeJobClient(fail_after=0, fail_with=ValueError("nope"))
        svc = ControlPlaneHeartbeat(
            client, ping_interval_seconds=10.0, on_error=errors.append,
        )
        await svc.start()
        await svc.stop()
        assert errors
        # Either the exception (raise path) or None (False path).
        assert isinstance(errors[0], (Exception, type(None)))


# ---------------------------------------------------------------------------
# 4. Counters
# ---------------------------------------------------------------------------


class TestCounters:
    @pytest.mark.asyncio
    async def test_success_counter_increments(self) -> None:
        # Minimum interval clamped to 1.0s, so we need to sleep >1.0s
        # to see the second ping.
        client = _FakeJobClient()
        svc = ControlPlaneHeartbeat(client, ping_interval_seconds=1.0)
        await svc.start()
        await asyncio.sleep(1.5)
        await svc.stop()
        assert svc.ping_success_count >= 2  # initial + at least one background

    @pytest.mark.asyncio
    async def test_failure_counter_separate(self) -> None:
        client = _FakeJobClient(return_false_after=0)
        svc = ControlPlaneHeartbeat(client, ping_interval_seconds=10.0)
        await svc.start()
        assert svc.ping_success_count == 0
        assert svc.ping_failure_count >= 1
        await svc.stop()


# ---------------------------------------------------------------------------
# 5. Boundary
# ---------------------------------------------------------------------------


class TestBoundary:
    @pytest.mark.asyncio
    async def test_minimum_interval_clamped(self) -> None:
        # < 1.0 s clamped up to 1.0 (avoids a busy loop).
        client = _FakeJobClient()
        svc = ControlPlaneHeartbeat(client, ping_interval_seconds=0.01)
        await svc.start()
        # Just verify it runs without crashing — the clamp is internal.
        assert svc.is_running
        await svc.stop()
