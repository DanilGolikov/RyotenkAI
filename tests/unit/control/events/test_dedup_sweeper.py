"""Tests for :class:`ryotenkai_control.events.dedup_sweeper.DedupTTLSweeper`.

Post-Phase-10 TODO #2 — background TTL eviction for the dedup set.

Coverage split (project policy):

1. Positive          — sweep fires evict_expired ≥ once after one interval
2. Negative          — evict_expired raising does not kill the daemon
3. Boundary          — stop immediately after start does not leak a thread
4. Invariants        — start is idempotent; only one daemon thread
5. DependencyErrors  — stop without prior start is a no-op
6. Regressions       — very short interval doesn't busy-loop / leak
7. LogicSpecific     — stop signal interrupts the wait before full interval
"""

from __future__ import annotations

import threading
import time

import pytest

from ryotenkai_control.events.dedup import EventDedup
from ryotenkai_control.events.dedup_sweeper import (
    DEFAULT_SWEEP_INTERVAL_SECONDS,
    DedupTTLSweeper,
)


# ---------------------------------------------------------------------------
# Test fakes
# ---------------------------------------------------------------------------


class _CountingDedup:
    """Stand-in for :class:`EventDedup` that counts sweep calls.

    NOT a Mock subclass — sentinel ``test_no_protocol_mocking`` forbids
    mocking Protocols and we follow the policy globally; using a fake
    keeps the test honest about what surface the sweeper relies on
    (``evict_expired``).
    """

    def __init__(self, *, raise_first: bool = False, raise_every: bool = False):
        self.calls = 0
        self.raise_first = raise_first
        self.raise_every = raise_every
        self._event = threading.Event()

    def evict_expired(self) -> int:
        self.calls += 1
        self._event.set()
        if self.raise_every or (self.raise_first and self.calls == 1):
            raise RuntimeError("synthetic sweep failure")
        return 0

    def wait_for_call(self, timeout: float) -> bool:
        return self._event.wait(timeout=timeout)


# ---------------------------------------------------------------------------
# 1. Positive
# ---------------------------------------------------------------------------


class TestPositive:
    def test_default_interval_matches_module_constant(self) -> None:
        # Guards against an accidental change to the default cadence
        # without a corresponding doc update.
        d = EventDedup()
        sweeper = DedupTTLSweeper(d)
        assert sweeper._interval == DEFAULT_SWEEP_INTERVAL_SECONDS

    def test_sweep_fires_after_one_interval(self) -> None:
        fake = _CountingDedup()
        sweeper = DedupTTLSweeper(fake, sweep_interval_seconds=0.05)
        sweeper.start()
        try:
            assert fake.wait_for_call(timeout=1.0)
            assert fake.calls >= 1
        finally:
            sweeper.stop()

    def test_evict_expired_called_on_real_dedup(self) -> None:
        # Integration-style: real EventDedup + real sweeper at a short
        # cadence. Verifies the sweeper's evict path reaches the real
        # implementation without a fake intermediary.
        clock_val = [100.0]

        def clock() -> float:
            return clock_val[0]

        dedup = EventDedup(ttl_seconds=0.0, clock=clock)
        dedup.remember("r", "src", 0)
        dedup.remember("r", "src", 1)
        clock_val[0] += 10  # move past TTL

        sweeper = DedupTTLSweeper(dedup, sweep_interval_seconds=0.05)
        sweeper.start()
        try:
            # Poll instead of sleeping a single window — keeps the test
            # robust against scheduler jitter on busy CI runners.
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline and dedup.size > 0:
                time.sleep(0.02)
            assert dedup.size == 0
            assert dedup.evicted_total == 2
        finally:
            sweeper.stop()


# ---------------------------------------------------------------------------
# 2. Negative
# ---------------------------------------------------------------------------


class TestNegative:
    def test_sweep_failure_does_not_kill_daemon(self) -> None:
        fake = _CountingDedup(raise_every=True)
        sweeper = DedupTTLSweeper(fake, sweep_interval_seconds=0.05)
        sweeper.start()
        try:
            # Wait for at least two calls — the second proves the
            # exception in the first did not terminate the loop.
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline and fake.calls < 2:
                time.sleep(0.02)
            assert fake.calls >= 2
            assert sweeper.is_running is True
        finally:
            sweeper.stop()

    def test_zero_interval_rejected(self) -> None:
        with pytest.raises(ValueError):
            DedupTTLSweeper(EventDedup(), sweep_interval_seconds=0.0)

    def test_negative_interval_rejected(self) -> None:
        with pytest.raises(ValueError):
            DedupTTLSweeper(EventDedup(), sweep_interval_seconds=-1.0)


# ---------------------------------------------------------------------------
# 3. Boundary
# ---------------------------------------------------------------------------


class TestBoundary:
    def test_stop_immediately_after_start_does_not_leak_thread(self) -> None:
        fake = _CountingDedup()
        sweeper = DedupTTLSweeper(fake, sweep_interval_seconds=10.0)
        sweeper.start()
        # Capture the thread handle BEFORE stop nulls/joins it.
        thread = sweeper._thread
        assert thread is not None and thread.is_alive()
        sweeper.stop(timeout=1.0)
        # Daemon thread must have exited inside the join window even
        # though the configured interval was 10 s — the stop event
        # short-circuits the wait.
        assert thread.is_alive() is False

    def test_no_sweep_called_when_stopped_before_interval(self) -> None:
        fake = _CountingDedup()
        sweeper = DedupTTLSweeper(fake, sweep_interval_seconds=10.0)
        sweeper.start()
        sweeper.stop(timeout=1.0)
        # Sweep is only invoked AFTER the wait returns successfully —
        # stopping inside the first wait prevents any evict call.
        assert fake.calls == 0


# ---------------------------------------------------------------------------
# 4. Invariants
# ---------------------------------------------------------------------------


class TestInvariants:
    def test_double_start_creates_one_thread(self) -> None:
        fake = _CountingDedup()
        sweeper = DedupTTLSweeper(fake, sweep_interval_seconds=5.0)
        sweeper.start()
        first = sweeper._thread
        sweeper.start()  # idempotent
        second = sweeper._thread
        try:
            assert first is second
            assert first is not None and first.is_alive()
            # Count of ryotenkai-dedup-ttl-sweeper threads is exactly 1.
            sweeper_threads = [
                t for t in threading.enumerate()
                if t.name == "ryotenkai-dedup-ttl-sweeper"
            ]
            assert len(sweeper_threads) == 1
        finally:
            sweeper.stop()

    def test_thread_is_daemon(self) -> None:
        # Daemon means a forgotten stop() doesn't block process exit —
        # critical for the orchestrator's sync close path.
        fake = _CountingDedup()
        sweeper = DedupTTLSweeper(fake, sweep_interval_seconds=5.0)
        sweeper.start()
        try:
            assert sweeper._thread is not None
            assert sweeper._thread.daemon is True
        finally:
            sweeper.stop()


# ---------------------------------------------------------------------------
# 5. Dependency errors
# ---------------------------------------------------------------------------


class TestDependencyErrors:
    def test_stop_without_start_is_no_op(self) -> None:
        sweeper = DedupTTLSweeper(EventDedup(), sweep_interval_seconds=5.0)
        # No exception, no thread.
        sweeper.stop()
        assert sweeper.is_running is False

    def test_double_stop_is_no_op(self) -> None:
        fake = _CountingDedup()
        sweeper = DedupTTLSweeper(fake, sweep_interval_seconds=5.0)
        sweeper.start()
        sweeper.stop()
        # Second stop must not raise or block.
        sweeper.stop()
        assert sweeper.is_running is False


# ---------------------------------------------------------------------------
# 6. Regressions
# ---------------------------------------------------------------------------


class TestRegressions:
    def test_short_interval_does_not_busy_loop(self) -> None:
        # A previous design had ``while not stop: evict(); sleep()``
        # which fires evict() on every iteration even when the interval
        # is tiny — easy to confuse with intentional behaviour. The
        # current design uses ``wait(timeout=interval)`` so a 0.05 s
        # interval bounds the call count by elapsed_time/interval.
        fake = _CountingDedup()
        sweeper = DedupTTLSweeper(fake, sweep_interval_seconds=0.05)
        sweeper.start()
        try:
            time.sleep(0.30)
        finally:
            sweeper.stop()
        # In 0.3 s with a 0.05 s interval expect roughly 4-7 calls;
        # we cap at 20 to catch a runaway loop without being flaky.
        assert 1 <= fake.calls <= 20, (
            f"sweep call count {fake.calls} suggests busy loop"
        )

    def test_recovers_after_one_failed_sweep(self) -> None:
        # Regression: a single sweep raising must not "stick" — the
        # next interval must call evict_expired again.
        fake = _CountingDedup(raise_first=True)
        sweeper = DedupTTLSweeper(fake, sweep_interval_seconds=0.05)
        sweeper.start()
        try:
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline and fake.calls < 2:
                time.sleep(0.02)
            assert fake.calls >= 2
        finally:
            sweeper.stop()


# ---------------------------------------------------------------------------
# 7. Logic-specific
# ---------------------------------------------------------------------------


class TestLogicSpecific:
    def test_stop_signal_interrupts_wait_before_full_interval(self) -> None:
        # The whole point of using Event.wait(timeout=interval) instead
        # of sleep(interval) is that stop() should not have to wait the
        # full interval before joining. Set a 5 s interval, stop after
        # 0.1 s, assert join completes well under 5 s.
        fake = _CountingDedup()
        sweeper = DedupTTLSweeper(fake, sweep_interval_seconds=5.0)
        sweeper.start()
        time.sleep(0.05)
        t0 = time.monotonic()
        sweeper.stop(timeout=2.0)
        elapsed = time.monotonic() - t0
        assert elapsed < 1.5, (
            f"stop took {elapsed:.2f}s — sweeper not respecting stop event"
        )

    def test_is_running_reflects_lifecycle(self) -> None:
        fake = _CountingDedup()
        sweeper = DedupTTLSweeper(fake, sweep_interval_seconds=5.0)
        assert sweeper.is_running is False
        sweeper.start()
        assert sweeper.is_running is True
        sweeper.stop()
        assert sweeper.is_running is False
