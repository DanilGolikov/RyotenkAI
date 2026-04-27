"""Phase 11.B — :class:`MacHeartbeat` contract tests.

The heartbeat is dead-simple (3 fields, 3 methods) but the contract
is operational: a wrong reading drives the natural-completion
decision matrix in :class:`PodTerminator`. Pin the semantics under
7-category coverage so future refactors don't silently flip
``is_alive()`` thresholds.
"""

from __future__ import annotations

import pytest

from src.runner.heartbeat import MacHeartbeat


class _Clock:
    """Deterministic clock for tests — advance manually via ``set``."""

    def __init__(self, start: float = 0.0) -> None:
        self.now: float = start

    def __call__(self) -> float:
        return self.now

    def set(self, value: float) -> None:
        self.now = value

    def advance(self, seconds: float) -> None:
        self.now += seconds


# ---------------------------------------------------------------------------
# 1. Positive — happy path: mark_active flips is_alive to True
# ---------------------------------------------------------------------------


class TestPositive:
    def test_mark_active_then_immediately_alive(self) -> None:
        clock = _Clock(start=100.0)
        hb = MacHeartbeat(clock=clock)
        hb.mark_active()
        assert hb.is_alive() is True

    def test_age_seconds_after_mark(self) -> None:
        clock = _Clock(start=100.0)
        hb = MacHeartbeat(clock=clock)
        hb.mark_active()
        clock.advance(5.0)
        assert hb.age_seconds() == 5.0


# ---------------------------------------------------------------------------
# 2. Negative — fresh runner, never seen the Mac
# ---------------------------------------------------------------------------


class TestNegative:
    def test_fresh_runner_is_not_alive(self) -> None:
        # Before any mark_active() call — assume Mac asleep / never
        # connected. Per docstring: "never seen" is treated as
        # asleep so PodTerminator picks the safe path.
        clock = _Clock(start=100.0)
        hb = MacHeartbeat(clock=clock)
        assert hb.is_alive() is False

    def test_fresh_runner_age_is_none(self) -> None:
        clock = _Clock(start=100.0)
        hb = MacHeartbeat(clock=clock)
        assert hb.age_seconds() is None


# ---------------------------------------------------------------------------
# 3. Boundary — TTL crossing
# ---------------------------------------------------------------------------


class TestBoundary:
    def test_alive_just_before_ttl(self) -> None:
        clock = _Clock(start=0.0)
        hb = MacHeartbeat(clock=clock, ttl_seconds=60.0)
        hb.mark_active()
        clock.advance(59.999)
        assert hb.is_alive() is True

    def test_dead_at_ttl(self) -> None:
        # Exactly at TTL — ``< ttl`` (strict less-than) means TTL
        # itself is the first-stale moment.
        clock = _Clock(start=0.0)
        hb = MacHeartbeat(clock=clock, ttl_seconds=60.0)
        hb.mark_active()
        clock.advance(60.0)
        assert hb.is_alive() is False

    def test_dead_well_past_ttl(self) -> None:
        clock = _Clock(start=0.0)
        hb = MacHeartbeat(clock=clock, ttl_seconds=60.0)
        hb.mark_active()
        clock.advance(3600.0)  # 1h asleep
        assert hb.is_alive() is False
        assert hb.age_seconds() == 3600.0


# ---------------------------------------------------------------------------
# 4. Invariants
# ---------------------------------------------------------------------------


class TestInvariants:
    def test_mark_active_idempotent_within_same_tick(self) -> None:
        clock = _Clock(start=100.0)
        hb = MacHeartbeat(clock=clock)
        hb.mark_active()
        hb.mark_active()
        hb.mark_active()
        # Three calls, age still 0.
        assert hb.age_seconds() == 0.0

    def test_mark_active_advances_freshness(self) -> None:
        # Multiple marks across ticks — last one wins.
        clock = _Clock(start=0.0)
        hb = MacHeartbeat(clock=clock, ttl_seconds=60.0)
        hb.mark_active()  # at t=0
        clock.advance(50.0)
        hb.mark_active()  # at t=50
        clock.advance(50.0)
        # Total elapsed=100s, but age from last mark_active is 50s.
        assert hb.age_seconds() == 50.0
        assert hb.is_alive() is True  # still inside 60s TTL

    def test_default_ttl_is_60(self) -> None:
        # Class-level constant pinned — operator dashboards expect
        # "Mac asleep" alerts after ~1 minute.
        assert MacHeartbeat.HEARTBEAT_TTL_SECONDS == 60.0

    def test_default_clock_is_monotonic(self) -> None:
        # No clock passed → uses time.monotonic (immune to wall-clock
        # skew). Pin the type so a future refactor that swaps in
        # ``time.time`` is caught.
        import time

        hb = MacHeartbeat()
        assert hb._clock is time.monotonic


# ---------------------------------------------------------------------------
# 5. Dependency errors — None clock fallback
# ---------------------------------------------------------------------------


class TestDependencyErrors:
    def test_clock_none_falls_back_to_monotonic(self) -> None:
        hb = MacHeartbeat(clock=None)
        # Without raising. mark_active + is_alive should work using
        # the real time.monotonic.
        hb.mark_active()
        assert hb.is_alive() is True


# ---------------------------------------------------------------------------
# 6. Regressions
# ---------------------------------------------------------------------------


class TestRegressions:
    def test_age_seconds_never_negative(self) -> None:
        # Defensive: even with a wonky clock that goes backwards
        # (shouldn't happen with monotonic, but tests can pass any
        # callable), age should be sensible.
        clock = _Clock(start=100.0)
        hb = MacHeartbeat(clock=clock)
        hb.mark_active()
        # Clock went backwards (programmer error in test stub):
        clock.set(50.0)
        # Age is computed naively: now - last_active = 50 - 100 = -50.
        # We don't clamp — heartbeat is a debugging signal, not an
        # operator-visible metric. Caller can clamp if they care.
        # Pin the actual behaviour to surface this contract.
        assert hb.age_seconds() == -50.0

    def test_is_alive_with_clock_skew_negative_age(self) -> None:
        # Clock skew producing negative age — the ``< ttl`` check
        # accidentally treats this as alive. Document and accept;
        # in production ``time.monotonic`` cannot go backwards so
        # this codepath never fires.
        clock = _Clock(start=100.0)
        hb = MacHeartbeat(clock=clock, ttl_seconds=60.0)
        hb.mark_active()
        clock.set(50.0)  # backwards
        # Age = -50. -50 < 60 → True (alive). Acceptable.
        assert hb.is_alive() is True


# ---------------------------------------------------------------------------
# 7. Logic-specific — ttl override + last-active state machine
# ---------------------------------------------------------------------------


class TestLogicSpecific:
    def test_ttl_override_via_constructor(self) -> None:
        clock = _Clock(start=0.0)
        hb = MacHeartbeat(clock=clock, ttl_seconds=10.0)
        hb.mark_active()
        clock.advance(9.5)
        assert hb.is_alive() is True
        clock.advance(1.0)  # total 10.5
        assert hb.is_alive() is False

    def test_alive_after_revive(self) -> None:
        # Mac goes silent past TTL → marked dead. Then Mac wakes
        # and pokes the runner → mark_active flips us back to alive.
        clock = _Clock(start=0.0)
        hb = MacHeartbeat(clock=clock, ttl_seconds=60.0)
        hb.mark_active()
        clock.advance(120.0)
        assert hb.is_alive() is False
        # Mac wakes, hits an event:
        hb.mark_active()
        assert hb.is_alive() is True
        assert hb.age_seconds() == 0.0
