"""Phase 11.E — :class:`MacHeartbeat` explicit-TTL contract.

The Phase 11.B :class:`MacHeartbeat` had a single class-level TTL.
Phase 11.E adds a per-mark TTL override so explicit control-plane
pings can request a longer freshness window (e.g. 120 s) than
implicit WS / REST pings (60 s).
"""

from __future__ import annotations

import pytest

from src.runner.heartbeat import MacHeartbeat


# ---------------------------------------------------------------------------
# 1. Positive — TTL kwargs accepted
# ---------------------------------------------------------------------------


class TestPositive:
    def test_default_mark_uses_class_ttl(self) -> None:
        clock = [0.0]
        hb = MacHeartbeat(clock=lambda: clock[0])
        hb.mark_active()
        # Default TTL = 60s. At t=59 still alive.
        clock[0] = 59.0
        assert hb.is_alive() is True
        # At t=61 stale.
        clock[0] = 61.0
        assert hb.is_alive() is False

    def test_explicit_ttl_extends_freshness(self) -> None:
        clock = [0.0]
        hb = MacHeartbeat(clock=lambda: clock[0])
        # Explicit ping with 120s TTL.
        hb.mark_active(ttl_seconds=120.0)
        # At t=100 still alive (would be stale on default 60s).
        clock[0] = 100.0
        assert hb.is_alive() is True
        # At t=121 stale.
        clock[0] = 121.0
        assert hb.is_alive() is False

    def test_class_constant_for_explicit(self) -> None:
        # Pin the documented default — operator dashboards may key on this.
        assert MacHeartbeat.EXPLICIT_HEARTBEAT_TTL_SECONDS == 120.0


# ---------------------------------------------------------------------------
# 2. Negative — None falls back to default
# ---------------------------------------------------------------------------


class TestNegative:
    def test_explicit_ttl_none_falls_back_to_default(self) -> None:
        clock = [0.0]
        hb = MacHeartbeat(clock=lambda: clock[0], ttl_seconds=30.0)
        hb.mark_active(ttl_seconds=None)
        clock[0] = 29.0
        assert hb.is_alive() is True
        clock[0] = 31.0
        assert hb.is_alive() is False


# ---------------------------------------------------------------------------
# 3. Boundary — TTL changes between marks
# ---------------------------------------------------------------------------


class TestBoundary:
    def test_short_ttl_then_long_ttl_picks_long(self) -> None:
        # First a short-TTL implicit ping, then a long-TTL explicit
        # ping. The most recent mark's TTL wins — Mac just told us
        # it'll be quiet for a while, we trust that.
        clock = [0.0]
        hb = MacHeartbeat(clock=lambda: clock[0])
        hb.mark_active()  # default 60s
        clock[0] = 30.0
        hb.mark_active(ttl_seconds=200.0)  # explicit long
        # At t=180 still alive (would be stale on 60s, alive on 200s).
        clock[0] = 180.0
        assert hb.is_alive() is True

    def test_long_ttl_then_short_ttl_picks_short(self) -> None:
        # Symmetric — long ping first, then short. New short TTL wins.
        clock = [0.0]
        hb = MacHeartbeat(clock=lambda: clock[0])
        hb.mark_active(ttl_seconds=200.0)
        clock[0] = 30.0
        hb.mark_active(ttl_seconds=10.0)
        # At t=45 stale (10s TTL since the short mark at t=30).
        clock[0] = 45.0
        assert hb.is_alive() is False


# ---------------------------------------------------------------------------
# 4. Invariants
# ---------------------------------------------------------------------------


class TestInvariants:
    def test_never_marked_is_dead_regardless_of_ttl(self) -> None:
        hb = MacHeartbeat()
        assert hb.is_alive() is False

    def test_age_seconds_unchanged_by_ttl(self) -> None:
        # TTL controls is_alive() boolean only — age_seconds reports
        # actual elapsed time regardless.
        clock = [0.0]
        hb = MacHeartbeat(clock=lambda: clock[0])
        hb.mark_active(ttl_seconds=200.0)
        clock[0] = 50.0
        assert hb.age_seconds() == pytest.approx(50.0)
