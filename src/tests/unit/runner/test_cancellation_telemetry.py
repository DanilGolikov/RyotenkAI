"""Phase 9.C — tests for cancellation telemetry constants module.

The constants module is the single source of truth for the event-kind
strings the cancellation chain emits. These tests pin the *contract*:

* Constant values are stable (operator dashboards / SLO alerts grep
  for them; renaming would silently break consumers).
* The convenience set ``CANCELLATION_EVENT_KINDS`` covers every kind
  defined as a top-level constant — no drift between exported names
  and the membership set.
* ``now_ms`` returns epoch milliseconds (NOT monotonic), so payloads
  round-trip through JSON / log scrapes / MLflow run metadata.
* ``latency_ms_since`` clamps negative deltas at 0 — clock skew on
  distributed deployments must never surface as "negative latency".
"""

from __future__ import annotations

import time
from unittest.mock import patch

from src.runner import cancellation_telemetry as ct


# ---------------------------------------------------------------------------
# Positive — constant values & exports
# ---------------------------------------------------------------------------


class TestConstantValues:
    """Pin the strings — operator dashboards depend on them verbatim."""

    def test_cancellation_started_string(self) -> None:
        assert ct.CANCELLATION_STARTED == "cancellation_started"

    def test_cancellation_finalized_string(self) -> None:
        assert ct.CANCELLATION_FINALIZED == "cancellation_finalized"

    def test_cancellation_completed_string(self) -> None:
        assert ct.CANCELLATION_COMPLETED == "cancellation_completed"

    def test_cancellation_requested_string(self) -> None:
        assert ct.CANCELLATION_REQUESTED == "cancellation_requested"

    def test_mlflow_reconciled_post_sigkill_string(self) -> None:
        assert (
            ct.MLFLOW_RECONCILED_POST_SIGKILL
            == "mlflow_reconciled_post_sigkill"
        )

    def test_cleanup_pod_failed_string(self) -> None:
        assert ct.CLEANUP_POD_FAILED == "cleanup_pod_failed"


class TestEventKindsSet:
    """The frozenset and the top-level constants must agree."""

    def test_set_is_frozenset(self) -> None:
        assert isinstance(ct.CANCELLATION_EVENT_KINDS, frozenset)

    def test_set_has_six_kinds(self) -> None:
        # If you add a new kind constant, bump this AND add it to the
        # set. The two-step bump is the point — silent drift fails this
        # test.
        assert len(ct.CANCELLATION_EVENT_KINDS) == 6

    def test_set_contains_every_constant(self) -> None:
        expected = {
            ct.CANCELLATION_REQUESTED,
            ct.CANCELLATION_STARTED,
            ct.CANCELLATION_FINALIZED,
            ct.CANCELLATION_COMPLETED,
            ct.MLFLOW_RECONCILED_POST_SIGKILL,
            ct.CLEANUP_POD_FAILED,
        }
        assert ct.CANCELLATION_EVENT_KINDS == expected


class TestModuleExports:
    """``__all__`` is the public-API contract — keep it explicit."""

    def test_all_includes_event_kinds(self) -> None:
        for name in (
            "CANCELLATION_REQUESTED",
            "CANCELLATION_STARTED",
            "CANCELLATION_FINALIZED",
            "CANCELLATION_COMPLETED",
            "MLFLOW_RECONCILED_POST_SIGKILL",
            "CLEANUP_POD_FAILED",
        ):
            assert name in ct.__all__

    def test_all_includes_helpers(self) -> None:
        assert "now_ms" in ct.__all__
        assert "latency_ms_since" in ct.__all__
        assert "CANCELLATION_EVENT_KINDS" in ct.__all__


# ---------------------------------------------------------------------------
# Positive / boundary — now_ms semantics
# ---------------------------------------------------------------------------


class TestNowMs:
    def test_returns_int(self) -> None:
        result = ct.now_ms()
        assert isinstance(result, int)

    def test_close_to_wall_clock_time(self) -> None:
        # Within ±100ms of wall clock — wide enough for slow CI, tight
        # enough to catch a regression where someone swaps in
        # ``time.monotonic`` (which would return a relative value, not
        # epoch).
        wall = int(time.time() * 1000)
        result = ct.now_ms()
        assert abs(result - wall) < 100

    def test_uses_time_dot_time_not_monotonic(self) -> None:
        # If ``now_ms`` ever switches to ``time.monotonic``, JSON
        # round-tripping breaks (monotonic is per-process, not epoch).
        # Patch ``time.time`` and confirm the return value follows.
        with patch("src.runner.cancellation_telemetry.time.time") as mock_time:
            mock_time.return_value = 1234567890.123  # epoch seconds
            result = ct.now_ms()
            assert result == 1234567890123  # ms

    def test_two_consecutive_calls_monotonic(self) -> None:
        # Even though we use ``time.time``, two consecutive calls in
        # the same test must not go backwards. Sanity check; catches
        # weird clock-rollback regressions.
        a = ct.now_ms()
        b = ct.now_ms()
        assert b >= a


# ---------------------------------------------------------------------------
# Positive — latency_ms_since semantics
# ---------------------------------------------------------------------------


class TestLatencyMsSince:
    def test_returns_int(self) -> None:
        start = ct.now_ms() - 100
        result = ct.latency_ms_since(start)
        assert isinstance(result, int)

    def test_recent_anchor_returns_small_positive(self) -> None:
        start = ct.now_ms() - 50  # 50 ms ago
        result = ct.latency_ms_since(start)
        # 50 ms ago → result ≥ 50, but bounded by test runtime
        assert 30 <= result < 5000  # generous CI budget

    def test_distant_past_anchor_returns_large_positive(self) -> None:
        start = ct.now_ms() - 60_000  # 60 s ago
        result = ct.latency_ms_since(start)
        assert result >= 60_000

    # ----- boundary -----

    def test_future_anchor_clamps_at_zero(self) -> None:
        # Anchor in the future (clock skew) → the function MUST NOT
        # return a negative value. Operator dashboards parse these as
        # u64 — negatives would either wrap or surface as
        # "suspiciously fast" outliers.
        future = ct.now_ms() + 10_000  # 10 s ahead
        result = ct.latency_ms_since(future)
        assert result == 0

    def test_anchor_equal_to_now_returns_near_zero(self) -> None:
        # Edge: anchor == now. ``latency_ms_since`` returns ≥ 0; a few
        # ms of accumulated time inside the function is acceptable.
        anchor = ct.now_ms()
        result = ct.latency_ms_since(anchor)
        assert result >= 0
        assert result < 100  # generous for slow CI

    # ----- invariant -----

    def test_never_returns_negative(self) -> None:
        # Combinatorial: across a range of anchors (past, equal,
        # future), the function never returns a negative value.
        now = ct.now_ms()
        for offset_ms in (-1000, -10, 0, 10, 1000, 60_000):
            anchor = now + offset_ms
            result = ct.latency_ms_since(anchor)
            assert result >= 0, f"negative latency for offset={offset_ms}"


# ---------------------------------------------------------------------------
# Regression — module shape
# ---------------------------------------------------------------------------


class TestModuleShape:
    """Pin the surface area; renaming a public symbol is a contract
    break that every grep'd consumer notices.
    """

    def test_no_unexpected_public_names(self) -> None:
        # Anything starting with a non-underscore should be either in
        # ``__all__`` or a stdlib symbol exported by import (``time``).
        public = {
            name for name in dir(ct)
            if not name.startswith("_")
        }
        allowed = set(ct.__all__) | {"time", "annotations"}
        unexpected = public - allowed
        assert not unexpected, (
            f"Unexpected public names in cancellation_telemetry: "
            f"{sorted(unexpected)}"
        )
