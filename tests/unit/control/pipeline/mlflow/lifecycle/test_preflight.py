"""Unit tests: ``PreflightConnectivityCheck``.

7-class structure (positive / negative / boundary / invariants /
dependency-errors / regressions / logic-specific) mandated by
``docs/testing/mock_policy.md``.

Coverage:

* Happy path - ping succeeds, returns None silently.
* Network failure - ProviderUnavailableError with context.
* Boundary - timeout <= 0 rejected at construction time.
* Invariant - check never opens a probe run (R-02 mitigation).
* Dependency errors - arbitrary exception classes from ping are
  wrapped into ProviderUnavailableError.
* Regression - exception chaining via __cause__ preserved.
* Logic-specific - context dict carries transport_uri / timeout_s /
  cause_class for ops triage.
"""

from __future__ import annotations

import pytest

from ryotenkai_control.pipeline.mlflow.lifecycle.preflight import (
    PreflightConnectivityCheck,
)
from ryotenkai_shared.errors import ProviderUnavailableError
from tests._fakes.mlflow_tracking_client import FakeTrackingClient


# ===========================================================================
# 1. Positive
# ===========================================================================


class TestPositive:
    """Happy path - ping succeeds; check returns silently."""

    def test_ping_success_returns_none(self) -> None:
        client = FakeTrackingClient()
        check = PreflightConnectivityCheck(client)

        result = check.run()

        assert result is None
        assert len(client.ping_calls) == 1

    def test_default_timeout_is_five_seconds(self) -> None:
        """Default mirrors legacy ``MLflowManager.ping_timeout_s``."""
        client = FakeTrackingClient()
        check = PreflightConnectivityCheck(client)

        check.run()

        assert client.ping_calls == [5.0]

    def test_custom_timeout_propagated(self) -> None:
        client = FakeTrackingClient()
        check = PreflightConnectivityCheck(client, timeout_s=12.5)

        check.run()

        assert client.ping_calls == [12.5]


# ===========================================================================
# 2. Negative
# ===========================================================================


class TestNegative:
    """Ping failures must raise typed ProviderUnavailableError."""

    def test_ping_failure_raises_provider_unavailable(self) -> None:
        client = FakeTrackingClient(ping_should_raise=ConnectionError("boom"))
        check = PreflightConnectivityCheck(client)

        with pytest.raises(ProviderUnavailableError):
            check.run()

    def test_oserror_wrapped_too(self) -> None:
        client = FakeTrackingClient(ping_should_raise=OSError("dns"))
        check = PreflightConnectivityCheck(client)

        with pytest.raises(ProviderUnavailableError):
            check.run()


# ===========================================================================
# 3. Boundary
# ===========================================================================


class TestBoundary:
    """Edge cases on the timeout parameter."""

    @pytest.mark.parametrize("bad", [0, -1.0, -5.0])
    def test_non_positive_timeout_rejected(self, bad: float) -> None:
        client = FakeTrackingClient()
        with pytest.raises(ValueError, match="timeout_s"):
            PreflightConnectivityCheck(client, timeout_s=bad)

    def test_very_small_positive_timeout_accepted(self) -> None:
        client = FakeTrackingClient()
        check = PreflightConnectivityCheck(client, timeout_s=0.001)
        check.run()
        assert client.ping_calls == [0.001]


# ===========================================================================
# 4. Invariants
# ===========================================================================


class TestInvariants:
    """Contract pins that hold across all preflight invocations."""

    def test_preflight_never_opens_probe_run(self) -> None:
        """R-02: preflight MUST NOT call start_run on the tracking client.

        Opening a probe run would leak artifacts in the experiment and
        race with the orchestrator's run counter.
        """
        client = FakeTrackingClient()
        check = PreflightConnectivityCheck(client)

        check.run()

        assert client.start_run_calls == []
        assert client.start_nested_run_calls == []

    def test_preflight_no_run_state_mutation(self) -> None:
        client = FakeTrackingClient()
        check = PreflightConnectivityCheck(client)

        check.run()

        # No tags, terminations, adopts.
        assert client.set_tags_calls == []
        assert client.terminated_calls == []
        assert client.adopt_run_calls == []

    def test_preflight_calls_ping_exactly_once_per_run(self) -> None:
        client = FakeTrackingClient()
        check = PreflightConnectivityCheck(client)

        check.run()
        check.run()
        check.run()

        assert len(client.ping_calls) == 3


# ===========================================================================
# 5. Dependency errors
# ===========================================================================


class TestDependencyErrors:
    """Various transport-level exceptions all collapse to one typed surface."""

    @pytest.mark.parametrize(
        "exc",
        [
            ConnectionError("net"),
            TimeoutError("slow"),
            OSError("dns"),
            RuntimeError("weird"),
            ValueError("auth"),
        ],
    )
    def test_arbitrary_exception_wrapped(self, exc: Exception) -> None:
        client = FakeTrackingClient(ping_should_raise=exc)
        check = PreflightConnectivityCheck(client)

        with pytest.raises(ProviderUnavailableError):
            check.run()


# ===========================================================================
# 6. Regressions
# ===========================================================================


class TestRegressions:
    """Tightly-scoped pins for behaviours that previously regressed."""

    def test_cause_chained_via_dunder_cause(self) -> None:
        """The original exception MUST be preserved via __cause__ so the
        traceback shows the network layer's error -- otherwise ops cannot
        diagnose 401 vs DNS-down from the structured log."""
        original = ConnectionError("dns failed")
        client = FakeTrackingClient(ping_should_raise=original)
        check = PreflightConnectivityCheck(client)

        with pytest.raises(ProviderUnavailableError) as excinfo:
            check.run()

        assert excinfo.value.__cause__ is original

    def test_failure_does_not_call_ping_twice(self) -> None:
        """No retry inside preflight - that is RetryPolicy's job."""
        client = FakeTrackingClient(ping_should_raise=ConnectionError("x"))
        check = PreflightConnectivityCheck(client)

        with pytest.raises(ProviderUnavailableError):
            check.run()

        assert len(client.ping_calls) == 1


# ===========================================================================
# 7. Logic-specific
# ===========================================================================


class TestLogicSpecific:
    """Pin the structured-context shape of the typed error."""

    def test_context_carries_timeout(self) -> None:
        client = FakeTrackingClient(ping_should_raise=ConnectionError("x"))
        check = PreflightConnectivityCheck(client, timeout_s=7.0)

        with pytest.raises(ProviderUnavailableError) as excinfo:
            check.run()

        assert excinfo.value.context["timeout_s"] == 7.0

    def test_context_carries_cause_class_name(self) -> None:
        client = FakeTrackingClient(ping_should_raise=ConnectionError("x"))
        check = PreflightConnectivityCheck(client)

        with pytest.raises(ProviderUnavailableError) as excinfo:
            check.run()

        assert excinfo.value.context["cause_class"] == "ConnectionError"

    def test_context_carries_transport_uri_key_even_when_unknown(self) -> None:
        """``FakeTrackingClient`` does not expose ``tracking_uri`` attribute;
        the preflight must still produce a populated context key (the
        ``getattr`` fallback)."""
        client = FakeTrackingClient(ping_should_raise=ConnectionError("x"))
        check = PreflightConnectivityCheck(client)

        with pytest.raises(ProviderUnavailableError) as excinfo:
            check.run()

        # The key is always present; the value is "<unknown>" sentinel.
        assert "transport_uri" in excinfo.value.context

    def test_detail_mentions_cause(self) -> None:
        client = FakeTrackingClient(ping_should_raise=ConnectionError("net"))
        check = PreflightConnectivityCheck(client)

        with pytest.raises(ProviderUnavailableError) as excinfo:
            check.run()

        assert excinfo.value.detail is not None
        assert "ConnectionError" in excinfo.value.detail
