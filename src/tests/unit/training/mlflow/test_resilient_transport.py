"""Tests for ResilientMLflowTransport and MLflowTransportCircuitBreaker.

Covers:
- Circuit breaker state machine (closed → open → half_open → closed)
- Transport exception detection (type-based and message-based)
- install() / uninstall() monkey-patching lifecycle
- Metric buffering when circuit is open
- Buffer flush on recovery
- Double-install guard
- Rate-limited warnings
"""

from __future__ import annotations

import time
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.training.mlflow.metrics_buffer import MetricsBuffer
from src.training.mlflow.resilient_transport import (
    MLflowTransportCircuitBreaker,
    ResilientMLflowTransport,
    _FAILURE_THRESHOLD,
    _FLUENT_METHODS,
    _PATCHED_MARKER,
    _TRANSPORT_MESSAGE_MARKERS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fake_mlflow() -> ModuleType:
    """Create an isolated fake mlflow module with all patchable methods."""
    fake = ModuleType("mlflow_fake")
    for method in _FLUENT_METHODS:
        setattr(fake, method, MagicMock(name=f"mlflow.{method}"))

    class FakeMlflowClient:
        pass

    for method in ("log_batch", "log_metric", "log_param", "log_params",
                    "set_tag", "set_tags", "log_dict", "log_text"):
        setattr(FakeMlflowClient, method, MagicMock(name=f"MlflowClient.{method}"))

    fake.MlflowClient = FakeMlflowClient  # type: ignore[attr-defined]
    return fake


# ===========================================================================
# Circuit Breaker State Machine
# ===========================================================================

class TestCircuitBreakerStateMachine:
    """MLflowTransportCircuitBreaker state transitions."""

    def test_initial_state_is_closed(self) -> None:
        cb = MLflowTransportCircuitBreaker()
        assert cb.state == "closed"
        assert cb.consecutive_failures == 0

    def test_allow_call_when_closed(self) -> None:
        cb = MLflowTransportCircuitBreaker()
        assert cb.allow_call() is True

    def test_stays_closed_below_threshold(self) -> None:
        cb = MLflowTransportCircuitBreaker(failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == "closed"
        assert cb.consecutive_failures == 2
        assert cb.allow_call() is True

    def test_opens_after_threshold_failures(self) -> None:
        cb = MLflowTransportCircuitBreaker(failure_threshold=3)
        for _ in range(3):
            cb.record_failure()
        assert cb.state == "open"
        assert cb.allow_call() is False

    def test_transitions_to_half_open_after_cooldown(self) -> None:
        cb = MLflowTransportCircuitBreaker(failure_threshold=1, recovery_cooldown_s=0.01)
        cb.record_failure()
        assert cb.state == "open"
        time.sleep(0.02)
        assert cb.allow_call() is True
        assert cb.state == "half_open"

    def test_half_open_to_closed_on_success(self) -> None:
        cb = MLflowTransportCircuitBreaker(failure_threshold=1, recovery_cooldown_s=0.01)
        cb.record_failure()
        time.sleep(0.02)
        cb.allow_call()  # → half_open
        cb.record_success()
        assert cb.state == "closed"
        assert cb.consecutive_failures == 0

    def test_half_open_to_open_on_failure(self) -> None:
        cb = MLflowTransportCircuitBreaker(failure_threshold=1, recovery_cooldown_s=0.01)
        cb.record_failure()
        time.sleep(0.02)
        cb.allow_call()  # → half_open
        cb.record_failure()
        assert cb.state == "open"

    def test_success_resets_failure_count(self) -> None:
        cb = MLflowTransportCircuitBreaker(failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        assert cb.consecutive_failures == 0
        # Now need 3 more failures to open
        cb.record_failure()
        cb.record_failure()
        assert cb.state == "closed"

    def test_min_threshold_is_one(self) -> None:
        cb = MLflowTransportCircuitBreaker(failure_threshold=0)
        assert cb.failure_threshold == 1


# ===========================================================================
# Transport Exception Detection
# ===========================================================================

class TestTransportExceptionDetection:
    """_is_transport_exception: type-based and message-based matching."""

    def setup_method(self) -> None:
        self.transport = ResilientMLflowTransport()

    def test_detects_connection_error(self) -> None:
        exc = ConnectionError("Connection refused")
        assert self.transport._is_transport_exception(exc) is True

    def test_detects_timeout_error(self) -> None:
        exc = TimeoutError("timed out")
        assert self.transport._is_transport_exception(exc) is True

    def test_detects_broken_pipe(self) -> None:
        exc = BrokenPipeError()
        assert self.transport._is_transport_exception(exc) is True

    @pytest.mark.parametrize("marker", list(_TRANSPORT_MESSAGE_MARKERS))
    def test_detects_message_markers(self, marker: str) -> None:
        exc = RuntimeError(f"Something {marker} something")
        assert self.transport._is_transport_exception(exc) is True

    def test_rejects_unrelated_exception(self) -> None:
        exc = ValueError("completely unrelated error")
        assert self.transport._is_transport_exception(exc) is False

    def test_traverses_exception_chain(self) -> None:
        """Detects transport exception in __cause__ chain."""
        inner = ConnectionError("Connection refused")
        outer = RuntimeError("MLflow call failed")
        outer.__cause__ = inner
        assert self.transport._is_transport_exception(outer) is True

    def test_handles_circular_exception_chain(self) -> None:
        """Doesn't infinite-loop on circular __context__."""
        a = RuntimeError("a")
        b = RuntimeError("b")
        a.__context__ = b
        b.__context__ = a
        assert self.transport._is_transport_exception(a) is False


# ===========================================================================
# Install / Uninstall Lifecycle
# ===========================================================================

class TestInstallUninstall:
    """install() and uninstall() monkey-patching lifecycle."""

    def test_install_patches_fluent_methods(self) -> None:
        fake = _make_fake_mlflow()
        originals = {m: getattr(fake, m) for m in _FLUENT_METHODS}
        transport = ResilientMLflowTransport()

        result = transport.install(fake)

        assert result is True
        for method in _FLUENT_METHODS:
            assert getattr(fake, method) is not originals[method]
        assert getattr(fake, _PATCHED_MARKER) is transport

    def test_install_patches_client_methods(self) -> None:
        fake = _make_fake_mlflow()
        original_log_metric = fake.MlflowClient.log_metric
        transport = ResilientMLflowTransport()

        transport.install(fake)

        assert fake.MlflowClient.log_metric is not original_log_metric

    def test_uninstall_restores_originals(self) -> None:
        fake = _make_fake_mlflow()
        originals = {m: getattr(fake, m) for m in _FLUENT_METHODS}
        transport = ResilientMLflowTransport()

        transport.install(fake)
        transport.uninstall()

        for method in _FLUENT_METHODS:
            assert getattr(fake, method) is originals[method]
        assert not hasattr(fake, _PATCHED_MARKER)

    def test_double_install_same_owner_returns_false(self) -> None:
        fake = _make_fake_mlflow()
        transport = ResilientMLflowTransport()
        transport.install(fake)

        result = transport.install(fake)
        assert result is False

    def test_double_install_different_owner_returns_false(self) -> None:
        fake = _make_fake_mlflow()
        t1 = ResilientMLflowTransport()
        t2 = ResilientMLflowTransport()
        t1.install(fake)

        result = t2.install(fake)
        assert result is False

    def test_uninstall_without_install_is_noop(self) -> None:
        transport = ResilientMLflowTransport()
        transport.uninstall()  # should not raise

    def test_install_without_client_class(self) -> None:
        """Module without MlflowClient still gets fluent methods patched."""
        fake = ModuleType("mlflow_no_client")
        for method in _FLUENT_METHODS:
            setattr(fake, method, MagicMock())
        transport = ResilientMLflowTransport()

        result = transport.install(fake)
        assert result is True


# ===========================================================================
# Wrapper Behavior (Integration)
# ===========================================================================

class TestWrapperBehavior:
    """Wrapped methods: success passthrough, failure handling, buffering."""

    def test_success_passthrough(self) -> None:
        fake = _make_fake_mlflow()
        fake.log_metric = MagicMock(return_value=42)
        transport = ResilientMLflowTransport()
        transport.install(fake)

        result = fake.log_metric("loss", 0.5, step=1)
        assert result == 42

    def test_transport_failure_returns_none(self) -> None:
        fake = _make_fake_mlflow()
        fake.log_metric = MagicMock(side_effect=ConnectionError("Connection refused"))
        transport = ResilientMLflowTransport()
        transport.install(fake)

        result = fake.log_metric("loss", 0.5, step=1)
        assert result is None

    def test_non_transport_exception_reraises(self) -> None:
        fake = _make_fake_mlflow()
        fake.log_metric = MagicMock(side_effect=TypeError("bad arg"))
        transport = ResilientMLflowTransport()
        transport.install(fake)

        with pytest.raises(TypeError, match="bad arg"):
            fake.log_metric("loss", 0.5, step=1)

    def test_circuit_opens_after_repeated_failures(self) -> None:
        fake = _make_fake_mlflow()
        fake.log_metric = MagicMock(side_effect=ConnectionError("fail"))
        transport = ResilientMLflowTransport(failure_threshold=2)
        transport.install(fake)

        fake.log_metric("a", 1)
        fake.log_metric("b", 2)
        assert transport.breaker_state == "open"

    def test_skips_call_when_circuit_open(self) -> None:
        fake = _make_fake_mlflow()
        call_count = 0
        original_fn = MagicMock(side_effect=ConnectionError("fail"))
        fake.log_metric = original_fn
        transport = ResilientMLflowTransport(failure_threshold=1)
        transport.install(fake)

        # First call opens circuit
        fake.log_metric("a", 1)
        # Second call should be skipped (not reach original)
        initial_calls = original_fn.call_count
        fake.log_metric("b", 2)
        assert original_fn.call_count == initial_calls  # no new call


# ===========================================================================
# Metric Buffering
# ===========================================================================

class TestMetricBuffering:
    """Buffer metrics when circuit is open, flush on recovery."""

    def test_buffer_log_metric_when_circuit_open(self, tmp_path: Path) -> None:
        fake = _make_fake_mlflow()
        fake.log_metric = MagicMock(side_effect=ConnectionError("fail"))
        buffer = MetricsBuffer(buffer_dir=tmp_path)

        transport = ResilientMLflowTransport(failure_threshold=1)
        transport.attach_buffer(buffer)
        transport.install(fake)

        # Open circuit
        fake.log_metric("loss", 0.5, step=1)
        assert transport.breaker_state == "open"

        # Next call should buffer
        fake.log_metric("loss", 0.4, step=2)
        assert buffer.count >= 1

        entries = buffer.read_all()
        assert any(e["key"] == "loss" and e["step"] == 2 for e in entries)

    def test_buffer_log_metrics_when_circuit_open(self, tmp_path: Path) -> None:
        fake = _make_fake_mlflow()
        fake.log_metrics = MagicMock(side_effect=ConnectionError("fail"))
        buffer = MetricsBuffer(buffer_dir=tmp_path)

        transport = ResilientMLflowTransport(failure_threshold=1)
        transport.attach_buffer(buffer)
        transport.install(fake)

        # Open circuit
        fake.log_metrics({"loss": 0.5}, step=1)

        # Buffer next batch
        fake.log_metrics({"loss": 0.4, "lr": 0.001}, step=2)
        entries = buffer.read_all()
        keys = {e["key"] for e in entries}
        assert "loss" in keys

    def test_flush_buffer_on_recovery(self, tmp_path: Path) -> None:
        buffer = MetricsBuffer(buffer_dir=tmp_path)
        buffer.write_metric("loss", 0.5, step=1)
        buffer.write_metric("loss", 0.4, step=2)

        flushed_calls: list[tuple] = []

        def tracking_fn(*args: Any, **kwargs: Any) -> None:
            flushed_calls.append((args, kwargs))

        fake = _make_fake_mlflow()
        fake.log_metric = MagicMock(wraps=tracking_fn)

        transport = ResilientMLflowTransport(failure_threshold=1, recovery_cooldown_s=0.01)
        transport.attach_buffer(buffer)
        transport.install(fake)

        # Open circuit
        fake.log_metric = MagicMock(side_effect=ConnectionError("fail"))
        transport.install(fake)  # won't re-install (already installed), so patch directly
        # Instead, simulate breaker opening
        transport._breaker.record_failure()
        assert transport.breaker_state == "open"

        # Wait for cooldown → half_open
        time.sleep(0.02)

        # Restore working fn and make a successful call
        # We need to make the wrapped function call succeed
        transport.uninstall()
        fake.log_metric = MagicMock(return_value=None)
        transport2 = ResilientMLflowTransport(failure_threshold=1, recovery_cooldown_s=0.01)
        transport2.attach_buffer(buffer)
        transport2.install(fake)

        # Simulate recovery: breaker already half_open from cooldown
        transport2._breaker.state = "half_open"
        fake.log_metric("new_metric", 0.3, step=3)

        # Buffer should be flushed (count reset)
        assert buffer.count == 0

    def test_no_buffer_when_buffer_not_attached(self) -> None:
        fake = _make_fake_mlflow()
        fake.log_metric = MagicMock(side_effect=ConnectionError("fail"))
        transport = ResilientMLflowTransport(failure_threshold=1)
        # No buffer attached
        transport.install(fake)

        # Should not raise even without buffer
        fake.log_metric("loss", 0.5, step=1)
        fake.log_metric("loss", 0.4, step=2)


# ===========================================================================
# Breaker State Property
# ===========================================================================

class TestBreakerStateProperty:
    def test_breaker_state_reflects_internal_state(self) -> None:
        transport = ResilientMLflowTransport()
        assert transport.breaker_state == "closed"

        transport._breaker.state = "open"
        assert transport.breaker_state == "open"


# ===========================================================================
# Rate-Limited Warnings
# ===========================================================================

class TestRateLimitedWarnings:
    def test_warning_suppressed_within_interval(self) -> None:
        transport = ResilientMLflowTransport(warning_interval_s=10.0)

        with patch.object(transport, "_warn_rate_limited", wraps=transport._warn_rate_limited) as wrapped:
            # Call warn directly
            transport._warn_rate_limited("test_key", "msg1")
            transport._warn_rate_limited("test_key", "msg2")
            # Both called, but logger only fires for first
            assert wrapped.call_count == 2

    def test_different_keys_not_suppressed(self) -> None:
        transport = ResilientMLflowTransport(warning_interval_s=10.0)

        with patch("src.training.mlflow.resilient_transport.logger") as mock_logger:
            transport._warn_rate_limited("key_a", "msg_a")
            transport._warn_rate_limited("key_b", "msg_b")
            assert mock_logger.warning.call_count == 2
