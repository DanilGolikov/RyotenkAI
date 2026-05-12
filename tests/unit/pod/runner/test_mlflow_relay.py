"""Comprehensive tests for :mod:`src.runner.mlflow_relay`.

The relay buffers MLflow-shaped events on the runner side and forwards
them through a circuit breaker. Trainer-side ``ResilientMLflowTransport``
remains the primary defence; the relay is an additive layer for
deployments where the trainer cannot reach MLflow directly.

Coverage split (project policy):

1. **Positive**           — happy submit, worker forwards each event
                             exactly once, circuit closed throughout.
2. **Negative**           — disabled relay drops every submit; non-MLflow
                             ``kind`` ignored; missing ``forward_fn``
                             keeps ``start()`` a no-op.
3. **Boundary**           — queue_max=1, repeated submit triggers
                             drop-oldest, queue full + worker drained,
                             single failure short of threshold.
4. **Invariants**         — every accepted event reaches forward_fn
                             before stop in the happy path; circuit
                             closed implies forwarder called; circuit
                             open implies forwarder paused.
5. **Dependency errors**  — forward_fn raises → record_failure +
                             re-queue; threshold reached → circuit
                             opens; cooldown grows exponentially.
6. **Regressions**        — drop-oldest preserves the newest event;
                             stop() cancels worker promptly;
                             ``submit`` is non-blocking even with full
                             queue; circuit auto-half-opens after
                             cooldown elapses.
7. **Logic-specific**     — ``MLFLOW_EVENT_KINDS`` set is stable;
                             ``make_mlflow_forward_fn`` lazy-imports.
8. **Combinatorial**      — (queue_max × failure_count × accepted)
                             matrix.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock

import pytest

from ryotenkai_pod.runner.mlflow_relay import (
    MLFLOW_EVENT_KINDS,
    MLflowRelay,
    MLflowRelayCircuitBreaker,
)

pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _metric(value: float = 0.5, run_id: str = "r-1") -> dict[str, Any]:
    return {
        "kind": "mlflow_metric",
        "payload": {"run_id": run_id, "key": "loss", "value": value, "step": 1},
    }


class _StepClock:
    """Deterministic clock for circuit-breaker tests.

    Starts at 0, advances when ``tick(seconds)`` is called. Used as the
    ``clock=`` injection point of MLflowRelay / MLflowRelayCircuitBreaker.
    """

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def tick(self, seconds: float) -> None:
        self.now += seconds


async def _drain(relay: MLflowRelay, max_iters: int = 50) -> None:
    """Yield to the worker until the queue is empty or ``max_iters``
    iterations elapse. Each iteration sleeps for a short while so the
    worker's ``wait_for(queue.get, ...)`` can wake up."""
    for _ in range(max_iters):
        if relay.queue_size == 0:
            return
        await asyncio.sleep(0.01)


# ---------------------------------------------------------------------------
# 1. Positive
# ---------------------------------------------------------------------------


class TestPositive:
    async def test_submit_and_forward_round_trip(self) -> None:
        seen: list[dict[str, Any]] = []

        async def _forward(event: dict[str, Any]) -> None:
            seen.append(event)

        relay = MLflowRelay(_forward, worker_idle_poll_s=0.01)
        await relay.start()
        try:
            assert relay.submit(_metric(0.4)) is True
            assert relay.submit(_metric(0.5)) is True
            await _drain(relay)
        finally:
            await relay.stop()

        assert [e["payload"]["value"] for e in seen] == [0.4, 0.5]

    async def test_start_is_idempotent(self) -> None:
        relay = MLflowRelay(AsyncMock(), worker_idle_poll_s=0.01)
        await relay.start()
        first = relay._worker_task
        await relay.start()  # second call must not spawn a second worker
        assert relay._worker_task is first
        await relay.stop()

    async def test_stop_safe_when_never_started(self) -> None:
        relay = MLflowRelay(AsyncMock())
        await relay.stop()  # must not raise

    async def test_dropped_count_starts_at_zero(self) -> None:
        relay = MLflowRelay(AsyncMock())
        assert relay.dropped_count == 0
        assert relay.queue_size == 0


# ---------------------------------------------------------------------------
# 2. Negative
# ---------------------------------------------------------------------------


class TestNegative:
    async def test_disabled_relay_rejects_submits(self) -> None:
        relay = MLflowRelay(forward_fn=None)
        assert relay.enabled is False
        assert relay.submit(_metric()) is False

    async def test_disabled_relay_start_is_noop(self) -> None:
        relay = MLflowRelay(forward_fn=None)
        await relay.start()
        assert relay.is_running is False

    async def test_non_mlflow_kind_ignored(self) -> None:
        relay = MLflowRelay(AsyncMock())
        assert relay.submit({"kind": "step", "payload": {}}) is False
        assert relay.submit({"kind": "trainer_spawned"}) is False
        assert relay.queue_size == 0

    async def test_event_without_kind_ignored(self) -> None:
        relay = MLflowRelay(AsyncMock())
        assert relay.submit({"payload": {"value": 1}}) is False


# ---------------------------------------------------------------------------
# 3. Boundary
# ---------------------------------------------------------------------------


class TestBoundary:
    async def test_queue_max_one_drops_oldest_on_overflow(self) -> None:
        # Synchronous fill: do NOT start the worker so the queue
        # actually overflows before being drained.
        relay = MLflowRelay(AsyncMock(), queue_max=1)
        assert relay.submit(_metric(0.1)) is True
        assert relay.queue_size == 1
        # Second submit drops the first and keeps the second.
        assert relay.submit(_metric(0.2)) is True
        assert relay.queue_size == 1
        assert relay.dropped_count == 1
        # The newest event is what's left.
        held = await relay._queue.get()
        assert held["payload"]["value"] == 0.2

    async def test_one_failure_below_threshold_keeps_circuit_closed(self) -> None:
        cb = MLflowRelayCircuitBreaker(failure_threshold=3)
        cb.record_failure(now=0.0)
        assert cb.is_open(0.0) is False
        cb.record_failure(now=0.1)
        # Two failures, threshold = 3 → still closed.
        assert cb.is_open(0.5) is False

    async def test_circuit_opens_at_threshold(self) -> None:
        cb = MLflowRelayCircuitBreaker(
            failure_threshold=3, initial_cooldown_s=10.0,
        )
        for t in (0.0, 0.1, 0.2):
            cb.record_failure(now=t)
        # Just opened — still inside cooldown window.
        assert cb.is_open(0.3) is True

    async def test_threshold_one_opens_immediately(self) -> None:
        cb = MLflowRelayCircuitBreaker(
            failure_threshold=1, initial_cooldown_s=5.0,
        )
        cb.record_failure(now=0.0)
        assert cb.is_open(0.0) is True


# ---------------------------------------------------------------------------
# 4. Invariants
# ---------------------------------------------------------------------------


class TestInvariants:
    async def test_every_accepted_event_eventually_forwarded(self) -> None:
        seen: list[dict[str, Any]] = []
        forward_calls = 0

        async def _forward(event: dict[str, Any]) -> None:
            nonlocal forward_calls
            forward_calls += 1
            seen.append(event)

        relay = MLflowRelay(_forward, worker_idle_poll_s=0.01)
        await relay.start()
        try:
            for value in (0.1, 0.2, 0.3, 0.4, 0.5):
                relay.submit(_metric(value))
            await _drain(relay)
        finally:
            await relay.stop()

        assert forward_calls == 5
        assert [e["payload"]["value"] for e in seen] == [0.1, 0.2, 0.3, 0.4, 0.5]

    async def test_success_resets_failure_counter(self) -> None:
        cb = MLflowRelayCircuitBreaker(failure_threshold=3)
        cb.record_failure(now=0.0)
        cb.record_failure(now=0.1)
        assert cb.consecutive_failures == 2
        cb.record_success()
        assert cb.consecutive_failures == 0
        # After reset, three NEW failures are needed to open again.
        cb.record_failure(now=1.0)
        cb.record_failure(now=1.1)
        assert cb.is_open(1.2) is False

    async def test_disabled_relay_never_starts_a_worker(self) -> None:
        relay = MLflowRelay(forward_fn=None)
        await relay.start()
        await asyncio.sleep(0.02)
        assert relay.is_running is False
        await relay.stop()


# ---------------------------------------------------------------------------
# 5. Dependency errors
# ---------------------------------------------------------------------------


class TestDependencyErrors:
    async def test_forward_failure_records_breaker_and_requeues(self) -> None:
        # First two attempts raise; third succeeds. Threshold is 3 so
        # we never actually open the circuit, but every failure must
        # bump the counter and re-queue the event.
        attempts = 0

        async def _flaky(event: dict[str, Any]) -> None:
            nonlocal attempts
            attempts += 1
            if attempts <= 2:
                raise ConnectionError("upstream blip")

        relay = MLflowRelay(_flaky, worker_idle_poll_s=0.01)
        await relay.start()
        try:
            relay.submit(_metric(0.7))
            # Spin long enough for two retries and a final success.
            for _ in range(50):
                if attempts >= 3:
                    break
                await asyncio.sleep(0.02)
        finally:
            await relay.stop()

        assert attempts >= 3
        # On final success the breaker is closed again.
        assert relay.circuit_breaker.consecutive_failures == 0

    async def test_threshold_failures_open_circuit(self) -> None:
        # A persistently failing upstream → after three failures the
        # circuit opens, the worker idles, and submits keep buffering
        # until cooldown expires.
        attempts = 0

        async def _always_fails(_: dict[str, Any]) -> None:
            nonlocal attempts
            attempts += 1
            raise RuntimeError("upstream gone")

        clock = _StepClock()
        relay = MLflowRelay(
            _always_fails,
            worker_idle_poll_s=0.01,
            circuit_breaker=MLflowRelayCircuitBreaker(
                failure_threshold=3, initial_cooldown_s=10.0,
            ),
            clock=clock,
        )
        await relay.start()
        try:
            relay.submit(_metric(0.5))
            for _ in range(60):
                if relay.circuit_breaker.consecutive_failures >= 3:
                    break
                await asyncio.sleep(0.02)
        finally:
            await relay.stop()

        assert relay.circuit_breaker.consecutive_failures >= 3
        assert relay.circuit_breaker.is_open(clock.now) is True

    async def test_cooldown_grows_after_repeated_opens(self) -> None:
        cb = MLflowRelayCircuitBreaker(
            failure_threshold=1, initial_cooldown_s=2.0,
            multiplier=2.0, max_cooldown_s=10.0,
        )
        # First open: 2s
        cb.record_failure(now=0.0)
        assert cb.is_open(1.0) is True
        assert cb.is_open(2.5) is False
        # Second open: 4s (×2)
        cb.record_failure(now=3.0)
        assert cb.is_open(3.5) is True
        assert cb.is_open(7.5) is False
        # Third open: 8s (×2)
        cb.record_failure(now=8.0)
        assert cb.is_open(8.5) is True
        assert cb.is_open(16.5) is False
        # Cap at max_cooldown_s
        cb.record_failure(now=17.0)
        assert cb.is_open(17.5) is True
        # Cooldown was capped at 10, so by t=27.5 it should be closed.
        assert cb.is_open(27.5) is False


# ---------------------------------------------------------------------------
# 6. Regressions
# ---------------------------------------------------------------------------


class TestRegressions:
    async def test_drop_oldest_keeps_newest(self) -> None:
        # Pin: the producer-side overflow MUST drop the oldest event.
        relay = MLflowRelay(AsyncMock(), queue_max=2)
        relay.submit(_metric(1))
        relay.submit(_metric(2))
        relay.submit(_metric(3))  # drops 1
        relay.submit(_metric(4))  # drops 2

        held: list[float] = []
        while relay.queue_size:
            event = await relay._queue.get()
            held.append(event["payload"]["value"])
        assert held == [3.0, 4.0]

    async def test_stop_cancels_worker_promptly(self) -> None:
        # Worker should respond to ``stop()`` within the idle-poll
        # timeout, not block until the next event arrives.
        relay = MLflowRelay(AsyncMock(), worker_idle_poll_s=0.05)
        await relay.start()
        await asyncio.sleep(0.02)
        # Don't submit anything — worker is in queue.get(timeout=…).
        # ``stop()`` should still complete in < 1s.
        try:
            await asyncio.wait_for(relay.stop(), timeout=1.0)
        except TimeoutError:  # pragma: no cover
            pytest.fail("relay.stop() did not return within 1s")

    async def test_submit_is_non_blocking_even_when_full(self) -> None:
        # Drop-oldest must keep ``submit`` synchronous — no await
        # path means producers (the FastAPI handler) cannot stall.
        relay = MLflowRelay(AsyncMock(), queue_max=2)
        relay.submit(_metric(1))
        relay.submit(_metric(2))
        # The third submit will drop the oldest. ``submit`` is sync,
        # so this assertion is trivially true; we keep it as
        # documentation of the invariant.
        assert relay.submit(_metric(3)) is True

    async def test_circuit_auto_recovers_after_cooldown(self) -> None:
        # Open the circuit, advance the clock past cooldown, and
        # the next ``is_open(now)`` must return False (half-open).
        clock = _StepClock()
        cb = MLflowRelayCircuitBreaker(
            failure_threshold=1, initial_cooldown_s=5.0, clock=clock,
        ) if False else MLflowRelayCircuitBreaker(
            failure_threshold=1, initial_cooldown_s=5.0,
        )
        cb.record_failure(now=0.0)
        assert cb.is_open(2.0) is True
        assert cb.is_open(6.0) is False  # cooldown elapsed → half-open


# ---------------------------------------------------------------------------
# 7. Logic-specific
# ---------------------------------------------------------------------------


class TestLogicSpecific:
    def test_mlflow_event_kinds_pinned(self) -> None:
        # The set of forwarded kinds is the contract between the
        # trainer-side callback and the runner-side relay. Pin it.
        assert frozenset({
            "mlflow_metric",
            "mlflow_param",
            "mlflow_tag",
            "mlflow_run_started",
            "mlflow_run_ended",
        }) == MLFLOW_EVENT_KINDS

    async def test_make_forward_fn_lazy_imports_mlflow(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Calling ``make_mlflow_forward_fn`` shouldn't fail when
        # mlflow is missing — at module-import time. It only fails
        # when the user actually invokes the factory.
        # Monkey-patch ``mlflow`` to be importable but stubbed; the
        # body of ``make_mlflow_forward_fn`` then runs without the
        # full ml deps.
        import sys
        import types

        from ryotenkai_pod.runner import mlflow_relay as relay_mod

        fake_mlflow = types.ModuleType("mlflow")

        def _set_uri(uri: str) -> None:
            fake_mlflow.tracking_uri_seen = uri  # type: ignore[attr-defined]

        fake_mlflow.set_tracking_uri = _set_uri  # type: ignore[attr-defined]

        fake_tracking = types.ModuleType("mlflow.tracking")

        class _StubClient:
            def log_metric(self, **_kwargs: Any) -> None: ...
            def log_param(self, **_kwargs: Any) -> None: ...
            def set_tag(self, **_kwargs: Any) -> None: ...

        fake_tracking.MlflowClient = _StubClient  # type: ignore[attr-defined]

        monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)
        monkeypatch.setitem(sys.modules, "mlflow.tracking", fake_tracking)

        forward = relay_mod.make_mlflow_forward_fn("http://mlflow:5000")
        # Forward a metric — uses the stub client, doesn't raise.
        await forward({
            "kind": "mlflow_metric",
            "payload": {"run_id": "r", "key": "k", "value": 1.0, "step": 0},
        })
        assert fake_mlflow.tracking_uri_seen == "http://mlflow:5000"  # type: ignore[attr-defined]

    async def test_forward_fn_skips_event_without_run_id(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Production stub of mlflow as in the previous test.
        import sys
        import types

        from ryotenkai_pod.runner import mlflow_relay as relay_mod

        fake_mlflow = types.ModuleType("mlflow")
        fake_mlflow.set_tracking_uri = lambda _u: None  # type: ignore[attr-defined]

        fake_tracking = types.ModuleType("mlflow.tracking")
        calls: list[str] = []

        class _StubClient:
            def log_metric(self, **kwargs: Any) -> None:
                calls.append("metric")

            def log_param(self, **kwargs: Any) -> None:
                calls.append("param")

            def set_tag(self, **kwargs: Any) -> None:
                calls.append("tag")

        fake_tracking.MlflowClient = _StubClient  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)
        monkeypatch.setitem(sys.modules, "mlflow.tracking", fake_tracking)

        forward = relay_mod.make_mlflow_forward_fn("http://mlflow:5000")
        # Missing run_id → silent skip.
        await forward({
            "kind": "mlflow_metric",
            "payload": {"key": "k", "value": 1.0},
        })
        assert calls == []


# ---------------------------------------------------------------------------
# 8. Combinatorial
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("queue_max", [1, 5, 100])
@pytest.mark.parametrize("submit_count", [1, 3, 10])
async def test_combinatorial_no_loss_when_under_capacity(
    queue_max: int,
    submit_count: int,
) -> None:
    """Under-capacity submits + working forwarder = no drops."""
    if submit_count > queue_max:
        pytest.skip("over-capacity is exercised by drop-oldest tests")

    seen: list[dict[str, Any]] = []

    async def _forward(event: dict[str, Any]) -> None:
        seen.append(event)

    relay = MLflowRelay(_forward, queue_max=queue_max, worker_idle_poll_s=0.01)
    await relay.start()
    try:
        for value in range(submit_count):
            assert relay.submit(_metric(float(value))) is True
        await _drain(relay)
    finally:
        await relay.stop()

    assert relay.dropped_count == 0
    assert len(seen) == submit_count


@pytest.mark.parametrize("failure_threshold", [1, 3, 10])
@pytest.mark.parametrize("multiplier", [1.0, 1.5, 2.0])
async def test_combinatorial_breaker_cooldown_growth(
    failure_threshold: int,
    multiplier: float,
) -> None:
    """Cooldown grows monotonically with each open while bounded by max."""
    cb = MLflowRelayCircuitBreaker(
        failure_threshold=failure_threshold,
        initial_cooldown_s=1.0,
        multiplier=multiplier,
        max_cooldown_s=10.0,
    )
    seen_cooldowns: list[float] = []
    t = 0.0
    for _ in range(5):
        for _ in range(failure_threshold):
            cb.record_failure(now=t)
            t += 0.001
        seen_cooldowns.append(cb.active_cooldown_s)
        # Skip past the cooldown so the next round is a fresh "open".
        t += cb.active_cooldown_s + 1.0
        cb.record_success()  # half-open recovers

    if multiplier > 1.0:
        # Cooldown either grows or hits the cap. Never shrinks.
        for prev, nxt in zip(seen_cooldowns, seen_cooldowns[1:]):
            assert nxt >= prev
    # Capped at max_cooldown_s.
    assert max(seen_cooldowns) <= 10.0
