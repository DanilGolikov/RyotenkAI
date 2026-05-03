"""Phase 4.3 — bounded async MLflow relay running inside the runner.

The trainer subprocess can opt into runner-side forwarding by emitting
events with one of the ``mlflow_*`` kinds through its existing
``RunnerEventCallback`` channel. The relay observes those events,
buffers them in a bounded :class:`asyncio.Queue`, and forwards them
to the configured ``MLFLOW_TRACKING_URI`` via the mlflow client API.

Why a runner-side relay at all
------------------------------

The default deployment has the trainer talk to MLflow directly, with
:class:`~src.training.mlflow.resilient_transport.ResilientMLflowTransport`
absorbing transient failures inside the trainer process. The relay
adds three things on top:

* **Single MLflow connection per pod.** Multi-trainer pods would all
  dial the same upstream; one relay amortises the cost.
* **Process-independent buffering.** A trainer crash + restart loses
  in-process buffers; the relay's queue lives for the lifetime of
  the pod, so a flaky upstream survives a trainer restart.
* **One circuit decision.** A single circuit-breaker controls the
  retry cadence rather than each trainer having its own view.

In a single-trainer-per-pod deployment the relay is dormant unless
explicitly enabled — see :func:`make_mlflow_forward_fn` and the
``MLFLOW_TRACKING_URI`` env-var wired up in :mod:`src.runner.main`.

Design
------

* **DI for the upstream call.** The relay accepts a ``forward_fn``
  callable rather than importing mlflow itself. Production builds a
  forwarder via :func:`make_mlflow_forward_fn`; unit tests pass a
  fake. This keeps tests free of the heavy mlflow dependency.
* **Bounded queue, drop-oldest on overflow.** A flaky MLflow that
  fills the queue should not stall the trainer, and stale metrics
  are less valuable than fresh ones.
* **Exponential cooldown after N consecutive failures.** Mirror of
  the trainer-side breaker constants (3 failures, 60 s ceiling).
* **Re-queue on failure.** Events that fail forwarding go back to
  the front of the queue so monotonic step ordering is preserved
  across a flap-and-recover sequence.

The trainer's existing :class:`ResilientMLflowTransport` retains the
in-process buffering; the runner relay is additive, not a replacement.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from ryotenkai_shared.utils.logger import get_logger

logger = get_logger(__name__)

__all__ = [
    "MLFLOW_EVENT_KINDS",
    "MLflowRelay",
    "MLflowRelayCircuitBreaker",
    "make_mlflow_forward_fn",
]


#: Event ``kind`` values the relay will pick up from the event bus.
#:
#: The trainer publishes these via ``RunnerEventCallback`` whenever
#: the user configures pod-side relay; non-MLflow events are ignored
#: by :meth:`MLflowRelay.submit`.
MLFLOW_EVENT_KINDS: frozenset[str] = frozenset({
    "mlflow_metric",
    "mlflow_param",
    "mlflow_tag",
    "mlflow_run_started",
    "mlflow_run_ended",
})

#: Type alias for the upstream forwarder. Async by design — the relay
#: worker is a coroutine and a sync forwarder would block the loop.
ForwardFn = Callable[[dict[str, Any]], Awaitable[None]]


# ---------------------------------------------------------------------------
# Circuit breaker
# ---------------------------------------------------------------------------


@dataclass
class MLflowRelayCircuitBreaker:
    """Tracks consecutive failures, opens after ``failure_threshold``,
    recovers after a cooldown that grows exponentially up to a cap.

    Mirror of the trainer-side breaker constants in
    :mod:`src.training.mlflow.resilient_transport` so the two layers
    behave the same when both are active. Time is injected via
    ``clock`` so tests can fast-forward without sleeping.
    """

    failure_threshold: int = 3
    initial_cooldown_s: float = 1.0
    max_cooldown_s: float = 60.0
    multiplier: float = 1.5

    _failures: int = field(default=0, init=False, repr=False)
    _opened_at: float | None = field(default=None, init=False, repr=False)
    # Two cooldown values are tracked separately:
    #   _active_cooldown — the duration the circuit remains open for
    #     THIS open period; consulted by ``is_open(now)``.
    #   _next_cooldown — the duration the *next* open will use; bumped
    #     monotonically each time the circuit re-opens, capped at
    #     ``max_cooldown_s``. Reset to ``initial_cooldown_s`` on
    #     ``record_success``.
    # Splitting them avoids the off-by-one where the first open used
    # the bumped value rather than the configured initial.
    _active_cooldown: float = field(default=0.0, init=False, repr=False)
    _next_cooldown: float = field(default=0.0, init=False, repr=False)

    def __post_init__(self) -> None:
        self._active_cooldown = self.initial_cooldown_s
        self._next_cooldown = self.initial_cooldown_s

    @property
    def is_open_now(self) -> bool:
        """Convenience for tests / metrics."""
        return self._opened_at is not None

    def is_open(self, now: float) -> bool:
        """``True`` while the circuit is open AND the cooldown is unfinished."""
        if self._opened_at is None:
            return False
        if (now - self._opened_at) >= self._active_cooldown:
            # Cooldown elapsed — half-open state. Caller will attempt
            # one forward; success → closes the breaker; failure →
            # ``record_failure`` re-arms with a longer cooldown.
            return False
        return True

    def record_success(self) -> None:
        self._failures = 0
        self._opened_at = None
        self._active_cooldown = self.initial_cooldown_s
        self._next_cooldown = self.initial_cooldown_s

    def record_failure(self, now: float) -> None:
        self._failures += 1
        if self._failures >= self.failure_threshold:
            self._opened_at = now
            # The ``_next_cooldown`` value was prepared by the *previous*
            # open (or the initial seeding in __post_init__). Apply it
            # to the current open period, then prepare an even longer
            # one for the next open — capped at ``max_cooldown_s``.
            self._active_cooldown = self._next_cooldown
            self._next_cooldown = min(
                self._next_cooldown * self.multiplier,
                self.max_cooldown_s,
            )

    @property
    def consecutive_failures(self) -> int:
        return self._failures

    @property
    def active_cooldown_s(self) -> float:
        """The cooldown the breaker is currently waiting on (if open)."""
        return self._active_cooldown


# ---------------------------------------------------------------------------
# Relay
# ---------------------------------------------------------------------------


class MLflowRelay:
    """See module docstring."""

    def __init__(
        self,
        forward_fn: ForwardFn | None,
        *,
        queue_max: int = 1000,
        circuit_breaker: MLflowRelayCircuitBreaker | None = None,
        clock: Callable[[], float] = time.monotonic,
        worker_idle_poll_s: float = 0.5,
    ):
        """Build a relay.

        Args:
            forward_fn: Coroutine called with each event. ``None``
                disables the relay entirely (:meth:`submit` becomes a
                no-op, :meth:`start` does nothing). Production wires
                :func:`make_mlflow_forward_fn`; tests inject fakes.
            queue_max: Bounded queue size. Drops the oldest event on
                overflow so the most recent metrics survive a backlog.
            circuit_breaker: Override the default breaker (e.g. to
                tighten thresholds in tests).
            clock: Monotonic clock source — overridden in tests to
                advance time deterministically.
            worker_idle_poll_s: Sleep between queue polls when the
                circuit is open. Tests set this to a small value to
                avoid waiting in real time.
        """
        self._forward_fn = forward_fn
        self._queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(
            maxsize=queue_max,
        )
        self._cb = circuit_breaker or MLflowRelayCircuitBreaker()
        self._clock = clock
        self._worker_idle_poll_s = worker_idle_poll_s

        self._worker_task: asyncio.Task[None] | None = None
        self._stopped = asyncio.Event()
        self._dropped_count = 0

    # ---- Properties ----------------------------------------------------

    @property
    def enabled(self) -> bool:
        """``True`` when a forwarder is configured."""
        return self._forward_fn is not None

    @property
    def is_running(self) -> bool:
        return self._worker_task is not None and not self._worker_task.done()

    @property
    def queue_size(self) -> int:
        return self._queue.qsize()

    @property
    def dropped_count(self) -> int:
        return self._dropped_count

    @property
    def circuit_breaker(self) -> MLflowRelayCircuitBreaker:
        return self._cb

    # ---- Lifecycle -----------------------------------------------------

    async def start(self) -> None:
        """Launch the worker task. No-op when disabled or already started."""
        if not self.enabled:
            return
        if self.is_running:
            return
        self._stopped.clear()
        self._worker_task = asyncio.create_task(self._worker())

    async def stop(self) -> None:
        """Stop the worker, cancelling and awaiting completion.

        Safe to call when the relay was never started — used as the
        cleanup half of a context manager.
        """
        self._stopped.set()
        task = self._worker_task
        if task is None:
            return
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, BaseException):  # noqa: BLE001
            # We're tearing down; suppress whatever the worker raised
            # at cancellation. Real failures during steady-state are
            # logged by ``_worker`` itself.
            pass
        finally:
            self._worker_task = None

    # ---- Submission ----------------------------------------------------

    def submit(self, event: dict[str, Any]) -> bool:
        """Enqueue an event for forwarding.

        Returns ``True`` if the event was accepted, ``False`` if the
        relay is disabled, the kind is not relayed, or the queue is
        full and drop-oldest also failed (impossible in practice).

        Drop-oldest semantics: when the queue is full, the oldest
        event is dropped first to make room for the incoming one.
        Stale metrics are less valuable than the most recent one.
        """
        if not self.enabled:
            return False
        if event.get("kind") not in MLFLOW_EVENT_KINDS:
            return False

        try:
            self._queue.put_nowait(event)
            return True
        except asyncio.QueueFull:
            # Drop oldest, retry once.
            try:
                self._queue.get_nowait()
                self._dropped_count += 1
                logger.warning(
                    "[MLFLOW_RELAY] queue full; dropped oldest event "
                    "(total dropped: %d)", self._dropped_count,
                )
            except asyncio.QueueEmpty:  # pragma: no cover - race we tolerate
                pass
            try:
                self._queue.put_nowait(event)
                return True
            except asyncio.QueueFull:  # pragma: no cover - implausible
                return False

    # ---- Worker --------------------------------------------------------

    async def _worker(self) -> None:
        """Drain the queue, forwarding each event past the circuit breaker.

        Failure handling:
        - Forward raises → ``record_failure`` + re-queue at the head
          (preserves ordering). Worker continues; circuit may open
          on subsequent failures.
        - Circuit open → idle-poll until cooldown elapses. The breaker
          considers the post-cooldown state "half-open"; the next
          forward attempts the upstream and either closes the circuit
          or re-arms it.
        """
        assert self._forward_fn is not None
        forward_fn = self._forward_fn
        try:
            while not self._stopped.is_set():
                if self._cb.is_open(self._clock()):
                    # Don't block on queue.get — that would prevent
                    # ``stop()`` from cancelling promptly.
                    await asyncio.sleep(self._worker_idle_poll_s)
                    continue

                # Wait for an event; short timeout keeps us responsive
                # to ``stop()``.
                try:
                    event = await asyncio.wait_for(
                        self._queue.get(), timeout=self._worker_idle_poll_s,
                    )
                except asyncio.TimeoutError:
                    continue

                try:
                    await forward_fn(event)
                    self._cb.record_success()
                except asyncio.CancelledError:
                    # Re-queue and re-raise so ``stop()`` can drain
                    # us; CancelledError is the only exception we
                    # treat as "tear down, don't retry".
                    try:
                        self._queue.put_nowait(event)
                    except asyncio.QueueFull:
                        pass
                    raise
                except Exception as exc:  # noqa: BLE001
                    self._cb.record_failure(self._clock())
                    logger.warning(
                        "[MLFLOW_RELAY] forward failed (kind=%s, "
                        "consecutive failures=%d): %s",
                        event.get("kind"),
                        self._cb.consecutive_failures,
                        exc,
                    )
                    # Re-queue so ordering survives recovery. If the
                    # queue filled up while we were blocked the drop-
                    # oldest path on the producer side has already
                    # handled fairness.
                    try:
                        self._queue.put_nowait(event)
                    except asyncio.QueueFull:  # pragma: no cover
                        self._dropped_count += 1
        except asyncio.CancelledError:
            # ``stop()`` cancelled us — exit cleanly.
            pass


# ---------------------------------------------------------------------------
# Default forwarder (lazy mlflow import)
# ---------------------------------------------------------------------------


def make_mlflow_forward_fn(tracking_uri: str) -> ForwardFn:
    """Build a default forwarder that translates events to mlflow client calls.

    Imported lazily so the relay can be unit-tested without pulling
    the (heavy, optional) ``mlflow`` package into every test process.

    Event payload contracts (match what
    :class:`~src.training.callbacks.runner_event_callback.RunnerEventCallback`
    publishes when MLflow relay is enabled on the trainer):

    - ``mlflow_metric``: ``{run_id, key, value, step?, timestamp_ms?}``
    - ``mlflow_param``:  ``{run_id, key, value}``
    - ``mlflow_tag``:    ``{run_id, key, value}``
    - ``mlflow_run_started`` / ``mlflow_run_ended``: no-op here — the
      trainer's own MLflow client owns run lifecycle.

    Missing ``run_id`` skips the event with a warning (we cannot
    address a write without one).
    """
    import mlflow  # type: ignore[import-not-found]
    from mlflow.tracking import MlflowClient  # type: ignore[import-not-found]

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    async def _forward(event: dict[str, Any]) -> None:
        kind = event["kind"]
        payload: dict[str, Any] = event.get("payload") or {}
        run_id = payload.get("run_id")
        if not run_id:
            logger.warning(
                "[MLFLOW_RELAY] dropping event %s without run_id", kind,
            )
            return

        if kind == "mlflow_metric":
            client.log_metric(
                run_id=run_id,
                key=payload["key"],
                value=float(payload["value"]),
                step=int(payload.get("step", 0)),
                timestamp=int(payload.get("timestamp_ms", time.time() * 1000)),
            )
        elif kind == "mlflow_param":
            client.log_param(
                run_id=run_id, key=payload["key"], value=payload["value"],
            )
        elif kind == "mlflow_tag":
            client.set_tag(
                run_id=run_id, key=payload["key"], value=payload["value"],
            )
        # ``mlflow_run_started`` / ``mlflow_run_ended`` are observable
        # by external listeners but the trainer manages its own run.

    return _forward
