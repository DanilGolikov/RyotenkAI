"""Canonical fake for :class:`IMetricSink` Protocol.

Use this in tests instead of ``unittest.mock.Mock(spec=IMetricSink)``
— the sentinel :mod:`tests._lint.test_no_protocol_mocking` forbids that.

The fake buffers metrics in memory and exposes both per-run histories
and a chronological call log. Tests can inject failures via
:meth:`fail_next_n_calls`, observe blocking vs non-blocking flush calls,
and reset state between cases.

Example::

    sink = FakeMetricSink()
    sink.log("run-1", {"loss": 0.5}, step=0)
    sink.log("run-1", {"loss": 0.4}, step=1)
    sink.flush("run-1", blocking=True)

    history = sink.get_history("run-1")
    assert [s.value for s in history] == [0.5, 0.4]
    assert sink.flush_calls == [("run-1", True)]
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


class TransientMetricError(Exception):
    """Default exception raised by :meth:`FakeMetricSink.fail_next_n_calls`."""


@dataclass(frozen=True)
class MetricSample:
    """A single metric observation captured by the fake.

    :param run_id: Owning run id.
    :param key: Metric name.
    :param value: Metric value (float).
    :param step: Training step (caller-supplied; ordering invariant).
    """

    run_id: str
    key: str
    value: float
    step: int


@dataclass(frozen=True)
class LogCall:
    """Captured invocation of :meth:`FakeMetricSink.log`.

    :param run_id: Target run id.
    :param metrics: Snapshot of the metric mapping at call time.
    :param step: Step index from the caller.
    """

    run_id: str
    metrics: dict[str, float]
    step: int


class FakeMetricSink:
    """In-memory fake for :class:`IMetricSink`.

    The buffer is exposed in two flavours:

    * :attr:`log_calls` / :attr:`flush_calls` — chronological call logs
      (in invocation order) for assertion of side-effect timing.
    * :attr:`samples` — flat list of every :class:`MetricSample` written
      so tests can compute aggregates without re-running production code.

    :param fail_on_unflushed_metrics: When ``True`` (default ``False``),
        :meth:`flush` raises if there is no recorded metric for the run.
        Useful for tests that assert flush ordering precedes log calls.
    """

    def __init__(
        self,
        *,
        fail_on_unflushed_metrics: bool = False,
    ) -> None:
        self._fail_on_unflushed = fail_on_unflushed_metrics
        # Per-run history maintained in insertion order.
        self._history: dict[str, list[MetricSample]] = {}
        # Pending metrics not yet flushed (per run).
        self._pending: dict[str, list[MetricSample]] = {}
        # Call logs.
        self.log_calls: list[LogCall] = []
        self.flush_calls: list[tuple[str, bool]] = []
        self.samples: list[MetricSample] = []
        # Chaos state.
        self._fail_remaining: int = 0
        self._fail_kind: type[Exception] = TransientMetricError

    # ------------------------------------------------------------------
    # Chaos surface
    # ------------------------------------------------------------------

    def fail_next_n_calls(
        self,
        n: int,
        kind: type[Exception] = TransientMetricError,
    ) -> None:
        """Program the next ``n`` :meth:`log` / :meth:`flush` calls to raise.

        :param n: Non-negative count of failures.
        :param kind: Exception class to raise.
        :raises ValueError: If ``n`` is negative.
        """
        if n < 0:
            raise ValueError("fail_next_n_calls requires non-negative count")
        self._fail_remaining = n
        self._fail_kind = kind

    def reset_chaos(self) -> None:
        """Clear chaos state."""
        self._fail_remaining = 0

    # ------------------------------------------------------------------
    # Inspection helpers (test convenience)
    # ------------------------------------------------------------------

    def get_history(self, run_id: str) -> list[MetricSample]:
        """Return the full ordered history of samples for ``run_id``.

        :param run_id: Target run id.
        :returns: Copy of the history list (safe to mutate).
        """
        return list(self._history.get(run_id, []))

    def get_pending(self, run_id: str) -> list[MetricSample]:
        """Return un-flushed samples for ``run_id``."""
        return list(self._pending.get(run_id, []))

    def has_pending(self, run_id: str) -> bool:
        """Return ``True`` iff there are un-flushed samples for ``run_id``."""
        return bool(self._pending.get(run_id))

    # ------------------------------------------------------------------
    # IMetricSink surface
    # ------------------------------------------------------------------

    def log(self, run_id: str, metrics: Mapping[str, float], step: int) -> None:
        """Record ``metrics`` as samples for ``run_id`` at ``step``.

        :param run_id: Target run id.
        :param metrics: Mapping of metric name -> float value.
        :param step: Step index (caller-supplied).
        """
        self._guard()
        metrics_copy = {k: float(v) for k, v in metrics.items()}
        self.log_calls.append(LogCall(run_id=run_id, metrics=metrics_copy, step=step))
        history = self._history.setdefault(run_id, [])
        pending = self._pending.setdefault(run_id, [])
        for key, value in metrics_copy.items():
            sample = MetricSample(run_id=run_id, key=key, value=value, step=step)
            history.append(sample)
            pending.append(sample)
            self.samples.append(sample)

    def flush(self, run_id: str, blocking: bool) -> None:
        """Drain the pending buffer for ``run_id``.

        :param run_id: Run to flush.
        :param blocking: Whether the flush is synchronous; recorded but
            behaviour is identical (fake is single-threaded).
        :raises RuntimeError: If ``fail_on_unflushed_metrics`` was set and
            no pending samples exist for ``run_id``.
        """
        self._guard()
        self.flush_calls.append((run_id, blocking))
        pending = self._pending.get(run_id, [])
        if not pending and self._fail_on_unflushed:
            raise RuntimeError(f"no pending metrics to flush for run_id={run_id!r}")
        self._pending[run_id] = []

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _guard(self) -> None:
        if self._fail_remaining > 0:
            self._fail_remaining -= 1
            raise self._fail_kind("fake_injected_failure")


__all__ = [
    "FakeMetricSink",
    "LogCall",
    "MetricSample",
    "TransientMetricError",
]
