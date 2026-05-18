"""Buffered, async metric writer (:class:`~.protocols.IMetricSink`).

Concrete implementation of :class:`~.protocols.IMetricSink`. Wraps
``MlflowClient.log_batch(synchronous=False)`` for backpressure-friendly
metric ingestion; falls back to a :class:`~.dead_letter.DeadLetterBuffer`
when the transport layer raises (network partition, server 5xx).

Replaces the implicit metric-routing logic of the legacy
``ResilientMLflowTransport``, which monkey-patched the ``mlflow`` module
to intercept every ``log_metric`` call. The new design is:

* DI'd into the producer (HF Trainer callback, control-side phase
  metrics) instead of monkey-patched.
* Dead-letter queue is an explicit dependency — caller injects it,
  inspects it, drains it on reconnect.
* No globals, no atexit hooks.

Per ``docs/plans/vectorized-fluttering-mist.md`` §Target architecture.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from ryotenkai_shared.utils.logger import get_logger

if TYPE_CHECKING:
    from ryotenkai_shared.infrastructure.mlflow.dead_letter import DeadLetterBuffer
    from ryotenkai_shared.infrastructure.mlflow.transport import MlflowTransport

logger = get_logger(__name__)

_MAX_BATCH_SIZE = 1000
"""MLflow ``log_batch`` REST limit (per the MLflow docs). Producers
that submit larger dicts get split into chunks of this size."""


class MetricSink:
    """Concrete :class:`~.protocols.IMetricSink`.

    :param client: Configured :class:`MlflowTransport`. The transport
        owns the URI stamping; this sink never re-stamps.
    :param dead_letter: Optional :class:`DeadLetterBuffer` to which
        unshippable batches are written. ``None`` disables fallback —
        in that mode :meth:`log` swallows transport errors with a
        warning (same as legacy behaviour).

    Thread-safety
    -------------
    All calls go through ``MlflowTransport.client`` which delegates to
    the underlying ``mlflow.tracking.MlflowClient``. MLflow's client
    is documented thread-safe for the log-write surface, so we don't
    add a lock here.
    """

    def __init__(
        self,
        client: MlflowTransport,
        dead_letter: DeadLetterBuffer | None = None,
    ) -> None:
        self._client = client
        self._dead_letter = dead_letter

    # -- IMetricSink methods ----------------------------------------

    def log(
        self,
        run_id: str,
        metrics: Mapping[str, float],
        step: int,
    ) -> None:
        """Submit a metric batch asynchronously.

        Values are wrapped in ``mlflow.entities.Metric`` proto objects
        with the current wall-clock timestamp. Submission uses
        ``synchronous=False`` — the call returns once MLflow has
        accepted the request locally; actual server delivery is
        deferred to MLflow's internal async log thread.

        On transport failure:

        * if a :class:`DeadLetterBuffer` is wired, the batch is
          appended there and the error is logged at WARNING;
        * otherwise the error is logged at WARNING and dropped.

        :param run_id: Active MLflow run id.
        :param metrics: ``{key: value}``.
        :param step: Optimization step (used as MLflow's ``Metric.step``).
        """
        if not metrics:
            return
        items = list(metrics.items())
        # Chunk to respect MLflow's 1000-metrics-per-batch ceiling.
        for chunk_start in range(0, len(items), _MAX_BATCH_SIZE):
            chunk = items[chunk_start : chunk_start + _MAX_BATCH_SIZE]
            self._log_chunk(run_id, dict(chunk), step)

    def flush(self, run_id: str, blocking: bool) -> None:
        """Best-effort flush of pending async writes for ``run_id``.

        ``MlflowClient.log_batch(synchronous=False)`` queues writes
        on an internal worker; MLflow exposes no public "flush this
        run" API. We approximate by:

        1. If a dead-letter buffer is configured, drain it
           (synchronously re-emit via the transport — bypasses the
           non-blocking path since we're trying to free disk).
        2. If ``blocking=True``, sleep a small grace period to let
           MLflow's worker thread settle. This is imperfect but
           bounded; the alternative — calling MLflow's private
           ``_log_batch_sync`` — couples us to internal API.

        :param run_id: Run whose pending writes we want flushed.
        :param blocking: When ``True``, wait briefly for the MLflow
            worker thread to settle (best-effort).
        """
        if self._dead_letter is not None and self._dead_letter.size_bytes() > 0:
            self._drain_dead_letter()
        if blocking:
            # MLflow's async log worker polls every ~1s; a small grace
            # window catches the common case without making flush
            # arbitrarily expensive.
            time.sleep(0.2)
        # Touch ``run_id`` so the lint doesn't flag an unused param;
        # MLflow's flush is process-wide, not run-scoped.
        _ = run_id

    # -- internal helpers -------------------------------------------

    def _log_chunk(
        self,
        run_id: str,
        metrics: Mapping[str, float],
        step: int,
    ) -> None:
        """Emit a single ``log_batch`` chunk; fall back to DLQ on failure."""
        try:
            metric_entities = self._build_metric_entities(metrics, step)
            client = self._client.client
            # ``synchronous=False`` is supported on MLflow >=2.7; older
            # servers ignore the kwarg, which is fine — we get the
            # back-compat synchronous behaviour.
            try:
                client.log_batch(
                    run_id,
                    metrics=metric_entities,
                    synchronous=False,
                )
            except TypeError:
                # Older MLflow without the kwarg.
                client.log_batch(run_id, metrics=metric_entities)
        except Exception as exc:  # noqa: BLE001 — boundary, classify below
            if self._is_transport_error(exc):
                self._handle_transport_failure(run_id, metrics, step, exc)
                return
            # Re-raise programmer errors (bad value type, etc.) so the
            # caller sees them in tests.
            raise

    def _handle_transport_failure(
        self,
        run_id: str,
        metrics: Mapping[str, float],
        step: int,
        exc: BaseException,
    ) -> None:
        """Route the batch to DLQ if configured; log otherwise."""
        if self._dead_letter is not None:
            try:
                self._dead_letter.write(run_id, metrics, step)
                logger.warning(
                    "[METRIC_SINK] Buffered %d metric(s) for run=%s step=%d to DLQ "
                    "after transport failure: %s",
                    len(metrics),
                    run_id,
                    step,
                    exc,
                )
                return
            except OSError as dlq_exc:
                logger.warning(
                    "[METRIC_SINK] DLQ write failed (%s); dropping batch "
                    "(run=%s step=%d, %d metrics): %s",
                    dlq_exc,
                    run_id,
                    step,
                    len(metrics),
                    exc,
                )
                return
        logger.warning(
            "[METRIC_SINK] Dropping batch (run=%s step=%d, %d metrics) — "
            "no DLQ configured: %s",
            run_id,
            step,
            len(metrics),
            exc,
        )

    def _drain_dead_letter(self) -> None:
        """Re-submit every buffered batch through the live transport."""
        if self._dead_letter is None:
            return
        drained = 0
        for run_id, metrics, step in self._dead_letter.drain():
            try:
                client = self._client.client
                metric_entities = self._build_metric_entities(metrics, step)
                client.log_batch(run_id, metrics=metric_entities)
                drained += 1
            except Exception as exc:  # noqa: BLE001 — best-effort drain
                logger.warning(
                    "[METRIC_SINK] DLQ replay failed at record %d (run=%s "
                    "step=%d): %s",
                    drained,
                    run_id,
                    step,
                    exc,
                )
                # Re-buffer remaining? For now we drop — the caller can
                # repeat the cycle on next flush.
                # The drained-so-far records are gone; this is the
                # at-most-once tradeoff documented in the SSOT journal
                # design (events.jsonl carries the source-of-truth).
                return
        if drained > 0:
            logger.info("[METRIC_SINK] DLQ drained: %d batch(es) replayed", drained)

    @staticmethod
    def _build_metric_entities(
        metrics: Mapping[str, float],
        step: int,
    ) -> list[Any]:
        """Construct ``mlflow.entities.Metric`` protos from a dict.

        Lazy-imports the ``Metric`` class to keep this module
        importable in trimmed CI venvs without ``mlflow``.
        """
        from mlflow.entities import Metric  # noqa: PLC0415

        ts_ms = int(time.time() * 1000)
        return [
            Metric(key=key, value=float(value), timestamp=ts_ms, step=int(step))
            for key, value in metrics.items()
        ]

    @staticmethod
    def _is_transport_error(exc: BaseException) -> bool:
        """Classify ``exc`` as transient transport vs programmer error.

        We treat as transport: ``ConnectionError``, ``TimeoutError``,
        ``OSError`` (DNS / socket), and any exception whose ``str()``
        contains MLflow's REST error markers. Anything else is
        programmer error and re-raised.
        """
        if isinstance(exc, ConnectionError | TimeoutError | OSError):
            return True
        message = str(exc)
        markers = (
            "API request to",
            "HTTPConnectionPool",
            "HTTPSConnectionPool",
            "Max retries exceeded",
            "Connection aborted",
            "Connection refused",
            "Read timed out",
            "ConnectTimeout",
        )
        return any(marker in message for marker in markers)


__all__ = ["MetricSink"]
