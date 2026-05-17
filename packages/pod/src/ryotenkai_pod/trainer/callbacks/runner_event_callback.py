"""HuggingFace ``TrainerCallback`` that pushes typed envelopes to the runner.

Phase 2 (ethereal-tumbling-patterson) rewrite — closes the
"trainer-side self-disable + silent drop" risk (R-02 in the plan) by:

* Building :class:`ryotenkai_shared.events.BaseEvent` envelopes inside
  each HF lifecycle hook (``on_train_begin``, ``on_step_end`` etc.)
  instead of the old free-form ``{kind, payload}`` dicts.
* Buffering envelopes in a bounded :class:`queue.Queue` (capacity
  10000, **drop-oldest** with a counter). The HF callback thread's
  emit path is non-blocking (queue ``put_nowait``); a single
  background ``threading.Thread`` daemon drains the queue and POSTs
  each envelope to ``/api/v1/internal/events`` with exponential
  backoff (1s/5s/30s). On HTTP failure the worker re-queues the
  envelope and retries forever — **never self-disables**. The
  drop-oldest counter is bumped (not the bus' per-consumer counter
  yet; Phase 4 will plumb that through).

The callback is a no-op unless ``RYOTENKAI_RUNNER_URL`` is set in the
trainer's environment (the supervisor sets this on spawn; standalone
local runs see it unset and short-circuit).
"""

from __future__ import annotations

import os
import queue
import threading
import time
import traceback
from typing import TYPE_CHECKING, Any

import httpx
from transformers import TrainerCallback

from ryotenkai_shared.events import UNKNOWN_OFFSET, BaseEvent, to_jsonl
from ryotenkai_shared.events.types.pod_training import (
    TrainingCheckpointSavedEvent,
    TrainingCheckpointSavedPayload,
    TrainingCompletedEvent,
    TrainingCompletedPayload,
    TrainingEpochCompletedEvent,
    TrainingEpochCompletedPayload,
    TrainingEpochStartedEvent,
    TrainingEpochStartedPayload,
    TrainingEvalMetricsEvent,
    TrainingEvalMetricsPayload,
    TrainingFailedEvent,
    TrainingFailedPayload,
    TrainingLogEvent,
    TrainingLogPayload,
    TrainingStartedEvent,
    TrainingStartedPayload,
    TrainingStepEvent,
    TrainingStepPayload,
)
from ryotenkai_shared.utils.logger import get_logger

if TYPE_CHECKING:
    from transformers import TrainerControl, TrainerState, TrainingArguments


__all__ = ["DEFAULT_QUEUE_CAP", "RUNNER_URL_ENV", "RunnerEventCallback"]


logger = get_logger(__name__)


# Env var the supervisor sets when spawning the trainer subprocess.
RUNNER_URL_ENV = "RYOTENKAI_RUNNER_URL"
# Run-ID env var (Phase 2) — used to stamp envelopes with the correct
# ``run_id``. Falls back to a constant sentinel when the env var is
# unset (standalone test runs).
RUN_ID_ENV = "RYOTENKAI_RUN_ID"

# Bounded queue capacity. 10k × ~1 KB ≈ 10 MB; trainer-side RAM stays
# well under control even on a long backpressure window.
DEFAULT_QUEUE_CAP = 10_000

# Retry backoff schedule (seconds) for transient HTTP failures.
_RETRY_BACKOFF_SECONDS: tuple[float, ...] = (1.0, 5.0, 30.0)

# Cap on ``traceback_excerpt`` payload size. Trainer crash traces can
# be megabytes deep (CUDA / bitsandbytes stacks plus C extensions); we
# truncate to 2 KB so the event bus and SSE consumers never see a
# pathological envelope. The truncated suffix is replaced with a marker
# so downstream tools can tell that the trace was cut.
_TRACEBACK_MAX_BYTES = 2048
_TRACEBACK_TRUNCATION_SUFFIX = "...[truncated]"

# Sentinel for ``TrainingFailedPayload.step`` when training failed
# before HF Trainer began (e.g. config / model load failure). The
# typed payload allows ``int | None`` but we prefer ``-1`` so consumers
# can treat the field uniformly without nullability branching.
_PRE_TRAIN_STEP_SENTINEL = -1


class RunnerEventCallback(TrainerCallback):
    """Push typed envelopes to the in-pod runner over loopback HTTP.

    Args:
        runner_url:   Base URL of the runner. ``None`` (default) reads
                       ``RYOTENKAI_RUNNER_URL`` from the environment.
                       Unset env var disables the callback for the
                       full lifetime of the trainer.
        run_id:        Run identifier stamped on every envelope. Reads
                       ``RYOTENKAI_RUN_ID`` from the environment by
                       default; falls back to ``"unknown"``.
        source:        Authoritative URI on each envelope's ``source``
                       field. Defaults to
                       ``f"pod://{run_id}/trainer"``.
        flush_every:   Emit a ``TrainingStepEvent`` every N steps.
        timeout_seconds: Per-request HTTP timeout. Same-machine
                          loopback, so 2s is generous.
        queue_cap:     Max in-memory queued envelopes. Excess drops
                       from the oldest end with a counter increment.
    """

    def __init__(
        self,
        runner_url: str | None = None,
        *,
        run_id: str | None = None,
        source: str | None = None,
        flush_every: int = 10,
        timeout_seconds: float = 2.0,
        queue_cap: int = DEFAULT_QUEUE_CAP,
    ) -> None:
        url = runner_url if runner_url is not None else os.environ.get(RUNNER_URL_ENV)
        self._enabled: bool = bool(url)
        self._url: str = (url or "").rstrip("/")
        self._run_id: str = run_id or os.environ.get(RUN_ID_ENV) or "unknown"
        self._source: str = source or f"pod://{self._run_id}/trainer"
        self._flush_every = max(1, int(flush_every))
        self._timeout = float(timeout_seconds)
        self._queue: queue.Queue[BaseEvent] = queue.Queue(maxsize=max(1, queue_cap))
        # Per-callback drop counter for trainer-side observability. A
        # future Phase 4 wires this to the bus' per-consumer table.
        self._dropped_total: int = 0
        # Tracks the most recent loss the Trainer reported through
        # ``on_log``; piggybacked onto the next step event.
        self._last_loss: float | None = None
        # Tracks the most recent ``global_step`` HF Trainer surfaced
        # through any callback hook. Used by ``emit_training_failed``
        # when the failure originates from ``run_training.py`` and
        # there is no live ``TrainerState`` to read from.
        self._last_global_step: int = _PRE_TRAIN_STEP_SENTINEL
        # Set by :meth:`emit_training_failed`. ``on_train_end`` checks
        # this flag and skips the success-path ``TrainingCompletedEvent``
        # so consumers never see a contradictory "completed after
        # failed" pair. HF Trainer's ``Trainer.train()`` calls
        # ``on_train_end`` from a ``try/finally``, so the hook fires on
        # both happy and unhappy paths and the flag is the only signal
        # we have to distinguish them at the callback layer.
        self._failed_flag: bool = False
        self._client: httpx.Client | None = None
        # Background worker — daemon so a stuck POST doesn't keep the
        # interpreter alive past the trainer process.
        self._stop_evt = threading.Event()
        self._worker: threading.Thread | None = None
        if self._enabled:
            self._worker = threading.Thread(
                target=self._worker_loop,
                name="runner-event-callback-worker",
                daemon=True,
            )
            self._worker.start()

    # ---- public introspection (used in tests) -----------------------

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def queue_size(self) -> int:
        return self._queue.qsize()

    @property
    def dropped_total(self) -> int:
        return self._dropped_total

    @property
    def failed(self) -> bool:
        """True once :meth:`emit_training_failed` has been called."""
        return self._failed_flag

    # ---- HF TrainerCallback hooks -----------------------------------

    def on_train_begin(  # type: ignore[override]
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        if not self._enabled:
            return
        # Once HF Trainer has entered the loop the pre-train sentinel
        # is no longer correct — even step 0 is more accurate than -1.
        self._last_global_step = int(getattr(state, "global_step", 0) or 0)
        algorithm = kwargs.get("algorithm", "sft")
        if algorithm not in ("sft", "cpt", "dpo", "grpo", "sapo"):
            algorithm = "sft"
        self._emit(
            TrainingStartedEvent(
                source=self._source,
                run_id=self._run_id,
                offset=UNKNOWN_OFFSET,
                payload=TrainingStartedPayload(
                    max_steps=int(state.max_steps),
                    num_train_epochs=int(args.num_train_epochs or 0),
                    per_device_batch_size=int(args.per_device_train_batch_size),
                    gradient_accumulation_steps=int(
                        getattr(args, "gradient_accumulation_steps", 1) or 1,
                    ),
                    learning_rate=float(getattr(args, "learning_rate", 0.0) or 0.0),
                    algorithm=algorithm,
                ),
            ),
        )

    def on_epoch_begin(  # type: ignore[override]
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        if not self._enabled:
            return
        epoch_raw = state.epoch
        epoch = int(epoch_raw) if epoch_raw is not None else 0
        self._emit(
            TrainingEpochStartedEvent(
                source=self._source,
                run_id=self._run_id,
                offset=UNKNOWN_OFFSET,
                payload=TrainingEpochStartedPayload(
                    epoch=epoch,
                    global_step=int(state.global_step),
                ),
            ),
        )

    def on_epoch_end(  # type: ignore[override]
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        if not self._enabled:
            return
        epoch_raw = state.epoch
        epoch = int(epoch_raw) if epoch_raw is not None else 0
        self._emit(
            TrainingEpochCompletedEvent(
                source=self._source,
                run_id=self._run_id,
                offset=UNKNOWN_OFFSET,
                payload=TrainingEpochCompletedPayload(
                    epoch=epoch,
                    global_step=int(state.global_step),
                    mean_loss=float(self._last_loss or 0.0),
                    duration_s=0.0,
                ),
            ),
        )

    def on_log(  # type: ignore[override]
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict[str, float] | None = None,
        **kwargs: Any,
    ) -> None:
        if not self._enabled or logs is None:
            return
        loss = logs.get("loss")
        if isinstance(loss, (int, float)):
            self._last_loss = float(loss)
        metrics = {
            k: float(v)
            for k, v in logs.items()
            if isinstance(v, (int, float))
        }
        self._emit(
            TrainingLogEvent(
                source=self._source,
                run_id=self._run_id,
                offset=UNKNOWN_OFFSET,
                payload=TrainingLogPayload(
                    step=int(state.global_step),
                    metrics=metrics,
                ),
            ),
        )

    def on_step_end(  # type: ignore[override]
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        if not self._enabled:
            return
        # Always cache the latest step so failure emission carries a
        # meaningful number even when flushing is gated by ``flush_every``.
        self._last_global_step = int(state.global_step)
        if state.global_step % self._flush_every != 0:
            return
        self._emit(
            TrainingStepEvent(
                source=self._source,
                run_id=self._run_id,
                offset=UNKNOWN_OFFSET,
                payload=TrainingStepPayload(
                    step=int(state.global_step),
                    loss=float(self._last_loss or 0.0),
                    learning_rate=float(
                        getattr(args, "learning_rate", 0.0) or 0.0,
                    ),
                ),
            ),
        )

    def on_evaluate(  # type: ignore[override]
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: dict[str, float] | None = None,
        **kwargs: Any,
    ) -> None:
        if not self._enabled:
            return
        metrics_safe = {
            k: float(v)
            for k, v in (metrics or {}).items()
            if isinstance(v, (int, float))
        }
        self._emit(
            TrainingEvalMetricsEvent(
                source=self._source,
                run_id=self._run_id,
                offset=UNKNOWN_OFFSET,
                payload=TrainingEvalMetricsPayload(
                    step=int(state.global_step),
                    metrics=metrics_safe,
                    dataset_name=str(kwargs.get("dataset_name") or "eval"),
                ),
            ),
        )

    def on_save(  # type: ignore[override]
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        if not self._enabled:
            return
        self._emit(
            TrainingCheckpointSavedEvent(
                source=self._source,
                run_id=self._run_id,
                offset=UNKNOWN_OFFSET,
                payload=TrainingCheckpointSavedPayload(
                    step=int(state.global_step),
                    local_path=str(kwargs.get("checkpoint_path") or ""),
                    size_bytes=int(kwargs.get("size_bytes") or 0),
                    is_best=bool(kwargs.get("is_best", False)),
                ),
            ),
        )

    def on_train_end(  # type: ignore[override]
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        if not self._enabled:
            return
        # HF Trainer's ``Trainer.train()`` wraps the inner loop in a
        # ``try/finally`` and always calls ``on_train_end`` regardless
        # of outcome. We use ``_failed_flag`` (set by
        # :meth:`emit_training_failed` upstream of HF's finally) to
        # avoid emitting a misleading ``TrainingCompletedEvent`` on
        # failure paths. The drain/stop housekeeping still runs so the
        # worker drains its queue and the httpx client closes.
        if not self._failed_flag:
            self._emit(
                TrainingCompletedEvent(
                    source=self._source,
                    run_id=self._run_id,
                    offset=UNKNOWN_OFFSET,
                    payload=TrainingCompletedPayload(
                        final_step=int(state.global_step),
                        mean_loss=float(self._last_loss or 0.0),
                        duration_s=0.0,
                        tokens_processed=0,
                    ),
                ),
            )
        # Give the worker a short window to drain whatever is still
        # queued; then signal stop. The daemon thread will be joined
        # via interpreter shutdown if we miss the bound.
        self._drain_with_deadline(deadline_s=10.0)
        self._stop_evt.set()
        if self._worker is not None:
            self._worker.join(timeout=1.0)
        self._close_client()

    # ---- failure emission --------------------------------------------

    def emit_training_failed(
        self,
        *,
        exc: BaseException | None = None,
        error_type: str | None = None,
        message: str | None = None,
        traceback_excerpt: str | None = None,
        step: int | None = None,
    ) -> None:
        """Enqueue a :class:`TrainingFailedEvent` envelope.

        Designed to be called from ``run_training.py``'s exception
        handler — explicit, structured, and idempotent. Two call
        styles are supported:

        * **Implicit** (preferred): pass ``exc`` and the method derives
          ``error_type``, ``message`` and ``traceback_excerpt`` from
          it using ``traceback.TracebackException``. Truncated to
          :data:`_TRACEBACK_MAX_BYTES` so the bus never sees a
          pathological envelope.
        * **Explicit**: pass any subset of ``error_type``, ``message``,
          ``traceback_excerpt`` to override the derivation. Useful
          when the caller already has structured error context
          (typed :class:`RyotenkAIError`) and wants to bypass the
          generic ``str(exc)`` form.

        ``step`` defaults to the most recent ``state.global_step``
        observed via the HF hooks, falling back to
        :data:`_PRE_TRAIN_STEP_SENTINEL` (``-1``) for failures
        originating before ``on_train_begin`` (config / model load
        crashes).

        The :attr:`failed` flag is set unconditionally so
        :meth:`on_train_end` skips the misleading
        ``TrainingCompletedEvent`` on the HF finally path.

        Idempotent — calling twice enqueues two events. The flag
        is set on first call. Callers are expected to invoke this
        once per failure; idempotency at the bus level is the
        consumer's job.

        No-op when ``self._enabled`` is False (mirrors the rest of
        the callback's contract under standalone / local runs).
        """
        # Always mark the failure even when disabled — keeping the
        # flag consistent means ``on_train_end`` skips
        # ``TrainingCompletedEvent`` if the bus turns out to be wired
        # up later in the same instance lifetime (defensive).
        self._failed_flag = True
        if not self._enabled:
            return

        resolved_error_type = error_type or (
            type(exc).__name__ if exc is not None else "UnknownError"
        )
        resolved_message = message if message is not None else (
            str(exc) if exc is not None else ""
        )
        resolved_traceback = traceback_excerpt if traceback_excerpt is not None else (
            self._format_traceback(exc) if exc is not None else ""
        )
        resolved_step = (
            int(step) if step is not None else int(self._last_global_step)
        )

        self._emit(
            TrainingFailedEvent(
                source=self._source,
                run_id=self._run_id,
                offset=UNKNOWN_OFFSET,
                payload=TrainingFailedPayload(
                    error_type=resolved_error_type,
                    message=resolved_message,
                    traceback_excerpt=self._truncate_traceback(resolved_traceback),
                    step=resolved_step,
                ),
            ),
        )

    @staticmethod
    def _format_traceback(exc: BaseException) -> str:
        """Render the last frames of ``exc`` as a string."""
        try:
            te = traceback.TracebackException.from_exception(exc)
            return "".join(te.format())
        except Exception:  # pragma: no cover — defensive
            # ``TracebackException`` parsing should never fail for a
            # real exception object, but pathological proxies or
            # mocked exception classes can surprise us. Fall back to
            # ``format_exception_only`` so callers always get a string.
            try:
                return "".join(
                    traceback.format_exception_only(type(exc), exc),
                )
            except Exception:
                return f"{type(exc).__name__}: {exc!s}"

    @staticmethod
    def _truncate_traceback(text: str) -> str:
        """Truncate ``text`` to :data:`_TRACEBACK_MAX_BYTES` (UTF-8)."""
        encoded = text.encode("utf-8", errors="replace")
        if len(encoded) <= _TRACEBACK_MAX_BYTES:
            return text
        marker_bytes = _TRACEBACK_TRUNCATION_SUFFIX.encode("utf-8")
        keep = _TRACEBACK_MAX_BYTES - len(marker_bytes)
        if keep <= 0:
            # Suffix itself larger than the cap — return only the
            # truncated suffix to stay within the limit.
            return _TRACEBACK_TRUNCATION_SUFFIX[:_TRACEBACK_MAX_BYTES]
        truncated = encoded[:keep].decode("utf-8", errors="ignore")
        return truncated + _TRACEBACK_TRUNCATION_SUFFIX

    # ---- internals ---------------------------------------------------

    def _emit(self, event: BaseEvent) -> None:
        """Non-blocking enqueue. Drop-oldest with counter on overflow."""
        try:
            self._queue.put_nowait(event)
            return
        except queue.Full:
            pass

        # Drop-oldest: pop the head to make room, then enqueue the new
        # event. The drop is recorded so an operator can spot a slow
        # runner via the trainer's metrics. The race here is acceptable
        # — if two emit calls overflow concurrently the worst case is
        # one extra drop.
        try:
            self._queue.get_nowait()
            self._dropped_total += 1
        except queue.Empty:
            pass
        try:
            self._queue.put_nowait(event)
        except queue.Full:
            # Worker drained between the get and put; just drop the
            # new event rather than blocking the trainer thread.
            self._dropped_total += 1

    def _drain_with_deadline(self, *, deadline_s: float) -> None:
        """Wait until the queue is empty or ``deadline_s`` elapses."""
        end = time.monotonic() + deadline_s
        while time.monotonic() < end:
            if self._queue.empty():
                return
            time.sleep(0.05)

    def _ensure_client(self) -> httpx.Client | None:
        if self._client is None:
            try:
                self._client = httpx.Client(timeout=self._timeout)
            except Exception:  # pragma: no cover — extreme env failure
                logger.warning("[RunnerEventCallback] httpx client init failed")
                return None
        return self._client

    def _close_client(self) -> None:
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None

    def _worker_loop(self) -> None:
        """Background worker — POSTs queued envelopes with backoff retry.

        Never self-disables; on transient HTTP failure the envelope is
        re-queued at the head (put it back in front of new arrivals so
        ordering is preserved across the retry).
        """
        endpoint = f"{self._url}/api/v1/internal/events"
        while not self._stop_evt.is_set():
            try:
                envelope = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if not self._post_with_retry(endpoint, envelope):
                # Final failure after retries — drop with counter so the
                # queue keeps making progress. Production never reaches
                # this branch because the retry loop is unbounded; we
                # only break out on interpreter shutdown.
                self._dropped_total += 1

    def _post_with_retry(
        self, endpoint: str, envelope: BaseEvent,
    ) -> bool:
        """POST an envelope; retry transient failures with backoff.

        Returns ``True`` on success, ``False`` on a non-retryable error
        (which is treated as a drop). Retries on transport / 5xx
        failures forever — the trainer's "fire-and-forget" semantics
        require that we never give up while the queue still has space
        for the next envelope.
        """
        client = self._ensure_client()
        if client is None:
            return False

        # Wire body: full envelope JSON as a single dict — the handler
        # validates via EVENT_ADAPTER. We could also use to_jsonl(line)
        # but JSON body is more idiomatic for HTTP and matches the
        # FastAPI route signature.
        import json
        body = json.loads(envelope.model_dump_json())

        attempt = 0
        while not self._stop_evt.is_set():
            try:
                resp = client.post(endpoint, json=body)
            except Exception as exc:
                logger.debug(
                    "[RunnerEventCallback] POST transport failure (attempt %d): %s",
                    attempt, exc,
                )
                resp = None

            if resp is not None and resp.status_code < 500:
                if resp.status_code >= 400:
                    # 4xx — non-retryable (validation error from the
                    # handler). Log + drop with counter.
                    logger.warning(
                        "[RunnerEventCallback] POST returned %d: %s",
                        resp.status_code, resp.text[:200],
                    )
                    return False
                return True

            # 5xx or transport failure → retry with backoff.
            backoff = _RETRY_BACKOFF_SECONDS[
                min(attempt, len(_RETRY_BACKOFF_SECONDS) - 1)
            ]
            self._stop_evt.wait(timeout=backoff)
            attempt += 1
        return False


# Optional helper for the trainer subprocess to serialise an envelope
# the same way the codec does, so smoke tests can compare bodies.
def envelope_to_post_body(event: BaseEvent) -> str:
    """Round-trip the envelope through the shared codec for diagnostics."""
    return to_jsonl(event)
