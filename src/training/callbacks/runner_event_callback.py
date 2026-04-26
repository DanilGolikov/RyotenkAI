"""HuggingFace ``TrainerCallback`` that pushes progress events to the
in-pod runner over loopback HTTP.

Lifecycle (mapped to ``transformers.TrainerCallback`` hooks):

==================== ============================================
Trainer hook         Event published
==================== ============================================
``on_train_begin``   ``training_started`` — once at start, with
                     ``max_steps`` and ``num_train_epochs`` from
                     ``TrainerState``.
``on_step_end``      ``step`` — every ``flush_every`` steps; carries
                     ``step``, ``epoch``, and the most recent
                     ``loss`` if it was logged this step.
``on_log``           ``log`` — every Trainer-issued log dict
                     (loss, learning_rate, etc.).
``on_evaluate``      ``eval_metrics`` — eval results.
``on_save``          ``checkpoint_saved`` — every checkpoint
                     write (incl. SIGTERM-triggered emergency).
``on_train_end``     ``training_complete`` — once at end; final
                     flush and HTTP client close.
==================== ============================================

Activation:
The callback is a no-op unless ``RYOTENKAI_RUNNER_URL`` is set in
the trainer's environment (the runner sets this when the supervisor
spawns the subprocess; locally-run training sees it unset and the
callback short-circuits without touching the network).

Backpressure / failure handling:
- A bounded local buffer holds outgoing events. ``_flush()`` posts
  them sequentially; failures keep events in the buffer for the
  next attempt.
- After ``MAX_CONSECUTIVE_FAILURES`` consecutive failed flushes the
  callback **disables itself for the rest of the run** — the
  supervisor's stdout pump still captures everything written to
  ``training.log``, and MLflow tracking continues independently.
  Training never blocks on the runner.

The HTTP client is constructed lazily on the first publish (so
import-time has no side effects) and closed in ``on_train_end``.
"""

from __future__ import annotations

import os
from collections import deque
from typing import TYPE_CHECKING, Any

import httpx
from transformers import TrainerCallback

if TYPE_CHECKING:
    from transformers import TrainerControl, TrainerState, TrainingArguments


__all__ = ["RunnerEventCallback", "RUNNER_URL_ENV"]


# Env var the supervisor sets when spawning the trainer subprocess.
# Looked up once in ``__init__``; absent → callback is a no-op.
RUNNER_URL_ENV = "RYOTENKAI_RUNNER_URL"

# How many consecutive POST failures before we give up for the run.
MAX_CONSECUTIVE_FAILURES = 3

# Default cap on the local buffer. Once exceeded, oldest events are
# dropped — keeps the trainer's RAM under control even if the runner
# is unresponsive for long stretches.
DEFAULT_BUFFER_CAP = 1000


class RunnerEventCallback(TrainerCallback):
    """Push HF Trainer progress to the in-pod runner.

    Args:
        runner_url: Base URL of the runner. ``None`` (the default)
            reads ``RYOTENKAI_RUNNER_URL`` from the environment;
            if that is also unset the callback is a no-op for the
            full lifetime of the trainer.
        flush_every: Emit a ``step`` event (and flush the buffer)
            every N training steps. Default 10.
        timeout_seconds: Per-request HTTP timeout. The runner is
            on the same machine over loopback, so 2 s is generous.
        buffer_cap: Max in-memory buffered events. Excess drops
            from the oldest end.
    """

    def __init__(
        self,
        runner_url: str | None = None,
        *,
        flush_every: int = 10,
        timeout_seconds: float = 2.0,
        buffer_cap: int = DEFAULT_BUFFER_CAP,
    ) -> None:
        url = runner_url if runner_url is not None else os.environ.get(RUNNER_URL_ENV)
        self._enabled: bool = bool(url)
        self._url: str = (url or "").rstrip("/")
        self._flush_every = max(1, int(flush_every))
        self._timeout = float(timeout_seconds)
        self._buffer: deque[dict[str, Any]] = deque(maxlen=buffer_cap)
        self._consecutive_failures = 0
        self._client: httpx.Client | None = None
        # Tracks the most recent loss the Trainer reported through
        # ``on_log``; piggybacked onto the next ``step`` event.
        self._last_loss: float | None = None

    # ---- public introspection (used in tests) -----------------------

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def buffer_size(self) -> int:
        return len(self._buffer)

    # ---- HF TrainerCallback hooks -----------------------------------

    def on_train_begin(  # type: ignore[override]
        self,
        args: "TrainingArguments",
        state: "TrainerState",
        control: "TrainerControl",
        **kwargs: Any,
    ) -> None:
        if not self._enabled:
            return
        self._publish(
            "training_started",
            {
                "max_steps": int(state.max_steps),
                "num_train_epochs": float(args.num_train_epochs or 0.0),
                "per_device_train_batch_size": int(args.per_device_train_batch_size),
            },
            flush_now=True,
        )

    def on_log(  # type: ignore[override]
        self,
        args: "TrainingArguments",
        state: "TrainerState",
        control: "TrainerControl",
        logs: dict[str, float] | None = None,
        **kwargs: Any,
    ) -> None:
        if not self._enabled:
            return
        if logs is None:
            return
        # Capture the loss for the next ``step`` event.
        loss = logs.get("loss")
        if isinstance(loss, (int, float)):
            self._last_loss = float(loss)
        self._publish(
            "log",
            {"step": int(state.global_step), "logs": dict(logs)},
        )

    def on_step_end(  # type: ignore[override]
        self,
        args: "TrainingArguments",
        state: "TrainerState",
        control: "TrainerControl",
        **kwargs: Any,
    ) -> None:
        if not self._enabled:
            return
        if state.global_step % self._flush_every != 0:
            return
        payload: dict[str, Any] = {
            "step": int(state.global_step),
            "epoch": float(state.epoch or 0.0),
        }
        if self._last_loss is not None:
            payload["loss"] = self._last_loss
        self._publish("step", payload, flush_now=True)

    def on_evaluate(  # type: ignore[override]
        self,
        args: "TrainingArguments",
        state: "TrainerState",
        control: "TrainerControl",
        metrics: dict[str, float] | None = None,
        **kwargs: Any,
    ) -> None:
        if not self._enabled:
            return
        self._publish(
            "eval_metrics",
            {"step": int(state.global_step), "metrics": dict(metrics or {})},
            flush_now=True,
        )

    def on_save(  # type: ignore[override]
        self,
        args: "TrainingArguments",
        state: "TrainerState",
        control: "TrainerControl",
        **kwargs: Any,
    ) -> None:
        if not self._enabled:
            return
        self._publish(
            "checkpoint_saved",
            {"step": int(state.global_step)},
            flush_now=True,
        )

    def on_train_end(  # type: ignore[override]
        self,
        args: "TrainingArguments",
        state: "TrainerState",
        control: "TrainerControl",
        **kwargs: Any,
    ) -> None:
        if not self._enabled:
            return
        self._publish(
            "training_complete",
            {"final_step": int(state.global_step)},
            flush_now=True,
        )
        # One more flush attempt to drain anything left in the buffer.
        self._flush()
        if self._client is not None:
            try:
                self._client.close()
            except Exception:  # pragma: no cover — best-effort cleanup
                pass
            self._client = None

    # ---- internals ---------------------------------------------------

    def _publish(
        self, kind: str, payload: dict[str, Any], *, flush_now: bool = False,
    ) -> None:
        """Append to the buffer; flush if it's a critical kind or the
        buffer's quota since last flush is reached."""
        self._buffer.append({"kind": kind, "payload": payload})
        if flush_now or len(self._buffer) >= self._flush_every:
            self._flush()

    def _ensure_client(self) -> httpx.Client | None:
        if self._client is None:
            try:
                self._client = httpx.Client(timeout=self._timeout)
            except Exception:  # pragma: no cover — extreme env failure
                self._enabled = False
                return None
        return self._client

    def _flush(self) -> None:
        """Drain the buffer one POST at a time; on failure keep events
        for the next attempt and bump the consecutive-failure counter."""
        if not self._enabled or not self._buffer:
            return
        client = self._ensure_client()
        if client is None:
            return

        endpoint = f"{self._url}/api/v1/internal/events"

        # Snapshot so we can rebuild the buffer if posts fail mid-way.
        pending = list(self._buffer)
        self._buffer.clear()

        i = 0
        try:
            for i, item in enumerate(pending):
                response = client.post(endpoint, json=item)
                response.raise_for_status()
            self._consecutive_failures = 0
        except Exception:
            # Re-buffer the rest (the failed item AND everything after).
            self._buffer.extendleft(reversed(pending[i:]))
            self._consecutive_failures += 1
            if self._consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                self._enabled = False
                # Drop accumulated buffer so we don't grow it forever.
                self._buffer.clear()
                # Best-effort: close the client so we don't keep the
                # socket alive for nothing.
                try:
                    client.close()
                except Exception:  # pragma: no cover
                    pass
                self._client = None
