"""Phase 2 — :class:`RunnerEventCallback` async-queue contract.

Coverage (slim — the heavy integration is in the runner-side internal
endpoint tests):

* The callback is a no-op when ``RYOTENKAI_RUNNER_URL`` is unset.
* :meth:`on_step_end` builds a typed envelope and enqueues it.
* Drop-oldest semantics bump the counter on overflow.

Heavy HF integration (full Trainer lifecycle, real httpx transport)
lives in the integration suite. This file only asserts the public
contract on the new envelope-based path without spinning up the
trainer.
"""

from __future__ import annotations

import importlib.util as _importlib_util
import pathlib as _pathlib
import sys as _sys
import types as _types
from dataclasses import dataclass


def _stub(name: str, attrs: dict[str, object] | None = None) -> None:
    if name in _sys.modules:
        return
    try:
        __import__(name)
    except ModuleNotFoundError:
        module = _types.ModuleType(name)
        for attr_name, attr_value in (attrs or {}).items():
            setattr(module, attr_name, attr_value)
        _sys.modules[name] = module


class _TrainerCallback:
    """Stand-in for ``transformers.TrainerCallback``."""


_stub("transformers", {"TrainerCallback": _TrainerCallback})
_stub("colorlog", {"ColoredFormatter": type})

_CALLBACK_PATH = (
    _pathlib.Path(__file__).resolve().parents[5]
    / "packages" / "pod" / "src" / "ryotenkai_pod" / "trainer"
    / "callbacks" / "runner_event_callback.py"
)
_spec = _importlib_util.spec_from_file_location(
    "_ryotenkai_runner_event_callback_under_test", _CALLBACK_PATH,
)
assert _spec is not None and _spec.loader is not None
_module = _importlib_util.module_from_spec(_spec)
_sys.modules["_ryotenkai_runner_event_callback_under_test"] = _module
_spec.loader.exec_module(_module)

RunnerEventCallback = _module.RunnerEventCallback
RUNNER_URL_ENV = _module.RUNNER_URL_ENV
DEFAULT_QUEUE_CAP = _module.DEFAULT_QUEUE_CAP


@dataclass
class _State:
    global_step: int = 0
    epoch: float = 0.0
    max_steps: int = 100


@dataclass
class _Args:
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    learning_rate: float = 0.001


@dataclass
class _Control:
    pass


class TestEnablement:
    def test_disabled_when_env_missing(self, monkeypatch) -> None:
        monkeypatch.delenv(RUNNER_URL_ENV, raising=False)
        cb = RunnerEventCallback()
        assert cb.enabled is False
        # Hooks return immediately and no envelope is enqueued.
        cb.on_step_end(_Args(), _State(global_step=10), _Control())
        assert cb.queue_size == 0

    def test_enabled_when_runner_url_set(self) -> None:
        cb = RunnerEventCallback(runner_url="http://127.0.0.1:9999")
        assert cb.enabled is True
        # Stop the daemon worker so the test doesn't hang on close.
        cb._stop_evt.set()


class TestEnqueue:
    def test_on_step_end_emits_typed_envelope(self) -> None:
        cb = RunnerEventCallback(runner_url="http://127.0.0.1:9999")
        cb._stop_evt.set()  # disable background worker for inspection
        cb.on_step_end(_Args(), _State(global_step=10), _Control())
        assert cb.queue_size == 1
        envelope = cb._queue.get()
        assert envelope.kind == "ryotenkai.pod.training.step"
        assert envelope.payload.step == 10

    def test_drop_oldest_increments_counter(self) -> None:
        cb = RunnerEventCallback(
            runner_url="http://127.0.0.1:9999",
            queue_cap=2,
        )
        cb._stop_evt.set()
        # Use ``_flush_every=1`` semantic by hitting the step end three
        # times. Each call enqueues; the third one forces a drop-oldest.
        cb.on_step_end(_Args(), _State(global_step=10), _Control())
        cb.on_step_end(_Args(), _State(global_step=20), _Control())
        cb.on_step_end(_Args(), _State(global_step=30), _Control())
        assert cb.dropped_total >= 1


# ---------------------------------------------------------------------------
# Phase 3 fix — typed ``TrainingFailedEvent`` emission contract.
# ---------------------------------------------------------------------------


def _silence_worker(cb) -> None:
    """Stop and join the background worker so it never drains the queue.

    The callback starts its daemon worker in ``__init__``; if we don't
    stop and join it, the worker can race the test and consume queued
    envelopes before we inspect them (no real HTTP server is listening,
    so it would just drop with a counter increment).
    """
    cb._stop_evt.set()
    if getattr(cb, "_worker", None) is not None:
        cb._worker.join(timeout=1.0)


def _drain_queue(cb) -> list:
    out = []
    while True:
        try:
            out.append(cb._queue.get_nowait())
        except Exception:
            break
    return out


class TestTrainingFailed:
    """Contract tests for :meth:`RunnerEventCallback.emit_training_failed`."""

    def test_emit_training_failed_builds_correct_envelope(self) -> None:
        cb = RunnerEventCallback(runner_url="http://127.0.0.1:9999")
        _silence_worker(cb)

        try:
            raise RuntimeError("boom")
        except RuntimeError as exc:
            cb.emit_training_failed(exc=exc, step=7)

        events = _drain_queue(cb)
        assert len(events) == 1
        envelope = events[0]
        assert envelope.kind == "ryotenkai.pod.training.failed"
        assert envelope.severity == "error"
        assert envelope.payload.error_type == "RuntimeError"
        assert envelope.payload.message == "boom"
        assert envelope.payload.step == 7
        assert envelope.payload.traceback_excerpt  # non-empty
        # ``RuntimeError`` should appear inside the formatted traceback.
        assert "RuntimeError" in envelope.payload.traceback_excerpt

    def test_emit_training_failed_then_on_train_end_does_not_emit_completed(
        self,
    ) -> None:
        cb = RunnerEventCallback(runner_url="http://127.0.0.1:9999")
        _silence_worker(cb)

        try:
            raise ValueError("config invalid")
        except ValueError as exc:
            cb.emit_training_failed(exc=exc)

        # Now simulate HF Trainer's try/finally calling on_train_end.
        cb.on_train_end(_Args(), _State(global_step=42), _Control())

        events = _drain_queue(cb)
        # Exactly ONE envelope: the failed event. on_train_end MUST NOT
        # have enqueued a TrainingCompletedEvent because ``failed`` is set.
        assert len(events) == 1
        assert events[0].kind == "ryotenkai.pod.training.failed"
        assert cb.failed is True

    def test_traceback_excerpt_truncated_to_2kb(self) -> None:
        cb = RunnerEventCallback(runner_url="http://127.0.0.1:9999")
        _silence_worker(cb)

        # Forge a giant traceback excerpt input directly to assert the
        # truncation cap — sidesteps platform / Python noise in real
        # tracebacks.
        giant = "x" * 5000  # 5 KB of body text
        cb.emit_training_failed(
            error_type="RuntimeError",
            message="big",
            traceback_excerpt=giant,
            step=0,
        )
        events = _drain_queue(cb)
        assert len(events) == 1
        excerpt = events[0].payload.traceback_excerpt
        # 2 KB cap (UTF-8 bytes). The truncation suffix lives inside
        # the cap, never past it.
        assert len(excerpt.encode("utf-8")) <= 2048
        # Sanity: truncation marker present so consumers can detect cut.
        assert "...[truncated]" in excerpt

    def test_step_reflects_current_global_step(self) -> None:
        cb = RunnerEventCallback(runner_url="http://127.0.0.1:9999")
        _silence_worker(cb)
        # Simulate HF reporting step 42 via on_step_end (flush_every=10
        # by default; the step is cached regardless of whether the
        # envelope is emitted).
        cb.on_step_end(_Args(), _State(global_step=42), _Control())
        # Drain the step event so only the failed event remains in queue.
        _drain_queue(cb)

        try:
            raise RuntimeError("after step 42")
        except RuntimeError as exc:
            cb.emit_training_failed(exc=exc)

        events = _drain_queue(cb)
        assert len(events) == 1
        assert events[0].payload.step == 42

    def test_step_minus_one_when_pre_train(self) -> None:
        cb = RunnerEventCallback(runner_url="http://127.0.0.1:9999")
        _silence_worker(cb)
        # No on_train_begin / on_step_end were ever called.
        try:
            raise RuntimeError("pre-loop crash")
        except RuntimeError as exc:
            cb.emit_training_failed(exc=exc)

        events = _drain_queue(cb)
        assert len(events) == 1
        assert events[0].payload.step == -1

    def test_failed_flag_set_even_when_disabled(self, monkeypatch) -> None:
        """``_failed_flag`` propagates even on no-op callbacks.

        Keeps ``on_train_end`` symmetric: when the callback was
        instantiated without ``RYOTENKAI_RUNNER_URL`` no envelope is
        emitted, but the flag still flips so any subsequent hook does
        not get a free pass.
        """
        monkeypatch.delenv(RUNNER_URL_ENV, raising=False)
        cb = RunnerEventCallback()
        assert cb.enabled is False
        cb.emit_training_failed(error_type="X", message="y", step=0)
        # No envelopes (disabled), but the flag tracks the failure.
        assert cb.queue_size == 0
        assert cb.failed is True
        # And ``on_train_end`` must not enqueue a Completed event either.
        cb.on_train_end(_Args(), _State(global_step=1), _Control())
        assert cb.queue_size == 0

    def test_explicit_fields_override_exception_derivation(self) -> None:
        cb = RunnerEventCallback(runner_url="http://127.0.0.1:9999")
        _silence_worker(cb)
        try:
            raise RuntimeError("generic")
        except RuntimeError as exc:
            cb.emit_training_failed(
                exc=exc,
                error_type="OutOfMemoryError",
                message="VRAM exhausted",
                step=99,
            )
        events = _drain_queue(cb)
        assert len(events) == 1
        payload = events[0].payload
        assert payload.error_type == "OutOfMemoryError"
        assert payload.message == "VRAM exhausted"
        assert payload.step == 99
