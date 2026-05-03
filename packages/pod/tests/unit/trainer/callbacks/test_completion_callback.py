"""Phase 11.A — :class:`CompletionCallback` contract.

Counterpart tests to :file:`test_cancellation_callback.py`. Same
slim-venv import pattern; same 7-category coverage; same fixture
shapes (``_StubMlflowManager``, ``_StubHandler``, ``_PublisherSpy``).

What we pin:

1. **Positive** — natural completion + manager + flush ok ⇒
   ``completion.marker`` written with ``reason="natural_completion"``,
   ``flush_timed_out=False``, count > 0.
2. **Negative** — cancellation flag raised ⇒ NO-OP (cancellation
   callback owns this path); manager unavailable ⇒ NO-OP silently.
3. **Boundary** — ``_signalled`` flips during ``on_train_end``.
4. **Invariants** — never raises, even when manager.flush_buffer
   raises; marker write failure ⇒ logs but no crash.
5. **Dependency errors** — handler missing ``should_stop`` ⇒
   defensive fallback (assume not cancelled).
6. **Regressions** — ``completion.marker`` written ALWAYS on natural
   end (success AND timeout); kept symmetric with cancellation
   callback contract.
7. **Logic-specific** — flush timeout ⇒ marker has
   ``reason="flush_budget_exceeded"``, ``flush_timed_out=true``;
   ``completion_finalized`` event still emitted with marker_written=True.

The test file uses the same slim-venv import trick as
``test_cancellation_callback.py`` so it runs in the dev venv without
pulling ``datasets`` / ``peft``.
"""

from __future__ import annotations

import importlib.util as _importlib_util
import json
import pathlib as _pathlib
import sys as _sys
import time
import types as _types
from types import SimpleNamespace
from typing import Any

import pytest


# ---------------------------------------------------------------------------
# Slim-venv import boilerplate (mirrors test_cancellation_callback.py)
# ---------------------------------------------------------------------------


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
    """Stand-in base class when ``transformers`` isn't installed."""


_stub("transformers", {"TrainerCallback": _TrainerCallback})
_stub("colorlog", {"ColoredFormatter": type})


_TRAINING_PKG = _pathlib.Path(__file__).resolve().parents[4] / "src" / "ryotenkai_pod" / "trainer"

# Make src.training a real package shell so other tests in the same
# session can still import sibling modules.
if "ryotenkai_pod.trainer" not in _sys.modules:
    _shell = _types.ModuleType("ryotenkai_pod.trainer")
    _shell.__path__ = [str(_TRAINING_PKG)]  # type: ignore[attr-defined]
    _sys.modules["ryotenkai_pod.trainer"] = _shell

# Pre-load _concurrent_helpers and _flush_helper under their real names
# so the lazy imports inside the callback body resolve.
for relpath, modname in (
    ("_concurrent_helpers.py", "ryotenkai_pod.trainer._concurrent_helpers"),
    ("callbacks/_flush_helper.py", "ryotenkai_pod.trainer.callbacks._flush_helper"),
):
    if modname not in _sys.modules:
        path = _TRAINING_PKG / relpath
        spec = _importlib_util.spec_from_file_location(modname, path)
        assert spec is not None and spec.loader is not None
        mod = _importlib_util.module_from_spec(spec)
        _sys.modules[modname] = mod
        spec.loader.exec_module(mod)


_CALLBACK_PATH = _TRAINING_PKG / "callbacks" / "completion_callback.py"
_spec = _importlib_util.spec_from_file_location(
    "_ryotenkai_completion_callback_under_test", _CALLBACK_PATH,
)
assert _spec is not None and _spec.loader is not None
_module = _importlib_util.module_from_spec(_spec)
_sys.modules["_ryotenkai_completion_callback_under_test"] = _module
_spec.loader.exec_module(_module)

CompletionCallback = _module.CompletionCallback


# ---------------------------------------------------------------------------
# Test doubles (kept identical to test_cancellation_callback.py for parity)
# ---------------------------------------------------------------------------


class _StubHandler:
    """In-memory stand-in for ShutdownHandler."""

    def __init__(self, *, requested: bool = False) -> None:
        self._requested = requested

    def should_stop(self) -> bool:
        return self._requested

    def request(self) -> None:
        self._requested = True


class _StubMlflowManager:
    """Minimal stand-in for :class:`MlflowManager`."""

    def __init__(
        self, *,
        flush_return: int = 0,
        flush_raises: Exception | None = None,
        flush_blocks_for: float = 0.0,
        run_id: str | None = None,
    ) -> None:
        self.flush_calls: int = 0
        self._flush_return = flush_return
        self._flush_raises = flush_raises
        self._flush_blocks_for = flush_blocks_for
        if run_id is not None:
            self.run_id = run_id

    def flush_buffer(self) -> int:
        self.flush_calls += 1
        if self._flush_blocks_for > 0:
            time.sleep(self._flush_blocks_for)
        if self._flush_raises is not None:
            raise self._flush_raises
        return self._flush_return


class _PublisherSpy:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, object]]] = []

    def __call__(self, kind: str, payload: dict[str, object]) -> None:
        self.events.append((kind, dict(payload)))


def _control() -> Any:
    return SimpleNamespace(should_save=False, should_training_stop=False)


def _state(global_step: int = 0) -> Any:
    return SimpleNamespace(global_step=global_step)


# ---------------------------------------------------------------------------
# 1. Positive
# ---------------------------------------------------------------------------


class TestPositive:
    def test_natural_completion_writes_marker_and_flushes(
        self, tmp_path: Any,
    ) -> None:
        manager = _StubMlflowManager(flush_return=42, run_id="run-001")
        cb = CompletionCallback(
            shutdown_handler=_StubHandler(requested=False),
            mlflow_manager=manager,
            workspace_dir=tmp_path,
        )
        cb.on_train_end(args=None, state=_state(), control=_control())

        # Flush ran exactly once.
        assert manager.flush_calls == 1

        # Marker exists with expected payload.
        marker = tmp_path / "completion.marker"
        assert marker.exists()
        body = json.loads(marker.read_text())
        assert body["run_id"] == "run-001"
        assert body["flushed_count"] == 42
        assert body["flush_timed_out"] is False
        assert body["reason"] == "natural_completion"
        assert isinstance(body["ts_ms"], int)

    def test_emits_completion_finalized_event(self, tmp_path: Any) -> None:
        publisher = _PublisherSpy()
        manager = _StubMlflowManager(flush_return=7)
        cb = CompletionCallback(
            shutdown_handler=_StubHandler(requested=False),
            mlflow_manager=manager,
            event_publisher=publisher,
            workspace_dir=tmp_path,
        )
        cb.on_train_end(args=None, state=_state(), control=_control())

        # Exactly one event, kind=completion_finalized.
        assert len(publisher.events) == 1
        kind, payload = publisher.events[0]
        assert kind == "completion_finalized"
        assert payload["flushed_count"] == 7
        assert payload["flush_timed_out"] is False
        assert payload["marker_written"] is True


# ---------------------------------------------------------------------------
# 2. Negative
# ---------------------------------------------------------------------------


class TestNegative:
    def test_cancellation_flag_raised_skips_flush(
        self, tmp_path: Any,
    ) -> None:
        manager = _StubMlflowManager(flush_return=10)
        cb = CompletionCallback(
            shutdown_handler=_StubHandler(requested=True),
            mlflow_manager=manager,
            workspace_dir=tmp_path,
        )
        cb.on_train_end(args=None, state=_state(), control=_control())

        # NO-OP — CancellationCallback owns this path.
        assert manager.flush_calls == 0
        marker = tmp_path / "completion.marker"
        assert not marker.exists()

    def test_no_manager_skips_silently(self, tmp_path: Any) -> None:
        cb = CompletionCallback(
            shutdown_handler=_StubHandler(requested=False),
            mlflow_manager=None,
            workspace_dir=tmp_path,
        )
        # Must not raise.
        cb.on_train_end(args=None, state=_state(), control=_control())
        # No marker since flush never ran.
        marker = tmp_path / "completion.marker"
        assert not marker.exists()


# ---------------------------------------------------------------------------
# 3. Boundary
# ---------------------------------------------------------------------------


class TestBoundary:
    def test_signalled_flips_during_on_train_end(
        self, tmp_path: Any,
    ) -> None:
        # If the cancellation flag flips DURING on_train_end (race
        # between user stop + natural exit), we still defer to
        # CancellationCallback. Implementation reads the flag once
        # at the start.
        handler = _StubHandler(requested=False)
        manager = _StubMlflowManager(flush_return=5)
        cb = CompletionCallback(
            shutdown_handler=handler,
            mlflow_manager=manager,
            workspace_dir=tmp_path,
        )

        # Flip the flag right BEFORE on_train_end is called.
        handler.request()
        cb.on_train_end(args=None, state=_state(), control=_control())

        # NO-OP — CancellationCallback owns.
        assert manager.flush_calls == 0


# ---------------------------------------------------------------------------
# 4. Invariants
# ---------------------------------------------------------------------------


class TestInvariants:
    def test_never_raises_when_flush_buffer_explodes(
        self, tmp_path: Any,
    ) -> None:
        manager = _StubMlflowManager(flush_raises=ValueError("manager bug"))
        cb = CompletionCallback(
            shutdown_handler=_StubHandler(requested=False),
            mlflow_manager=manager,
            workspace_dir=tmp_path,
        )
        # MUST NOT raise.
        cb.on_train_end(args=None, state=_state(), control=_control())
        # Marker still written (always-write contract).
        marker = tmp_path / "completion.marker"
        assert marker.exists()

    def test_never_raises_when_marker_write_fails(self) -> None:
        # workspace_dir is a non-existent path → marker write fails →
        # callback must still complete cleanly.
        manager = _StubMlflowManager(flush_return=3)
        cb = CompletionCallback(
            shutdown_handler=_StubHandler(requested=False),
            mlflow_manager=manager,
            workspace_dir="/nonexistent/path/that/cannot/exist",
        )
        # MUST NOT raise.
        cb.on_train_end(args=None, state=_state(), control=_control())
        # Flush ran.
        assert manager.flush_calls == 1


# ---------------------------------------------------------------------------
# 5. Dependency errors
# ---------------------------------------------------------------------------


class TestDependencyErrors:
    def test_handler_missing_should_stop_defaults_to_not_cancelled(
        self, tmp_path: Any,
    ) -> None:
        class _NoMethod:
            pass

        manager = _StubMlflowManager(flush_return=4)
        cb = CompletionCallback(
            shutdown_handler=_NoMethod(),
            mlflow_manager=manager,
            workspace_dir=tmp_path,
        )
        cb.on_train_end(args=None, state=_state(), control=_control())

        # Defensive: handler raised → assume not cancelled → flush runs.
        assert manager.flush_calls == 1
        marker = tmp_path / "completion.marker"
        assert marker.exists()

    def test_publisher_exception_does_not_propagate(
        self, tmp_path: Any,
    ) -> None:
        def _bad_publisher(kind: str, payload: dict) -> None:
            raise RuntimeError("event bus offline")

        manager = _StubMlflowManager(flush_return=2)
        cb = CompletionCallback(
            shutdown_handler=_StubHandler(requested=False),
            mlflow_manager=manager,
            event_publisher=_bad_publisher,
            workspace_dir=tmp_path,
        )
        # MUST NOT raise.
        cb.on_train_end(args=None, state=_state(), control=_control())
        assert manager.flush_calls == 1


# ---------------------------------------------------------------------------
# 6. Regressions
# ---------------------------------------------------------------------------


class TestRegressions:
    def test_marker_always_written_even_on_zero_drained(
        self, tmp_path: Any,
    ) -> None:
        # Empty buffer (drained=0) → marker still written. Per
        # Phase 11.A "always-write" contract.
        manager = _StubMlflowManager(flush_return=0)
        cb = CompletionCallback(
            shutdown_handler=_StubHandler(requested=False),
            mlflow_manager=manager,
            workspace_dir=tmp_path,
        )
        cb.on_train_end(args=None, state=_state(), control=_control())
        marker = tmp_path / "completion.marker"
        assert marker.exists()
        body = json.loads(marker.read_text())
        assert body["flushed_count"] == 0

    def test_workspace_resolution_falls_back_to_env(
        self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # No workspace_dir injected → fall back to HELIX_WORKSPACE.
        monkeypatch.setenv("HELIX_WORKSPACE", str(tmp_path))

        manager = _StubMlflowManager(flush_return=1)
        cb = CompletionCallback(
            shutdown_handler=_StubHandler(requested=False),
            mlflow_manager=manager,
            # No workspace_dir argument.
        )
        cb.on_train_end(args=None, state=_state(), control=_control())
        marker = tmp_path / "completion.marker"
        assert marker.exists()


# ---------------------------------------------------------------------------
# 7. Logic-specific
# ---------------------------------------------------------------------------


class TestLogicSpecific:
    def test_flush_timeout_marker_has_correct_reason(
        self, tmp_path: Any,
    ) -> None:
        # Slow flush exceeds the budget → marker has reason=
        # "flush_budget_exceeded" and flush_timed_out=True.
        manager = _StubMlflowManager(flush_blocks_for=0.5, run_id="run-tt")
        cb = CompletionCallback(
            shutdown_handler=_StubHandler(requested=False),
            mlflow_manager=manager,
            flush_timeout_seconds=0.05,
            workspace_dir=tmp_path,
        )
        cb.on_train_end(args=None, state=_state(), control=_control())

        marker = tmp_path / "completion.marker"
        assert marker.exists()
        body = json.loads(marker.read_text())
        assert body["reason"] == "flush_budget_exceeded"
        assert body["flush_timed_out"] is True
        # We never saw the return value → drained_count stays 0.
        assert body["flushed_count"] == 0

    def test_flush_timeout_emits_event_with_marker_written_true(
        self, tmp_path: Any,
    ) -> None:
        publisher = _PublisherSpy()
        manager = _StubMlflowManager(flush_blocks_for=0.5)
        cb = CompletionCallback(
            shutdown_handler=_StubHandler(requested=False),
            mlflow_manager=manager,
            event_publisher=publisher,
            flush_timeout_seconds=0.05,
            workspace_dir=tmp_path,
        )
        cb.on_train_end(args=None, state=_state(), control=_control())

        assert len(publisher.events) == 1
        _, payload = publisher.events[0]
        assert payload["flush_timed_out"] is True
        assert payload["marker_written"] is True

    def test_cancellation_path_emits_no_event(
        self, tmp_path: Any,
    ) -> None:
        # If cancellation owns the path, completion stays silent —
        # no double-event on the bus.
        publisher = _PublisherSpy()
        manager = _StubMlflowManager(flush_return=10)
        cb = CompletionCallback(
            shutdown_handler=_StubHandler(requested=True),
            mlflow_manager=manager,
            event_publisher=publisher,
            workspace_dir=tmp_path,
        )
        cb.on_train_end(args=None, state=_state(), control=_control())

        assert publisher.events == []

    def test_injected_marker_writer_is_used(self) -> None:
        captured: dict[str, object] = {}

        def _writer(payload: dict[str, object]) -> str:
            captured["payload"] = payload
            return "/sentinel/completion.marker"

        manager = _StubMlflowManager(flush_return=5)
        cb = CompletionCallback(
            shutdown_handler=_StubHandler(requested=False),
            mlflow_manager=manager,
            marker_writer=_writer,
        )
        cb.on_train_end(args=None, state=_state(), control=_control())

        assert "payload" in captured
        payload = captured["payload"]  # type: ignore[index]
        assert isinstance(payload, dict)
        assert payload["flushed_count"] == 5
        assert payload["flush_timed_out"] is False
