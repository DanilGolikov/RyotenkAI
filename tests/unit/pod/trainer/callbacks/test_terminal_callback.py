"""Unified :class:`TerminalCallback` contract -- cancel + complete reasons.

Consolidated test suite for the merged terminal-state HF Trainer
callback. Parametric over ``reason``:

* ``reason="cancel"`` -- bridges :class:`ShutdownHandler`'s flag (set
  by SIGTERM) to HF Trainer's ``TrainerControl(should_save,
  should_training_stop)`` on each ``on_step_end`` so the trainer
  cooperatively saves a checkpoint at the next save boundary; on
  ``on_train_end`` (only when signalled) drains the MLflow buffer and
  writes ``cancelled.marker`` on flush-budget overrun.
* ``reason="complete"`` -- on ``on_train_end`` (only when no
  cancellation observed) drains the MLflow buffer and unconditionally
  writes ``completion.marker``.

Categories covered:

1. Positive
2. Negative
3. Boundary
4. Invariants
5. Dependency errors
6. Regressions
7. Logic-specific (flush, marker, telemetry)
"""

from __future__ import annotations

import importlib.util as _importlib_util
import pathlib as _pathlib
import sys as _sys
import types as _types
from types import SimpleNamespace
from typing import Any

import pytest


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


# Slim-venv pre-load of the helper module so the callback's lazy
# import resolves without triggering ``src.training/__init__.py``'s
# heavy dependency chain.
_HELPER_PATH = (
    _pathlib.Path(__file__).resolve().parents[5]
    / "packages" / "pod" / "src" / "ryotenkai_pod" / "trainer" / "_concurrent_helpers.py"
)
_helper_spec = _importlib_util.spec_from_file_location(
    "ryotenkai_pod.trainer._concurrent_helpers", _HELPER_PATH,
)
assert _helper_spec is not None and _helper_spec.loader is not None
_helper_module = _importlib_util.module_from_spec(_helper_spec)
if "ryotenkai_pod.trainer" not in _sys.modules:
    _training_shell = _types.ModuleType("ryotenkai_pod.trainer")
    _training_shell.__path__ = [  # type: ignore[attr-defined]
        str(_pathlib.Path(__file__).resolve().parents[5] / "packages" / "pod" / "src" / "ryotenkai_pod" / "trainer"),
    ]
    _sys.modules["ryotenkai_pod.trainer"] = _training_shell
_sys.modules["ryotenkai_pod.trainer._concurrent_helpers"] = _helper_module
_helper_spec.loader.exec_module(_helper_module)


_CALLBACK_PATH = (
    _pathlib.Path(__file__).resolve().parents[5]
    / "packages" / "pod" / "src" / "ryotenkai_pod" / "trainer" / "callbacks" / "terminal_callback.py"
)
_spec = _importlib_util.spec_from_file_location(
    "_ryotenkai_terminal_callback_under_test", _CALLBACK_PATH,
)
assert _spec is not None and _spec.loader is not None
_module = _importlib_util.module_from_spec(_spec)
_sys.modules["_ryotenkai_terminal_callback_under_test"] = _module
_spec.loader.exec_module(_module)

TerminalCallback = _module.TerminalCallback


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _StubHandler:
    def __init__(self, *, requested: bool = False) -> None:
        self._requested = requested

    def should_stop(self) -> bool:
        return self._requested

    def request(self) -> None:
        self._requested = True


class _StubMlflowManager:
    """Minimal stand-in for the MLflow manager used by flush tests."""

    def __init__(
        self, *,
        flush_return: int = 0,
        flush_raises: Exception | None = None,
        flush_blocks_for: float = 0.0,
        run_id: str | None = "run-XYZ",
    ) -> None:
        self.flush_calls: int = 0
        self._flush_return = flush_return
        self._flush_raises = flush_raises
        self._flush_blocks_for = flush_blocks_for
        self.run_id = run_id

    def flush_buffer(self) -> int:
        self.flush_calls += 1
        if self._flush_blocks_for > 0:
            import time
            time.sleep(self._flush_blocks_for)
        if self._flush_raises is not None:
            raise self._flush_raises
        return self._flush_return


def _control() -> Any:
    return SimpleNamespace(should_save=False, should_training_stop=False)


def _state(global_step: int = 0) -> Any:
    return SimpleNamespace(global_step=global_step)


# ---------------------------------------------------------------------------
# 1. Positive
# ---------------------------------------------------------------------------


class TestPositiveCancel:
    def test_flag_set_raises_both_controls(self) -> None:
        cb = TerminalCallback(reason="cancel", shutdown_handler=_StubHandler(requested=True))
        ctrl = _control()
        cb.on_step_end(args=None, state=_state(global_step=42), control=ctrl)
        assert ctrl.should_save is True
        assert ctrl.should_training_stop is True

    def test_returns_none(self) -> None:
        cb = TerminalCallback(reason="cancel", shutdown_handler=_StubHandler(requested=True))
        result = cb.on_step_end(args=None, state=_state(), control=_control())
        assert result is None


class TestPositiveComplete:
    def test_on_train_end_flushes_on_natural_exit(self) -> None:
        manager = _StubMlflowManager(flush_return=7)
        cb = TerminalCallback(
            reason="complete",
            shutdown_handler=_StubHandler(requested=False),
            mlflow_manager=manager,
            marker_writer=lambda payload: None,
        )
        cb.on_train_end(args=None, state=_state(), control=_control())
        assert manager.flush_calls == 1


# ---------------------------------------------------------------------------
# 2. Negative
# ---------------------------------------------------------------------------


class TestNegativeCancel:
    def test_flag_unset_leaves_control_untouched(self) -> None:
        cb = TerminalCallback(reason="cancel", shutdown_handler=_StubHandler(requested=False))
        ctrl = _control()
        cb.on_step_end(args=None, state=_state(), control=ctrl)
        assert ctrl.should_save is False
        assert ctrl.should_training_stop is False

    def test_on_train_end_silent_on_clean_exit(self) -> None:
        manager = _StubMlflowManager(flush_return=99)
        cb = TerminalCallback(
            reason="cancel",
            shutdown_handler=_StubHandler(requested=False),
            mlflow_manager=manager,
        )
        cb.on_train_end(args=None, state=_state(), control=_control())
        assert manager.flush_calls == 0


class TestNegativeComplete:
    def test_on_train_end_is_noop_when_cancellation_active(self) -> None:
        manager = _StubMlflowManager(flush_return=5)
        cb = TerminalCallback(
            reason="complete",
            shutdown_handler=_StubHandler(requested=True),
            mlflow_manager=manager,
        )
        cb.on_train_end(args=None, state=_state(), control=_control())
        assert manager.flush_calls == 0

    def test_on_step_end_is_strict_noop_for_complete(self) -> None:
        cb = TerminalCallback(reason="complete", shutdown_handler=_StubHandler(requested=True))
        ctrl = _control()
        cb.on_step_end(args=None, state=_state(), control=ctrl)
        assert ctrl.should_save is False
        assert ctrl.should_training_stop is False


# ---------------------------------------------------------------------------
# 3. Boundary
# ---------------------------------------------------------------------------


class TestBoundaryCancel:
    def test_flag_flips_mid_run(self) -> None:
        h = _StubHandler(requested=False)
        cb = TerminalCallback(reason="cancel", shutdown_handler=h)
        for step in (1, 2, 3):
            ctrl = _control()
            cb.on_step_end(args=None, state=_state(step), control=ctrl)
            assert ctrl.should_save is False
            assert ctrl.should_training_stop is False
        h.request()
        ctrl = _control()
        cb.on_step_end(args=None, state=_state(4), control=ctrl)
        assert ctrl.should_save is True
        assert ctrl.should_training_stop is True

    def test_idempotent_across_subsequent_steps(self) -> None:
        h = _StubHandler(requested=True)
        cb = TerminalCallback(reason="cancel", shutdown_handler=h)
        for step in (10, 11, 12):
            ctrl = _control()
            cb.on_step_end(args=None, state=_state(step), control=ctrl)
            assert ctrl.should_save is True
            assert ctrl.should_training_stop is True


# ---------------------------------------------------------------------------
# 4. Invariants
# ---------------------------------------------------------------------------


class TestInvariantsCancel:
    def test_save_order_is_save_then_stop(self) -> None:
        observed: list[str] = []

        class _RecorderControl:
            _save = False
            _stop = False

            @property
            def should_save(self) -> bool:
                return self._save

            @should_save.setter
            def should_save(self, value: bool) -> None:
                observed.append("save")
                self._save = value

            @property
            def should_training_stop(self) -> bool:
                return self._stop

            @should_training_stop.setter
            def should_training_stop(self, value: bool) -> None:
                observed.append("stop")
                self._stop = value

        cb = TerminalCallback(reason="cancel", shutdown_handler=_StubHandler(requested=True))
        cb.on_step_end(args=None, state=_state(), control=_RecorderControl())
        assert observed == ["save", "stop"]

    def test_callback_never_raises_on_handler_error(self) -> None:
        class _BrokenHandler:
            def should_stop(self) -> bool:
                raise RuntimeError("handler internals exploded")

        cb = TerminalCallback(reason="cancel", shutdown_handler=_BrokenHandler())
        ctrl = _control()
        cb.on_step_end(args=None, state=_state(), control=ctrl)
        assert ctrl.should_save is False
        assert ctrl.should_training_stop is False


class TestInvariantsConstructor:
    def test_invalid_reason_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            TerminalCallback(reason="bogus")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 5. Dependency errors
# ---------------------------------------------------------------------------


class TestDependencyErrors:
    def test_handler_missing_attribute(self) -> None:
        class _NoMethod:
            pass

        cb = TerminalCallback(reason="cancel", shutdown_handler=_NoMethod())
        ctrl = _control()
        cb.on_step_end(args=None, state=_state(), control=ctrl)
        assert ctrl.should_save is False

    def test_no_manager_skips_flush_silently_cancel(self) -> None:
        cb = TerminalCallback(
            reason="cancel",
            shutdown_handler=_StubHandler(requested=True),
            mlflow_manager=None,
        )
        cb.on_step_end(args=None, state=_state(1), control=_control())
        cb.on_train_end(args=None, state=_state(2), control=_control())

    def test_no_manager_skips_flush_silently_complete(self) -> None:
        cb = TerminalCallback(
            reason="complete",
            shutdown_handler=_StubHandler(requested=False),
            mlflow_manager=None,
        )
        cb.on_train_end(args=None, state=_state(), control=_control())


# ---------------------------------------------------------------------------
# 6. Regressions
# ---------------------------------------------------------------------------


class TestRegressions:
    def test_handler_not_resolved_when_injected(self) -> None:
        h = _StubHandler(requested=False)
        cb = TerminalCallback(reason="cancel", shutdown_handler=h)
        assert cb._resolve_handler() is h
        assert cb._handler is h


# ---------------------------------------------------------------------------
# 7. Logic-specific
# ---------------------------------------------------------------------------


class TestFlushBehaviour:
    @pytest.mark.parametrize("reason", ["cancel", "complete"])
    def test_flush_drains_buffer(
        self, reason: str, caplog: pytest.LogCaptureFixture,
    ) -> None:
        import logging
        caplog.set_level(logging.INFO, logger="ryotenkai_pod.trainer.callbacks.terminal_callback")

        manager = _StubMlflowManager(flush_return=42)
        cb = TerminalCallback(
            reason=reason,  # type: ignore[arg-type]
            shutdown_handler=_StubHandler(requested=(reason == "cancel")),
            mlflow_manager=manager,
            marker_writer=lambda payload: None,
        )
        if reason == "cancel":
            cb.on_step_end(args=None, state=_state(1), control=_control())
        cb.on_train_end(args=None, state=_state(2), control=_control())

        assert manager.flush_calls == 1

    @pytest.mark.parametrize("reason", ["cancel", "complete"])
    def test_flush_timeout_logs_warning_and_continues(
        self, reason: str, caplog: pytest.LogCaptureFixture,
    ) -> None:
        import logging
        caplog.set_level(logging.WARNING, logger="ryotenkai_pod.trainer.callbacks.terminal_callback")

        manager = _StubMlflowManager(flush_blocks_for=0.5)
        cb = TerminalCallback(
            reason=reason,  # type: ignore[arg-type]
            shutdown_handler=_StubHandler(requested=(reason == "cancel")),
            mlflow_manager=manager,
            flush_timeout_seconds=0.05,
            marker_writer=lambda payload: None,
        )
        if reason == "cancel":
            cb.on_step_end(args=None, state=_state(1), control=_control())
        cb.on_train_end(args=None, state=_state(2), control=_control())

        timeout_logs = [
            r for r in caplog.records
            if "MLflow flush exceeded" in r.message
        ]
        assert len(timeout_logs) == 1

    @pytest.mark.parametrize("reason", ["cancel", "complete"])
    def test_flush_raises_does_not_propagate(self, reason: str) -> None:
        manager = _StubMlflowManager(flush_raises=RuntimeError("upstream MLflow down"))
        cb = TerminalCallback(
            reason=reason,  # type: ignore[arg-type]
            shutdown_handler=_StubHandler(requested=(reason == "cancel")),
            mlflow_manager=manager,
            marker_writer=lambda payload: None,
        )
        if reason == "cancel":
            cb.on_step_end(args=None, state=_state(1), control=_control())
        # Must not raise.
        cb.on_train_end(args=None, state=_state(2), control=_control())


class TestMarkerBehaviour:
    def test_complete_writes_marker_on_natural_completion(self) -> None:
        written: list[dict[str, Any]] = []

        def _writer(payload: dict[str, Any]) -> object:
            written.append(payload)
            return "/tmp/completion.marker"

        manager = _StubMlflowManager(flush_return=3, run_id="run-A")
        cb = TerminalCallback(
            reason="complete",
            shutdown_handler=_StubHandler(requested=False),
            mlflow_manager=manager,
            marker_writer=_writer,
        )
        cb.on_train_end(args=None, state=_state(), control=_control())
        assert len(written) == 1
        assert written[0]["run_id"] == "run-A"
        assert written[0]["flushed_count"] == 3
        assert written[0]["flush_timed_out"] is False

    def test_cancel_writes_marker_only_on_flush_timeout(self) -> None:
        written: list[dict[str, Any]] = []

        def _writer(payload: dict[str, Any]) -> object:
            written.append(payload)
            return "/tmp/cancelled.marker"

        # Fast flush -- no timeout, no marker.
        manager = _StubMlflowManager(flush_return=1)
        cb = TerminalCallback(
            reason="cancel",
            shutdown_handler=_StubHandler(requested=True),
            mlflow_manager=manager,
            marker_writer=_writer,
        )
        cb.on_step_end(args=None, state=_state(1), control=_control())
        cb.on_train_end(args=None, state=_state(2), control=_control())
        assert written == []

        # Slow flush -- timeout, marker written.
        manager2 = _StubMlflowManager(flush_blocks_for=0.5)
        cb2 = TerminalCallback(
            reason="cancel",
            shutdown_handler=_StubHandler(requested=True),
            mlflow_manager=manager2,
            flush_timeout_seconds=0.05,
            marker_writer=_writer,
        )
        cb2.on_step_end(args=None, state=_state(1), control=_control())
        cb2.on_train_end(args=None, state=_state(2), control=_control())
        assert len(written) == 1


class TestTelemetry:
    @pytest.mark.parametrize("reason", ["cancel", "complete"])
    def test_emit_finalized_event(self, reason: str) -> None:
        published: list[tuple[str, dict[str, Any]]] = []

        def _pub(kind: str, payload: dict[str, Any]) -> None:
            published.append((kind, payload))

        manager = _StubMlflowManager(flush_return=11)
        cb = TerminalCallback(
            reason=reason,  # type: ignore[arg-type]
            shutdown_handler=_StubHandler(requested=(reason == "cancel")),
            mlflow_manager=manager,
            event_publisher=_pub,
            marker_writer=lambda payload: None,
        )
        if reason == "cancel":
            cb.on_step_end(args=None, state=_state(1), control=_control())
        cb.on_train_end(args=None, state=_state(2), control=_control())
        assert len(published) == 1
        kind, payload = published[0]
        # Don't pin the exact kind constant -- just shape.
        assert "finalized" in kind
        assert payload["flushed_count"] == 11
        assert payload["flush_timed_out"] is False
