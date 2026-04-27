"""Phase 9.A — :class:`CancellationCallback` contract.

The callback bridges :class:`ShutdownHandler`'s flag (set by SIGTERM)
to HF Trainer's ``TrainerControl(should_save, should_training_stop)``
on each ``on_step_end`` so the trainer cooperatively saves a
checkpoint at the next save boundary and exits the train loop —
instead of running until the supervisor's SIGKILL escalation.

7-category coverage (project policy):

1. **Positive**           — flag set → control flags raised.
2. **Negative**           — flag unset → control unchanged.
3. **Boundary**           — flag flips to true mid-run; idempotent
                             across subsequent steps.
4. **Invariants**         — order: ``should_save = True`` BEFORE
                             ``should_training_stop = True``;
                             callback never raises (handler errors
                             are logged, not propagated);
                             ``on_train_end`` is no-op in 9.A.
5. **Dependency errors**  — handler attribute missing / raises →
                             callback degrades gracefully.
6. **Regressions**        — global handler resolved lazily (no
                             import-time side effect on transformers).
7. **Logic-specific**     — exactly one log on transition (idempotent
                             after multiple subsequent step calls).
"""

from __future__ import annotations

# Slim-venv import pattern — same trick as
# ``test_runner_event_callback.py``. The callback module imports
# ``transformers.TrainerCallback`` and lives under ``src.training`` —
# ``src.training/__init__`` pulls in the orchestrator cascade which
# imports ``datasets`` (the HuggingFace one). The dev venv used for
# fast unit tests doesn't have it. We stub ``transformers`` and load
# the callback module by file path, bypassing ``src.training``'s
# ``__init__``.

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

_CALLBACK_PATH = (
    _pathlib.Path(__file__).resolve().parents[4]
    / "training" / "callbacks" / "cancellation_callback.py"
)
_spec = _importlib_util.spec_from_file_location(
    "_ryotenkai_cancellation_callback_under_test", _CALLBACK_PATH,
)
assert _spec is not None and _spec.loader is not None
_module = _importlib_util.module_from_spec(_spec)
_sys.modules["_ryotenkai_cancellation_callback_under_test"] = _module
_spec.loader.exec_module(_module)

CancellationCallback = _module.CancellationCallback


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _StubHandler:
    """In-memory stand-in for ShutdownHandler.

    Bare flag — no signal registration, no thread state. Lets us
    flip ``should_stop`` deterministically per test without
    touching the real signal machinery (which pytest disallows
    in worker threads).
    """

    def __init__(self, *, requested: bool = False) -> None:
        self._requested = requested

    def should_stop(self) -> bool:
        return self._requested

    def request(self) -> None:
        self._requested = True


def _control() -> Any:
    """Build a fake ``TrainerControl`` with the two fields the callback touches."""
    return SimpleNamespace(should_save=False, should_training_stop=False)


def _state(global_step: int = 0) -> Any:
    return SimpleNamespace(global_step=global_step)


# ---------------------------------------------------------------------------
# 1. Positive
# ---------------------------------------------------------------------------


class TestPositive:
    def test_flag_set_raises_both_controls(self) -> None:
        h = _StubHandler(requested=True)
        cb = CancellationCallback(shutdown_handler=h)
        ctrl = _control()
        cb.on_step_end(args=None, state=_state(global_step=42), control=ctrl)
        assert ctrl.should_save is True
        assert ctrl.should_training_stop is True

    def test_returns_none(self) -> None:
        # HF Trainer expects callbacks to return ``None`` (mutates
        # control in place). Pin the return shape.
        cb = CancellationCallback(shutdown_handler=_StubHandler(requested=True))
        result = cb.on_step_end(args=None, state=_state(), control=_control())
        assert result is None


# ---------------------------------------------------------------------------
# 2. Negative
# ---------------------------------------------------------------------------


class TestNegative:
    def test_flag_unset_leaves_control_untouched(self) -> None:
        cb = CancellationCallback(shutdown_handler=_StubHandler(requested=False))
        ctrl = _control()
        cb.on_step_end(args=None, state=_state(), control=ctrl)
        assert ctrl.should_save is False
        assert ctrl.should_training_stop is False

    def test_on_train_end_is_noop_in_9A(self) -> None:
        # Phase 9.A intentionally leaves on_train_end empty — MLflow
        # finalize wires in 9.B. Pin the no-op so an accidental future
        # mutation here surfaces as a test failure.
        cb = CancellationCallback(shutdown_handler=_StubHandler(requested=True))
        ctrl = _control()
        result = cb.on_train_end(args=None, state=_state(), control=ctrl)
        assert result is None
        # No mutation through on_train_end.
        assert ctrl.should_save is False
        assert ctrl.should_training_stop is False


# ---------------------------------------------------------------------------
# 3. Boundary
# ---------------------------------------------------------------------------


class TestBoundary:
    def test_flag_flips_mid_run(self) -> None:
        h = _StubHandler(requested=False)
        cb = CancellationCallback(shutdown_handler=h)

        # Steps 1-3: flag clean, no mutation.
        for step in (1, 2, 3):
            ctrl = _control()
            cb.on_step_end(args=None, state=_state(step), control=ctrl)
            assert ctrl.should_save is False
            assert ctrl.should_training_stop is False

        # Step 4: SIGTERM observed.
        h.request()
        ctrl = _control()
        cb.on_step_end(args=None, state=_state(4), control=ctrl)
        assert ctrl.should_save is True
        assert ctrl.should_training_stop is True

    def test_idempotent_across_subsequent_steps(self) -> None:
        # Once the flag is set, every following step keeps the controls
        # raised (without re-logging — see TestLogicSpecific).
        h = _StubHandler(requested=True)
        cb = CancellationCallback(shutdown_handler=h)
        for step in (10, 11, 12):
            ctrl = _control()
            cb.on_step_end(args=None, state=_state(step), control=ctrl)
            assert ctrl.should_save is True
            assert ctrl.should_training_stop is True


# ---------------------------------------------------------------------------
# 4. Invariants
# ---------------------------------------------------------------------------


class TestInvariants:
    def test_save_order_is_save_then_stop(self) -> None:
        """``should_save = True`` must be set BEFORE ``should_training_stop``.

        HF Trainer reads ``should_training_stop`` first on the next
        loop iteration; if we set it without ``should_save`` HF would
        exit without saving. Pin the order via ordered attribute
        observation.
        """
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

        cb = CancellationCallback(shutdown_handler=_StubHandler(requested=True))
        cb.on_step_end(args=None, state=_state(), control=_RecorderControl())
        assert observed == ["save", "stop"]

    def test_callback_never_raises_on_handler_error(self) -> None:
        class _BrokenHandler:
            def should_stop(self) -> bool:
                raise RuntimeError("handler internals exploded")

        cb = CancellationCallback(shutdown_handler=_BrokenHandler())
        ctrl = _control()
        # Must not raise — orchestrator-level shutdown is the
        # safety-net.
        cb.on_step_end(args=None, state=_state(), control=ctrl)
        # Control left untouched on error — defensive default.
        assert ctrl.should_save is False
        assert ctrl.should_training_stop is False


# ---------------------------------------------------------------------------
# 5. Dependency errors
# ---------------------------------------------------------------------------


class TestDependencyErrors:
    def test_handler_missing_attribute(self) -> None:
        # If something injects a handler shape that doesn't have
        # ``should_stop`` (e.g. mock typo) — still don't crash.
        class _NoMethod:
            pass

        cb = CancellationCallback(shutdown_handler=_NoMethod())
        ctrl = _control()
        cb.on_step_end(args=None, state=_state(), control=ctrl)
        assert ctrl.should_save is False  # graceful degrade


# ---------------------------------------------------------------------------
# 6. Regressions
# ---------------------------------------------------------------------------


class TestRegressions:
    def test_handler_not_resolved_when_injected(self) -> None:
        """When tests inject a handler, ``_resolve_handler`` returns it
        directly without falling back to the global lazy import.

        This is the unit-level guarantee behind the production lazy
        contract: ``get_shutdown_handler`` (which has heavier deps via
        ``src.training.orchestrator``) is touched ONLY when no handler
        was passed to the constructor. Tests inject — production
        defaults to None and the lazy path runs once on first poll.
        """
        h = _StubHandler(requested=False)
        cb = CancellationCallback(shutdown_handler=h)
        # Calling the resolver returns our stub, never the global one.
        assert cb._resolve_handler() is h
        # And ``_handler`` stays exactly the injected stub — no swap.
        assert cb._handler is h

    def test_construction_with_explicit_handler_skips_lazy_resolve(self) -> None:
        # When tests inject a handler, we must NEVER reach into the
        # global one. This guards against accidental dual resolution.
        h = _StubHandler(requested=False)
        cb = CancellationCallback(shutdown_handler=h)
        # ``_handler`` is set to the injected stub, not None.
        assert cb._handler is h


# ---------------------------------------------------------------------------
# 7. Logic-specific
# ---------------------------------------------------------------------------


class TestLogicSpecific:
    def test_logs_once_on_transition(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Idempotent logging across subsequent steps once flag is set.

        The operator should see exactly one ``[CANCELLATION]`` line
        per cancellation event in the trainer log, not one per step
        after the SIGTERM. Pin via log capture: 5 step calls after
        flag raised → exactly 1 INFO message.
        """
        import logging

        caplog.set_level(logging.INFO, logger="src.training.callbacks.cancellation_callback")

        h = _StubHandler(requested=True)
        cb = CancellationCallback(shutdown_handler=h)

        for step in range(5):
            cb.on_step_end(args=None, state=_state(step), control=_control())

        cancel_logs = [
            rec for rec in caplog.records
            if "[CANCELLATION] shutdown flag observed" in rec.message
        ]
        assert len(cancel_logs) == 1, (
            f"expected exactly 1 transition log, got {len(cancel_logs)}: "
            f"{[r.message for r in cancel_logs]}"
        )
