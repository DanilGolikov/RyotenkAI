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


# Pre-load ``src.training._concurrent_helpers`` under its real module
# name (and stub ``src.training`` parent) so the lazy import inside
# ``CancellationCallback.on_train_end`` resolves without triggering
# ``src.training/__init__.py`` (which pulls the orchestrator → datasets
# chain and fails in the slim dev venv).
_HELPER_PATH = (
    _pathlib.Path(__file__).resolve().parents[4]
    / "training" / "_concurrent_helpers.py"
)
_helper_spec = _importlib_util.spec_from_file_location(
    "src.training._concurrent_helpers", _HELPER_PATH,
)
assert _helper_spec is not None and _helper_spec.loader is not None
_helper_module = _importlib_util.module_from_spec(_helper_spec)
# Stub the parent package shell so ``from src.training._concurrent_helpers
# import ...`` finds both the parent and the leaf in sys.modules
# without running ``src.training/__init__.py``.
#
# IMPORTANT: set ``__path__`` so Python treats the stub as a package
# (not a plain module). Without ``__path__`` other tests in the same
# pytest collection — e.g. ``test_system_metrics_callback.py`` which
# does ``from src.training.callbacks.system_metrics_callback import X``
# — fail because Python rejects sub-attribute lookup on a non-package
# module shell. Pointing ``__path__`` at the real source dir lets the
# regular import machinery still find the modules on disk.
if "src.training" not in _sys.modules:
    _training_shell = _types.ModuleType("src.training")
    _training_shell.__path__ = [  # type: ignore[attr-defined]
        str(_pathlib.Path(__file__).resolve().parents[4] / "training"),
    ]
    _sys.modules["src.training"] = _training_shell
_sys.modules["src.training._concurrent_helpers"] = _helper_module
_helper_spec.loader.exec_module(_helper_module)


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


class _StubMlflowManager:
    """Minimal stand-in for :class:`MlflowManager` used by 9.B tests."""

    def __init__(
        self, *,
        flush_return: int = 0,
        flush_raises: Exception | None = None,
        flush_blocks_for: float = 0.0,
    ) -> None:
        self.flush_calls: int = 0
        self._flush_return = flush_return
        self._flush_raises = flush_raises
        self._flush_blocks_for = flush_blocks_for

    def flush_buffer(self) -> int:
        self.flush_calls += 1
        if self._flush_blocks_for > 0:
            import time
            time.sleep(self._flush_blocks_for)
        if self._flush_raises is not None:
            raise self._flush_raises
        return self._flush_return


class TestNegative:
    def test_flag_unset_leaves_control_untouched(self) -> None:
        cb = CancellationCallback(shutdown_handler=_StubHandler(requested=False))
        ctrl = _control()
        cb.on_step_end(args=None, state=_state(), control=ctrl)
        assert ctrl.should_save is False
        assert ctrl.should_training_stop is False

    def test_on_train_end_silent_on_clean_exit(self) -> None:
        """Phase 9.B regression: on_train_end is a no-op when
        cancellation was NEVER signalled during the run.

        This is the happy path — training completed naturally,
        ``self._signalled`` stays False, so we don't bother flushing
        the buffer (steady-state drain handled it on the live path).
        """
        manager = _StubMlflowManager(flush_return=99)
        cb = CancellationCallback(
            shutdown_handler=_StubHandler(requested=False),
            mlflow_manager=manager,
        )
        # Don't call on_step_end — _signalled stays False.
        cb.on_train_end(args=None, state=_state(), control=_control())
        # Manager never touched.
        assert manager.flush_calls == 0


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
    def test_phase_9b_flush_drains_buffer_after_cancellation(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Phase 9.B happy path: on_train_end after a signalled
        cancellation calls ``mlflow_manager.flush_buffer()`` exactly
        once."""
        import logging
        caplog.set_level(logging.INFO, logger="src.training.callbacks.cancellation_callback")

        manager = _StubMlflowManager(flush_return=42)
        cb = CancellationCallback(
            shutdown_handler=_StubHandler(requested=True),
            mlflow_manager=manager,
        )
        # Step 1 raises the flag.
        cb.on_step_end(args=None, state=_state(1), control=_control())
        # Now train ends — flush should run.
        cb.on_train_end(args=None, state=_state(2), control=_control())

        assert manager.flush_calls == 1
        # Operator-visible log mentions the drained count.
        flush_logs = [
            r for r in caplog.records
            if "flushed 42 buffered MLflow records" in r.message
        ]
        assert len(flush_logs) == 1

    def test_phase_9b_no_manager_skips_flush_silently(self) -> None:
        """No MLflow manager configured (e.g. tracking disabled) →
        on_train_end is a silent no-op even after cancellation.
        Defensive — env-gate in factory.py prevents this in
        production but the callback handles it gracefully."""
        cb = CancellationCallback(
            shutdown_handler=_StubHandler(requested=True),
            mlflow_manager=None,
        )
        cb.on_step_end(args=None, state=_state(1), control=_control())
        # Resolver returns None because we didn't inject anything and
        # the lazy import path returns None when MLflowManagerHolder
        # isn't reachable.
        cb._mlflow_manager = None  # explicit override against lazy resolve
        cb.on_train_end(args=None, state=_state(2), control=_control())
        # No raise, no crash.

    def test_phase_9b_flush_timeout_logs_warning_and_continues(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Slow flush exceeds 5s budget → :class:`TimeoutExceededError`
        caught, warning logged, flow continues to HF's end_run.

        Tight 0.05s budget so the test takes <1s to fail-out the
        sleeping flush. The actual flush thread keeps running in the
        background; we just stopped waiting.
        """
        import logging
        caplog.set_level(logging.WARNING, logger="src.training.callbacks.cancellation_callback")

        # Flush sleeps longer than the budget — guaranteed timeout.
        manager = _StubMlflowManager(flush_blocks_for=0.5)
        cb = CancellationCallback(
            shutdown_handler=_StubHandler(requested=True),
            mlflow_manager=manager,
            flush_timeout_seconds=0.05,
        )
        cb.on_step_end(args=None, state=_state(1), control=_control())
        # Must not raise.
        cb.on_train_end(args=None, state=_state(2), control=_control())

        timeout_logs = [
            r for r in caplog.records
            if "MLflow flush exceeded" in r.message
        ]
        assert len(timeout_logs) == 1, (
            f"expected exactly one timeout warning, got "
            f"{len(timeout_logs)}: {[r.message for r in timeout_logs]}"
        )

    def test_phase_9b_flush_exception_logs_and_continues(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        """``flush_buffer`` raising unexpectedly → caught + logged,
        flow continues. Prevents a programmer bug in the manager
        surface from crashing the trainer's exit path."""
        import logging
        caplog.set_level(logging.WARNING, logger="src.training.callbacks.cancellation_callback")

        manager = _StubMlflowManager(
            flush_raises=ValueError("manager bug"),
        )
        cb = CancellationCallback(
            shutdown_handler=_StubHandler(requested=True),
            mlflow_manager=manager,
        )
        cb.on_step_end(args=None, state=_state(1), control=_control())
        # Must not raise — best-effort by contract.
        cb.on_train_end(args=None, state=_state(2), control=_control())

        unexpected_logs = [
            r for r in caplog.records
            if "flush_buffer raised unexpectedly" in r.message
        ]
        assert len(unexpected_logs) == 1

    def test_phase_9b_callback_does_not_call_end_run(self) -> None:
        """Critical contract: our callback NEVER calls ``end_run``.
        That belongs to HF Trainer's own MLflow callback. Pin via
        spy that the manager's ``set_run_terminated`` (which would
        be the mistake-magnet) is not touched on the on_train_end
        path."""

        class _SpyManager(_StubMlflowManager):
            def __init__(self) -> None:
                super().__init__(flush_return=0)
                self.set_terminated_calls: int = 0

            def set_run_terminated(self, run_id: str, status: str = "KILLED") -> bool:
                self.set_terminated_calls += 1
                return True

        manager = _SpyManager()
        cb = CancellationCallback(
            shutdown_handler=_StubHandler(requested=True),
            mlflow_manager=manager,
        )
        cb.on_step_end(args=None, state=_state(1), control=_control())
        cb.on_train_end(args=None, state=_state(2), control=_control())

        # Flush ran, but set_run_terminated did NOT.
        assert manager.flush_calls == 1
        assert manager.set_terminated_calls == 0

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


# ---------------------------------------------------------------------------
# Phase 9.C — cancellation_finalized telemetry + cancelled.marker
# ---------------------------------------------------------------------------


class _PublisherSpy:
    """Records ``(kind, payload)`` invocations.

    Mirrors the runtime contract: the callback hands us a ``(str, dict)``
    tuple via the publisher callable and we just append it. Tests
    assert on the kind strings + payload schema.
    """

    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, object]]] = []

    def __call__(self, kind: str, payload: dict[str, object]) -> None:
        self.events.append((kind, dict(payload)))


class TestPhase9CFinalizedEvent:
    """``on_train_end`` MUST emit ``cancellation_finalized`` after the
    flush attempt — both on success and on timeout. Operator
    dashboards key off this event for "trainer finished its end-of-run
    bookkeeping" — without it Mac side can't tell whether the trainer
    crashed mid-finalize or just hadn't reached on_train_end yet.
    """

    def test_emits_finalized_event_on_successful_flush(self) -> None:
        from src.runner.cancellation_telemetry import CANCELLATION_FINALIZED

        publisher = _PublisherSpy()
        manager = _StubMlflowManager(flush_return=7)
        cb = CancellationCallback(
            shutdown_handler=_StubHandler(requested=True),
            mlflow_manager=manager,
            event_publisher=publisher,
        )
        cb.on_step_end(args=None, state=_state(1), control=_control())
        cb.on_train_end(args=None, state=_state(2), control=_control())

        assert len(publisher.events) == 1
        kind, payload = publisher.events[0]
        assert kind == CANCELLATION_FINALIZED
        assert payload["flushed_count"] == 7
        assert payload["flush_timed_out"] is False
        assert payload["marker_written"] is False
        assert "flush_budget_seconds" in payload

    def test_emits_finalized_event_on_timeout_with_marker_flag(
        self, tmp_path: Any,
    ) -> None:
        from src.runner.cancellation_telemetry import CANCELLATION_FINALIZED

        publisher = _PublisherSpy()
        # Slow flush forces timeout.
        manager = _StubMlflowManager(flush_blocks_for=0.5)
        # Inject explicit workspace dir → marker writer goes to the
        # default ``atomic_write_text`` path against this tmp_path.
        cb = CancellationCallback(
            shutdown_handler=_StubHandler(requested=True),
            mlflow_manager=manager,
            event_publisher=publisher,
            flush_timeout_seconds=0.05,
            workspace_dir=tmp_path,
        )
        cb.on_step_end(args=None, state=_state(1), control=_control())
        cb.on_train_end(args=None, state=_state(2), control=_control())

        # Exactly one finalized event, with timed_out=True and
        # marker_written=True.
        assert len(publisher.events) == 1
        kind, payload = publisher.events[0]
        assert kind == CANCELLATION_FINALIZED
        assert payload["flush_timed_out"] is True
        assert payload["marker_written"] is True
        assert payload["flushed_count"] == 0

    def test_no_event_when_clean_exit_without_signal(self) -> None:
        # Training ended naturally — _signalled stays False → no flush
        # → no finalized event. Keep dashboards focused on real
        # cancellation chains.
        publisher = _PublisherSpy()
        cb = CancellationCallback(
            shutdown_handler=_StubHandler(requested=False),
            mlflow_manager=_StubMlflowManager(),
            event_publisher=publisher,
        )
        cb.on_train_end(args=None, state=_state(2), control=_control())
        assert publisher.events == []

    def test_publisher_exception_does_not_propagate(self) -> None:
        # Best-effort contract: a publisher that explodes must not
        # crash the trainer's exit path. Use a publisher that raises
        # on every call.
        def _bad_publisher(kind: str, payload: dict) -> None:
            raise RuntimeError("event bus offline")

        manager = _StubMlflowManager(flush_return=3)
        cb = CancellationCallback(
            shutdown_handler=_StubHandler(requested=True),
            mlflow_manager=manager,
            event_publisher=_bad_publisher,
        )
        cb.on_step_end(args=None, state=_state(1), control=_control())
        # MUST NOT raise.
        cb.on_train_end(args=None, state=_state(2), control=_control())
        # Flush still happened (publisher fires AFTER flush).
        assert manager.flush_calls == 1

    def test_no_publisher_means_no_telemetry_no_crash(self) -> None:
        # Local-mode runs (no runner attached) pass publisher=None.
        # Callback must still work — flush + log only, no event.
        manager = _StubMlflowManager(flush_return=2)
        cb = CancellationCallback(
            shutdown_handler=_StubHandler(requested=True),
            mlflow_manager=manager,
            event_publisher=None,
        )
        cb.on_step_end(args=None, state=_state(1), control=_control())
        cb.on_train_end(args=None, state=_state(2), control=_control())
        assert manager.flush_calls == 1


class TestPhase9CMarkerWrite:
    """When the flush budget is exceeded, the callback writes
    ``cancelled.marker`` to the workspace so the Mac-side
    TrainingMonitor can run reconciliation against the upstream MLflow
    run on its next poll."""

    def test_marker_file_written_to_workspace_on_timeout(
        self, tmp_path: Any,
    ) -> None:
        manager = _StubMlflowManager(flush_blocks_for=0.5)

        class _ManagerWithRunId(_StubMlflowManager):
            run_id = "abc123"

        manager = _ManagerWithRunId(flush_blocks_for=0.5)

        cb = CancellationCallback(
            shutdown_handler=_StubHandler(requested=True),
            mlflow_manager=manager,
            flush_timeout_seconds=0.05,
            workspace_dir=tmp_path,
        )
        cb.on_step_end(args=None, state=_state(1), control=_control())
        cb.on_train_end(args=None, state=_state(2), control=_control())

        marker = tmp_path / "cancelled.marker"
        assert marker.exists()
        import json as _json
        body = _json.loads(marker.read_text())
        assert body["run_id"] == "abc123"
        assert body["reason"] == "flush_budget_exceeded"
        assert isinstance(body["ts_ms"], int)

    def test_no_marker_on_successful_flush(self, tmp_path: Any) -> None:
        manager = _StubMlflowManager(flush_return=5)
        cb = CancellationCallback(
            shutdown_handler=_StubHandler(requested=True),
            mlflow_manager=manager,
            workspace_dir=tmp_path,
        )
        cb.on_step_end(args=None, state=_state(1), control=_control())
        cb.on_train_end(args=None, state=_state(2), control=_control())

        marker = tmp_path / "cancelled.marker"
        assert not marker.exists()

    def test_no_marker_on_clean_exit(self, tmp_path: Any) -> None:
        # Clean exit (no signal): on_train_end is no-op → no marker.
        cb = CancellationCallback(
            shutdown_handler=_StubHandler(requested=False),
            mlflow_manager=_StubMlflowManager(),
            workspace_dir=tmp_path,
        )
        cb.on_train_end(args=None, state=_state(2), control=_control())
        marker = tmp_path / "cancelled.marker"
        assert not marker.exists()

    def test_workspace_resolution_falls_back_to_env(
        self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # No ``workspace_dir`` injected → callback must fall back to
        # ``HELIX_WORKSPACE`` env var (the production wiring).
        monkeypatch.setenv("HELIX_WORKSPACE", str(tmp_path))

        manager = _StubMlflowManager(flush_blocks_for=0.5)
        cb = CancellationCallback(
            shutdown_handler=_StubHandler(requested=True),
            mlflow_manager=manager,
            flush_timeout_seconds=0.05,
            # No workspace_dir argument.
        )
        cb.on_step_end(args=None, state=_state(1), control=_control())
        cb.on_train_end(args=None, state=_state(2), control=_control())

        marker = tmp_path / "cancelled.marker"
        assert marker.exists()

    def test_no_workspace_resolved_skips_marker_silently(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Neither injected workspace nor HELIX_WORKSPACE env →
        # marker write skipped; finalized event still emitted with
        # marker_written=False.
        monkeypatch.delenv("HELIX_WORKSPACE", raising=False)
        publisher = _PublisherSpy()
        manager = _StubMlflowManager(flush_blocks_for=0.5)
        cb = CancellationCallback(
            shutdown_handler=_StubHandler(requested=True),
            mlflow_manager=manager,
            event_publisher=publisher,
            flush_timeout_seconds=0.05,
        )
        cb.on_step_end(args=None, state=_state(1), control=_control())
        cb.on_train_end(args=None, state=_state(2), control=_control())

        # Single finalized event, marker_written=False.
        assert len(publisher.events) == 1
        _, payload = publisher.events[0]
        assert payload["flush_timed_out"] is True
        assert payload["marker_written"] is False

    def test_injected_marker_writer_is_used_when_provided(self) -> None:
        # Custom marker writer takes precedence over default
        # atomic_write_text path. Pin DI seam.
        captured: dict[str, object] = {}

        def _writer(payload: dict[str, object]) -> str:
            captured["payload"] = payload
            return "/sentinel/cancelled.marker"

        manager = _StubMlflowManager(flush_blocks_for=0.5)
        cb = CancellationCallback(
            shutdown_handler=_StubHandler(requested=True),
            mlflow_manager=manager,
            flush_timeout_seconds=0.05,
            marker_writer=_writer,
        )
        cb.on_step_end(args=None, state=_state(1), control=_control())
        cb.on_train_end(args=None, state=_state(2), control=_control())

        # Custom writer was invoked with the expected payload shape.
        assert "payload" in captured
        assert "ts_ms" in captured["payload"]  # type: ignore[operator]
        assert "flushed_count" in captured["payload"]  # type: ignore[operator]
