"""Phase 11.A — :func:`run_flush_with_deadline` shared helper.

The helper extracts the deadline + branching logic that
:class:`CancellationCallback` (Phase 9.B) and
:class:`CompletionCallback` (Phase 11.A) both need. Behaviour pinned
here is the contract both callbacks rely on:

* Returns a :class:`FlushOutcome` tuple with three booleans
  (`drained_count`, `timed_out`, `raised`) — never raises.
* Success path: returns the integer ``flush_fn`` returned, no
  timeout, no raise.
* Timeout: ``timed_out=True``, ``drained_count=0`` (we never saw
  the return).
* Exception: ``raised=True``, ``drained_count=0``.
* Logger is best-effort — None is acceptable, callable is acceptable.

Slim-venv pattern: this module imports ``_flush_helper`` directly
without going through ``src.training/__init__.py``, so the test runs
in the dev venv without ``datasets`` / ``peft``.
"""

from __future__ import annotations

import importlib.util as _importlib_util
import logging
import pathlib as _pathlib
import sys as _sys
import time
import types as _types

import pytest


# ---------------------------------------------------------------------------
# Slim-venv loader — pre-populate src.training package shell + helpers
# ---------------------------------------------------------------------------

_TRAINING_PKG = _pathlib.Path(__file__).resolve().parents[4] / "training"

if "ryotenkai_pod.trainer" not in _sys.modules:
    _shell = _types.ModuleType("ryotenkai_pod.trainer")
    _shell.__path__ = [str(_TRAINING_PKG)]  # type: ignore[attr-defined]
    _sys.modules["ryotenkai_pod.trainer"] = _shell

# Pre-load _concurrent_helpers under its real module name so the lazy
# import inside _flush_helper's body resolves without dragging the
# package init.
_CONCURRENT = _TRAINING_PKG / "_concurrent_helpers.py"
_concurrent_spec = _importlib_util.spec_from_file_location(
    "ryotenkai_pod.trainer._concurrent_helpers", _CONCURRENT,
)
assert _concurrent_spec is not None and _concurrent_spec.loader is not None
_concurrent_mod = _importlib_util.module_from_spec(_concurrent_spec)
_sys.modules["ryotenkai_pod.trainer._concurrent_helpers"] = _concurrent_mod
_concurrent_spec.loader.exec_module(_concurrent_mod)


_HELPER_PATH = _TRAINING_PKG / "callbacks" / "_flush_helper.py"
_helper_spec = _importlib_util.spec_from_file_location(
    "_ryotenkai_flush_helper_under_test", _HELPER_PATH,
)
assert _helper_spec is not None and _helper_spec.loader is not None
_helper_mod = _importlib_util.module_from_spec(_helper_spec)
_sys.modules["_ryotenkai_flush_helper_under_test"] = _helper_mod
_helper_spec.loader.exec_module(_helper_mod)

run_flush_with_deadline = _helper_mod.run_flush_with_deadline
FlushOutcome = _helper_mod.FlushOutcome


# ---------------------------------------------------------------------------
# 1. Positive — happy path returns drained count, no flags set
# ---------------------------------------------------------------------------


class TestPositive:
    def test_success_returns_drained_count(self) -> None:
        outcome = run_flush_with_deadline(
            lambda: 42, timeout_seconds=1.0,
        )
        assert isinstance(outcome, FlushOutcome)
        assert outcome.drained_count == 42
        assert outcome.timed_out is False
        assert outcome.raised is False

    def test_zero_drained_is_valid(self) -> None:
        # Empty buffer → 0 records drained → still success.
        outcome = run_flush_with_deadline(
            lambda: 0, timeout_seconds=1.0,
        )
        assert outcome.drained_count == 0
        assert outcome.timed_out is False
        assert outcome.raised is False


# ---------------------------------------------------------------------------
# 2. Negative — timeout + exception paths
# ---------------------------------------------------------------------------


class TestNegative:
    def test_timeout_sets_timed_out_flag(self) -> None:
        # Sleep longer than the budget → timed_out=True.
        def slow() -> int:
            time.sleep(0.5)
            return 99

        outcome = run_flush_with_deadline(slow, timeout_seconds=0.05)
        assert outcome.timed_out is True
        assert outcome.raised is False
        # drained_count is 0 — we never saw the return value.
        assert outcome.drained_count == 0

    def test_exception_sets_raised_flag(self) -> None:
        def buggy() -> int:
            raise ValueError("manager bug")

        outcome = run_flush_with_deadline(buggy, timeout_seconds=1.0)
        assert outcome.raised is True
        assert outcome.timed_out is False
        assert outcome.drained_count == 0


# ---------------------------------------------------------------------------
# 3. Boundary — invariants on output type
# ---------------------------------------------------------------------------


class TestBoundary:
    def test_outcome_is_frozen_dataclass(self) -> None:
        outcome = run_flush_with_deadline(
            lambda: 1, timeout_seconds=1.0,
        )
        # FlushOutcome is frozen — mutation should fail.
        with pytest.raises((AttributeError, Exception)):  # FrozenInstanceError
            outcome.drained_count = 99  # type: ignore[misc]

    def test_drained_count_is_int_even_when_fn_returns_float(self) -> None:
        outcome = run_flush_with_deadline(
            lambda: 3.7, timeout_seconds=1.0,  # type: ignore[arg-type,return-value]
        )
        # int() coercion in helper for safety.
        assert outcome.drained_count == 3
        assert isinstance(outcome.drained_count, int)


# ---------------------------------------------------------------------------
# 4. Invariants — never raises, regardless of fn behavior
# ---------------------------------------------------------------------------


class TestInvariants:
    def test_helper_never_raises_on_function_exception(self) -> None:
        def buggy() -> int:
            raise RuntimeError("explosion")

        # MUST NOT raise.
        outcome = run_flush_with_deadline(buggy, timeout_seconds=1.0)
        assert outcome.raised is True

    def test_helper_never_raises_on_timeout(self) -> None:
        def slow() -> int:
            time.sleep(0.5)
            return 0

        # MUST NOT raise.
        outcome = run_flush_with_deadline(slow, timeout_seconds=0.05)
        assert outcome.timed_out is True

    def test_zero_timeout_swallowed_as_raised(self) -> None:
        # ``with_timeout`` rejects timeout_seconds <= 0 with ValueError.
        # By the helper's never-raise contract, that ValueError is
        # caught and surfaced as ``raised=True``. Programmer error
        # gets logged via the optional logger but does NOT crash the
        # trainer's exit path — same logic as a flush_buffer
        # exception.
        outcome = run_flush_with_deadline(lambda: 0, timeout_seconds=0.0)
        assert outcome.raised is True
        assert outcome.timed_out is False
        assert outcome.drained_count == 0


# ---------------------------------------------------------------------------
# 5. Dependency errors — logger is optional
# ---------------------------------------------------------------------------


class TestDependencyErrors:
    def test_no_logger_works(self) -> None:
        # logger=None → silent path, but outcome still correct.
        outcome = run_flush_with_deadline(
            lambda: 5, timeout_seconds=1.0, logger=None,
        )
        assert outcome.drained_count == 5

    def test_logger_called_on_timeout(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        log = logging.getLogger("test_flush_helper.timeout")
        caplog.set_level(logging.WARNING, logger=log.name)

        def slow() -> int:
            time.sleep(0.5)
            return 0

        run_flush_with_deadline(
            slow, timeout_seconds=0.05, logger=log, label="TEST",
        )

        timeout_logs = [
            r for r in caplog.records
            if r.name == log.name and "exceeded" in r.message
        ]
        assert len(timeout_logs) == 1
        # Label appears in message for grepability.
        assert "[TEST]" in timeout_logs[0].message

    def test_logger_called_on_exception(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        log = logging.getLogger("test_flush_helper.exc")
        caplog.set_level(logging.WARNING, logger=log.name)

        def buggy() -> int:
            raise ValueError("manager bug")

        run_flush_with_deadline(
            buggy, timeout_seconds=1.0, logger=log, label="TEST",
        )

        exc_logs = [
            r for r in caplog.records
            if r.name == log.name and "raised unexpectedly" in r.message
        ]
        assert len(exc_logs) == 1


# ---------------------------------------------------------------------------
# 6. Regressions — surface match
# ---------------------------------------------------------------------------


class TestRegressions:
    def test_outcome_has_three_fields(self) -> None:
        # Pin the dataclass fields — operator dashboards key off these
        # by name. Adding a field is OK; renaming or removing breaks
        # downstream consumers (telemetry events).
        outcome = run_flush_with_deadline(lambda: 1, timeout_seconds=1.0)
        assert hasattr(outcome, "drained_count")
        assert hasattr(outcome, "timed_out")
        assert hasattr(outcome, "raised")

    def test_default_label_is_flush(self) -> None:
        # Pin the default label — if a caller forgets to pass one,
        # logs still parse cleanly.
        log = logging.getLogger("test_flush_helper.default_label")
        with pytest.MonkeyPatch.context() as mp:
            captured: list[str] = []

            class _CapturingLogger:
                def warning(self, fmt: str, *args: object) -> None:
                    captured.append(fmt % args)

                def info(self, *args: object, **kwargs: object) -> None:
                    pass

            run_flush_with_deadline(
                lambda: (_ for _ in ()).throw(ValueError("x")),
                timeout_seconds=1.0,
                logger=_CapturingLogger(),  # type: ignore[arg-type]
            )
            assert any("[flush]" in m for m in captured), (
                f"expected default label '[flush]', got {captured!r}"
            )


# ---------------------------------------------------------------------------
# 7. Logic-specific — successful flush bypasses timeout flag
# ---------------------------------------------------------------------------


class TestLogicSpecific:
    def test_fast_function_does_not_set_timed_out(self) -> None:
        # Function returns immediately — no timer fires, timed_out
        # stays False even if the budget was tight.
        outcome = run_flush_with_deadline(
            lambda: 10, timeout_seconds=0.001,
        )
        # Race-y by definition (very tight budget); accept either
        # outcome shape but both flags should be consistent.
        if outcome.timed_out:
            assert outcome.drained_count == 0
        else:
            assert outcome.drained_count == 10
            assert outcome.raised is False
