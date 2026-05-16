"""Tests for :mod:`ryotenkai_pod.trainer.exit_reporter` (Phase D).

The reporter is the trainer-side half of the trainer-exit.json
protocol: it consumes a captured ``Exception`` and writes a structured
:class:`TrainerExitPayload` atomically to ``<workdir>/trainer-exit.json``.
Supervisor-side reading is exercised separately in
:mod:`tests.unit.pod.runner.test_supervisor_exit_payload`.

Seven test classes (mirrors the contract's coverage):

* construction (``build_failure_payload``) — RyotenkAIError mapping
* construction — generic Exception mapping
* construction — wall-clock seconds
* write_failure_payload — happy path
* write_failure_payload — missing workdir / disk errors
* sanitisation pin (no raw paths leak through)
* integration regression — full pipeline
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from ryotenkai_pod.trainer.exit_reporter import (
    build_failure_payload,
    write_failure_payload,
)
from ryotenkai_shared.contracts.problem_details import ErrorCode
from ryotenkai_shared.contracts.trainer_exit import (
    TRAINER_EXIT_FILENAME,
    TrainerExitPayload,
)
from ryotenkai_shared.errors import (
    InternalError,
    TrainingFailedError,
    TrainingOOMError,
)


def _raise_and_capture(exc: BaseException) -> BaseException:
    """Raise ``exc`` so it has a proper ``__traceback__`` attached.

    ``build_failure_payload`` reads the traceback off the exception's
    ``__traceback__`` (via :func:`traceback.format_exception`); a
    freshly-constructed exception has none and the resulting payload
    has ``traceback_summary=None`` (or empty). Tests that exercise
    the traceback path must therefore raise+catch.
    """
    try:
        raise exc
    except BaseException as caught:  # noqa: BLE001
        return caught


# ---------------------------------------------------------------------------
# Class 1: build_failure_payload — RyotenkAIError mapping
# ---------------------------------------------------------------------------


class TestBuildRyotenkAIError:
    """Typed exception → pinned ``code``, detail / title fallback."""

    def test_training_oom_maps_to_training_oom_code(self) -> None:
        exc = _raise_and_capture(TrainingOOMError(detail="step 5 OOM"))
        p = build_failure_payload(exc, started_at=100.0, exit_code=1, now=101.5)
        assert p.code == ErrorCode.TRAINING_OOM
        assert p.message == "step 5 OOM"

    def test_training_failed_maps_to_training_failed_code(self) -> None:
        exc = _raise_and_capture(TrainingFailedError(detail="non-finite loss"))
        p = build_failure_payload(exc, started_at=0.0, exit_code=1, now=1.0)
        assert p.code == ErrorCode.TRAINING_FAILED

    def test_falls_back_to_title_when_detail_missing(self) -> None:
        exc = _raise_and_capture(TrainingFailedError())
        p = build_failure_payload(exc, started_at=0.0, exit_code=1, now=0.0)
        # Title is configured for TRAINING_FAILED in _DEFAULT_TITLES.
        assert p.message  # non-empty
        assert isinstance(p.message, str)

    def test_includes_traceback_when_raised_through_try_except(self) -> None:
        exc = _raise_and_capture(TrainingFailedError(detail="x"))
        p = build_failure_payload(exc, started_at=0.0, exit_code=1, now=0.0)
        assert p.traceback_summary is not None
        # The frame name comes from this test module.
        assert "_raise_and_capture" in p.traceback_summary

    def test_internal_error_explicit_carries_internal_code(self) -> None:
        exc = _raise_and_capture(InternalError(detail="generic boom"))
        p = build_failure_payload(exc, started_at=0.0, exit_code=1, now=0.0)
        assert p.code == ErrorCode.INTERNAL_ERROR


# ---------------------------------------------------------------------------
# Class 2: build_failure_payload — generic Exception mapping
# ---------------------------------------------------------------------------


class TestBuildGenericException:
    """Anything not derived from ``RyotenkAIError`` → INTERNAL_ERROR."""

    def test_runtime_error_maps_to_internal_error(self) -> None:
        exc = _raise_and_capture(RuntimeError("untyped boom"))
        p = build_failure_payload(exc, started_at=0.0, exit_code=1, now=0.0)
        assert p.code == ErrorCode.INTERNAL_ERROR
        assert p.message == "untyped boom"

    def test_value_error_message_preserved(self) -> None:
        exc = _raise_and_capture(ValueError("bad config"))
        p = build_failure_payload(exc, started_at=0.0, exit_code=2, now=0.0)
        assert p.message == "bad config"

    def test_empty_exception_message_uses_class_name(self) -> None:
        exc = _raise_and_capture(RuntimeError())
        p = build_failure_payload(exc, started_at=0.0, exit_code=1, now=0.0)
        assert p.message == "RuntimeError"

    def test_keyboard_interrupt_maps_to_internal_error(self) -> None:
        """KeyboardInterrupt isn't a RyotenkAIError; still produces a
        valid payload (won't crash the reporter)."""
        exc = _raise_and_capture(KeyboardInterrupt())
        p = build_failure_payload(exc, started_at=0.0, exit_code=130, now=0.0)
        assert p.code == ErrorCode.INTERNAL_ERROR


# ---------------------------------------------------------------------------
# Class 3: build_failure_payload — wall_seconds + edge cases
# ---------------------------------------------------------------------------


class TestBuildWallSeconds:
    """``wall_seconds`` computation invariants."""

    def test_wall_seconds_is_now_minus_started(self) -> None:
        exc = _raise_and_capture(RuntimeError("x"))
        p = build_failure_payload(exc, started_at=10.0, exit_code=1, now=15.5)
        assert p.wall_seconds == pytest.approx(5.5)

    def test_wall_seconds_floored_at_zero(self) -> None:
        """Pathological clock skew (now < started) → 0, not negative."""
        exc = _raise_and_capture(RuntimeError("x"))
        p = build_failure_payload(exc, started_at=100.0, exit_code=1, now=50.0)
        assert p.wall_seconds == 0.0

    def test_default_now_uses_time_monotonic(self) -> None:
        exc = _raise_and_capture(RuntimeError("x"))
        # Won't be exact, but should be a non-negative number close to 0
        # because ``started_at`` is set just before.
        started = time.monotonic()
        p = build_failure_payload(exc, started_at=started, exit_code=1)
        assert p.wall_seconds >= 0.0
        # Loose upper bound — sane test machine completes the call
        # well under a second.
        assert p.wall_seconds < 10.0


# ---------------------------------------------------------------------------
# Class 4: write_failure_payload — happy path
# ---------------------------------------------------------------------------


class TestWriteFailurePayload:
    """``write_failure_payload`` writes the file and returns its path."""

    def test_writes_file_to_workdir(self, tmp_path: Path) -> None:
        exc = _raise_and_capture(TrainingFailedError(detail="x"))
        path = write_failure_payload(
            tmp_path, exc, started_at=0.0, exit_code=1,
        )
        assert path == tmp_path / TRAINER_EXIT_FILENAME
        assert path.exists()
        restored = TrainerExitPayload.read_from(path)
        assert restored is not None
        assert restored.code == ErrorCode.TRAINING_FAILED

    def test_writes_full_payload_round_trip(self, tmp_path: Path) -> None:
        exc = _raise_and_capture(TrainingOOMError(detail="OOM at 90%"))
        path = write_failure_payload(
            tmp_path, exc, started_at=0.0, exit_code=137,
        )
        assert path is not None
        restored = TrainerExitPayload.read_from(path)
        assert restored is not None
        assert restored.code == ErrorCode.TRAINING_OOM
        assert restored.message == "OOM at 90%"
        assert restored.exit_code == 137


# ---------------------------------------------------------------------------
# Class 5: write_failure_payload — error tolerance
# ---------------------------------------------------------------------------


class TestWriteFailurePayloadResilience:
    """Reporter never crashes the trainer's exit path."""

    def test_returns_none_when_workdir_is_none(self) -> None:
        exc = _raise_and_capture(RuntimeError("x"))
        assert write_failure_payload(
            None, exc, started_at=0.0, exit_code=1,
        ) is None

    def test_returns_none_when_workdir_unwritable(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A disk-full / permission error returns ``None`` rather than
        propagating — atexit must not raise harder than the original
        exception."""
        exc = _raise_and_capture(RuntimeError("x"))

        def _boom(*args, **kwargs):
            raise OSError("disk full")

        monkeypatch.setattr(
            "ryotenkai_shared.contracts.trainer_exit.TrainerExitPayload.write_to",
            _boom,
        )
        result = write_failure_payload(
            tmp_path, exc, started_at=0.0, exit_code=1,
        )
        assert result is None


# ---------------------------------------------------------------------------
# Class 6: Sanitisation pin — paths must not leak through the reporter
# ---------------------------------------------------------------------------


class TestSanitisationPin:
    """Pin that the reporter routes its traceback through the
    sanitiser. Defends against an agent regressing the chain by
    swapping ``traceback.format_exc()`` raw into the payload."""

    def test_payload_traceback_has_no_users_path(self, tmp_path: Path) -> None:
        # Build an exception via a function that lives in a path we
        # can grep for. The path of THIS test file IS under
        # ``/Users/...`` on the CI runner, so the un-sanitised
        # traceback would contain that path.
        exc = _raise_and_capture(TrainingFailedError(detail="leak check"))
        path = write_failure_payload(
            tmp_path, exc, started_at=0.0, exit_code=1,
        )
        assert path is not None
        restored = TrainerExitPayload.read_from(path)
        assert restored is not None
        assert restored.traceback_summary is not None
        # The sanitiser collapses ``/Users/<user>/`` → ``<home>/``. So
        # neither the literal ``/Users/`` nor a tester's username
        # should leak through.
        assert "/Users/" not in restored.traceback_summary

    def test_payload_traceback_capped(self, tmp_path: Path) -> None:
        """Confirm the cap is plumbed end-to-end. We need a long
        traceback — build one by chaining many calls."""

        def _recurse(n: int) -> None:
            if n <= 0:
                raise TrainingFailedError(detail="deep")
            _recurse(n - 1)

        try:
            _recurse(60)
        except TrainingFailedError as exc:
            path = write_failure_payload(
                tmp_path, exc, started_at=0.0, exit_code=1,
            )
        assert path is not None
        restored = TrainerExitPayload.read_from(path)
        assert restored is not None
        assert restored.traceback_summary is not None
        # 30-line cap + 1 truncation marker (when truncation happens).
        lines = restored.traceback_summary.splitlines()
        assert len(lines) <= 31


# ---------------------------------------------------------------------------
# Class 7: Integration regression — full pipeline trainer→file→reader
# ---------------------------------------------------------------------------


class TestIntegration:
    """End-to-end: trainer reports → supervisor-equivalent read."""

    def test_round_trip_via_disk_typed_error(self, tmp_path: Path) -> None:
        exc = _raise_and_capture(TrainingOOMError(detail="40 GB VRAM exhausted"))
        write_failure_payload(tmp_path, exc, started_at=0.0, exit_code=1)
        # Supervisor-equivalent read.
        restored = TrainerExitPayload.read_from(tmp_path / TRAINER_EXIT_FILENAME)
        assert restored is not None
        assert restored.code == ErrorCode.TRAINING_OOM
        assert restored.message == "40 GB VRAM exhausted"

    def test_round_trip_via_disk_generic_error(self, tmp_path: Path) -> None:
        exc = _raise_and_capture(RuntimeError("generic"))
        write_failure_payload(tmp_path, exc, started_at=0.0, exit_code=1)
        restored = TrainerExitPayload.read_from(tmp_path / TRAINER_EXIT_FILENAME)
        assert restored is not None
        assert restored.code == ErrorCode.INTERNAL_ERROR
        assert restored.message == "generic"

    def test_no_file_on_success_path(self, tmp_path: Path) -> None:
        """The trainer never calls write_failure_payload on rc=0 — the
        function is only invoked from the except branch. The
        supervisor's reader returns ``None`` when the file is absent.
        Pin the absence so a future agent doesn't add a success-side
        write that would change the supervisor's interpretation."""
        # Don't call write_failure_payload at all — the supervisor
        # side must be happy with a missing file.
        restored = TrainerExitPayload.read_from(tmp_path / TRAINER_EXIT_FILENAME)
        assert restored is None
