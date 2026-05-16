"""Tests for :mod:`ryotenkai_shared.contracts.trainer_exit` (Phase D).

Seven test classes — covering the Pydantic model, the on-disk
atomic write, the regex-based traceback sanitiser, the JSON
round-trip, and behaviour invariants enumerated in the plan.

This module also exercises the security boundary
(``sanitize_traceback`` strips paths) so the
``test_no_traceback_in_context`` sentinel can rely on it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from ryotenkai_shared.contracts.problem_details import ErrorCode
from ryotenkai_shared.contracts.trainer_exit import (
    DEFAULT_TRACEBACK_MAX_LINES,
    TRAINER_EXIT_FILENAME,
    TRAINER_EXIT_SCHEMA_VERSION,
    TrainerExitPayload,
    sanitize_traceback,
)


# ---------------------------------------------------------------------------
# Class 1: Model construction + field validation
# ---------------------------------------------------------------------------


class TestTrainerExitPayloadModel:
    """Pydantic field-level invariants on :class:`TrainerExitPayload`."""

    def test_constructs_with_required_fields(self) -> None:
        p = TrainerExitPayload(
            code=ErrorCode.TRAINING_OOM,
            message="Out of memory",
            exit_code=137,
            wall_seconds=42.5,
        )
        assert p.code == ErrorCode.TRAINING_OOM
        assert p.message == "Out of memory"
        assert p.exit_code == 137
        assert p.wall_seconds == 42.5
        assert p.schema_version == TRAINER_EXIT_SCHEMA_VERSION
        assert p.traceback_summary is None

    def test_extra_fields_rejected(self) -> None:
        """``extra="forbid"`` — unknown keys raise immediately so a
        newer payload sent to an older runner fails closed."""
        with pytest.raises(ValidationError):
            TrainerExitPayload.model_validate({
                "code": ErrorCode.TRAINING_FAILED.value,
                "message": "x",
                "exit_code": 1,
                "wall_seconds": 0.5,
                "future_field": "boom",
            })

    def test_negative_wall_seconds_rejected(self) -> None:
        with pytest.raises(ValidationError):
            TrainerExitPayload(
                code=ErrorCode.TRAINING_FAILED,
                message="x",
                exit_code=1,
                wall_seconds=-1.0,
            )

    def test_schema_version_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            TrainerExitPayload(
                code=ErrorCode.TRAINING_FAILED,
                message="x",
                exit_code=1,
                wall_seconds=0.0,
                schema_version=0,
            )

    def test_invalid_error_code_rejected(self) -> None:
        with pytest.raises(ValidationError):
            TrainerExitPayload.model_validate({
                "code": "NOT_A_REAL_CODE",
                "message": "x",
                "exit_code": 1,
                "wall_seconds": 0.0,
            })


# ---------------------------------------------------------------------------
# Class 2: JSON round-trip
# ---------------------------------------------------------------------------


class TestRoundTrip:
    """The payload survives JSON round-trip without field drift."""

    def test_json_round_trip_preserves_fields(self) -> None:
        original = TrainerExitPayload(
            code=ErrorCode.TRAINING_OOM,
            message="cgroup OOM",
            traceback_summary="frame 1\nframe 2",
            exit_code=137,
            wall_seconds=12.345,
        )
        raw = original.model_dump_json()
        restored = TrainerExitPayload.model_validate_json(raw)
        assert restored == original

    def test_null_traceback_serialises_as_null(self) -> None:
        p = TrainerExitPayload(
            code=ErrorCode.TRAINING_FAILED,
            message="x",
            exit_code=1,
            wall_seconds=0.0,
        )
        data = json.loads(p.model_dump_json())
        # Must serialise the key (round-trip determinism) — ``None``
        # in JSON becomes ``null``, not missing.
        assert data["traceback_summary"] is None


# ---------------------------------------------------------------------------
# Class 3: Atomic write semantics
# ---------------------------------------------------------------------------


class TestAtomicWrite:
    """``write_to`` writes atomically (tmp + rename) and is recoverable."""

    def test_write_creates_target_file(self, tmp_path: Path) -> None:
        target = tmp_path / TRAINER_EXIT_FILENAME
        payload = TrainerExitPayload(
            code=ErrorCode.INTERNAL_ERROR,
            message="boom",
            exit_code=1,
            wall_seconds=0.5,
        )
        payload.write_to(target)
        assert target.exists()
        restored = TrainerExitPayload.model_validate_json(target.read_text())
        assert restored == payload

    def test_write_creates_parent_dir(self, tmp_path: Path) -> None:
        target = tmp_path / "subdir" / "deep" / TRAINER_EXIT_FILENAME
        payload = TrainerExitPayload(
            code=ErrorCode.TRAINING_FAILED,
            message="x",
            exit_code=1,
            wall_seconds=0.0,
        )
        payload.write_to(target)
        assert target.exists()

    def test_write_overwrites_existing_file(self, tmp_path: Path) -> None:
        target = tmp_path / TRAINER_EXIT_FILENAME
        target.write_text('{"stale": true}', encoding="utf-8")
        payload = TrainerExitPayload(
            code=ErrorCode.TRAINING_FAILED,
            message="fresh",
            exit_code=2,
            wall_seconds=1.0,
        )
        payload.write_to(target)
        assert TrainerExitPayload.model_validate_json(target.read_text()) == payload

    def test_write_leaves_no_tmp_files_on_success(self, tmp_path: Path) -> None:
        """The .tmp sibling must be cleaned by ``os.replace``."""
        target = tmp_path / TRAINER_EXIT_FILENAME
        payload = TrainerExitPayload(
            code=ErrorCode.TRAINING_FAILED,
            message="x",
            exit_code=1,
            wall_seconds=0.0,
        )
        payload.write_to(target)
        leftover = [
            p for p in tmp_path.iterdir()
            if p.name.startswith(".trainer-exit.") and p.name.endswith(".tmp")
        ]
        assert leftover == []


# ---------------------------------------------------------------------------
# Class 4: read_from / TrainerExitPayload.read_from
# ---------------------------------------------------------------------------


class TestReadFrom:
    """``read_from`` returns ``None`` for missing files, validates content."""

    def test_returns_none_when_file_absent(self, tmp_path: Path) -> None:
        result = TrainerExitPayload.read_from(tmp_path / "missing.json")
        assert result is None

    def test_returns_payload_for_valid_file(self, tmp_path: Path) -> None:
        target = tmp_path / TRAINER_EXIT_FILENAME
        original = TrainerExitPayload(
            code=ErrorCode.TRAINING_OOM,
            message="OOM",
            exit_code=137,
            wall_seconds=10.0,
        )
        original.write_to(target)
        restored = TrainerExitPayload.read_from(target)
        assert restored == original

    def test_raises_validation_error_on_malformed_json(self, tmp_path: Path) -> None:
        target = tmp_path / TRAINER_EXIT_FILENAME
        target.write_text("{ not valid json", encoding="utf-8")
        with pytest.raises((ValidationError, ValueError)):
            TrainerExitPayload.read_from(target)

    def test_raises_validation_error_on_missing_required_field(self, tmp_path: Path) -> None:
        target = tmp_path / TRAINER_EXIT_FILENAME
        # Missing ``exit_code``.
        target.write_text(
            '{"code": "TRAINING_FAILED", "message": "x", "wall_seconds": 1.0}',
            encoding="utf-8",
        )
        with pytest.raises(ValidationError):
            TrainerExitPayload.read_from(target)


# ---------------------------------------------------------------------------
# Class 5: Traceback sanitisation
# ---------------------------------------------------------------------------


class TestSanitizeTraceback:
    """``sanitize_traceback`` strips paths and caps line count."""

    def test_strips_users_path(self) -> None:
        tb = '  File "/Users/alice/projects/foo/bar.py", line 5, in f'
        result = sanitize_traceback(tb)
        assert "/Users/alice" not in result
        assert "<home>/" in result

    def test_strips_home_path(self) -> None:
        tb = '  File "/home/bob/code/x.py", line 1, in f'
        result = sanitize_traceback(tb)
        assert "/home/bob" not in result
        assert "<home>/" in result

    def test_strips_site_packages_path(self) -> None:
        tb = (
            '  File "/Users/alice/.venv/lib/python3.12/site-packages/torch/nn.py", '
            'line 100, in forward'
        )
        result = sanitize_traceback(tb)
        assert "/Users/alice" not in result
        assert "<sp>/" in result

    def test_strips_tmp_path(self) -> None:
        tb = '  File "/tmp/pytest-abc/test.py", line 1, in f'
        result = sanitize_traceback(tb)
        assert "/tmp/pytest-abc" not in result
        assert "<tmp>" in result

    def test_caps_at_max_lines(self) -> None:
        tb = "\n".join(f"line {i}" for i in range(100))
        result = sanitize_traceback(tb, max_lines=10)
        lines = result.splitlines()
        # ``max_lines`` retained + a 1-line truncation marker prepended.
        assert len(lines) == 11
        assert lines[0].startswith("...")
        assert "line 99" in result  # tail kept
        assert "line 0" not in result  # head dropped

    def test_default_max_lines_constant(self) -> None:
        assert DEFAULT_TRACEBACK_MAX_LINES == 30

    def test_empty_string_returns_empty(self) -> None:
        assert sanitize_traceback("") == ""

    def test_no_paths_no_changes(self) -> None:
        tb = "Traceback (most recent call last):\n  ValueError: x"
        assert sanitize_traceback(tb) == tb


# ---------------------------------------------------------------------------
# Class 6: Sanitiser security regression tests
# ---------------------------------------------------------------------------


class TestSanitizerSecurity:
    """Regressions that protect the security boundary the sanitiser owns.

    Each test pins a path-leak vector the sanitiser is responsible for
    closing. If a real traceback in production manages to slip a path
    through, add a test here first, then teach :func:`sanitize_traceback`
    about it.
    """

    def test_realistic_python_traceback_redacted(self) -> None:
        # Synthesise a full traceback shape — multiple ``File "..."``
        # lines + a message. Verify all path-revealing tokens are
        # replaced and the structure is intact.
        tb = (
            "Traceback (most recent call last):\n"
            '  File "/Users/jane.doe/.venv/lib/python3.12/'
            'site-packages/transformers/trainer.py", line 1234, in train\n'
            "    self._inner_training_loop(args)\n"
            '  File "/home/jane/work/my_project/run.py", line 42, in main\n'
            "    train(model)\n"
            "torch.cuda.OutOfMemoryError: CUDA out of memory\n"
        )
        result = sanitize_traceback(tb)
        assert "/Users/jane.doe" not in result
        assert "/home/jane" not in result
        assert "site-packages" not in result  # collapsed to <sp>/
        # Structural cues remain so the trace stays readable.
        assert "Traceback" in result
        assert "trainer.py" in result
        assert "out of memory" in result.lower()

    def test_does_not_leak_username_in_var_folders(self) -> None:
        tb = '  File "/var/folders/zz/abc/T/foo.py", line 1, in f'
        result = sanitize_traceback(tb)
        assert "/var/folders" not in result

    def test_strips_control_chars(self) -> None:
        tb = "frame\x00with\x01control\nchars"
        result = sanitize_traceback(tb)
        assert "\x00" not in result
        assert "\x01" not in result
        assert "frame" in result and "with" in result

    def test_truncation_marker_does_not_leak_count_via_long_path(self) -> None:
        """Even with paths in the tail, sanitiser must run BEFORE the cap.

        Otherwise truncated output could leak the original tail's
        path content. Pin the ordering.
        """
        tb = "\n".join(
            f'  File "/Users/secret-username/run.py", line {i}, in f'
            for i in range(50)
        )
        result = sanitize_traceback(tb, max_lines=10)
        assert "secret-username" not in result


# ---------------------------------------------------------------------------
# Class 7: Invariants / regressions
# ---------------------------------------------------------------------------


class TestInvariants:
    """Cross-cutting invariants over the contract module."""

    def test_filename_constant_pinned(self) -> None:
        # The filename is shared between trainer and supervisor via
        # this constant — must not drift.
        assert TRAINER_EXIT_FILENAME == "trainer-exit.json"

    def test_schema_version_default_is_one(self) -> None:
        # First release pins schema_version=1; bumping it is a
        # cross-process protocol change that requires updating both
        # sides + this assertion in lockstep.
        assert TRAINER_EXIT_SCHEMA_VERSION == 1

    def test_round_trip_via_disk(self, tmp_path: Path) -> None:
        target = tmp_path / TRAINER_EXIT_FILENAME
        original = TrainerExitPayload(
            code=ErrorCode.TRAINING_FAILED,
            message="fail",
            traceback_summary="frame 1\nframe 2",
            exit_code=2,
            wall_seconds=3.0,
        )
        original.write_to(target)
        restored = TrainerExitPayload.read_from(target)
        assert restored == original

    def test_sanitised_traceback_idempotent(self) -> None:
        """Running the sanitiser twice should be a no-op on the
        already-clean output (no double <home>/ <home>/ collapses)."""
        tb = '  File "/Users/x/foo.py", line 1, in f'
        once = sanitize_traceback(tb)
        twice = sanitize_traceback(once)
        assert once == twice

    def test_extra_forbidden_pin(self) -> None:
        # Defensive pin: an agent rewriting the model to ``extra="allow"``
        # would break the "fail closed on unknown fields" invariant the
        # supervisor relies on for forward-compat.
        assert TrainerExitPayload.model_config["extra"] == "forbid"
