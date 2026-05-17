"""Tests for the Phase H3 ``_write_init_error`` startup-failure writer.

Coverage split by 7-class policy:

1. **Positive**          — file written with expected format.
2. **Negative**          — non-RyotenkAIError exceptions still produce a record.
3. **Boundary**          — empty context / detail / request_id.
4. **Invariants**        — atomic write (no partial file on tempfile rename).
5. **Regression**        — overwrites prior file (postmortem = latest only).
6. **Dependency error**  — read-only filesystem swallowed silently.
7. **Logic-specific**    — private (``_xxx``) context keys stripped.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path
from typing import Any

import pytest

from ryotenkai_control.pipeline.worker import _write_init_error
from ryotenkai_shared.errors import (
    InternalError,
    PipelineStageFailedError,
    RyotenkAIError,
)


# ---------------------------------------------------------------------------
# 1. Positive — well-formed init_error.log
# ---------------------------------------------------------------------------


class TestPositive:
    def test_ryotenkai_error_writes_full_block(self, tmp_path: Path) -> None:
        path = tmp_path / "init_error.log"
        exc = InternalError(
            detail="RUNPOD_API_KEY missing",
            context={"provider": "runpod"},
        )
        _write_init_error(
            path,
            exc,
            request_id="req-abc",
            command_argv=["python", "-m", "x.y.worker", "-c", "cfg.yaml"],
        )
        assert path.exists()
        content = path.read_text(encoding="utf-8")
        assert "Pipeline init FAILED" in content
        assert "  command:  python -m x.y.worker -c cfg.yaml" in content
        assert "  error:    Internal Server Error" in content
        assert "  code:     INTERNAL_ERROR" in content
        assert "  detail:   RUNPOD_API_KEY missing" in content
        assert "  request:  req-abc" in content
        assert '"provider": "runpod"' in content

    def test_record_has_separator_markers(self, tmp_path: Path) -> None:
        path = tmp_path / "init_error.log"
        _write_init_error(
            path,
            InternalError(detail="x"),
            request_id="r",
            command_argv=["cmd"],
        )
        content = path.read_text(encoding="utf-8")
        # The format opens with ``timestamp Pipeline init FAILED`` then
        # uses the 80-char separator three times (open/close).
        assert content.count("=" * 80) == 2


# ---------------------------------------------------------------------------
# 2. Negative — non-RyotenkAIError exceptions
# ---------------------------------------------------------------------------


class TestNegative:
    def test_plain_exception_records_with_internal_error_code(self, tmp_path: Path) -> None:
        path = tmp_path / "init_error.log"
        _write_init_error(
            path,
            RuntimeError("plain text"),
            request_id="r-1",
            command_argv=["cmd"],
        )
        content = path.read_text(encoding="utf-8")
        assert "  error:    RuntimeError" in content
        assert "  code:     INTERNAL_ERROR" in content
        assert "  detail:   plain text" in content


# ---------------------------------------------------------------------------
# 3. Boundary — empty context / no detail / no request_id
# ---------------------------------------------------------------------------


class TestBoundary:
    def test_no_request_id_omits_request_line(self, tmp_path: Path) -> None:
        path = tmp_path / "init_error.log"
        _write_init_error(
            path,
            InternalError(detail="x"),
            request_id=None,
            command_argv=["cmd"],
        )
        content = path.read_text(encoding="utf-8")
        assert "  request:" not in content

    def test_empty_context_omits_context_line(self, tmp_path: Path) -> None:
        path = tmp_path / "init_error.log"
        _write_init_error(
            path,
            InternalError(detail="x", context={}),
            request_id="r",
            command_argv=["cmd"],
        )
        content = path.read_text(encoding="utf-8")
        assert "  context:" not in content

    def test_no_detail_omits_detail_line(self, tmp_path: Path) -> None:
        path = tmp_path / "init_error.log"
        _write_init_error(
            path,
            InternalError(),
            request_id="r",
            command_argv=["cmd"],
        )
        content = path.read_text(encoding="utf-8")
        assert "  detail:" not in content


# ---------------------------------------------------------------------------
# 4. Invariants — atomic write (no half-written file)
# ---------------------------------------------------------------------------


class TestInvariants:
    def test_atomic_write_no_leftover_tempfile_on_success(self, tmp_path: Path) -> None:
        path = tmp_path / "init_error.log"
        _write_init_error(
            path,
            InternalError(detail="x"),
            request_id="r",
            command_argv=["cmd"],
        )
        # Only the target file lives in the directory — no half-written
        # ``tmp*`` siblings from tempfile.NamedTemporaryFile.
        siblings = [p for p in tmp_path.iterdir() if p != path]
        assert siblings == []

    def test_creates_parent_directory(self, tmp_path: Path) -> None:
        deep = tmp_path / "a" / "b" / "c"
        path = deep / "init_error.log"
        _write_init_error(
            path,
            InternalError(detail="x"),
            request_id="r",
            command_argv=["cmd"],
        )
        assert path.exists()


# ---------------------------------------------------------------------------
# 5. Regression — overwrite (NOT append)
# ---------------------------------------------------------------------------


class TestRegression:
    def test_second_call_overwrites_first(self, tmp_path: Path) -> None:
        path = tmp_path / "init_error.log"
        _write_init_error(
            path,
            InternalError(detail="first error"),
            request_id="r",
            command_argv=["cmd"],
        )
        _write_init_error(
            path,
            InternalError(detail="second error"),
            request_id="r",
            command_argv=["cmd"],
        )
        content = path.read_text(encoding="utf-8")
        assert "first error" not in content
        assert "second error" in content


# ---------------------------------------------------------------------------
# 6. Dependency error — best-effort failure path
# ---------------------------------------------------------------------------


class TestDependencyError:
    @pytest.mark.skipif(
        os.name == "nt", reason="chmod-based read-only test is POSIX-only"
    )
    def test_read_only_parent_silently_swallowed(self, tmp_path: Path) -> None:
        ro_dir = tmp_path / "readonly"
        ro_dir.mkdir()
        ro_dir.chmod(stat.S_IRUSR | stat.S_IXUSR)  # remove write bit
        try:
            # MUST NOT raise — failure to write must not mask the
            # original exception.
            _write_init_error(
                ro_dir / "init_error.log",
                InternalError(detail="x"),
                request_id="r",
                command_argv=["cmd"],
            )
        finally:
            ro_dir.chmod(stat.S_IRWXU)


# ---------------------------------------------------------------------------
# 7. Logic-specific — private context keys stripped
# ---------------------------------------------------------------------------


class TestLogicSpecific:
    def test_underscore_prefixed_keys_stripped(self, tmp_path: Path) -> None:
        path = tmp_path / "init_error.log"
        exc = InternalError(
            detail="x",
            context={"_internal": "secret", "visible": "ok"},
        )
        _write_init_error(
            path,
            exc,
            request_id="r",
            command_argv=["cmd"],
        )
        content = path.read_text(encoding="utf-8")
        assert "_internal" not in content
        assert "secret" not in content
        assert "visible" in content

    def test_trace_id_falls_back_to_none_marker(self, tmp_path: Path) -> None:
        path = tmp_path / "init_error.log"
        _write_init_error(
            path,
            InternalError(detail="x"),
            request_id="r",
            command_argv=["cmd"],
        )
        content = path.read_text(encoding="utf-8")
        # Raise-site RyotenkAIError has no trace_id ⇒ the "(none — ...)"
        # marker is emitted.
        assert "  trace:    (none" in content
