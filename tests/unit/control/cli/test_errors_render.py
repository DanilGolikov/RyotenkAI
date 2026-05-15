"""Unit tests for the Phase F kubectl/Terraform-style CLI rendering.

Covers :func:`die_from_ryotenkai` and :func:`wrap_command` in
:mod:`ryotenkai_control.cli.errors`.

Test categories (per CLAUDE.md mandatory 7-class coverage):

* TestPositive       -- typed exception -> formatted multi-line output
* TestNegative       -- wrong input type fails fast
* TestBoundary       -- empty detail / long detail / non-ASCII / no trace_id
* TestInvariants     -- exit-code mapping table (4xx -> 2, 5xx/etc -> 1)
* TestDependencyErrors -- request_id contextvar absent vs present
* TestRegressions    -- title fallback respects wire-level value
* TestLogicSpecific  -- parametrised (ErrorCode, expected_format) tuples
"""

from __future__ import annotations

import io
from typing import Any

import pytest
import typer
import yaml
from pydantic import BaseModel, ValidationError
from rich.console import Console

from ryotenkai_control.cli import errors as cli_errors
from ryotenkai_shared.api.request_id import REQUEST_ID, current_request_id
from ryotenkai_shared.contracts.problem_details import (
    ErrorCode,
    ProblemDetails,
)
from ryotenkai_shared.errors.base import (
    InternalError,
    RyotenkAIError,
    TransportError,
)
from ryotenkai_shared.errors.domain import (
    JobSpecInvalidError,
    JobStateInvalidError,
    StateLockedError,
)
from ryotenkai_shared.errors.infra import TrainingFailedError


@pytest.fixture()
def captured_stderr(monkeypatch: pytest.MonkeyPatch) -> io.StringIO:
    """Replace ``err_console`` with one writing to an in-memory buffer.

    Rich auto-detects width based on terminal columns; force a wide
    fixed width so the layout assertions are deterministic.
    """
    buf = io.StringIO()
    fake = Console(file=buf, force_terminal=False, no_color=True, width=200)
    monkeypatch.setattr(cli_errors, "err_console", fake)
    return buf


def _reset_request_id(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make sure each test starts with no inherited REQUEST_ID."""
    monkeypatch.setattr(
        REQUEST_ID,
        "set",
        REQUEST_ID.set,  # leave the real one
    )
    # Ensure the ContextVar is empty -- contextvars are per-task, but
    # pytest runs tests in the same task, so previous tests can leak.
    REQUEST_ID.set(None)


# ===========================================================================
# TestPositive -- happy path rendering
# ===========================================================================


class TestPositive:
    """Typed exception -> three-line kubectl/Terraform output."""

    def test_renders_title_hint_and_code_line(
        self, captured_stderr: io.StringIO,
    ) -> None:
        exc = JobStateInvalidError(detail='current_state="running"')
        with pytest.raises(typer.Exit) as excinfo:
            cli_errors.die_from_ryotenkai(exc, request_id="r-abc123")
        out = captured_stderr.getvalue()
        # Three lines: title / hint / code
        assert "error: Invalid job state transition" in out
        assert 'hint: current_state="running"' in out
        assert "code: JOB_STATE_INVALID" in out
        assert "request=r-abc123" in out
        # Exit code: 409 is 4xx -> exit 2
        assert excinfo.value.exit_code == 2

    def test_renders_trace_id_when_present(
        self, captured_stderr: io.StringIO,
    ) -> None:
        # from_problem populates trace_id from the wire body.
        problem = ProblemDetails(
            title="Job not found",
            status=404,
            code=ErrorCode.JOB_NOT_FOUND,
            detail='job_id="abc123" is not active',
            trace_id="a3b1c2d4",
        )
        exc = RyotenkAIError.from_problem(problem)
        with pytest.raises(typer.Exit):
            cli_errors.die_from_ryotenkai(exc, request_id="8e7f6c5b4a3d2e1f")
        out = captured_stderr.getvalue()
        assert "error: Job not found" in out
        assert 'hint: job_id="abc123" is not active' in out
        assert "code: JOB_NOT_FOUND" in out
        assert "trace=a3b1c2d4" in out
        assert "request=8e7f6c5b4a3d2e1f" in out

    def test_no_hint_line_when_detail_is_none(
        self, captured_stderr: io.StringIO,
    ) -> None:
        exc = StateLockedError()  # detail defaults to None
        with pytest.raises(typer.Exit):
            cli_errors.die_from_ryotenkai(exc)
        out = captured_stderr.getvalue()
        assert "error: " in out
        # No "hint:" line on a None detail
        assert "hint:" not in out
        assert "code: STATE_LOCKED" in out


# ===========================================================================
# TestNegative -- bad inputs
# ===========================================================================


class TestNegative:
    """Wrong input type rejected at the boundary."""

    def test_non_ryotenkai_error_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="RyotenkAIError"):
            cli_errors.die_from_ryotenkai(ValueError("nope"))  # type: ignore[arg-type]

    def test_string_input_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="RyotenkAIError"):
            cli_errors.die_from_ryotenkai("oops")  # type: ignore[arg-type]


# ===========================================================================
# TestBoundary -- edge cases of the input shape
# ===========================================================================


class TestBoundary:
    """Empty / very long / non-ASCII detail; no trace_id; no request_id."""

    def test_empty_detail_skips_hint_line(
        self, captured_stderr: io.StringIO,
    ) -> None:
        exc = InternalError(detail=None)
        with pytest.raises(typer.Exit):
            cli_errors.die_from_ryotenkai(exc)
        out = captured_stderr.getvalue()
        assert "hint:" not in out
        assert "code: INTERNAL_ERROR" in out

    def test_very_long_detail_renders_intact(
        self, captured_stderr: io.StringIO,
    ) -> None:
        long_detail = "x" * 500
        exc = JobSpecInvalidError(detail=long_detail)
        with pytest.raises(typer.Exit):
            cli_errors.die_from_ryotenkai(exc)
        out = captured_stderr.getvalue()
        # Rich may wrap but the full string is in there.
        assert long_detail.replace(" ", "") in out.replace("\n", "").replace(
            " ", "",
        )

    def test_non_ascii_detail_renders_intact(
        self, captured_stderr: io.StringIO,
    ) -> None:
        exc = JobSpecInvalidError(detail="строка с кириллицей и emoji [bug]")
        with pytest.raises(typer.Exit):
            cli_errors.die_from_ryotenkai(exc)
        out = captured_stderr.getvalue()
        assert "кириллицей" in out
        assert "[bug]" in out

    def test_no_request_id_omits_request_part(
        self, captured_stderr: io.StringIO,
    ) -> None:
        exc = JobStateInvalidError(detail="x")
        with pytest.raises(typer.Exit):
            cli_errors.die_from_ryotenkai(exc, request_id=None)
        out = captured_stderr.getvalue()
        assert "request=" not in out


# ===========================================================================
# TestInvariants -- exit-code table holds for every status class
# ===========================================================================


class TestInvariants:
    """Exit-code mapping is stable across the HTTP status surface."""

    @pytest.mark.parametrize(
        ("status", "expected_exit"),
        [
            (400, 2),
            (401, 2),
            (404, 2),
            (409, 2),
            (422, 2),
            (429, 2),
            (499, 2),
            (500, 1),
            (502, 1),
            (503, 1),
            (599, 1),  # TransportError lives here
        ],
    )
    def test_exit_code_mapping(
        self,
        captured_stderr: io.StringIO,
        status: int,
        expected_exit: int,
    ) -> None:
        # Synthesise via from_problem so we get any status independent of
        # subclass ClassVars.
        problem = ProblemDetails(
            title="x", status=status, code=ErrorCode.INTERNAL_ERROR,
        )
        exc = RyotenkAIError.from_problem(problem)
        with pytest.raises(typer.Exit) as excinfo:
            cli_errors.die_from_ryotenkai(exc)
        assert excinfo.value.exit_code == expected_exit

    def test_internal_error_always_exit_1(
        self, captured_stderr: io.StringIO,
    ) -> None:
        with pytest.raises(typer.Exit) as excinfo:
            cli_errors.die_from_ryotenkai(InternalError(detail="bug"))
        assert excinfo.value.exit_code == 1

    def test_transport_error_renders_status_part(
        self, captured_stderr: io.StringIO,
    ) -> None:
        # TransportError is special-cased: the code line carries an
        # additional ``status=<n>`` part.
        problem = ProblemDetails(
            title="Transport error",
            status=502,
            code=ErrorCode.TRANSPORT_UNREACHABLE,
            detail="tunnel down",
        )
        exc = RyotenkAIError.from_problem(problem)
        assert isinstance(exc, TransportError)
        with pytest.raises(typer.Exit) as excinfo:
            cli_errors.die_from_ryotenkai(exc)
        out = captured_stderr.getvalue()
        assert "status=502" in out
        assert excinfo.value.exit_code == 1


# ===========================================================================
# TestDependencyErrors -- contextvar plumbing
# ===========================================================================


class TestDependencyErrors:
    """``request_id`` contextvar present vs absent."""

    def test_wrap_command_mints_request_id_when_absent(
        self,
        captured_stderr: io.StringIO,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Ensure the contextvar starts empty.
        REQUEST_ID.set(None)

        observed: dict[str, str | None] = {}

        @cli_errors.wrap_command
        def cmd() -> None:
            observed["rid"] = current_request_id()
            raise JobStateInvalidError(detail="oops")

        with pytest.raises(typer.Exit):
            cmd()

        # Decorator generated and set a non-empty request id inside the
        # command body so the error rendering can surface it.
        assert observed["rid"] is not None
        assert len(observed["rid"]) >= 8
        # And it was passed through to the rendered output.
        assert f"request={observed['rid']}" in captured_stderr.getvalue()

    def test_wrap_command_inherits_existing_request_id(
        self,
        captured_stderr: io.StringIO,
    ) -> None:
        # Outer scope already has a request id (e.g. nested call from
        # the in-process FastAPI test surface). The decorator must NOT
        # mint a fresh one -- it must inherit so logs stay correlated.
        REQUEST_ID.set("outer-id-12345")

        observed: dict[str, str | None] = {}

        @cli_errors.wrap_command
        def cmd() -> None:
            observed["rid"] = current_request_id()
            raise JobStateInvalidError(detail="oops")

        try:
            with pytest.raises(typer.Exit):
                cmd()
            assert observed["rid"] == "outer-id-12345"
            assert "request=outer-id-12345" in captured_stderr.getvalue()
        finally:
            REQUEST_ID.set(None)

    def test_wrap_command_resets_request_id_on_exit(
        self,
        captured_stderr: io.StringIO,
    ) -> None:
        # Before the call: empty.
        REQUEST_ID.set(None)

        @cli_errors.wrap_command
        def cmd() -> None:
            assert current_request_id() is not None  # set by wrapper
            raise JobStateInvalidError(detail="x")

        with pytest.raises(typer.Exit):
            cmd()
        # After the wrapper unwinds the contextvar is reset to its
        # pre-call value (None) -- crucial for not leaking ids between
        # subsequent CLI invocations within a single process.
        assert current_request_id() is None


# ===========================================================================
# TestRegressions -- bug guards
# ===========================================================================


class TestRegressions:
    """Title fallback, escape characters, idempotency on re-raise."""

    def test_title_falls_back_to_registry_when_wire_title_absent(
        self,
        captured_stderr: io.StringIO,
    ) -> None:
        # A code raised in-process (not via from_problem) has no wire
        # title; the rendered title comes from the _DEFAULT_TITLES
        # registry.
        exc = JobStateInvalidError(detail="x")
        with pytest.raises(typer.Exit):
            cli_errors.die_from_ryotenkai(exc)
        out = captured_stderr.getvalue()
        # The default title for JOB_STATE_INVALID lives in _DEFAULT_TITLES.
        # If this assertion breaks, that registry drifted -- consult
        # ryotenkai_shared/errors/_render.py.
        assert "Invalid job state transition" in out

    def test_rendered_brackets_do_not_break_rich_markup(
        self,
        captured_stderr: io.StringIO,
    ) -> None:
        # Detail strings can legitimately carry square brackets (e.g.
        # f"job_id={obj!r}") that look like Rich markup tags. The
        # renderer must NOT crash on them.
        exc = JobStateInvalidError(detail="job_id=[bug] something [escape]")
        with pytest.raises(typer.Exit):
            cli_errors.die_from_ryotenkai(exc)
        # No exception, output is well-formed text.
        out = captured_stderr.getvalue()
        assert "code: JOB_STATE_INVALID" in out

    def test_verbose_flag_appends_context_block(
        self,
        captured_stderr: io.StringIO,
    ) -> None:
        exc = JobStateInvalidError(
            detail="x", context={"job_id": "abc", "state": "running"},
        )
        with pytest.raises(typer.Exit):
            cli_errors.die_from_ryotenkai(exc, verbose=True)
        out = captured_stderr.getvalue()
        # Context appears as a fourth line when verbose=True.
        assert "context:" in out
        assert "job_id" in out
        assert "abc" in out

    def test_verbose_false_omits_context_block(
        self,
        captured_stderr: io.StringIO,
    ) -> None:
        exc = JobStateInvalidError(
            detail="x", context={"job_id": "abc"},
        )
        with pytest.raises(typer.Exit):
            cli_errors.die_from_ryotenkai(exc, verbose=False)
        out = captured_stderr.getvalue()
        assert "context:" not in out


# ===========================================================================
# TestLogicSpecific -- (code, exit_code, expected_title_fragment) tuples
# ===========================================================================


class TestLogicSpecific:
    """Per-code rendering contract."""

    @pytest.mark.parametrize(
        ("exc_cls", "code_value", "exit_code"),
        [
            (JobStateInvalidError, "JOB_STATE_INVALID", 2),
            (JobSpecInvalidError, "JOB_SPEC_INVALID", 2),
            (StateLockedError, "STATE_LOCKED", 2),
            (InternalError, "INTERNAL_ERROR", 1),
            (TrainingFailedError, "TRAINING_FAILED", 1),
        ],
    )
    def test_render_per_code(
        self,
        captured_stderr: io.StringIO,
        exc_cls: type[RyotenkAIError],
        code_value: str,
        exit_code: int,
    ) -> None:
        exc = exc_cls(detail="d")
        with pytest.raises(typer.Exit) as excinfo:
            cli_errors.die_from_ryotenkai(exc)
        out = captured_stderr.getvalue()
        assert f"code: {code_value}" in out
        assert excinfo.value.exit_code == exit_code


# ===========================================================================
# wrap_command -- end-to-end behaviour for non-RyotenkAI exception paths
# ===========================================================================


class _TestModel(BaseModel):
    n: int


class TestWrapCommand:
    """`wrap_command` translates four exception families into ``die``."""

    def test_passthrough_for_non_caught_exit(
        self,
        captured_stderr: io.StringIO,
    ) -> None:
        # typer.Exit must NOT be wrapped -- it's the controlled-exit
        # primitive die() already uses.
        @cli_errors.wrap_command
        def cmd() -> None:
            raise typer.Exit(code=0)

        with pytest.raises(typer.Exit) as excinfo:
            cmd()
        assert excinfo.value.exit_code == 0

    def test_keyboard_interrupt_passthrough(
        self, captured_stderr: io.StringIO,
    ) -> None:
        @cli_errors.wrap_command
        def cmd() -> None:
            raise KeyboardInterrupt

        with pytest.raises(KeyboardInterrupt):
            cmd()

    def test_validation_error_routed_to_die(
        self, captured_stderr: io.StringIO,
    ) -> None:
        @cli_errors.wrap_command
        def cmd() -> None:
            _TestModel.model_validate({"n": "not-an-int"})

        with pytest.raises(typer.Exit) as excinfo:
            cmd()
        out = captured_stderr.getvalue()
        assert "error: invalid input" in out
        # exit 1 because die() uses code=1 by default
        assert excinfo.value.exit_code == 1

    def test_file_not_found_routed_to_die(
        self, captured_stderr: io.StringIO,
    ) -> None:
        @cli_errors.wrap_command
        def cmd() -> None:
            raise FileNotFoundError("missing.yaml")

        with pytest.raises(typer.Exit):
            cmd()
        out = captured_stderr.getvalue()
        assert "error: file not found" in out
        assert "missing.yaml" in out

    def test_yaml_error_routed_to_die(
        self, captured_stderr: io.StringIO,
    ) -> None:
        @cli_errors.wrap_command
        def cmd() -> None:
            raise yaml.YAMLError("bad mapping")

        with pytest.raises(typer.Exit):
            cmd()
        out = captured_stderr.getvalue()
        assert "error: invalid YAML" in out
        assert "bad mapping" in out

    def test_returns_value_on_happy_path(self) -> None:
        @cli_errors.wrap_command
        def cmd() -> int:
            return 42

        assert cmd() == 42

    def test_decorator_preserves_function_signature(self) -> None:
        @cli_errors.wrap_command
        def cmd(a: int, b: int = 0) -> int:
            return a + b

        # functools.wraps preserves __name__, __wrapped__, etc. which
        # is critical for Typer's introspection.
        assert cmd.__name__ == "cmd"
        assert cmd(1, b=2) == 3
