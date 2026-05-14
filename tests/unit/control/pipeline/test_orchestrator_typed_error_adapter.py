"""Tests for ``_typed_error_to_app_error`` — orchestrator's compat adapter.

Phase A2 Batch 6 introduced a one-point adapter that converts the typed
:class:`RyotenkAIError` raised by ``StageExecutionLoop.run_attempt`` into
the legacy :class:`AppError` shape that ``worker.py`` / ``run_pipeline``
still consume via ``Result[dict, AppError]``.

Coverage by category (7-class):
- 1. Positive: typed exception → AppError with correct code/message
- 2. Negative: typed exception without legacy_code falls back to wire code
- 3. Boundary: empty context; ``None`` detail; non-dict context
- 4. Invariants: typed_error_class always present in details
- 5. Dependency errors: granular legacy_code preserved on AppError.code
- 6. Regressions: PipelineStageFailedError wraps as STAGE_FAILED
- 7. Combinatorial: parametrised RyotenkAIError subclasses
"""

from __future__ import annotations

import pytest

from ryotenkai_control.pipeline.orchestrator import _typed_error_to_app_error
from ryotenkai_shared.errors import (
    ConfigInvalidError,
    InternalError,
    PipelineStageFailedError,
    ProviderUnavailableError,
    RyotenkAIError,
)


# ===========================================================================
# 1. POSITIVE
# ===========================================================================


class TestPositive:
    def test_internal_error_with_legacy_code_preserves_it(self) -> None:
        exc = InternalError(
            detail="something blew up",
            context={"legacy_code": "PIPELINE_INTERRUPTED", "foo": "bar"},
        )
        app_err = _typed_error_to_app_error(exc)
        assert app_err.code == "PIPELINE_INTERRUPTED"
        assert app_err.message == "something blew up"
        assert app_err.details is not None
        assert app_err.details["foo"] == "bar"
        assert app_err.details["typed_error_class"] == "InternalError"
        # legacy_code should not be duplicated in details
        assert "legacy_code" not in app_err.details

    def test_pipeline_state_error_legacy_code_preserved(self) -> None:
        exc = InternalError(
            detail="disk full",
            context={"legacy_code": "PIPELINE_STATE_ERROR"},
        )
        app_err = _typed_error_to_app_error(exc)
        assert app_err.code == "PIPELINE_STATE_ERROR"


# ===========================================================================
# 2. NEGATIVE
# ===========================================================================


class TestNegative:
    def test_no_legacy_code_falls_back_to_wire_code(self) -> None:
        exc = ProviderUnavailableError(detail="503 returned")
        app_err = _typed_error_to_app_error(exc)
        # ErrorCode enum value used when legacy_code is absent
        assert app_err.code == "PROVIDER_UNAVAILABLE"
        assert app_err.message == "503 returned"

    def test_empty_legacy_code_string_falls_back(self) -> None:
        exc = InternalError(detail="x", context={"legacy_code": ""})
        app_err = _typed_error_to_app_error(exc)
        # Empty string is falsy → fallback to wire code
        assert app_err.code == "INTERNAL_ERROR"

    def test_non_string_legacy_code_falls_back(self) -> None:
        exc = InternalError(detail="x", context={"legacy_code": 42})
        app_err = _typed_error_to_app_error(exc)
        # Non-str legacy_code is ignored
        assert app_err.code == "INTERNAL_ERROR"


# ===========================================================================
# 3. BOUNDARY
# ===========================================================================


class TestBoundary:
    def test_no_context_at_all(self) -> None:
        exc = InternalError(detail="solo")
        app_err = _typed_error_to_app_error(exc)
        assert app_err.code == "INTERNAL_ERROR"
        assert app_err.message == "solo"
        # typed_error_class always recorded
        assert app_err.details == {"typed_error_class": "InternalError"}

    def test_none_detail_uses_str_repr(self) -> None:
        exc = InternalError()  # detail=None
        app_err = _typed_error_to_app_error(exc)
        # str(exc) is "<code>: <title>" for typed exceptions
        assert app_err.message
        assert "INTERNAL_ERROR" in app_err.message

    def test_empty_context_dict(self) -> None:
        exc = InternalError(detail="x", context={})
        app_err = _typed_error_to_app_error(exc)
        assert app_err.code == "INTERNAL_ERROR"
        assert app_err.details == {"typed_error_class": "InternalError"}


# ===========================================================================
# 4. INVARIANTS
# ===========================================================================


class TestInvariants:
    def test_typed_error_class_always_in_details(self) -> None:
        for exc in [
            InternalError(detail="a"),
            PipelineStageFailedError(detail="b"),
            ProviderUnavailableError(detail="c"),
            ConfigInvalidError(detail="d"),
        ]:
            app_err = _typed_error_to_app_error(exc)
            assert app_err.details is not None
            assert app_err.details["typed_error_class"] == type(exc).__name__


# ===========================================================================
# 5. DEPENDENCY ERRORS
# ===========================================================================


class TestDependencyErrors:
    def test_unexpected_error_legacy_code_round_trips(self) -> None:
        exc = InternalError(
            detail="oh no",
            context={"legacy_code": "UNEXPECTED_ERROR", "exception_type": "RuntimeError"},
        )
        app_err = _typed_error_to_app_error(exc)
        assert app_err.code == "UNEXPECTED_ERROR"
        assert app_err.details is not None
        assert app_err.details["exception_type"] == "RuntimeError"


# ===========================================================================
# 6. REGRESSIONS
# ===========================================================================


class TestRegressions:
    def test_pipeline_stage_failed_without_legacy_code_wraps_as_stage_failed(self) -> None:
        """Stage-direct PipelineStageFailedError → AppError(code=STAGE_FAILED)."""
        exc = PipelineStageFailedError(
            detail="stage X broke", context={"stage_name": "X"}
        )
        app_err = _typed_error_to_app_error(exc)
        # No legacy_code present → wrap as STAGE_FAILED with original_code
        assert app_err.code == "STAGE_FAILED"
        assert app_err.message.startswith("Stage failed:")
        assert app_err.details is not None
        assert app_err.details["original_code"] == "PIPELINE_STAGE_FAILED"
        assert app_err.details["stage_name"] == "X"

    def test_pipeline_stage_failed_with_legacy_code_preserves_legacy(self) -> None:
        """Prereq-violation PipelineStageFailedError carries legacy_code; preserve it."""
        exc = PipelineStageFailedError(
            detail="prereq failed",
            context={"legacy_code": "MISSING_PREREQ", "prereq_failure": True},
        )
        app_err = _typed_error_to_app_error(exc)
        assert app_err.code == "MISSING_PREREQ"
        assert app_err.message == "prereq failed"


# ===========================================================================
# 7. COMBINATORIAL
# ===========================================================================


@pytest.mark.parametrize(
    "exc_cls,detail,expected_code_when_no_legacy",
    [
        (InternalError, "i", "INTERNAL_ERROR"),
        (ProviderUnavailableError, "p", "PROVIDER_UNAVAILABLE"),
        (ConfigInvalidError, "c", "CONFIG_INVALID"),
    ],
)
def test_subclass_fallback_to_wire_code(
    exc_cls: type[RyotenkAIError], detail: str, expected_code_when_no_legacy: str
) -> None:
    exc = exc_cls(detail=detail)
    app_err = _typed_error_to_app_error(exc)
    assert app_err.code == expected_code_when_no_legacy
    assert app_err.message == detail
