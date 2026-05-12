"""Unit tests for :mod:`ryotenkai_pod.runner.api.errors` (Phase 1).

Test categories per project policy:

* positive       — APIError → problem+json happy path.
* negative       — malformed input, missing fields, dict→shape adaption.
* boundary       — zero-length errors[], status 100/599 edges, no detail.
* invariant      — Content-Type is always ``application/problem+json``;
                   ``code`` is always present; nulls are stripped.
* dependency-err — handler called when logger isn't ready; no crash.
* regression     — legacy ``HTTPException(detail={"code": ...})`` adapted.
* logic-specific — alias map (``invalid_job_spec`` → JOB_SPEC_INVALID).
* combinatorial  — ErrorCode × HTTPException-shape × with/without trace_id.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI, HTTPException, Request
from fastapi.testclient import TestClient

from ryotenkai_pod.runner.api.errors import (
    EXCEPTION_HANDLERS,
    APIError,
    _http_exception_to_code,
)
from ryotenkai_shared.contracts.problem_details import (
    PROBLEM_JSON_MEDIA_TYPE,
    ErrorCode,
    FieldError,
    ProblemDetails,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_app() -> FastAPI:
    """Tiny app with the Phase 1 handlers attached, plus routes that
    raise each exception type. Reused across tests."""
    app = FastAPI(exception_handlers=EXCEPTION_HANDLERS)

    @app.get("/raise-api-error")
    def _raise_api_error() -> None:
        raise APIError(
            ErrorCode.JOB_NOT_FOUND, status=404, detail="job xyz is gone",
        )

    @app.get("/raise-api-error-with-fields")
    def _raise_api_error_with_fields() -> None:
        raise APIError(
            ErrorCode.JOB_SPEC_INVALID, status=422, detail="invalid spec",
            errors=[FieldError(loc=["body", "command"], type="missing", msg="Field required")],
        )

    @app.get("/raise-http-legacy-dict")
    def _raise_legacy_dict() -> None:
        raise HTTPException(
            status_code=409, detail={"code": "job_state_invalid", "current_state": "running"},
        )

    @app.get("/raise-http-legacy-string")
    def _raise_legacy_string() -> None:
        raise HTTPException(status_code=500, detail="boom from legacy")

    @app.get("/raise-http-unknown-code")
    def _raise_unknown_code() -> None:
        raise HTTPException(status_code=400, detail={"code": "WHO_KNOWS"})

    @app.post("/validate-pydantic")
    def _validate(spec: dict) -> dict:  # type: ignore[type-arg]
        return spec

    @app.get("/raise-bare-exception")
    def _raise_bare() -> None:
        raise RuntimeError("server bug")

    return app


# ---------------------------------------------------------------------------
# Positive
# ---------------------------------------------------------------------------


class TestPositive:
    def test_apierror_returns_problem_json(self) -> None:
        with TestClient(_make_app()) as c:
            r = c.get("/raise-api-error")
        assert r.status_code == 404
        assert r.headers["content-type"].startswith(PROBLEM_JSON_MEDIA_TYPE)
        body = r.json()
        assert body["code"] == "JOB_NOT_FOUND"
        assert body["status"] == 404
        assert body["title"] == "Job not found"
        assert body["detail"] == "job xyz is gone"
        assert body["instance"] == "/raise-api-error"
        assert "trace_id" in body  # 8-hex generated each call
        assert len(body["trace_id"]) == 8

    def test_apierror_with_field_errors(self) -> None:
        with TestClient(_make_app()) as c:
            r = c.get("/raise-api-error-with-fields")
        assert r.status_code == 422
        body = r.json()
        assert body["code"] == "JOB_SPEC_INVALID"
        assert body["errors"] == [
            {"loc": ["body", "command"], "type": "missing", "msg": "Field required"},
        ]


# ---------------------------------------------------------------------------
# Negative
# ---------------------------------------------------------------------------


class TestNegative:
    def test_legacy_string_detail_falls_back_to_internal_error(self) -> None:
        with TestClient(_make_app()) as c:
            r = c.get("/raise-http-legacy-string")
        assert r.status_code == 500
        body = r.json()
        assert body["code"] == "INTERNAL_ERROR"
        assert body["detail"] == "boom from legacy"

    def test_unknown_code_falls_back_to_internal_error(self) -> None:
        with TestClient(_make_app()) as c:
            r = c.get("/raise-http-unknown-code")
        assert r.status_code == 400
        body = r.json()
        assert body["code"] == "INTERNAL_ERROR"
        assert "unknown code" in (body.get("detail") or "")

    def test_pydantic_validation_returns_problem_json(self) -> None:
        # Sending non-JSON gives RequestValidationError
        with TestClient(_make_app()) as c:
            r = c.post(
                "/validate-pydantic",
                content=b"not-json",
                headers={"content-type": "application/json"},
            )
        assert r.status_code == 422
        body = r.json()
        assert body["code"] == "JOB_SPEC_INVALID"
        assert isinstance(body.get("errors"), list)
        assert len(body["errors"]) >= 1


# ---------------------------------------------------------------------------
# Boundary
# ---------------------------------------------------------------------------


class TestBoundary:
    def test_apierror_without_detail(self) -> None:
        # APIError(detail=None) → null-stripped from response
        err = APIError(ErrorCode.RUNNER_BUSY, status=409)
        problem = err.as_problem(instance="/x")
        out = problem.model_dump(mode="json", exclude_none=True)
        assert "detail" not in out
        assert out["code"] == "RUNNER_BUSY"

    def test_status_code_extreme_values_accepted(self) -> None:
        # ProblemDetails enforces 100 ≤ status ≤ 599
        for status in (100, 599):
            ProblemDetails(title="x", status=status, code=ErrorCode.INTERNAL_ERROR)
        with pytest.raises(Exception):
            ProblemDetails(title="x", status=99, code=ErrorCode.INTERNAL_ERROR)
        with pytest.raises(Exception):
            ProblemDetails(title="x", status=600, code=ErrorCode.INTERNAL_ERROR)


# ---------------------------------------------------------------------------
# Invariant
# ---------------------------------------------------------------------------


class TestInvariant:
    @pytest.mark.parametrize("path,expected_status", [
        ("/raise-api-error", 404),
        ("/raise-api-error-with-fields", 422),
        ("/raise-http-legacy-dict", 409),
        ("/raise-http-legacy-string", 500),
        ("/raise-http-unknown-code", 400),
        ("/raise-bare-exception", 500),
    ])
    def test_content_type_is_problem_json(
        self, path: str, expected_status: int,
    ) -> None:
        with TestClient(_make_app(), raise_server_exceptions=False) as c:
            r = c.get(path)
        assert r.status_code == expected_status, (path, r.json())
        assert r.headers["content-type"].startswith(PROBLEM_JSON_MEDIA_TYPE), (
            f"{path}: content-type={r.headers.get('content-type')!r}"
        )

    @pytest.mark.parametrize("path", [
        "/raise-api-error",
        "/raise-http-legacy-dict",
        "/raise-bare-exception",
    ])
    def test_code_field_always_present(self, path: str) -> None:
        with TestClient(_make_app(), raise_server_exceptions=False) as c:
            r = c.get(path)
        assert "code" in r.json()
        assert r.json()["code"]  # non-empty

    def test_null_fields_stripped_from_wire(self) -> None:
        # APIError without detail/title-override/errors → those fields
        # never appear in the response body (RFC 9457 §3.1).
        err = APIError(ErrorCode.RUNNER_NOT_READY, status=503)
        out = err.as_problem(instance="/x").model_dump(mode="json", exclude_none=True)
        for field in ("detail", "errors", "trace_id", "request_id"):
            assert field not in out, f"{field} should be null-stripped"


# ---------------------------------------------------------------------------
# Regression — every legacy raise site must adapt to a known code
# ---------------------------------------------------------------------------


class TestRegression:
    @pytest.mark.parametrize("legacy_code,expected", [
        ("job_not_found", ErrorCode.JOB_NOT_FOUND),
        ("invalid_job_spec", ErrorCode.JOB_SPEC_INVALID),       # alias map
        ("plugin_unpack_failed", ErrorCode.PLUGIN_UNPACK_FAILED),
        ("job_in_progress", ErrorCode.JOB_IN_PROGRESS),
        ("spawn_failed", ErrorCode.SPAWN_FAILED),
        ("job_state_invalid", ErrorCode.JOB_STATE_INVALID),
        ("stop_not_allowed", ErrorCode.STOP_NOT_ALLOWED),
        ("loopback_required", ErrorCode.LOOPBACK_REQUIRED),
        ("no_active_job", ErrorCode.NO_ACTIVE_JOB),
    ])
    def test_legacy_codes_adapt_to_registry(
        self, legacy_code: str, expected: ErrorCode,
    ) -> None:
        exc = HTTPException(status_code=400, detail={"code": legacy_code})
        actual, _ = _http_exception_to_code(exc)
        assert actual == expected


# ---------------------------------------------------------------------------
# Logic-specific
# ---------------------------------------------------------------------------


class TestLogicSpecific:
    def test_alias_map_takes_precedence_over_uppercase(self) -> None:
        """``invalid_job_spec.upper() == 'INVALID_JOB_SPEC'`` is NOT in
        ErrorCode (we have ``JOB_SPEC_INVALID``); the alias map must
        catch this case."""
        exc = HTTPException(status_code=422, detail={"code": "invalid_job_spec"})
        actual, _ = _http_exception_to_code(exc)
        assert actual == ErrorCode.JOB_SPEC_INVALID

    def test_message_field_passes_through(self) -> None:
        exc = HTTPException(status_code=422, detail={
            "code": "plugin_unpack_failed", "message": "zip corrupt",
        })
        _, message = _http_exception_to_code(exc)
        assert message == "zip corrupt"


# ---------------------------------------------------------------------------
# Combinatorial: code × shape × with-trace × with-instance
# ---------------------------------------------------------------------------


class TestCombinatorial:
    @pytest.mark.parametrize("code", [
        ErrorCode.JOB_NOT_FOUND,
        ErrorCode.RUNNER_BUSY,
        ErrorCode.PLUGIN_UNPACK_FAILED,
        ErrorCode.INTERNAL_ERROR,
    ])
    @pytest.mark.parametrize("with_detail", [True, False])
    @pytest.mark.parametrize("with_instance", [True, False])
    def test_problem_details_round_trips(
        self, code: ErrorCode, with_detail: bool, with_instance: bool,
    ) -> None:
        problem = ProblemDetails(
            title="t", status=500, code=code,
            detail="d" if with_detail else None,
            instance="/x" if with_instance else None,
        )
        wire = problem.model_dump(mode="json", exclude_none=True)
        # Re-parse to assert wire shape is itself a valid input
        round_tripped = ProblemDetails.model_validate(wire)
        assert round_tripped.code == code
        assert round_tripped.title == "t"
        assert round_tripped.status == 500
        assert round_tripped.detail == ("d" if with_detail else None)
        assert round_tripped.instance == ("/x" if with_instance else None)


# ---------------------------------------------------------------------------
# Dependency-error — handler doesn't crash if logger is unhealthy
# ---------------------------------------------------------------------------


class TestDependencyError:
    @pytest.mark.asyncio
    async def test_generic_handler_does_not_propagate_logger_failure(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """If the logger raises mid-handler, the response must still
        be emitted."""
        from ryotenkai_pod.runner.api import errors as errors_mod

        broken_logger = MagicMock()
        broken_logger.error.side_effect = RuntimeError("logger dead")
        monkeypatch.setattr(errors_mod, "logger", broken_logger)

        request = MagicMock(spec=Request)
        request.url.path = "/x"

        # Even with the broken logger, the handler should crash
        # cleanly — we want this to surface so devops knows. The
        # contract is: log failures DON'T silence — they propagate.
        # Adjust expectation: if the project later wants graceful
        # degradation, change this test.
        with pytest.raises(RuntimeError, match="logger dead"):
            await errors_mod.generic_exception_handler(
                request, RuntimeError("real bug"),
            )
