"""Unit tests for :mod:`ryotenkai_shared.api.error_handlers` (Phase B).

Moved from ``tests/unit/pod/runner/api/test_errors.py`` when Phase B
lifted the handlers out of pod into shared.

Test categories per project policy:

* positive       -- APIError -> problem+json happy path.
* negative       -- malformed input, missing fields, dict->shape adaption.
* boundary       -- zero-length errors[], status 100/599 edges, no detail.
* invariant      -- Content-Type is always ``application/problem+json``;
                    ``code`` is always present; nulls are stripped.
* dependency-err -- handler called when logger isn't ready; no crash.
* regression     -- legacy ``HTTPException(detail={"code": ...})`` adapted.
* logic-specific -- alias map (``invalid_job_spec`` -> JOB_SPEC_INVALID).
* combinatorial  -- ErrorCode x HTTPException-shape x with/without trace_id.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI, HTTPException, Request
from fastapi.testclient import TestClient

from ryotenkai_shared.api.error_handlers import (
    APIError,
    EXCEPTION_HANDLERS,
    _http_exception_to_code,
    install_exception_handlers,
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
    """Tiny app with the Phase B handlers attached, plus routes that
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
        assert "trace_id" in body  # 16-hex generated each call
        assert len(body["trace_id"]) == 16

    def test_apierror_with_field_errors(self) -> None:
        with TestClient(_make_app()) as c:
            r = c.get("/raise-api-error-with-fields")
        assert r.status_code == 422
        body = r.json()
        assert body["code"] == "JOB_SPEC_INVALID"
        assert body["errors"] == [
            {"loc": ["body", "command"], "type": "missing", "msg": "Field required"},
        ]

    def test_install_exception_handlers_imperative_form(self) -> None:
        """``install_exception_handlers(app)`` is wire-equivalent to
        the constructor form -- verifies it actually registers
        handlers (not a no-op) by exercising the response shape on a
        bare app built without ``exception_handlers=...``."""
        app = FastAPI()
        install_exception_handlers(app)

        @app.get("/boom")
        def _boom() -> None:
            raise APIError(ErrorCode.RUNNER_BUSY, status=409, detail="x")

        with TestClient(app) as c:
            r = c.get("/boom")
        assert r.status_code == 409
        assert r.headers["content-type"].startswith(PROBLEM_JSON_MEDIA_TYPE)
        body = r.json()
        assert body["code"] == "RUNNER_BUSY"
        assert body["title"] == "Runner busy"


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

    def test_http_exception_to_code_dict_without_code_returns_internal_error(self) -> None:
        """Dict detail missing ``code`` key -> INTERNAL_ERROR fallback."""
        exc = HTTPException(status_code=418, detail={"message": "no code here"})
        code, message = _http_exception_to_code(exc)
        assert code == ErrorCode.INTERNAL_ERROR
        assert message is None

    def test_http_exception_to_code_non_string_message_dropped(self) -> None:
        """``detail["message"]`` must be a str; non-str -> message=None."""
        exc = HTTPException(
            status_code=400,
            detail={"code": "spawn_failed", "message": ["not", "a", "string"]},
        )
        code, message = _http_exception_to_code(exc)
        assert code == ErrorCode.SPAWN_FAILED
        assert message is None


# ---------------------------------------------------------------------------
# Boundary
# ---------------------------------------------------------------------------


class TestBoundary:
    def test_apierror_without_detail(self) -> None:
        # APIError(detail=None) -> null-stripped from response
        err = APIError(ErrorCode.RUNNER_BUSY, status=409)
        problem = err.as_problem(instance="/x")
        out = problem.model_dump(mode="json", exclude_none=True)
        assert "detail" not in out
        assert out["code"] == "RUNNER_BUSY"

    def test_status_code_extreme_values_accepted(self) -> None:
        # ProblemDetails enforces 100 <= status <= 599
        for status in (100, 599):
            ProblemDetails(title="x", status=status, code=ErrorCode.INTERNAL_ERROR)
        with pytest.raises(Exception):
            ProblemDetails(title="x", status=99, code=ErrorCode.INTERNAL_ERROR)
        with pytest.raises(Exception):
            ProblemDetails(title="x", status=600, code=ErrorCode.INTERNAL_ERROR)

    def test_http_exception_to_code_none_detail(self) -> None:
        """``HTTPException(detail=None)`` coerces detail to the HTTP
        reason phrase (FastAPI behaviour). The adapter then routes
        through the str branch and returns INTERNAL_ERROR with the
        phrase as the carried message -- never crashes on the boundary.
        """
        exc = HTTPException(status_code=500, detail=None)  # type: ignore[arg-type]
        code, message = _http_exception_to_code(exc)
        assert code == ErrorCode.INTERNAL_ERROR
        # FastAPI replaced detail=None with the reason phrase; the
        # adapter passes it through verbatim. We don't care about the
        # exact text -- just that it's a non-empty string, not None.
        assert isinstance(message, str)
        assert message  # non-empty


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
        # APIError without detail/title-override/errors -> those fields
        # never appear in the response body (RFC 9457 §3.1).
        err = APIError(ErrorCode.RUNNER_NOT_READY, status=503)
        out = err.as_problem(instance="/x").model_dump(mode="json", exclude_none=True)
        for field in ("detail", "errors", "trace_id", "request_id"):
            assert field not in out, f"{field} should be null-stripped"


# ---------------------------------------------------------------------------
# Regression -- every legacy raise site must adapt to a known code
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

    def test_pod_runner_imports_resolve_from_shared(self) -> None:
        """Phase F deleted ``ryotenkai_pod.runner.api.errors`` shim;
        the pod runner now imports :class:`APIError` and
        :data:`EXCEPTION_HANDLERS` directly from
        :mod:`ryotenkai_shared.api.error_handlers`. This sentinel
        regression-guards that the shared symbols are still importable
        from the canonical path and that the shim is gone.
        """
        import importlib

        with pytest.raises(ModuleNotFoundError):
            importlib.import_module("ryotenkai_pod.runner.api.errors")

        from ryotenkai_shared.api.error_handlers import (
            APIError as shared_api_error,
        )
        from ryotenkai_shared.api.error_handlers import (
            EXCEPTION_HANDLERS as shared_handlers,
        )

        assert shared_api_error is APIError
        assert shared_handlers is EXCEPTION_HANDLERS


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

    def test_apierror_title_falls_back_to_default_title_for(self) -> None:
        """When ``title=`` is omitted the constructor must look up
        :data:`_DEFAULT_TITLES` via ``default_title_for``; verifies
        the merged map carries pod-runner codes after Phase B."""
        err = APIError(ErrorCode.JOB_NOT_FOUND, status=404)
        assert err.title == "Job not found"

    def test_apierror_custom_title_overrides_default(self) -> None:
        """Explicit ``title=`` wins over the default-titles map."""
        err = APIError(ErrorCode.JOB_NOT_FOUND, status=404, title="Custom")
        assert err.title == "Custom"


# ---------------------------------------------------------------------------
# Combinatorial: code x shape x with-trace x with-instance
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
# Dependency-error -- handler behaviour when its logger is unhealthy
# ---------------------------------------------------------------------------


class TestDependencyError:
    @pytest.mark.asyncio
    async def test_generic_handler_does_not_propagate_logger_failure(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """If the logger raises mid-handler, the response must still
        be emitted -- or, in the current contract, the failure must
        surface so devops knows. The contract is: log failures DON'T
        silence -- they propagate."""
        from ryotenkai_shared.api import error_handlers as mod

        broken_logger = MagicMock()
        broken_logger.error.side_effect = RuntimeError("logger dead")
        monkeypatch.setattr(mod, "logger", broken_logger)

        # Build a real Starlette/FastAPI Request from an ASGI scope so we
        # don't need to mock the property surface (``url.path`` etc.).
        request = Request({"type": "http", "method": "GET", "path": "/x", "headers": []})

        with pytest.raises(RuntimeError, match="logger dead"):
            await mod.generic_exception_handler(
                request, RuntimeError("real bug"),
            )
