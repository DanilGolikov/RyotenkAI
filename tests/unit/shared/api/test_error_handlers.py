"""Unit tests for :mod:`ryotenkai_shared.api.error_handlers`.

Originally lifted from ``tests/unit/pod/runner/api/test_errors.py`` in
Phase B; the post-Phase G fix-up retired the :class:`APIError` class
and migrated every raise site to a typed :class:`RyotenkAIError`
subclass. The test categories below now cover the four remaining
handlers: ``ryotenkai_error_handler``, ``http_exception_handler``
(kept for FastAPI's own raises), ``validation_exception_handler``,
and ``generic_exception_handler``.

Test categories per project policy:

* positive       -- typed RyotenkAIError -> problem+json happy path.
* negative       -- malformed input, missing fields, dict->shape adaption.
* boundary       -- zero-length errors[], status 100/599 edges, no detail.
* invariant      -- Content-Type is always ``application/problem+json``;
                    ``code`` is always present; nulls are stripped.
* dependency-err -- handler called when logger isn't ready; no crash.
* regression     -- legacy ``HTTPException(detail={"code": ...})`` adapted;
                    ``APIError`` symbol must not reappear.
* logic-specific -- alias map (``invalid_job_spec`` -> JOB_SPEC_INVALID).
* combinatorial  -- ErrorCode x HTTPException-shape x with/without trace_id.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI, HTTPException, Request
from fastapi.testclient import TestClient

from ryotenkai_shared.api.error_handlers import (
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
from ryotenkai_shared.errors import (
    JobNotFoundError,
    JobSpecInvalidError,
    RunnerBusyError,
    RunnerNotReadyError,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_app() -> FastAPI:
    """Tiny app with the unified handlers attached, plus routes that raise
    each exception type. Reused across tests."""
    app = FastAPI(exception_handlers=EXCEPTION_HANDLERS)

    @app.get("/raise-typed-error")
    def _raise_typed() -> None:
        raise JobNotFoundError(detail="job xyz is gone")

    @app.get("/raise-typed-with-fields")
    def _raise_typed_with_fields() -> None:
        raise JobSpecInvalidError(
            detail="invalid spec",
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
    def test_typed_error_returns_problem_json(self) -> None:
        with TestClient(_make_app()) as c:
            r = c.get("/raise-typed-error")
        assert r.status_code == 404
        assert r.headers["content-type"].startswith(PROBLEM_JSON_MEDIA_TYPE)
        body = r.json()
        assert body["code"] == "JOB_NOT_FOUND"
        assert body["status"] == 404
        assert body["title"] == "Job not found"
        assert body["detail"] == "job xyz is gone"
        assert body["instance"] == "/raise-typed-error"
        assert "trace_id" in body
        assert len(body["trace_id"]) == 16

    def test_typed_error_with_field_errors(self) -> None:
        with TestClient(_make_app()) as c:
            r = c.get("/raise-typed-with-fields")
        assert r.status_code == 422
        body = r.json()
        assert body["code"] == "JOB_SPEC_INVALID"
        assert body["errors"] == [
            {"loc": ["body", "command"], "type": "missing", "msg": "Field required"},
        ]

    def test_install_exception_handlers_imperative_form(self) -> None:
        """``install_exception_handlers(app)`` is wire-equivalent to the
        constructor form -- verifies it actually registers handlers (not
        a no-op) by exercising the response shape on a bare app built
        without ``exception_handlers=...``."""
        app = FastAPI()
        install_exception_handlers(app)

        @app.get("/boom")
        def _boom() -> None:
            raise RunnerBusyError(detail="x")

        with TestClient(app) as c:
            r = c.get("/boom")
        assert r.status_code == 503
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
        exc = HTTPException(status_code=418, detail={"message": "no code here"})
        code, message = _http_exception_to_code(exc)
        assert code == ErrorCode.INTERNAL_ERROR
        assert message is None

    def test_http_exception_to_code_non_string_message_dropped(self) -> None:
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
    def test_typed_error_without_detail(self) -> None:
        err = JobNotFoundError()
        problem = err.as_problem(instance="/x")
        out = problem.model_dump(mode="json", exclude_none=True)
        assert "detail" not in out
        assert out["code"] == "JOB_NOT_FOUND"

    def test_status_code_extreme_values_accepted(self) -> None:
        for status in (100, 599):
            ProblemDetails(title="x", status=status, code=ErrorCode.INTERNAL_ERROR)
        with pytest.raises(Exception):
            ProblemDetails(title="x", status=99, code=ErrorCode.INTERNAL_ERROR)
        with pytest.raises(Exception):
            ProblemDetails(title="x", status=600, code=ErrorCode.INTERNAL_ERROR)

    def test_http_exception_to_code_none_detail(self) -> None:
        exc = HTTPException(status_code=500, detail=None)  # type: ignore[arg-type]
        code, message = _http_exception_to_code(exc)
        assert code == ErrorCode.INTERNAL_ERROR
        assert isinstance(message, str)
        assert message


# ---------------------------------------------------------------------------
# Invariant
# ---------------------------------------------------------------------------


class TestInvariant:
    @pytest.mark.parametrize("path,expected_status", [
        ("/raise-typed-error", 404),
        ("/raise-typed-with-fields", 422),
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
        "/raise-typed-error",
        "/raise-http-legacy-dict",
        "/raise-bare-exception",
    ])
    def test_code_field_always_present(self, path: str) -> None:
        with TestClient(_make_app(), raise_server_exceptions=False) as c:
            r = c.get(path)
        assert "code" in r.json()
        assert r.json()["code"]

    def test_null_fields_stripped_from_wire(self) -> None:
        err = RunnerNotReadyError()
        out = err.as_problem(instance="/x").model_dump(mode="json", exclude_none=True)
        for field in ("detail", "errors", "trace_id", "request_id"):
            assert field not in out, f"{field} should be null-stripped"


# ---------------------------------------------------------------------------
# Regression -- every legacy raise site must adapt to a known code
# ---------------------------------------------------------------------------


class TestRegression:
    @pytest.mark.parametrize("legacy_code,expected", [
        ("job_not_found", ErrorCode.JOB_NOT_FOUND),
        ("invalid_job_spec", ErrorCode.JOB_SPEC_INVALID),
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
        post-Phase G fix-up retired ``APIError`` itself. The runner now
        raises typed :class:`RyotenkAIError` subclasses directly. This
        sentinel regression-guards both: the legacy shim stays gone, and
        ``APIError`` is no longer exported from
        :mod:`ryotenkai_shared.api.error_handlers`.
        """
        import importlib

        with pytest.raises(ModuleNotFoundError):
            importlib.import_module("ryotenkai_pod.runner.api.errors")

        from ryotenkai_shared.api import error_handlers as mod

        assert not hasattr(mod, "APIError"), (
            "APIError must not be re-introduced; raise sites use "
            "RyotenkAIError subclasses instead."
        )

        from ryotenkai_shared.api.error_handlers import (
            EXCEPTION_HANDLERS as shared_handlers,
        )

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

    def test_typed_error_title_defaults_via_render(self) -> None:
        """The title is pulled from ``_DEFAULT_TITLES`` via the base
        ``RyotenkAIError.title`` property; verifies the merged map
        carries pod-runner codes."""
        err = JobNotFoundError()
        assert err.title == "Job not found"


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
        """If the logger raises mid-handler, the response must still be
        emitted -- or, in the current contract, the failure must surface
        so devops knows. The contract is: log failures DON'T silence --
        they propagate."""
        from ryotenkai_shared.api import error_handlers as mod

        broken_logger = MagicMock()
        broken_logger.error.side_effect = RuntimeError("logger dead")
        monkeypatch.setattr(mod, "logger", broken_logger)

        request = Request({"type": "http", "method": "GET", "path": "/x", "headers": []})

        with pytest.raises(RuntimeError, match="logger dead"):
            await mod.generic_exception_handler(
                request, RuntimeError("real bug"),
            )
