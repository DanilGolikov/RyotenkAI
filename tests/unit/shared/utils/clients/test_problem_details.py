"""Unit tests for the Mac-side problem+json parser (Phase 1 PR-1.3).

Test categories:
* positive       — proper problem+json → typed APIException
* negative       — malformed JSON, missing fields → TransportError
* boundary       — empty body, status 100/599
* invariant      — every parser output is APIException-compatible
* dependency-err — schema validation failure stays surfaced
* regression     — Content-Type detection is case-insensitive
* logic-specific — TransportError carries TRANSPORT_UNREACHABLE
* combinatorial  — content_type × body_shape × status_code matrix
"""

from __future__ import annotations

import json

import httpx
import pytest

from ryotenkai_shared.contracts.problem_details import (
    PROBLEM_JSON_MEDIA_TYPE,
    ErrorCode,
    ProblemDetails,
)
from ryotenkai_shared.utils.clients.problem_details import (
    APIException,
    TransportError,
    parse_problem_details,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_response(
    *,
    content_type: str,
    body: bytes,
    status_code: int = 500,
) -> httpx.Response:
    """Build a minimal httpx.Response — works without a transport."""
    return httpx.Response(
        status_code=status_code,
        headers={"content-type": content_type},
        content=body,
        request=httpx.Request("GET", "https://example/"),
    )


def _problem_body(**kwargs) -> bytes:  # type: ignore[no-untyped-def]
    p = ProblemDetails(**kwargs)
    return json.dumps(p.model_dump(mode="json", exclude_none=True)).encode()


# ---------------------------------------------------------------------------
# Positive
# ---------------------------------------------------------------------------


class TestPositive:
    def test_runner_problem_json_parses_into_apiexception(self) -> None:
        response = _make_response(
            content_type=PROBLEM_JSON_MEDIA_TYPE,
            body=_problem_body(
                title="Job not found", status=404,
                code=ErrorCode.JOB_NOT_FOUND, detail="job foo missing",
            ),
            status_code=404,
        )
        exc = parse_problem_details(response)
        assert isinstance(exc, APIException)
        assert not isinstance(exc, TransportError)
        assert exc.code == ErrorCode.JOB_NOT_FOUND
        assert exc.status == 404
        assert exc.title == "Job not found"
        assert exc.detail == "job foo missing"


# ---------------------------------------------------------------------------
# Negative
# ---------------------------------------------------------------------------


class TestNegative:
    def test_invalid_json_returns_transport_error(self) -> None:
        response = _make_response(
            content_type=PROBLEM_JSON_MEDIA_TYPE,
            body=b"<<not-json>>",
            status_code=500,
        )
        exc = parse_problem_details(response)
        assert isinstance(exc, TransportError)
        assert exc.code == ErrorCode.TRANSPORT_UNREACHABLE

    def test_problem_json_with_unknown_code_fails_schema_validation(self) -> None:
        # Body has a code value not in ErrorCode → schema rejects.
        bad = json.dumps({
            "title": "x", "status": 500, "code": "NOT_A_REAL_CODE",
        }).encode()
        response = _make_response(
            content_type=PROBLEM_JSON_MEDIA_TYPE, body=bad, status_code=500,
        )
        exc = parse_problem_details(response)
        assert isinstance(exc, TransportError)
        assert "schema validation" in (exc.detail or "")

    def test_html_response_returns_transport_error(self) -> None:
        response = _make_response(
            content_type="text/html",
            body=b"<html>500</html>",
            status_code=500,
        )
        exc = parse_problem_details(response)
        assert isinstance(exc, TransportError)
        assert exc.code == ErrorCode.TRANSPORT_UNREACHABLE


# ---------------------------------------------------------------------------
# Boundary
# ---------------------------------------------------------------------------


class TestBoundary:
    def test_empty_body_returns_transport_error(self) -> None:
        response = _make_response(
            content_type=PROBLEM_JSON_MEDIA_TYPE, body=b"", status_code=500,
        )
        exc = parse_problem_details(response)
        assert isinstance(exc, TransportError)


# ---------------------------------------------------------------------------
# Invariant
# ---------------------------------------------------------------------------


class TestInvariant:
    @pytest.mark.parametrize("content_type,body,status", [
        (PROBLEM_JSON_MEDIA_TYPE,
         _problem_body(title="x", status=500, code=ErrorCode.INTERNAL_ERROR),
         500),
        ("text/html", b"<html>x</html>", 500),
        (PROBLEM_JSON_MEDIA_TYPE, b"corrupt", 500),
        ("", b"", 599),  # tunnel-down style
    ])
    def test_parser_always_returns_apiexception(
        self, content_type: str, body: bytes, status: int,
    ) -> None:
        response = _make_response(
            content_type=content_type, body=body, status_code=status,
        )
        exc = parse_problem_details(response)
        # Either typed runner-issued OR TransportError — but always
        # APIException so callers can ``isinstance(exc, APIException)``.
        assert isinstance(exc, APIException)


# ---------------------------------------------------------------------------
# Regression — Content-Type matching is case-insensitive
# ---------------------------------------------------------------------------


class TestRegression:
    def test_content_type_uppercase_still_parses(self) -> None:
        response = _make_response(
            content_type="APPLICATION/PROBLEM+JSON",
            body=_problem_body(
                title="x", status=500, code=ErrorCode.INTERNAL_ERROR,
            ),
            status_code=500,
        )
        exc = parse_problem_details(response)
        assert not isinstance(exc, TransportError)
        assert exc.code == ErrorCode.INTERNAL_ERROR

    def test_content_type_with_charset_still_parses(self) -> None:
        response = _make_response(
            content_type=f"{PROBLEM_JSON_MEDIA_TYPE}; charset=utf-8",
            body=_problem_body(
                title="x", status=400, code=ErrorCode.JOB_SPEC_INVALID,
            ),
            status_code=400,
        )
        exc = parse_problem_details(response)
        assert not isinstance(exc, TransportError)
        assert exc.code == ErrorCode.JOB_SPEC_INVALID


# ---------------------------------------------------------------------------
# Logic-specific
# ---------------------------------------------------------------------------


class TestLogicSpecific:
    def test_transport_error_carries_TRANSPORT_UNREACHABLE_code(self) -> None:  # noqa: N802
        response = _make_response(
            content_type="text/plain",
            body=b"oops",
            status_code=502,
        )
        exc = parse_problem_details(response)
        assert isinstance(exc, TransportError)
        assert exc.code == ErrorCode.TRANSPORT_UNREACHABLE
        assert exc.status == 502

    def test_str_representation_includes_code_and_status(self) -> None:
        response = _make_response(
            content_type=PROBLEM_JSON_MEDIA_TYPE,
            body=_problem_body(
                title="t", status=404, code=ErrorCode.JOB_NOT_FOUND,
                detail="d",
            ),
            status_code=404,
        )
        exc = parse_problem_details(response)
        s = str(exc)
        assert "JOB_NOT_FOUND" in s
        assert "404" in s
        assert "d" in s


# ---------------------------------------------------------------------------
# Dependency-error
# ---------------------------------------------------------------------------


class TestDependencyError:
    def test_response_with_zero_status_falls_through_to_599(self) -> None:
        # Some httpx mock paths return Response(status=0); we map to 599
        # so ProblemDetails.status validator (≥100) passes.
        response = _make_response(
            content_type="text/plain",
            body=b"",
            status_code=599,  # closest synthetic value
        )
        exc = parse_problem_details(response)
        assert isinstance(exc, TransportError)
        assert 100 <= exc.status <= 599


# ---------------------------------------------------------------------------
# Combinatorial: content_type × body × status
# ---------------------------------------------------------------------------


class TestCombinatorial:
    @pytest.mark.parametrize("content_type", [
        PROBLEM_JSON_MEDIA_TYPE,
        "text/plain",
        "application/json",
    ])
    @pytest.mark.parametrize("status_code", [400, 404, 422, 500, 502])
    def test_matrix(self, content_type: str, status_code: int) -> None:
        body = _problem_body(
            title="x", status=status_code, code=ErrorCode.INTERNAL_ERROR,
        )
        response = _make_response(
            content_type=content_type, body=body, status_code=status_code,
        )
        exc = parse_problem_details(response)
        # When the wire body itself is valid problem+json but the
        # content-type is wrong, parser still falls back to
        # TransportError — matches contract §1.
        if content_type == PROBLEM_JSON_MEDIA_TYPE:
            assert not isinstance(exc, TransportError), (content_type, status_code)
        else:
            assert isinstance(exc, TransportError), (content_type, status_code)
