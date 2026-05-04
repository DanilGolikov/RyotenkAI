"""Mac-side parser for runner ``application/problem+json`` responses.

The Mac client (``JobClient``) calls :func:`parse_problem_details`
on every non-2xx response so the caller gets a typed exception
instead of a bare ``raise_for_status``-style error.

Contract:

* ``Content-Type: application/problem+json`` → parse JSON into
  :class:`ProblemDetails`, raise :class:`APIException` (or a
  subclass) carrying the typed :class:`ErrorCode`.
* Any other content type or unparseable body → raise
  :class:`TransportError(code=TRANSPORT_UNREACHABLE)` so callers
  can still ``isinstance(exc, APIException)`` and switch on
  ``exc.code`` without speccial-casing wire-shape failures.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ryotenkai_shared.contracts.problem_details import (
    PROBLEM_JSON_MEDIA_TYPE,
    ErrorCode,
    ProblemDetails,
)

if TYPE_CHECKING:
    import httpx


class APIException(Exception):
    """Mac-side typed exception parsed from a runner ``problem+json``
    response.

    Public attributes mirror the wire fields a Mac handler typically
    branches on:

        ``code``       — :class:`ErrorCode` (machine-readable switch key)
        ``status``     — HTTP status (mirrored in the body per RFC 9457)
        ``title``      — short human summary (stable across occurrences)
        ``detail``     — occurrence-specific human explanation
        ``trace_id``   — server-side correlation id (or None)
        ``problem``    — full :class:`ProblemDetails` payload (extras)

    Subclassed by :class:`TransportError` when the failure mode is
    the tunnel itself, not anything the runner emitted.
    """

    def __init__(self, problem: ProblemDetails) -> None:
        super().__init__(f"{problem.code.value} ({problem.status}): {problem.detail or problem.title}")
        self.problem: ProblemDetails = problem

    @property
    def code(self) -> ErrorCode:
        return self.problem.code

    @property
    def status(self) -> int:
        return self.problem.status

    @property
    def title(self) -> str:
        return self.problem.title

    @property
    def detail(self) -> str | None:
        return self.problem.detail

    @property
    def trace_id(self) -> str | None:
        return self.problem.trace_id


class TransportError(APIException):
    """The runner did not produce a ``problem+json`` response.

    Caused by:
    * SSH tunnel down (httpx ``ConnectError``).
    * Network reset / timeout (httpx ``ReadTimeout``).
    * Server bug returning HTML / plain text 500 (rare — every error
      path now goes through the unified handlers, but defence-in-depth).

    The carried :class:`ProblemDetails` is synthesised on the Mac side
    so the surface ``isinstance(exc, APIException)`` and
    ``exc.code == ErrorCode.TRANSPORT_UNREACHABLE`` keep working
    uniformly with runner-issued errors.
    """


def parse_problem_details(response: "httpx.Response") -> APIException:
    """Build a typed :class:`APIException` from an httpx response.

    Behaviour:

    1. ``Content-Type`` starts with ``application/problem+json`` →
       parse the body, return :class:`APIException`.
    2. Anything else (including missing/empty body) → return
       :class:`TransportError`. Body is summarised in the synthesised
       ``detail`` field so the caller can still log something useful.

    Note this function **never raises** — it builds and returns the
    exception, leaving the caller free to ``raise`` it where it
    makes sense in the call chain.
    """
    content_type = (response.headers.get("content-type") or "").lower()
    if content_type.startswith(PROBLEM_JSON_MEDIA_TYPE):
        try:
            payload = response.json()
        except ValueError:
            return _make_transport_error(
                response,
                detail="problem+json content-type but body is not valid JSON",
            )
        try:
            problem = ProblemDetails.model_validate(payload)
        except Exception as exc:  # noqa: BLE001 — defensive, exotic shapes
            return _make_transport_error(
                response,
                detail=f"problem+json failed schema validation: {exc}",
            )
        return APIException(problem)

    return _make_transport_error(
        response,
        detail=(
            f"non-problem+json response (content-type={content_type!r}, "
            f"status={response.status_code})"
        ),
    )


def _make_transport_error(
    response: "httpx.Response", *, detail: str,
) -> TransportError:
    """Synthesise :class:`ProblemDetails` for a transport-level failure
    so the wire-shape switch still works on the Mac side.
    """
    problem = ProblemDetails(
        title="Transport error",
        status=response.status_code if response.status_code else 599,
        detail=detail,
        code=ErrorCode.TRANSPORT_UNREACHABLE,
    )
    return TransportError(problem)


__all__ = [
    "APIException",
    "TransportError",
    "parse_problem_details",
]
