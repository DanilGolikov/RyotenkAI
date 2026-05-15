"""Mac-side parser for runner ``application/problem+json`` responses.

The Mac client (``JobClient``) calls :func:`parse_problem_details` on
every non-2xx response so the caller gets a typed :class:`RyotenkAIError`
(the unified root) instead of a bare ``raise_for_status``-style error.

Contract:

* ``Content-Type: application/problem+json`` -- parse JSON into
  :class:`ProblemDetails`, return a typed :class:`RyotenkAIError`
  subclass via :meth:`RyotenkAIError.from_problem` (factory keyed on
  :class:`ErrorCode`).
* Any other content type or unparseable body -- return
  :class:`TransportError` (a :class:`RyotenkAIError` subclass with
  ``code=TRANSPORT_UNREACHABLE``) so callers can still
  ``isinstance(exc, RyotenkAIError)`` and switch on ``exc.code``
  without special-casing wire-shape failures.

Phase F (sharded-stargazing-wigderson, 2026-05-16) renamed the
return type from ``APIException`` to :class:`RyotenkAIError`. The two
names referred to the same wire-parsed object historically; Phase F
unifies the in-process raise hierarchy and the wire-parsed surface
under a single name so callers do not need to ``isinstance`` against
two unrelated classes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ryotenkai_shared.contracts.problem_details import (
    PROBLEM_JSON_MEDIA_TYPE,
    ErrorCode,
    ProblemDetails,
)
from ryotenkai_shared.errors.base import RyotenkAIError, TransportError

if TYPE_CHECKING:
    import httpx


def parse_problem_details(response: "httpx.Response") -> RyotenkAIError:
    """Build a typed :class:`RyotenkAIError` from an httpx response.

    Behaviour:

    1. ``Content-Type`` starts with ``application/problem+json`` ->
       parse the body, return :class:`RyotenkAIError` (concrete
       subclass picked by :meth:`RyotenkAIError.from_problem`).
    2. Anything else (including missing/empty body) -> return
       :class:`TransportError`. Body is summarised in the synthesised
       ``detail`` field so the caller can still log something useful.

    Note this function **never raises** -- it builds and returns the
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
        except Exception as exc:  # noqa: BLE001 -- defensive, exotic shapes
            return _make_transport_error(
                response,
                detail=f"problem+json failed schema validation: {exc}",
            )
        return RyotenkAIError.from_problem(problem)

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

    Returns a :class:`TransportError` instance with mirrored
    ``trace_id`` / ``request_id`` populated from the round-tripped
    ``ProblemDetails`` (so callers can still grep server logs by the
    correlation id even when the body failed to decode).
    """
    problem = ProblemDetails(
        title="Transport error",
        status=response.status_code if response.status_code else 599,
        detail=detail,
        code=ErrorCode.TRANSPORT_UNREACHABLE,
    )
    # Build via from_problem so .trace_id / .request_id round-trip the
    # same way as a regular runner-issued error.
    exc = RyotenkAIError.from_problem(problem)
    assert isinstance(exc, TransportError), (
        "from_problem must route TRANSPORT_UNREACHABLE to TransportError"
    )
    return exc


__all__ = [
    "TransportError",
    "parse_problem_details",
]
