"""Root exception types for the unified error model.

``RyotenkAIError`` is the only base for in-repo raised exceptions.
Concrete subclasses live in :mod:`.domain` (4xx) and :mod:`.infra` (5xx).

Key properties:

* ``code: ClassVar[ErrorCode]`` -- machine-readable identifier; pinned
  per subclass; sentinel-enforced.
* ``status: ClassVar[int]`` -- HTTP-suggested status; pinned per subclass.
* ``title``: property reading from :data:`_DEFAULT_TITLES` (override via
  ``title_default`` ClassVar if a subclass wants a custom title).
* ``detail: str | None`` -- occurrence-specific human explanation; never
  contains a traceback (sentinel-enforced).
* ``context: dict[str, Any]`` -- structured per-occurrence metadata; MUST
  NOT contain ``traceback.format_exc()`` output.
* ``cause: Exception | None`` -- chained original exception (stored via
  ``__cause__`` for native Python traceback chaining).

The hierarchy is wired to RFC 9457 problem+json via
:meth:`as_problem`. Conversion from a parsed ``ProblemDetails`` (Mac-side
client) uses the :meth:`from_problem` factory.
"""

from __future__ import annotations

from typing import Any, ClassVar

from ryotenkai_shared.contracts.problem_details import (
    ErrorCode,
    FieldError,
    ProblemDetails,
)
from ryotenkai_shared.errors._render import default_title_for


class RyotenkAIError(Exception):
    """Single root for every typed exception in the RyotenkAI monorepo.

    Subclass instead of raising directly: concrete subclasses pin
    ``code``/``status`` as :class:`ClassVar` so the boundary protocol
    (RFC 9457 problem+json) can be produced via :meth:`as_problem`
    without ad-hoc construction at raise sites.
    """

    code: ClassVar[ErrorCode] = ErrorCode.INTERNAL_ERROR
    status: ClassVar[int] = 500
    title_default: ClassVar[str | None] = None

    def __init__(
        self,
        detail: str | None = None,
        *,
        context: dict[str, Any] | None = None,
        cause: Exception | None = None,
        errors: list[FieldError] | None = None,
    ) -> None:
        if context is not None and not isinstance(context, dict):
            raise TypeError(
                f"context must be a dict or None, got {type(context).__name__}"
            )
        if cause is not None and not isinstance(cause, BaseException):
            raise TypeError(
                f"cause must be an Exception or None, got {type(cause).__name__}"
            )
        super().__init__(f"{self.code.value}: {detail or self.title}")
        self.detail = detail
        self.context: dict[str, Any] = dict(context) if context else {}
        self.errors = errors
        if cause is not None:
            self.__cause__ = cause

    @property
    def title(self) -> str:
        """Short human-readable title (per RFC 9457 §3, stable per code)."""
        return self.title_default or default_title_for(self.code)

    def as_problem(
        self,
        *,
        instance: str | None = None,
        trace_id: str | None = None,
        request_id: str | None = None,
    ) -> ProblemDetails:
        """Render this exception as an RFC 9457 ``ProblemDetails`` body.

        Pure transformation -- never raises (so error handlers can call
        it without try/except wrapping). ``instance``/``trace_id``/
        ``request_id`` are passed through verbatim; null fields are
        stripped on the wire by callers via
        ``model_dump(exclude_none=True)`` per RFC 9457 §3.1.
        """
        return ProblemDetails(
            title=self.title,
            status=self.status,
            detail=self.detail,
            instance=instance,
            code=self.code,
            trace_id=trace_id,
            request_id=request_id,
            errors=self.errors,
        )

    @classmethod
    def from_problem(cls, problem: ProblemDetails) -> "RyotenkAIError":
        """Reconstruct a typed exception from a parsed ProblemDetails.

        Used by the Mac-side client to convert wire payloads into
        ``raise``. The returned exception's class is selected from a
        registry keyed on :class:`ErrorCode`; unknown codes fall back
        to :class:`InternalError` so callers can still ``isinstance``
        against :class:`RyotenkAIError`.
        """
        # Local import to avoid a cycle: _factory imports the concrete
        # subclasses from this module's sibling files.
        from ryotenkai_shared.errors._factory import code_to_class

        target_cls = code_to_class(problem.code)
        inst = target_cls(detail=problem.detail, context={})
        # Mirror wire-side state for round-trip fidelity. Stored on the
        # instance (not class) because the class is shared across
        # occurrences and trace_id/request_id are per-occurrence.
        inst._trace_id = problem.trace_id  # type: ignore[attr-defined]
        inst._request_id = problem.request_id  # type: ignore[attr-defined]
        if problem.errors:
            inst.errors = list(problem.errors)
        return inst

    @property
    def trace_id(self) -> str | None:
        """Server-side correlation id, populated by :meth:`from_problem`."""
        return getattr(self, "_trace_id", None)

    @property
    def request_id(self) -> str | None:
        """Per-request id (set by middleware), populated by :meth:`from_problem`."""
        return getattr(self, "_request_id", None)


class DomainError(RyotenkAIError):
    """4xx semantics -- caller's fault, recoverable, surface to user.

    Abstract marker -- do not raise directly. Subclass instead.
    """


class InfrastructureError(RyotenkAIError):
    """5xx semantics -- external/transient/bug; retry-or-report.

    Abstract marker -- do not raise directly. Subclass instead.
    """


class InternalError(RyotenkAIError):
    """500 catch-all for unknown / unhandled defects.

    Concrete (raisable) -- used by the factory when the wire payload
    carries a code not in the local registry, and by handlers for the
    "unhandled ``Exception``" case.
    """

    code: ClassVar[ErrorCode] = ErrorCode.INTERNAL_ERROR
    status: ClassVar[int] = 500


class TransportError(InfrastructureError):
    """Tunnel-itself failure (Mac client cannot reach the runner).

    Synthesised by ``parse_problem_details`` when the response is not
    ``application/problem+json``. The carried problem details are
    fabricated on the Mac side; ``code`` is always
    ``ErrorCode.TRANSPORT_UNREACHABLE``.
    """

    code: ClassVar[ErrorCode] = ErrorCode.TRANSPORT_UNREACHABLE
    status: ClassVar[int] = 599


__all__ = [
    "DomainError",
    "InfrastructureError",
    "InternalError",
    "RyotenkAIError",
    "TransportError",
]
