"""Registry mapping :class:`ErrorCode` to its concrete typed-exception class.

Used by :meth:`RyotenkAIError.from_problem` to reconstruct a typed
exception from a Mac-side parsed ``ProblemDetails``. Unknown codes
fall back to :class:`InternalError` so wire-shape mismatches degrade
gracefully (callers can still ``isinstance`` against
:class:`RyotenkAIError`).
"""

from __future__ import annotations

from ryotenkai_shared.contracts.problem_details import ErrorCode
from ryotenkai_shared.errors import domain as _domain
from ryotenkai_shared.errors import infra as _infra
from ryotenkai_shared.errors.base import (
    InternalError,
    RyotenkAIError,
    TransportError,
)


def _build_registry() -> dict[ErrorCode, type[RyotenkAIError]]:
    """Walk ``domain``/``infra`` modules and harvest concrete subclasses.

    Iterates ``__all__`` (falling back to ``dir()``) so anything listed
    in the module's public API and inheriting from
    :class:`RyotenkAIError` lands in the registry keyed on its
    class-level ``code``.

    Concrete-by-design classes from ``base`` (TransportError,
    InternalError) are added explicitly to guarantee they win even if
    a sibling module shadowed them.
    """
    out: dict[ErrorCode, type[RyotenkAIError]] = {}
    for module in (_domain, _infra):
        for name in getattr(module, "__all__", dir(module)):
            cls = getattr(module, name, None)
            if isinstance(cls, type) and issubclass(cls, RyotenkAIError):
                # Last-writer-wins within the module's own __all__; we
                # don't have duplicate codes within either file in
                # Phase A1, so this is well-defined.
                out[cls.code] = cls
    # Pin the base-module concretes after harvest (so they always win).
    out[ErrorCode.TRANSPORT_UNREACHABLE] = TransportError
    out[ErrorCode.INTERNAL_ERROR] = InternalError
    return out


_REGISTRY: dict[ErrorCode, type[RyotenkAIError]] = _build_registry()


def code_to_class(code: ErrorCode) -> type[RyotenkAIError]:
    """Return the registered class for ``code`` or :class:`InternalError`.

    The fallback is intentional: a wire payload may carry an
    :class:`ErrorCode` value that this process doesn't have a typed
    subclass for (e.g. older Mac client, newer server). Returning
    :class:`InternalError` lets callers raise/catch via the
    :class:`RyotenkAIError` hierarchy without crashing.
    """
    return _REGISTRY.get(code, InternalError)


__all__ = ["code_to_class"]
