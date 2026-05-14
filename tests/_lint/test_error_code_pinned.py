"""Sentinel: every concrete :class:`RyotenkAIError` subclass pins ``code`` + ``status``.

Phase A1 (sharded-stargazing-wigderson, 2026-05-14) introduces the
unified exception root. The boundary protocol (RFC 9457 problem+json)
relies on class-level ``code`` and ``status`` ClassVars so handlers
can call :meth:`RyotenkAIError.as_problem` without per-instance setup.

The sentinel enforces:

1. Every concrete (raisable) subclass of :class:`RyotenkAIError` has
   ``"code" in cls.__dict__`` AND ``"status" in cls.__dict__`` (i.e.
   pinned at class level, not inherited).
2. The :class:`ErrorCode` of every Phase-A1-new code is mapped to a
   real entry in ``_DEFAULT_TITLES`` (no enum-value fallback).

The three abstract markers (:class:`RyotenkAIError`,
:class:`DomainError`, :class:`InfrastructureError`) are exempt -- they
are NOT raisable; they pin defaults only.

:class:`InternalError` and :class:`TransportError` ARE concrete (callers
raise them directly when no more-specific subclass applies) so they DO
need to pin both.
"""

from __future__ import annotations

from ryotenkai_shared.contracts.problem_details import ErrorCode
from ryotenkai_shared.errors import (
    DomainError,
    InfrastructureError,
    RyotenkAIError,
)
from ryotenkai_shared.errors._render import _DEFAULT_TITLES, default_title_for

# Abstract markers (NOT raisable directly).
_ABSTRACT_MARKERS: frozenset[type] = frozenset({
    RyotenkAIError,
    DomainError,
    InfrastructureError,
})


# Codes introduced by Phase A1 -- must have a real title (not fallback).
# Pod-runner codes (JOB_NOT_FOUND etc.) keep their titles in the pod's
# own ``_DEFAULT_TITLES`` until Phase B unifies. INTERNAL_ERROR and
# TRANSPORT_UNREACHABLE are seeded in the new module too (they're
# needed by the construction path of the new concrete classes).
_PHASE_A1_NEW_CODES: frozenset[ErrorCode] = frozenset({
    ErrorCode.CONFIG_INVALID,
    ErrorCode.CONFIG_DRIFT,
    ErrorCode.CONFIG_FILE_NOT_FOUND,
    ErrorCode.PROJECT_NOT_FOUND,
    ErrorCode.PROJECT_ALREADY_EXISTS,
    ErrorCode.PROVIDER_NOT_FOUND,
    ErrorCode.INTEGRATION_NOT_FOUND,
    ErrorCode.WORKSPACE_STORE_FAILED,
    ErrorCode.STATE_LOAD_FAILED,
    ErrorCode.STATE_LOCKED,
    ErrorCode.LAUNCH_IN_PROGRESS,
    ErrorCode.LAUNCH_PREPARATION_FAILED,
    ErrorCode.PIPELINE_STAGE_FAILED,
    ErrorCode.RUN_IS_ACTIVE,
    ErrorCode.TRAINING_FAILED,
    ErrorCode.TRAINING_OOM,
    ErrorCode.DATASET_LOAD_FAILED,
    ErrorCode.DATASET_VALIDATION_FAILED,
    ErrorCode.MODEL_LOAD_FAILED,
    ErrorCode.INFERENCE_UNAVAILABLE,
    ErrorCode.PROVIDER_UNAVAILABLE,
    ErrorCode.PROVIDER_RATE_LIMITED,
    ErrorCode.PROVIDER_AUTH_FAILED,
    ErrorCode.SSH_CONNECTION_FAILED,
    ErrorCode.SSH_EXEC_FAILED,
    ErrorCode.SSH_TRANSFER_FAILED,
    ErrorCode.HF_AUTH_FAILED,
    ErrorCode.HF_NOT_FOUND,
    ErrorCode.ENGINE_NOT_REGISTERED,
    ErrorCode.ENGINE_CONFIG_INVALID,
})


def _walk_subclasses(root: type) -> set[type]:
    """Return ``root`` plus every direct/transitive subclass."""
    seen: set[type] = set()
    stack: list[type] = [root]
    while stack:
        cls = stack.pop()
        if cls in seen:
            continue
        seen.add(cls)
        stack.extend(cls.__subclasses__())
    return seen


def _concrete_subclasses() -> list[type]:
    """All concrete (non-abstract-marker) subclasses of RyotenkAIError."""
    # Touch the public ``errors`` module so all subclasses are imported
    # before we walk __subclasses__.
    import ryotenkai_shared.errors  # noqa: F401  -- side-effect import

    all_subs = _walk_subclasses(RyotenkAIError)
    return sorted(
        (c for c in all_subs if c not in _ABSTRACT_MARKERS),
        key=lambda c: c.__name__,
    )


def test_concrete_subclasses_pin_code_and_status_at_class_level() -> None:
    """Every concrete subclass MUST define ``code``/``status`` in its own ``__dict__``."""
    failures: list[str] = []
    for cls in _concrete_subclasses():
        own = vars(cls)
        if "code" not in own:
            failures.append(f"{cls.__module__}.{cls.__name__} missing class-level `code`")
        if "status" not in own:
            failures.append(f"{cls.__module__}.{cls.__name__} missing class-level `status`")
    assert not failures, (
        "Concrete RyotenkAIError subclasses must pin `code` and `status` "
        "as ClassVars at class level. Inheritance is not enough -- the "
        "factory and the wire-rendering logic introspect `vars(cls)`.\n"
        "Offenders:\n  " + "\n  ".join(failures)
    )


def test_concrete_subclass_codes_unique() -> None:
    """No two concrete subclasses share the same ``ErrorCode``.

    The factory registry (``_factory.py::_REGISTRY``) keys on
    ``cls.code``; duplicates would silently collapse to last-writer
    and break ``from_problem`` round-trips.
    """
    seen: dict[ErrorCode, str] = {}
    duplicates: list[str] = []
    for cls in _concrete_subclasses():
        key = cls.code
        if key in seen:
            duplicates.append(f"{key.value}: {seen[key]} vs {cls.__module__}.{cls.__name__}")
        else:
            seen[key] = f"{cls.__module__}.{cls.__name__}"
    assert not duplicates, "Duplicate ErrorCode pins:\n  " + "\n  ".join(duplicates)


def test_concrete_subclass_status_in_http_range() -> None:
    """Status codes must be valid HTTP / extended (100..599)."""
    failures: list[str] = []
    for cls in _concrete_subclasses():
        if not (100 <= cls.status <= 599):
            failures.append(f"{cls.__module__}.{cls.__name__}.status={cls.status}")
    assert not failures, (
        "Status must be a valid HTTP status code (100..599). "
        "Offenders:\n  " + "\n  ".join(failures)
    )


def test_phase_a1_codes_have_registered_titles() -> None:
    """Every Phase-A1-new ``ErrorCode`` has a real title (not the fallback).

    Pod-runner codes that pre-existed (JOB_NOT_FOUND, etc.) are exempt
    -- their titles live in ``packages/pod/.../runner/api/errors.py``
    until Phase B unifies the registries.
    """
    missing: list[str] = []
    for code in _PHASE_A1_NEW_CODES:
        if code not in _DEFAULT_TITLES:
            missing.append(code.value)
            continue
        title = _DEFAULT_TITLES[code]
        if title == code.value:
            # Fallback collision -- title equals the enum value, which
            # means whoever added the entry just typed the name again.
            # That's strictly worse than no title at all; force a real
            # human-readable string.
            missing.append(f"{code.value}: title is identical to enum value (no human-readable text)")
    assert not missing, (
        "Every Phase A1 ErrorCode must have a registered human-readable "
        "title in ``_DEFAULT_TITLES``. Missing/fallback:\n  "
        + "\n  ".join(missing)
    )


def test_default_title_for_returns_registered_title() -> None:
    """``default_title_for(code)`` returns the registered title verbatim."""
    for code, expected in _DEFAULT_TITLES.items():
        assert default_title_for(code) == expected, (
            f"default_title_for({code!r}) did not return registered title"
        )


def test_default_title_for_falls_back_to_enum_value_for_unregistered() -> None:
    """An unregistered ``ErrorCode`` falls back to the enum value (no exception)."""
    # JOB_NOT_FOUND is intentionally not in the Phase A1 _DEFAULT_TITLES
    # (it lives in pod's map). Verify the fallback path.
    fallback = default_title_for(ErrorCode.JOB_NOT_FOUND)
    assert fallback == ErrorCode.JOB_NOT_FOUND.value


def test_concrete_subclass_count_matches_phase_a1_catalog() -> None:
    """Phase A1 ships exactly the 32 concrete subclasses listed in the plan.

    Regression test: makes adding a NEW subclass in a later phase
    (which is fine) visible by failing this test until the constant
    below is bumped. That forces a conscious update to the catalog.
    """
    expected_phase_a1_concrete_count = 34  # 20 domain + 12 infra + InternalError + TransportError
    actual = len(_concrete_subclasses())
    assert actual == expected_phase_a1_concrete_count, (
        f"Phase A1 ships {expected_phase_a1_concrete_count} concrete subclasses; "
        f"got {actual}. If you added or removed a subclass in this phase, "
        "bump the expected count here and update the plan."
    )
