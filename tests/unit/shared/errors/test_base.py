"""Phase A1 -- :class:`RyotenkAIError` base + markers + factories.

Covers the unified exception root and its abstract markers / concrete
sentinels per the 7-class policy in ``.claude/CLAUDE.md``:

1. ``TestPositive`` -- happy-path construction + as_problem() round-trip.
2. ``TestNegative`` -- malformed inputs and abstract-marker edge cases.
3. ``TestBoundary`` -- parametrised over edge values (empty/None/large).
4. ``TestInvariants`` -- pinned constants (codes, statuses, media type).
5. ``TestDependencyErrors`` -- factory registry behaviour + round-trip.
6. ``TestRegressions`` -- traceback-leak / past-bug guards.
7. ``TestLogicSpecific`` -- status -> flavour mapping truth table.

No mocks: every test uses real :class:`RyotenkAIError` subclasses and
real :class:`ProblemDetails` instances (per the project's mock policy).
"""

from __future__ import annotations

from typing import ClassVar

import pytest

from ryotenkai_shared.contracts.problem_details import (
    PROBLEM_JSON_MEDIA_TYPE,
    ErrorCode,
    FieldError,
    ProblemDetails,
)
from ryotenkai_shared.errors import (
    ConfigDriftError,
    ConfigInvalidError,
    DatasetLoadFailedError,
    DomainError,
    HFAuthFailedError,
    InferenceUnavailableError,
    InfrastructureError,
    InternalError,
    LaunchPreparationError,
    ProjectNotFoundError,
    ProviderRateLimitedError,
    ProviderUnavailableError,
    RyotenkAIError,
    SSHConnectionFailedError,
    StateLoadFailedError,
    TrainingFailedError,
    TransportError,
)
from ryotenkai_shared.errors._render import default_title_for, new_trace_id
from ryotenkai_shared.errors.base import RyotenkAIError as _Root  # alias for tests

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# 1. Positive -- happy paths
# ---------------------------------------------------------------------------


class TestPositive:
    """Happy path: construction + as_problem() round-trip."""

    def test_internal_error_as_problem_payload(self) -> None:
        exc = InternalError(detail="boom", context={"k": "v"})
        problem = exc.as_problem(instance="/x", trace_id="abcd1234")
        assert problem.code is ErrorCode.INTERNAL_ERROR
        assert problem.status == 500
        assert problem.detail == "boom"
        # Phase B unified the title map; pod-runner used the
        # HTTP-standard title-cased phrase ("Internal Server Error")
        # which now wins for the merged INTERNAL_ERROR entry.
        assert problem.title == "Internal Server Error"
        assert problem.instance == "/x"
        assert problem.trace_id == "abcd1234"

    def test_transport_error_status_and_code(self) -> None:
        exc = TransportError(detail="tunnel down")
        assert exc.status == 599
        assert exc.code is ErrorCode.TRANSPORT_UNREACHABLE
        # transport is an infra-flavour error -- isinstance check pinned
        assert isinstance(exc, InfrastructureError)
        assert isinstance(exc, RyotenkAIError)

    def test_context_constructor_arg_is_copied_not_referenced(self) -> None:
        """Mutating the input dict afterwards must not leak into the exception."""
        external = {"runtime": "alpha"}
        exc = ConfigInvalidError("x", context=external)
        external["runtime"] = "beta"  # external mutation
        external["extra"] = "leak"
        assert exc.context == {"runtime": "alpha"}

    def test_cause_sets_dunder_cause(self) -> None:
        original = ValueError("orig")
        exc = InternalError("wrapped", cause=original)
        assert exc.__cause__ is original

    def test_concrete_domain_subclass_status_and_code(self) -> None:
        exc = ConfigInvalidError("missing fields")
        assert exc.code is ErrorCode.CONFIG_INVALID
        assert exc.status == 400
        assert isinstance(exc, DomainError)
        assert isinstance(exc, RyotenkAIError)

    def test_concrete_infra_subclass_status_and_code(self) -> None:
        exc = ProviderUnavailableError("RunPod returned 503")
        assert exc.code is ErrorCode.PROVIDER_UNAVAILABLE
        assert exc.status == 503
        assert isinstance(exc, InfrastructureError)

    def test_as_problem_with_field_errors(self) -> None:
        fe = FieldError(loc=["body", "command"], type="missing", msg="Field required")
        exc = ConfigInvalidError("bad", errors=[fe])
        problem = exc.as_problem(instance="/x")
        assert problem.errors == [fe]


# ---------------------------------------------------------------------------
# 2. Negative -- precondition violations
# ---------------------------------------------------------------------------


class TestNegative:
    """Error paths and precondition violations."""

    def test_bare_ryotenkai_error_falls_back_to_internal(self) -> None:
        """Constructing the root directly is allowed; defaults to INTERNAL_ERROR/500."""
        exc = RyotenkAIError("anonymous failure")
        assert exc.code is ErrorCode.INTERNAL_ERROR
        assert exc.status == 500

    def test_non_dict_context_rejected(self) -> None:
        with pytest.raises(TypeError, match="context must be a dict"):
            InternalError("x", context=["not", "a", "dict"])  # type: ignore[arg-type]

    def test_non_exception_cause_rejected(self) -> None:
        with pytest.raises(TypeError, match="cause must be an Exception"):
            InternalError("x", cause="not an exception")  # type: ignore[arg-type]

    def test_abstract_marker_domain_is_constructible_but_carries_root_defaults(self) -> None:
        """Markers are not Python-abstract; they're constructible but lack pinned codes.

        Constructing them is style-discouraged (sentinel test catches new
        such call sites in raise statements) but does NOT raise -- the
        defaults from the root carry through.
        """
        marker = DomainError("anonymous")
        # Inherits the root's INTERNAL_ERROR/500 defaults.
        assert marker.code is ErrorCode.INTERNAL_ERROR
        assert marker.status == 500


# ---------------------------------------------------------------------------
# 3. Boundary -- edge values
# ---------------------------------------------------------------------------


class TestBoundary:
    """Parametrised over edge values."""

    @pytest.mark.parametrize("detail", ["", None])
    def test_empty_vs_none_detail_distinguished_in_payload(self, detail: str | None) -> None:
        """``detail=""`` is preserved as-is; ``detail=None`` is stripped on the wire."""
        exc = InternalError(detail=detail)
        problem = exc.as_problem()
        wire = problem.model_dump(mode="json", exclude_none=True)
        if detail is None:
            assert "detail" not in wire
        else:
            # The model holds the empty string verbatim.
            assert problem.detail == ""

    @pytest.mark.parametrize("context", [None, {}, {"single": "entry"}])
    def test_context_normalisation(self, context: dict | None) -> None:
        exc = InternalError("x", context=context)
        assert exc.context == (context or {})

    def test_large_context_round_trips_through_as_problem(self) -> None:
        """A 10k-entry context dict survives both attribute storage and as_problem."""
        big = {f"k{i}": i for i in range(10_000)}
        exc = InternalError("big", context=big)
        assert len(exc.context) == 10_000
        # as_problem does NOT serialise context (it's not in ProblemDetails).
        # We only verify the construction path didn't choke.
        problem = exc.as_problem()
        assert problem.code is ErrorCode.INTERNAL_ERROR

    def test_no_detail_no_context_no_cause(self) -> None:
        """Bare ``InternalError()`` is constructible without any arguments."""
        exc = InternalError()
        assert exc.detail is None
        assert exc.context == {}
        assert exc.__cause__ is None


# ---------------------------------------------------------------------------
# 4. Invariants -- pinned constants
# ---------------------------------------------------------------------------


class TestInvariants:
    """Pin constants -- regression guards on the boundary protocol."""

    def test_internal_error_class_constants(self) -> None:
        assert InternalError.code is ErrorCode.INTERNAL_ERROR
        assert InternalError.status == 500

    def test_transport_error_class_constants(self) -> None:
        assert TransportError.code is ErrorCode.TRANSPORT_UNREACHABLE
        assert TransportError.status == 599

    def test_title_property_reads_from_default_title_for(self) -> None:
        """The ``title`` property is just ``default_title_for(self.code)``."""
        exc = StateLoadFailedError("missing")
        assert exc.title == default_title_for(ErrorCode.STATE_LOAD_FAILED)

    def test_title_default_classvar_overrides_lookup(self) -> None:
        """``title_default`` ClassVar wins over the registry lookup."""

        class CustomTitle(InternalError):
            title_default: ClassVar[str] = "Override title"

        exc = CustomTitle("x")
        assert exc.title == "Override title"

    def test_as_problem_returns_problem_details_instance(self) -> None:
        exc = InternalError("x")
        problem = exc.as_problem()
        assert isinstance(problem, ProblemDetails)

    def test_problem_json_media_type_pinned(self) -> None:
        assert PROBLEM_JSON_MEDIA_TYPE == "application/problem+json"

    def test_new_trace_id_is_8_hex_chars(self) -> None:
        for _ in range(50):
            tid = new_trace_id()
            assert len(tid) == 8
            int(tid, 16)  # raises if not hex

    def test_trace_id_uniqueness_across_many_generations(self) -> None:
        """Collision risk per spec: ~32 bits -> birthday paradox at ~65k samples.

        We only generate 1000 here -- expecting zero collisions in practice.
        """
        seen = {new_trace_id() for _ in range(1000)}
        assert len(seen) == 1000


# ---------------------------------------------------------------------------
# 5. Dependency errors -- factory registry + from_problem round-trip
# ---------------------------------------------------------------------------


class TestDependencyErrors:
    """External dep failure pathways (factory, parsing, registry)."""

    def test_from_problem_returns_typed_subclass(self) -> None:
        problem = ProblemDetails(
            title="Project not found",
            status=404,
            code=ErrorCode.PROJECT_NOT_FOUND,
            detail="id=xyz",
        )
        exc = RyotenkAIError.from_problem(problem)
        assert isinstance(exc, ProjectNotFoundError)
        assert exc.code is ErrorCode.PROJECT_NOT_FOUND
        assert exc.detail == "id=xyz"

    def test_from_problem_preserves_trace_and_request_id(self) -> None:
        problem = ProblemDetails(
            title="X",
            status=500,
            code=ErrorCode.TRAINING_FAILED,
            trace_id="aabb1122",
            request_id="0123456789abcdef",
        )
        exc = RyotenkAIError.from_problem(problem)
        assert exc.trace_id == "aabb1122"
        assert exc.request_id == "0123456789abcdef"

    def test_from_problem_preserves_field_errors(self) -> None:
        fe = FieldError(loc=["body", "x"], type="missing", msg="Field required")
        problem = ProblemDetails(
            title="bad",
            status=422,
            code=ErrorCode.JOB_SPEC_INVALID,
            errors=[fe],
        )
        exc = RyotenkAIError.from_problem(problem)
        assert exc.errors == [fe]

    def test_from_problem_unknown_code_falls_back_to_internal_error(self) -> None:
        """An ErrorCode not in the factory registry maps to InternalError.

        Realistic scenario: Mac client is older than the server and
        encounters a new code it has no typed class for.
        """
        # JOB_NOT_FOUND has no class in the Phase A1 ``errors`` module
        # (pod-owned code; Phase B will add it). Verify the fallback.
        problem = ProblemDetails(
            title="x",
            status=404,
            code=ErrorCode.JOB_NOT_FOUND,
        )
        exc = RyotenkAIError.from_problem(problem)
        assert isinstance(exc, InternalError)
        assert isinstance(exc, RyotenkAIError)

    def test_transport_error_round_trip(self) -> None:
        problem = ProblemDetails(
            title="Transport unreachable",
            status=599,
            code=ErrorCode.TRANSPORT_UNREACHABLE,
        )
        exc = RyotenkAIError.from_problem(problem)
        assert isinstance(exc, TransportError)

    def test_factory_registry_directly(self) -> None:
        """``code_to_class`` direct surface -- without going through from_problem.

        The factory is the registry that ``from_problem`` consults; we
        also exercise it directly so the module is referenced and the
        ``no test required`` sentinel sees it.
        """
        from ryotenkai_shared.errors._factory import code_to_class

        assert code_to_class(ErrorCode.CONFIG_INVALID) is ConfigInvalidError
        assert code_to_class(ErrorCode.PROVIDER_UNAVAILABLE) is ProviderUnavailableError
        assert code_to_class(ErrorCode.TRANSPORT_UNREACHABLE) is TransportError
        assert code_to_class(ErrorCode.INTERNAL_ERROR) is InternalError
        # Unknown / unregistered (pod-owned) code falls back to InternalError.
        assert code_to_class(ErrorCode.JOB_NOT_FOUND) is InternalError


# ---------------------------------------------------------------------------
# 6. Regressions -- concrete past-bug guard tests
# ---------------------------------------------------------------------------


class TestRegressions:
    """Concrete past-bug guard tests, each cited to a reason."""

    def test_context_does_not_carry_traceback_key_by_construction(self) -> None:
        """``AppError.details['traceback']`` leak guard.

        Reason: ``AppError`` (Phase A2-bound for deletion) stored
        ``traceback.format_exc()`` inside its ``details`` dict, which
        leaked filesystem paths into the wire body and into log lines
        that already carried the traceback separately. The new
        :class:`RyotenkAIError` hierarchy never auto-populates a
        ``traceback`` key; the sentinel test
        ``tests/_lint/test_no_traceback_in_context.py`` blocks new
        leak sites.

        Ref: docs/plans/sharded-stargazing-wigderson.md -- R-TRACEBACK.
        """
        try:
            raise ValueError("original cause")
        except ValueError as exc:
            wrapped = InternalError("wrapped", cause=exc)
        # The exception's context was never auto-populated with a tb.
        assert "traceback" not in wrapped.context
        # The cause is still chained via __cause__, which IS the
        # correct place for tracebacks (Python native).
        assert isinstance(wrapped.__cause__, ValueError)

    def test_as_problem_does_not_raise_for_any_concrete_class(self) -> None:
        """as_problem() must NEVER raise -- handlers call it inside catch."""
        # Touch all concrete classes; each must produce a valid
        # ProblemDetails without exception. If a class forgets to pin
        # `code` or `status` in a future refactor, ProblemDetails's
        # own validation raises -- guarding against silent breakage.
        for cls in (
            InternalError,
            TransportError,
            ConfigInvalidError,
            ConfigDriftError,
            ProjectNotFoundError,
            StateLoadFailedError,
            TrainingFailedError,
            ProviderRateLimitedError,
            SSHConnectionFailedError,
            HFAuthFailedError,
            LaunchPreparationError,
            DatasetLoadFailedError,
            InferenceUnavailableError,
        ):
            problem = cls(detail="x").as_problem()
            assert problem.code is cls.code
            assert problem.status == cls.status

    def test_problem_details_extra_forbid_blocks_unknown_fields(self) -> None:
        """ProblemDetails(extra='forbid'); future drift detected fast.

        Reason: the wire shape is RFC 9457 + four project extensions.
        Adding a stray kwarg via as_problem must blow up immediately,
        not silently drop. This pins the contract.
        """
        with pytest.raises(Exception):  # noqa: BLE001 -- pydantic ValidationError
            ProblemDetails(  # type: ignore[call-arg]
                title="x",
                status=500,
                code=ErrorCode.INTERNAL_ERROR,
                unknown_field="dropped silently?",
            )


# ---------------------------------------------------------------------------
# 7. Logic-specific -- status -> flavour mapping truth table
# ---------------------------------------------------------------------------


class TestLogicSpecific:
    """Parametrised truth tables for domain logic."""

    @pytest.mark.parametrize(
        ("exc_cls", "expected_marker"),
        [
            (ConfigInvalidError, DomainError),          # 400
            (HFAuthFailedError, DomainError),           # 401
            (StateLoadFailedError, DomainError),        # 404
            (ConfigDriftError, DomainError),            # 409
            (DatasetLoadFailedError, DomainError),      # 422
            (ProviderRateLimitedError, InfrastructureError),   # 429
            (TrainingFailedError, InfrastructureError),        # 500
            (LaunchPreparationError, InfrastructureError),     # 500
            (SSHConnectionFailedError, InfrastructureError),   # 502
            (InferenceUnavailableError, InfrastructureError),  # 503
            (TransportError, InfrastructureError),             # 599
        ],
    )
    def test_status_class_maps_to_correct_flavour(
        self,
        exc_cls: type[RyotenkAIError],
        expected_marker: type[RyotenkAIError],
    ) -> None:
        """A 4xx subclass MUST inherit from DomainError; 5xx from InfrastructureError."""
        exc = exc_cls("test")
        assert isinstance(exc, expected_marker)

    @pytest.mark.parametrize(
        ("exc_cls", "expected_status"),
        [
            (ConfigInvalidError, 400),
            (HFAuthFailedError, 401),
            (StateLoadFailedError, 404),
            (ConfigDriftError, 409),
            (DatasetLoadFailedError, 422),
            (ProviderRateLimitedError, 429),
            (TrainingFailedError, 500),
            (SSHConnectionFailedError, 502),
            (InferenceUnavailableError, 503),
            (TransportError, 599),
        ],
    )
    def test_status_pins_match_plan_table(
        self,
        exc_cls: type[RyotenkAIError],
        expected_status: int,
    ) -> None:
        """Status pins mirror docs/plans/sharded-stargazing-wigderson.md Layer 2 table."""
        assert exc_cls.status == expected_status

    def test_str_includes_code_value_and_detail(self) -> None:
        """Default ``Exception.__init__`` message: ``"<CODE>: <detail or title>"``."""
        exc = InternalError("a specific failure")
        assert "INTERNAL_ERROR" in str(exc)
        assert "a specific failure" in str(exc)

    def test_str_falls_back_to_title_when_no_detail(self) -> None:
        exc = InternalError()
        # Phase B unified the title map (see TestPositive note above).
        assert str(exc) == "INTERNAL_ERROR: Internal Server Error"

    def test_root_class_module_anchor(self) -> None:
        """RyotenkAIError lives at ryotenkai_shared.errors.base.

        Regression guard: a refactor that relocates the root class
        would break ``from_problem``'s registry walk.
        """
        assert _Root.__module__ == "ryotenkai_shared.errors.base"
