"""Phase 14.B — :mod:`src.runner.runtime.lifecycle_client` contract.

Pure-stdlib unit tests for the new types introduced by Phase 14.B:

* :class:`LifecycleActionResult` — frozen dataclass, post-init
  truncation invariant.
* :class:`IPodLifecycleClient` — runtime_checkable Protocol.

Slim-venv compatible — no provider impls or transports exercised
here (those live in their own test files).
"""

from __future__ import annotations

import pytest

from ryotenkai_shared.infrastructure.lifecycle import (
    IPodLifecycleClient,
    LifecycleActionResult,
    PodTerminalOutcome,
)

# ---------------------------------------------------------------------------
# 1. Positive — happy-path construction + values
# ---------------------------------------------------------------------------


class TestLifecycleActionResultPositive:
    def test_minimal_construction(self) -> None:
        result = LifecycleActionResult(outcome="terminated", attempts_made=1)
        assert result.outcome == "terminated"
        assert result.attempts_made == 1
        assert result.last_error is None
        assert result.raw_response_excerpt is None

    def test_full_construction(self) -> None:
        result = LifecycleActionResult(
            outcome="failed",
            attempts_made=3,
            last_error="HTTPError 503",
            raw_response_excerpt="<html>service unavailable</html>",
        )
        assert result.outcome == "failed"
        assert result.attempts_made == 3
        assert result.last_error == "HTTPError 503"
        assert result.raw_response_excerpt == "<html>service unavailable</html>"


# ---------------------------------------------------------------------------
# 2. Negative — frozen dataclass rejects mutation
# ---------------------------------------------------------------------------


class TestLifecycleActionResultNegative:
    def test_frozen_attribute_assignment_rejected(self) -> None:
        result = LifecycleActionResult(outcome="terminated", attempts_made=1)
        with pytest.raises(Exception):  # FrozenInstanceError
            result.outcome = "failed"  # type: ignore[misc]

    def test_runtime_does_not_validate_outcome_string(self) -> None:
        # Pin: dataclass doesn't constrain ``outcome`` to a fixed set
        # at runtime — that's the type checker's job at the callsite.
        # Phase 14.B § 1.2 documents the choice (outcome is a closed
        # vocabulary at the design level, but resume's "resumed" /
        # "already_running" are NOT on PodTerminalOutcome enum).
        result = LifecycleActionResult(outcome="not_a_real_outcome", attempts_made=1)
        assert result.outcome == "not_a_real_outcome"


# ---------------------------------------------------------------------------
# 3. Boundary — truncation cap, attempts_made edge values
# ---------------------------------------------------------------------------


class TestLifecycleActionResultBoundary:
    def test_raw_response_excerpt_truncated_at_300_chars(self) -> None:
        # The cap is silent: longer payloads get truncated, not raised.
        big_body = "x" * 1000
        result = LifecycleActionResult(
            outcome="failed",
            attempts_made=1,
            raw_response_excerpt=big_body,
        )
        assert result.raw_response_excerpt is not None
        assert len(result.raw_response_excerpt) == 300

    def test_raw_response_excerpt_at_cap_unchanged(self) -> None:
        # Exactly 300 chars: no truncation, byte-identical.
        body = "y" * 300
        result = LifecycleActionResult(
            outcome="failed",
            attempts_made=1,
            raw_response_excerpt=body,
        )
        assert result.raw_response_excerpt == body
        assert len(result.raw_response_excerpt) == 300

    def test_raw_response_excerpt_below_cap_unchanged(self) -> None:
        body = "z" * 50
        result = LifecycleActionResult(
            outcome="failed",
            attempts_made=1,
            raw_response_excerpt=body,
        )
        assert result.raw_response_excerpt == body

    def test_attempts_made_zero_for_noop(self) -> None:
        # NoOp clients return ``attempts_made=0`` because there's no
        # transport to attempt against.
        result = LifecycleActionResult(outcome="skipped", attempts_made=0)
        assert result.attempts_made == 0


# ---------------------------------------------------------------------------
# 4. Invariants — Protocol conformance
# ---------------------------------------------------------------------------


class TestProtocolInvariants:
    def test_runtime_checkable_recognises_full_impl(self) -> None:
        # A class with all four methods (plus the property) matches
        # at runtime — no explicit inheritance needed.
        class _Stub:
            @property
            def provider_name(self) -> str:
                return "stub"

            async def terminate(self, *, resource_id: str) -> LifecycleActionResult:
                return LifecycleActionResult(outcome="terminated", attempts_made=1)

            async def pause(self, *, resource_id: str) -> LifecycleActionResult:
                return LifecycleActionResult(outcome="stopped", attempts_made=1)

            async def resume(self, *, resource_id: str) -> LifecycleActionResult:
                return LifecycleActionResult(outcome="resumed", attempts_made=1)

        assert isinstance(_Stub(), IPodLifecycleClient)

    def test_runtime_checkable_rejects_missing_method(self) -> None:
        class _Partial:
            @property
            def provider_name(self) -> str:
                return "partial"

            async def terminate(self, *, resource_id: str) -> LifecycleActionResult:
                return LifecycleActionResult(outcome="terminated", attempts_made=1)

            async def pause(self, *, resource_id: str) -> LifecycleActionResult:
                return LifecycleActionResult(outcome="stopped", attempts_made=1)

            # Note: runtime_checkable Protocol only checks method NAMES,
            # not signatures. ``_Partial`` is missing ``resume`` → fails
            # isinstance.

        assert not isinstance(_Partial(), IPodLifecycleClient)

    def test_runtime_checkable_rejects_missing_property(self) -> None:
        class _NoName:
            async def terminate(self, *, resource_id: str) -> LifecycleActionResult:
                return LifecycleActionResult(outcome="terminated", attempts_made=1)

            async def pause(self, *, resource_id: str) -> LifecycleActionResult:
                return LifecycleActionResult(outcome="stopped", attempts_made=1)

            async def resume(self, *, resource_id: str) -> LifecycleActionResult:
                return LifecycleActionResult(outcome="resumed", attempts_made=1)

        # ``provider_name`` missing entirely.
        assert not isinstance(_NoName(), IPodLifecycleClient)


# ---------------------------------------------------------------------------
# 5. Dependency errors — N/A (pure types, no I/O)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 6. Regressions — :class:`PodTerminalOutcome` strings are still valid
# ---------------------------------------------------------------------------


class TestRegressionsPodTerminalOutcomeStrings:
    @pytest.mark.parametrize(
        "outcome_string",
        [
            PodTerminalOutcome.TERMINATED,
            PodTerminalOutcome.ALREADY_TERMINATED,
            PodTerminalOutcome.STOPPED,
            PodTerminalOutcome.ALREADY_STOPPED,
            PodTerminalOutcome.FAILED,
            PodTerminalOutcome.SKIPPED,
        ],
    )
    def test_all_pod_terminal_outcome_strings_round_trip(
        self, outcome_string: str,
    ) -> None:
        # Pin: every existing :class:`PodTerminalOutcome` action-stage
        # string is a valid :class:`LifecycleActionResult.outcome`
        # value. Phase 14.B preserves the vocabulary 1:1.
        result = LifecycleActionResult(outcome=outcome_string, attempts_made=1)
        assert result.outcome == outcome_string


# ---------------------------------------------------------------------------
# 7. Logic-specific — resume's "resumed" / "already_running" outcomes
# ---------------------------------------------------------------------------


class TestLogicSpecificResumeOutcomes:
    @pytest.mark.parametrize("outcome_string", ["resumed", "already_running"])
    def test_resume_outcomes_are_NOT_on_pod_terminal_outcome(
        self, outcome_string: str,
    ) -> None:
        # Phase 14.B § 1.2 + § OQ-2: resume introduces TWO new
        # outcome strings that are deliberately NOT promoted to
        # :class:`PodTerminalOutcome` (the runner doesn't self-resume
        # today, so broadening the public vocabulary would be
        # premature). They live as raw strings on
        # :class:`LifecycleActionResult.outcome` only.
        existing = {
            value
            for name, value in vars(PodTerminalOutcome).items()
            if not name.startswith("_") and isinstance(value, str)
        }
        assert outcome_string not in existing

    @pytest.mark.parametrize("outcome_string", ["resumed", "already_running"])
    def test_resume_outcomes_construct_lifecycle_result(
        self, outcome_string: str,
    ) -> None:
        result = LifecycleActionResult(outcome=outcome_string, attempts_made=1)
        assert result.outcome == outcome_string
