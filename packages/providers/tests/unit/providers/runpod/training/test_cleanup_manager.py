"""``RunPodCleanupManager`` — bounded-retry pod termination.

Coverage categories: positive, negative, boundary, invariant,
dependency-error, regression (legacy single-attempt parity), logic-
specific (backoff doubling / sleep injection / last-error preservation),
combinatorial (success-on-Nth-attempt matrix).

The injectable ``sleep_fn`` parameter means these tests run instantly —
no actual ``time.sleep`` calls.
"""

from __future__ import annotations

import pytest

from ryotenkai_providers.runpod.training.cleanup_manager import (
    RunPodCleanupManager,
)
from ryotenkai_shared.utils.result import Err, Ok, ProviderError, Result

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeApiClient:
    """Records calls and returns a queue of canned results.

    Each call to ``terminate_pod`` pops the next response. ``responses``
    can contain ``Ok(None)`` for success or ``Err(ProviderError(...))``
    for transient failures.
    """

    def __init__(self, responses: list[Result[None, ProviderError]]) -> None:
        self.responses = list(responses)
        self.calls: list[str] = []

    def terminate_pod(self, pod_id: str) -> Result[None, ProviderError]:
        self.calls.append(pod_id)
        if not self.responses:
            raise AssertionError(
                f"FakeApiClient ran out of responses on call {len(self.calls)} "
                f"(pod_id={pod_id!r})"
            )
        return self.responses.pop(0)


class _RecordingSleeper:
    """``sleep_fn`` substitute that records durations without sleeping."""

    def __init__(self) -> None:
        self.durations: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.durations.append(seconds)


def _err(code: str = "RUNPOD_TRANSIENT", msg: str = "boom") -> Err[ProviderError]:
    return Err(ProviderError(message=msg, code=code))


# ===========================================================================
# Positive — happy paths
# ===========================================================================


class TestPositive:
    def test_first_attempt_succeeds_no_sleep(self) -> None:
        api = _FakeApiClient([Ok(None)])
        sleeper = _RecordingSleeper()
        mgr = RunPodCleanupManager(api, sleep_fn=sleeper)

        result = mgr.cleanup_pod("pod_abc")

        assert result.is_ok()
        assert api.calls == ["pod_abc"]
        # Success on first try — no backoff sleeps.
        assert sleeper.durations == []

    def test_succeeds_on_second_attempt(self) -> None:
        api = _FakeApiClient([_err("NET"), Ok(None)])
        sleeper = _RecordingSleeper()
        mgr = RunPodCleanupManager(api, sleep_fn=sleeper)

        result = mgr.cleanup_pod("pod_abc")

        assert result.is_ok()
        assert api.calls == ["pod_abc", "pod_abc"]
        # One sleep between attempts 1 and 2.
        assert sleeper.durations == [2.0]

    def test_succeeds_on_third_attempt(self) -> None:
        api = _FakeApiClient([_err("NET"), _err("5XX"), Ok(None)])
        sleeper = _RecordingSleeper()
        mgr = RunPodCleanupManager(api, sleep_fn=sleeper)

        result = mgr.cleanup_pod("pod_abc")

        assert result.is_ok()
        assert api.calls == ["pod_abc"] * 3
        # Two sleeps with exponential backoff: 2s then 4s.
        assert sleeper.durations == [2.0, 4.0]


# ===========================================================================
# Negative — exhaustion
# ===========================================================================


class TestNegative:
    def test_all_attempts_fail_returns_last_error(self) -> None:
        api = _FakeApiClient([
            _err("FIRST"),
            _err("SECOND"),
            _err("THIRD_FINAL"),
        ])
        sleeper = _RecordingSleeper()
        mgr = RunPodCleanupManager(api, sleep_fn=sleeper)

        result = mgr.cleanup_pod("pod_abc")

        assert result.is_err()
        # Last error is the most actionable signal — verify it's preserved.
        assert result.unwrap_err().code == "THIRD_FINAL"
        # All 3 attempts ran.
        assert len(api.calls) == 3
        # Sleeps only happen BETWEEN attempts (n-1 of them), not after the last.
        assert sleeper.durations == [2.0, 4.0]

    def test_zero_max_attempts_rejected(self) -> None:
        api = _FakeApiClient([Ok(None)])
        with pytest.raises(ValueError, match="max_attempts"):
            RunPodCleanupManager(api, max_attempts=0)

    def test_negative_initial_backoff_rejected(self) -> None:
        api = _FakeApiClient([Ok(None)])
        with pytest.raises(ValueError, match="initial_backoff"):
            RunPodCleanupManager(api, initial_backoff_s=-1.0)

    def test_max_backoff_below_initial_rejected(self) -> None:
        api = _FakeApiClient([Ok(None)])
        with pytest.raises(ValueError, match="max_backoff"):
            RunPodCleanupManager(api, initial_backoff_s=10.0, max_backoff_s=5.0)


# ===========================================================================
# Boundary — edge configurations
# ===========================================================================


class TestBoundary:
    def test_max_attempts_one_means_no_retry(self) -> None:
        """``max_attempts=1`` is the legacy "one shot" behavior. Useful for
        callers who want different policy (e.g. fast-fail + manual retry)."""
        api = _FakeApiClient([_err("ONLY")])
        sleeper = _RecordingSleeper()
        mgr = RunPodCleanupManager(api, max_attempts=1, sleep_fn=sleeper)

        result = mgr.cleanup_pod("pod_abc")

        assert result.is_err()
        assert result.unwrap_err().code == "ONLY"
        assert len(api.calls) == 1
        # No backoff at all — failed first and only attempt.
        assert sleeper.durations == []

    def test_zero_initial_backoff_skips_sleep(self) -> None:
        """``initial_backoff_s=0`` is valid — useful in tests / fast retries."""
        api = _FakeApiClient([_err("NET"), Ok(None)])
        sleeper = _RecordingSleeper()
        mgr = RunPodCleanupManager(
            api,
            initial_backoff_s=0.0,
            max_backoff_s=0.0,
            sleep_fn=sleeper,
        )

        result = mgr.cleanup_pod("pod_abc")

        assert result.is_ok()
        # Sleep was called but with 0 — still records the call (proves
        # the loop runs the sleep step) but no actual wait.
        assert sleeper.durations == [0.0]

    def test_backoff_capped_at_max(self) -> None:
        """Doubling stops at ``max_backoff_s``."""
        api = _FakeApiClient([_err("a"), _err("b"), _err("c"), _err("d"), Ok(None)])
        sleeper = _RecordingSleeper()
        mgr = RunPodCleanupManager(
            api,
            max_attempts=5,
            initial_backoff_s=10.0,
            max_backoff_s=15.0,  # cap below 2*10 = 20
            sleep_fn=sleeper,
        )

        result = mgr.cleanup_pod("pod_abc")

        assert result.is_ok()
        # Attempts: 1→2 (sleep 10), 2→3 (sleep 15 — capped), 3→4 (sleep 15),
        # 4→5 (sleep 15). Successive doublings 10→20→40→80 all clamp to 15
        # except the first (which is < cap).
        assert sleeper.durations == [10.0, 15.0, 15.0, 15.0]


# ===========================================================================
# Invariants
# ===========================================================================


class TestInvariants:
    def test_pod_id_passed_unchanged_on_every_attempt(self) -> None:
        """Retry MUST NOT mutate the pod_id between attempts (defensive
        against future refactors that try to be 'smart')."""
        api = _FakeApiClient([_err("a"), _err("b"), Ok(None)])
        mgr = RunPodCleanupManager(api, sleep_fn=lambda _: None)

        mgr.cleanup_pod("special-pod-id-with-dashes_42")

        assert api.calls == ["special-pod-id-with-dashes_42"] * 3

    def test_n_minus_one_sleeps_for_n_attempts(self) -> None:
        """Invariant: with ``max_attempts=N``, on full failure exactly
        ``N-1`` sleeps occur — no trailing sleep after the last attempt."""
        for n in (1, 2, 3, 5, 10):
            api = _FakeApiClient([_err(f"E{i}") for i in range(n)])
            sleeper = _RecordingSleeper()
            mgr = RunPodCleanupManager(api, max_attempts=n, sleep_fn=sleeper)

            mgr.cleanup_pod("pod_abc")

            assert len(sleeper.durations) == n - 1, (
                f"max_attempts={n}: expected {n-1} sleeps, got "
                f"{len(sleeper.durations)}"
            )

    def test_no_sleep_after_success(self) -> None:
        """Invariant: as soon as Ok is returned, no further sleep happens."""
        # Ok comes on attempt 2 of allowed 5 — should sleep once (between 1→2),
        # then return immediately without sleeping for attempts 3/4/5.
        api = _FakeApiClient([_err("NET"), Ok(None)])
        sleeper = _RecordingSleeper()
        mgr = RunPodCleanupManager(api, max_attempts=5, sleep_fn=sleeper)

        mgr.cleanup_pod("pod_abc")

        assert len(sleeper.durations) == 1


# ===========================================================================
# Dependency-error
# ===========================================================================


class TestDependencyError:
    def test_api_client_exception_propagates(self) -> None:
        """If the API client raises (rather than returning ``Err``), the
        exception bubbles up — retry only applies to ``Result`` failures.
        Catching arbitrary exceptions would mask real bugs."""

        class _RaisingApi:
            def terminate_pod(self, pod_id: str):  # type: ignore[no-untyped-def]
                raise RuntimeError("transport blew up")

        mgr = RunPodCleanupManager(_RaisingApi(), sleep_fn=lambda _: None)
        with pytest.raises(RuntimeError, match="transport blew up"):
            mgr.cleanup_pod("pod_abc")


# ===========================================================================
# Regression — parity with legacy single-attempt behavior
# ===========================================================================


class TestRegression:
    def test_legacy_single_attempt_via_max_attempts_one(self) -> None:
        """The pre-PR behavior was effectively ``max_attempts=1``. Callers
        who must keep that (e.g. tests, dry-runs) can pass it explicitly
        and get byte-for-byte the old semantics."""
        api = _FakeApiClient([Ok(None)])
        mgr = RunPodCleanupManager(api, max_attempts=1, sleep_fn=lambda _: None)

        result = mgr.cleanup_pod("pod_abc")

        assert result.is_ok()
        assert len(api.calls) == 1

    def test_default_constructor_has_three_attempts(self) -> None:
        """Default tuning regression: don't accidentally drop retry from
        the public ctor signature."""
        api = _FakeApiClient([_err("a"), _err("b"), _err("c")])
        mgr = RunPodCleanupManager(api, sleep_fn=lambda _: None)

        result = mgr.cleanup_pod("pod_abc")

        assert result.is_err()
        assert len(api.calls) == 3


# ===========================================================================
# Logic-specific — backoff math + last-error preservation
# ===========================================================================


class TestLogic:
    def test_backoff_doubles_each_attempt(self) -> None:
        api = _FakeApiClient(
            [_err("a"), _err("b"), _err("c"), _err("d"), Ok(None)]
        )
        sleeper = _RecordingSleeper()
        mgr = RunPodCleanupManager(
            api,
            max_attempts=5,
            initial_backoff_s=1.0,
            max_backoff_s=100.0,  # high cap so we observe pure doubling
            sleep_fn=sleeper,
        )

        mgr.cleanup_pod("pod_abc")

        # 4 sleeps for 5 attempts; pure exponential 1→2→4→8.
        assert sleeper.durations == [1.0, 2.0, 4.0, 8.0]

    def test_last_error_preserved_not_first(self) -> None:
        """Operators read ``unwrap_err`` to diagnose. The most recent
        error is the most diagnostic — it's the state of the world right
        now, not 6 seconds ago."""
        api = _FakeApiClient([
            _err("AUTH_BLIP", "old auth state"),
            _err("NET_TIMEOUT", "intermediate"),
            _err("FINAL_5XX", "what's actually wrong now"),
        ])
        mgr = RunPodCleanupManager(api, sleep_fn=lambda _: None)

        result = mgr.cleanup_pod("pod_abc")

        err = result.unwrap_err()
        assert err.code == "FINAL_5XX"
        assert err.message == "what's actually wrong now"

    def test_sleep_function_is_injectable(self) -> None:
        """``sleep_fn`` is a hard contract for testability — make sure
        the constructor honours it (catches a regression where someone
        hardcodes ``time.sleep`` inside the loop)."""
        called: list[float] = []

        def _custom_sleeper(d: float) -> None:
            called.append(d)

        api = _FakeApiClient([_err("a"), Ok(None)])
        mgr = RunPodCleanupManager(api, sleep_fn=_custom_sleeper)

        mgr.cleanup_pod("pod_abc")

        assert called == [2.0]


# ===========================================================================
# Combinatorial — success-on-Nth-attempt matrix
# ===========================================================================


class TestCombinatorial:
    @pytest.mark.parametrize(
        ("success_attempt", "expected_sleeps"),
        [
            (1, []),                # No retry needed.
            (2, [2.0]),             # 1 sleep before retry.
            (3, [2.0, 4.0]),        # 2 sleeps, exponential.
        ],
    )
    def test_success_on_attempt_n(
        self, success_attempt: int, expected_sleeps: list[float]
    ) -> None:
        """For each ``success_attempt`` ∈ {1,2,3} verify exactly the right
        number of API calls and exactly the right backoff schedule."""
        responses: list[Result[None, ProviderError]] = [
            _err(f"E{i}") for i in range(success_attempt - 1)
        ]
        responses.append(Ok(None))

        api = _FakeApiClient(responses)
        sleeper = _RecordingSleeper()
        mgr = RunPodCleanupManager(api, sleep_fn=sleeper)

        result = mgr.cleanup_pod("pod_abc")

        assert result.is_ok()
        assert len(api.calls) == success_attempt
        assert sleeper.durations == expected_sleeps
