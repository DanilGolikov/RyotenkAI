"""``RunPodCleanupManager`` — bounded-retry pod termination.

Coverage categories: positive, negative, boundary, invariant,
dependency-error, regression (legacy single-attempt parity), logic-
specific (backoff doubling / sleep injection / last-error preservation),
combinatorial (success-on-Nth-attempt matrix).

Phase A2 Batch 11 (2026-05-15): migrated to raise-based contract.
``cleanup_pod`` returns ``None`` on success and re-raises the last
typed exception on exhaustion.
"""

from __future__ import annotations

import pytest

from ryotenkai_providers.runpod.training.cleanup_manager import (
    RunPodCleanupManager,
)
from ryotenkai_shared.errors import ProviderUnavailableError

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeApiClient:
    """Records calls and yields a queue of canned actions.

    Each call to ``terminate_pod`` pops the next action: ``None`` means
    success; an ``Exception`` instance is raised.
    """

    def __init__(self, actions: list[Exception | None]) -> None:
        self.actions = list(actions)
        self.calls: list[str] = []

    def terminate_pod(self, pod_id: str) -> None:
        self.calls.append(pod_id)
        if not self.actions:
            raise AssertionError(
                f"FakeApiClient ran out of responses on call {len(self.calls)} "
                f"(pod_id={pod_id!r})"
            )
        action = self.actions.pop(0)
        if isinstance(action, Exception):
            raise action


class _RecordingSleeper:
    """``sleep_fn`` substitute that records durations without sleeping."""

    def __init__(self) -> None:
        self.durations: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.durations.append(seconds)


def _err(code: str = "RUNPOD_TRANSIENT", msg: str = "boom") -> ProviderUnavailableError:
    return ProviderUnavailableError(detail=msg, context={"code": code})


# ===========================================================================
# Positive — happy paths
# ===========================================================================


class TestPositive:
    def test_first_attempt_succeeds_no_sleep(self) -> None:
        api = _FakeApiClient([None])
        sleeper = _RecordingSleeper()
        mgr = RunPodCleanupManager(api, sleep_fn=sleeper)

        mgr.cleanup_pod("pod_abc")  # returns None

        assert api.calls == ["pod_abc"]
        assert sleeper.durations == []

    def test_succeeds_on_second_attempt(self) -> None:
        api = _FakeApiClient([_err("NET"), None])
        sleeper = _RecordingSleeper()
        mgr = RunPodCleanupManager(api, sleep_fn=sleeper)

        mgr.cleanup_pod("pod_abc")

        assert api.calls == ["pod_abc", "pod_abc"]
        assert sleeper.durations == [2.0]

    def test_succeeds_on_third_attempt(self) -> None:
        api = _FakeApiClient([_err("NET"), _err("5XX"), None])
        sleeper = _RecordingSleeper()
        mgr = RunPodCleanupManager(api, sleep_fn=sleeper)

        mgr.cleanup_pod("pod_abc")

        assert api.calls == ["pod_abc"] * 3
        assert sleeper.durations == [2.0, 4.0]


# ===========================================================================
# Negative — exhaustion
# ===========================================================================


class TestNegative:
    def test_all_attempts_fail_raises_last_error(self) -> None:
        api = _FakeApiClient([
            _err("FIRST"),
            _err("SECOND"),
            _err("THIRD_FINAL"),
        ])
        sleeper = _RecordingSleeper()
        mgr = RunPodCleanupManager(api, sleep_fn=sleeper)

        with pytest.raises(ProviderUnavailableError) as ei:
            mgr.cleanup_pod("pod_abc")
        assert ei.value.context["code"] == "THIRD_FINAL"
        assert len(api.calls) == 3
        assert sleeper.durations == [2.0, 4.0]

    def test_zero_max_attempts_rejected(self) -> None:
        api = _FakeApiClient([None])
        with pytest.raises(ValueError, match="max_attempts"):
            RunPodCleanupManager(api, max_attempts=0)

    def test_negative_initial_backoff_rejected(self) -> None:
        api = _FakeApiClient([None])
        with pytest.raises(ValueError, match="initial_backoff"):
            RunPodCleanupManager(api, initial_backoff_s=-1.0)

    def test_max_backoff_below_initial_rejected(self) -> None:
        api = _FakeApiClient([None])
        with pytest.raises(ValueError, match="max_backoff"):
            RunPodCleanupManager(api, initial_backoff_s=10.0, max_backoff_s=5.0)


# ===========================================================================
# Boundary — edge configurations
# ===========================================================================


class TestBoundary:
    def test_max_attempts_one_means_no_retry(self) -> None:
        api = _FakeApiClient([_err("ONLY")])
        sleeper = _RecordingSleeper()
        mgr = RunPodCleanupManager(api, max_attempts=1, sleep_fn=sleeper)

        with pytest.raises(ProviderUnavailableError) as ei:
            mgr.cleanup_pod("pod_abc")
        assert ei.value.context["code"] == "ONLY"
        assert len(api.calls) == 1
        assert sleeper.durations == []

    def test_zero_initial_backoff_skips_sleep(self) -> None:
        """``initial_backoff_s=0`` is valid — useful in tests / fast retries."""
        api = _FakeApiClient([_err("NET"), None])
        sleeper = _RecordingSleeper()
        mgr = RunPodCleanupManager(
            api,
            initial_backoff_s=0.0,
            max_backoff_s=0.0,
            sleep_fn=sleeper,
        )

        mgr.cleanup_pod("pod_abc")

        assert sleeper.durations == [0.0]

    def test_backoff_capped_at_max(self) -> None:
        """Doubling stops at ``max_backoff_s``."""
        api = _FakeApiClient([_err("a"), _err("b"), _err("c"), _err("d"), None])
        sleeper = _RecordingSleeper()
        mgr = RunPodCleanupManager(
            api,
            max_attempts=5,
            initial_backoff_s=10.0,
            max_backoff_s=15.0,  # cap below 2*10 = 20
            sleep_fn=sleeper,
        )

        mgr.cleanup_pod("pod_abc")

        assert sleeper.durations == [10.0, 15.0, 15.0, 15.0]


# ===========================================================================
# Invariants
# ===========================================================================


class TestInvariants:
    def test_pod_id_passed_unchanged_on_every_attempt(self) -> None:
        api = _FakeApiClient([_err("a"), _err("b"), None])
        mgr = RunPodCleanupManager(api, sleep_fn=lambda _: None)

        mgr.cleanup_pod("special-pod-id-with-dashes_42")

        assert api.calls == ["special-pod-id-with-dashes_42"] * 3

    def test_n_minus_one_sleeps_for_n_attempts(self) -> None:
        for n in (1, 2, 3, 5, 10):
            api = _FakeApiClient([_err(f"E{i}") for i in range(n)])
            sleeper = _RecordingSleeper()
            mgr = RunPodCleanupManager(api, max_attempts=n, sleep_fn=sleeper)

            with pytest.raises(ProviderUnavailableError):
                mgr.cleanup_pod("pod_abc")

            assert len(sleeper.durations) == n - 1, (
                f"max_attempts={n}: expected {n-1} sleeps, got "
                f"{len(sleeper.durations)}"
            )

    def test_no_sleep_after_success(self) -> None:
        """Invariant: as soon as success, no further sleep happens."""
        api = _FakeApiClient([_err("NET"), None])
        sleeper = _RecordingSleeper()
        mgr = RunPodCleanupManager(api, max_attempts=5, sleep_fn=sleeper)

        mgr.cleanup_pod("pod_abc")

        assert len(sleeper.durations) == 1


# ===========================================================================
# Dependency-error
# ===========================================================================


class TestDependencyError:
    def test_api_client_arbitrary_exception_propagates(self) -> None:
        """Non-RyotenkAIError exceptions (e.g. ``RuntimeError`` from a buggy
        transport) propagate without retry — retry only applies to typed
        failures we know how to interpret."""

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
        api = _FakeApiClient([None])
        mgr = RunPodCleanupManager(api, max_attempts=1, sleep_fn=lambda _: None)

        mgr.cleanup_pod("pod_abc")

        assert len(api.calls) == 1

    def test_default_constructor_has_three_attempts(self) -> None:
        api = _FakeApiClient([_err("a"), _err("b"), _err("c")])
        mgr = RunPodCleanupManager(api, sleep_fn=lambda _: None)

        with pytest.raises(ProviderUnavailableError):
            mgr.cleanup_pod("pod_abc")
        assert len(api.calls) == 3


# ===========================================================================
# Logic-specific — backoff math + last-error preservation
# ===========================================================================


class TestLogic:
    def test_backoff_doubles_each_attempt(self) -> None:
        api = _FakeApiClient(
            [_err("a"), _err("b"), _err("c"), _err("d"), None]
        )
        sleeper = _RecordingSleeper()
        mgr = RunPodCleanupManager(
            api,
            max_attempts=5,
            initial_backoff_s=1.0,
            max_backoff_s=100.0,
            sleep_fn=sleeper,
        )

        mgr.cleanup_pod("pod_abc")

        assert sleeper.durations == [1.0, 2.0, 4.0, 8.0]

    def test_last_error_preserved_not_first(self) -> None:
        """Operators read the raised exception to diagnose. The most recent
        error is the most diagnostic — it's the state of the world right
        now, not 6 seconds ago."""
        api = _FakeApiClient([
            _err("AUTH_BLIP", "old auth state"),
            _err("NET_TIMEOUT", "intermediate"),
            _err("FINAL_5XX", "what's actually wrong now"),
        ])
        mgr = RunPodCleanupManager(api, sleep_fn=lambda _: None)

        with pytest.raises(ProviderUnavailableError) as ei:
            mgr.cleanup_pod("pod_abc")
        assert ei.value.context["code"] == "FINAL_5XX"
        assert ei.value.detail == "what's actually wrong now"

    def test_sleep_function_is_injectable(self) -> None:
        called: list[float] = []

        def _custom_sleeper(d: float) -> None:
            called.append(d)

        api = _FakeApiClient([_err("a"), None])
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
        actions: list[Exception | None] = [
            _err(f"E{i}") for i in range(success_attempt - 1)
        ]
        actions.append(None)

        api = _FakeApiClient(actions)
        sleeper = _RecordingSleeper()
        mgr = RunPodCleanupManager(api, sleep_fn=sleeper)

        mgr.cleanup_pod("pod_abc")

        assert len(api.calls) == success_attempt
        assert sleeper.durations == expected_sleeps
