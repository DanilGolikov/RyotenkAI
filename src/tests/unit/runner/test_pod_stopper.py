"""Phase 4.4 — :class:`PodStopper` contract.

Money-critical: a regression here means the runner finishes a job
but never tells RunPod to stop billing. The test matrix covers
every branch of the env-driven decision plus the GraphQL retry
loop's success / idempotent / failure paths.

Coverage matrix:

- TestShouldStop          pure decision function
- TestEnvShortCircuits    AUTO_STOP=false / KEEP_ON_ERROR=true / missing creds
- TestGraphQLSuccess      first attempt returns ``id`` → ``stopped``
- TestGraphQLIdempotent   already-stopped error string → ``already_stopped``
- TestGraphQLRetry        first attempt fails → second succeeds
- TestGraphQLAllFail      every attempt fails → ``failed``
- TestNetworkError        httpx raises → counted as a failed attempt
- TestStopOnTerminalWrap  convenience wrapper publishes outcome
"""

from __future__ import annotations

import httpx
import pytest

from src.runner.pod_stopper import (
    PodStopOutcome,
    PodStopper,
    should_stop_pod,
    stop_pod_on_terminal,
)

# Async tests opt in per-class via @pytest.mark.asyncio decorator;
# applying pytestmark file-wide produced spurious warnings on the
# synchronous decision-table tests below.


# ---------------------------------------------------------------------------
# Pure decision function
# ---------------------------------------------------------------------------


class TestShouldStop:
    @pytest.mark.parametrize(
        ("auto", "failed", "keep", "expected"),
        [
            ("true", False, "false", True),  # happy path: completed
            ("true", True, "false", True),  # failed without keep flag → stop
            ("true", True, "true", False),  # failed + keep_on_error=true → keep
            ("false", False, "false", False),  # AUTO_STOP off
            (None, False, None, False),  # AUTO_STOP unset → off
            ("True", True, "TRUE", False),  # case-insensitive keep
            ("TRUE", False, "false", True),  # case-insensitive auto
        ],
    )
    def test_decision_table(
        self,
        auto: str | None,
        failed: bool,
        keep: str | None,
        expected: bool,
    ) -> None:
        assert (
            should_stop_pod(
                auto_stop=auto,
                failed=failed,
                keep_on_error=keep,
            )
            == expected
        )

    def test_user_stop_ignores_keep_on_error(self) -> None:
        """Phase 9.1.B regression: user-initiated stop (terminal_state
        ``cancelled`` → ``failed=False``) **always** terminates the pod,
        regardless of ``KEEP_ON_ERROR``.

        ``KEEP_ON_ERROR`` is a debug affordance for AUTOMATIC failures
        (idle, crash, OOM) only. An explicit user click in the CLI / Web
        UI overrides it — pin that decision so a future tweak to the
        decision table doesn't accidentally re-introduce the keep-alive
        path on user-stop.
        """
        # User-stop equivalent: terminal_state="cancelled" surfaces as
        # ``failed=False`` to ``should_stop_pod``.
        assert should_stop_pod(
            auto_stop="true",
            failed=False,  # cancelled, NOT failed
            keep_on_error="true",  # debug affordance set
        ) is True

    def test_failed_with_keep_on_error_is_only_keep_path(self) -> None:
        """Tighten the contract: KEEP_ON_ERROR has effect ONLY when
        ``failed=True``. Any other combination terminates."""
        # Only this exact combination keeps the pod alive.
        assert should_stop_pod(
            auto_stop="true", failed=True, keep_on_error="true",
        ) is False
        # All neighbours flip back to "terminate".
        assert should_stop_pod(
            auto_stop="true", failed=True, keep_on_error="false",
        ) is True
        assert should_stop_pod(
            auto_stop="true", failed=False, keep_on_error="true",
        ) is True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_stopper(handler) -> PodStopper:
    """Build a stopper whose HTTP calls go through ``handler``."""

    def _factory() -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(handler))

    async def _no_sleep(_seconds: float) -> None:
        return None  # zero-time backoff for fast tests

    return PodStopper(
        http_client_factory=_factory,
        sleep=_no_sleep,
        max_attempts=3,
    )


# ---------------------------------------------------------------------------
# Env short-circuits
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestEnvShortCircuits:
    async def test_auto_stop_disabled_returns_disabled(self) -> None:
        stopper = _make_stopper(
            lambda r: httpx.Response(500, text="should not reach"),
        )
        outcome = await stopper.stop_if_needed(
            terminal_state="completed",
            env={"RUNPOD_AUTO_STOP": "false"},
        )
        assert outcome == PodStopOutcome.DISABLED

    async def test_keep_on_error_with_failed_state(self) -> None:
        stopper = _make_stopper(
            lambda r: httpx.Response(500, text="should not reach"),
        )
        outcome = await stopper.stop_if_needed(
            terminal_state="failed",
            env={
                "RUNPOD_AUTO_STOP": "true",
                "RUNPOD_KEEP_ON_ERROR": "true",
                "RUNPOD_API_KEY": "k",
                "RUNPOD_POD_ID": "p",
            },
        )
        assert outcome == PodStopOutcome.DISABLED

    async def test_keep_on_error_ignored_for_completed(self) -> None:
        # KEEP_ON_ERROR only matters when the terminal state IS failed.
        # A completed run still stops even with the flag set.
        responses = []

        def _h(request: httpx.Request) -> httpx.Response:
            responses.append(request)
            return httpx.Response(200, json={"data": {"podTerminate": None}})

        stopper = _make_stopper(_h)
        outcome = await stopper.stop_if_needed(
            terminal_state="completed",
            env={
                "RUNPOD_AUTO_STOP": "true",
                "RUNPOD_KEEP_ON_ERROR": "true",
                "RUNPOD_API_KEY": "k",
                "RUNPOD_POD_ID": "p",
            },
        )
        assert outcome == PodStopOutcome.TERMINATED
        assert len(responses) == 1

    async def test_missing_api_key_skips(self) -> None:
        stopper = _make_stopper(
            lambda r: httpx.Response(500, text="should not reach"),
        )
        outcome = await stopper.stop_if_needed(
            terminal_state="completed",
            env={"RUNPOD_AUTO_STOP": "true", "RUNPOD_POD_ID": "p"},
        )
        assert outcome == PodStopOutcome.SKIPPED

    async def test_missing_pod_id_skips(self) -> None:
        stopper = _make_stopper(
            lambda r: httpx.Response(500, text="should not reach"),
        )
        outcome = await stopper.stop_if_needed(
            terminal_state="completed",
            env={"RUNPOD_AUTO_STOP": "true", "RUNPOD_API_KEY": "k"},
        )
        assert outcome == PodStopOutcome.SKIPPED


# ---------------------------------------------------------------------------
# GraphQL happy / idempotent / failure paths
# ---------------------------------------------------------------------------


def _success_handler(request: httpx.Request) -> httpx.Response:
    # ``podTerminate`` returns null on success — RunPod signals success
    # via ``data.podTerminate`` key presence + absence of ``errors``.
    return httpx.Response(200, json={"data": {"podTerminate": None}})


def _already_stopped_handler(request: httpx.Request) -> httpx.Response:
    # Idempotency on ``podTerminate``: post-deletion the API returns
    # "not found" / "does not exist" — Phase 9.A widened the regex.
    return httpx.Response(
        200,
        json={"errors": [{"message": "Pod not found — does not exist"}]},
    )


class _FlakyHandler:
    """Fails N times with synthetic 500, then succeeds."""

    def __init__(self, fail_count: int) -> None:
        self.fail_count = fail_count
        self.calls = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.calls += 1
        if self.calls <= self.fail_count:
            return httpx.Response(500, text="transient backend error")
        return httpx.Response(200, json={"data": {"podTerminate": None}})


_FULL_ENV = {
    "RUNPOD_AUTO_STOP": "true",
    "RUNPOD_API_KEY": "test-key",
    "RUNPOD_POD_ID": "pod-xyz",
}


@pytest.mark.asyncio
class TestGraphQLSuccess:
    async def test_first_attempt_returns_stopped(self) -> None:
        stopper = _make_stopper(_success_handler)
        outcome = await stopper.stop_if_needed(
            terminal_state="completed", env=_FULL_ENV,
        )
        assert outcome == PodStopOutcome.TERMINATED


@pytest.mark.asyncio
class TestGraphQLIdempotent:
    async def test_already_stopped_message_counts_as_success(self) -> None:
        stopper = _make_stopper(_already_stopped_handler)
        outcome = await stopper.stop_if_needed(
            terminal_state="completed", env=_FULL_ENV,
        )
        assert outcome == PodStopOutcome.ALREADY_TERMINATED


@pytest.mark.asyncio
class TestGraphQLRetry:
    async def test_retry_after_transient_failure(self) -> None:
        flaky = _FlakyHandler(fail_count=2)
        stopper = _make_stopper(flaky)
        outcome = await stopper.stop_if_needed(
            terminal_state="completed", env=_FULL_ENV,
        )
        assert outcome == PodStopOutcome.TERMINATED
        assert flaky.calls == 3


@pytest.mark.asyncio
class TestGraphQLAllFail:
    async def test_three_failures_returns_failed(self) -> None:
        flaky = _FlakyHandler(fail_count=10)  # always fails
        stopper = _make_stopper(flaky)
        outcome = await stopper.stop_if_needed(
            terminal_state="completed", env=_FULL_ENV,
        )
        assert outcome == PodStopOutcome.FAILED
        assert flaky.calls == 3  # max_attempts


@pytest.mark.asyncio
class TestNetworkError:
    async def test_httpx_exception_counted_as_failed_attempt(self) -> None:
        def _raise(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("dns lookup failed")

        stopper = _make_stopper(_raise)
        outcome = await stopper.stop_if_needed(
            terminal_state="completed", env=_FULL_ENV,
        )
        assert outcome == PodStopOutcome.FAILED


# ---------------------------------------------------------------------------
# Convenience wrapper publishes structured outcome event
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestStopOnTerminalWrap:
    async def test_publishes_outcome(self) -> None:
        stopper = _make_stopper(_success_handler)
        events: list[tuple[str, dict]] = []

        def _publish(kind: str, payload: dict) -> None:
            events.append((kind, payload))

        await stop_pod_on_terminal(
            stopper,
            terminal_state="completed",
            bus_publish=_publish,
            env=_FULL_ENV,
        )
        assert ("pod_stop_attempt", {
            "terminal_state": "completed",
            "outcome": PodStopOutcome.TERMINATED,
        }) in events

    async def test_swallows_unexpected_exception(self) -> None:
        # Synthetic stopper that raises an unexpected (non-httpx) error.
        class _BadStopper:
            async def stop_if_needed(self, **_: object) -> str:
                raise RuntimeError("boom")

        events: list[tuple[str, dict]] = []
        await stop_pod_on_terminal(
            _BadStopper(),  # type: ignore[arg-type]
            terminal_state="failed",
            bus_publish=lambda k, p: events.append((k, p)),
            env={},
        )
        assert any(k == "pod_stop_error" for k, _ in events)
