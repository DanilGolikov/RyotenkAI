"""PR-C — :data:`PodTerminalOutcome.TERMINATED_AFTER_DIAGNOSTIC_GRACE`
+ :meth:`PodTerminator._wait_diagnostic_grace` dispatch tests.

Asymmetry guard for the decision matrix: pre PR-C ``failed + mac_alive``
returned :data:`TERMINATED_SAFETY` (zero grace), forcing a race between
Mac's post-mortem SCP and provider teardown. The 15-crash incident on
2026-05-02 surfaced this as ``<<MISSING>>`` postmortems for every
single failure. PR-C inserts a brief, heartbeat-aware diagnostic grace
on that one quadrant. These tests pin the new outcome and the
dispatch behaviour (terminate fires AFTER the grace, abort early on
heartbeat loss).
"""

from __future__ import annotations

from typing import Any

import pytest

from ryotenkai_pod.runner.pod_terminator import (
    PodTerminalOutcome,
    PodTerminator,
    decide_terminal_outcome,
)
from ryotenkai_shared.constants import PROVIDER_RUNPOD
from ryotenkai_shared.infrastructure.lifecycle import (
    IPodLifecycleClient,
    LifecycleActionResult,
)

# TestDiagnosticGraceDispatch + TestConfigKnobs run with the asyncio
# marker individually below. TestDecisionMatrix32Cells is a pure-sync
# pin of the decision function — keeping it outside the asyncio marker
# avoids spurious pytest-asyncio warnings on parametrized sync tests.


# ---------------------------------------------------------------------------
# Test doubles (mirrored shape from test_pod_terminator.py)
# ---------------------------------------------------------------------------


class _PublisherSpy:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    def __call__(self, kind: str, payload: dict[str, Any]) -> None:
        self.events.append((kind, dict(payload)))


class _AlwaysAliveHeartbeat:
    def is_alive(self) -> bool:
        return True

    def age_seconds(self) -> float | None:
        return 5.0

    def mark_active(self) -> None:
        pass


class _ScriptedHeartbeat:
    """Heartbeat that reads is_alive() from a list, advancing per call.

    Used to script "alive at decision time, dies during grace tick N"
    scenarios.
    """

    def __init__(self, schedule: list[bool]) -> None:
        self._schedule = list(schedule)
        self.calls = 0

    def is_alive(self) -> bool:
        if not self._schedule:
            return False
        v = self._schedule.pop(0)
        self.calls += 1
        return v

    def age_seconds(self) -> float | None:
        return 1.0

    def mark_active(self) -> None:
        pass


class _FakeLifecycleClient:
    def __init__(
        self,
        *,
        terminate_outcome: str = PodTerminalOutcome.TERMINATED,
        attempts_made: int = 1,
    ) -> None:
        self._terminate_outcome = terminate_outcome
        self._attempts_made = attempts_made
        self.calls: list[tuple[str, str]] = []

    @property
    def provider_name(self) -> str:
        return PROVIDER_RUNPOD

    async def terminate(self, *, resource_id: str) -> LifecycleActionResult:
        self.calls.append(("terminate", resource_id))
        return LifecycleActionResult(
            outcome=self._terminate_outcome, attempts_made=self._attempts_made,
        )

    async def pause(self, *, resource_id: str) -> LifecycleActionResult:
        self.calls.append(("pause", resource_id))
        return LifecycleActionResult(
            outcome=PodTerminalOutcome.STOPPED, attempts_made=self._attempts_made,
        )

    async def resume(self, *, resource_id: str) -> LifecycleActionResult:
        self.calls.append(("resume", resource_id))
        return LifecycleActionResult(
            outcome="resumed", attempts_made=self._attempts_made,
        )


def _mk_terminator(
    *,
    client: IPodLifecycleClient | None = None,
    diagnostic_grace_seconds: float = 0.05,
    diagnostic_grace_tick_seconds: float = 0.01,
    sleep: Any = None,
) -> PodTerminator:
    async def _no_sleep(_: float) -> None:
        return

    return PodTerminator(
        client=client or _FakeLifecycleClient(),
        resource_id="pod-abc",
        volume_kind="persistent",
        keep_on_error=False,
        sleep=sleep or _no_sleep,
        # Tests don't use the SHORT_GRACE knobs but keep them small for safety.
        grace_base_seconds=0.05,
        grace_max_seconds=0.5,
        grace_tick_seconds=0.01,
        heartbeat_retry_attempts=0,
        heartbeat_retry_tick_seconds=0.0,
        diagnostic_grace_seconds=diagnostic_grace_seconds,
        diagnostic_grace_tick_seconds=diagnostic_grace_tick_seconds,
    )


# ---------------------------------------------------------------------------
# 1. Decision matrix — combinatorial 32-cell coverage including new outcome
# ---------------------------------------------------------------------------


class TestDecisionMatrix32Cells:
    """All 4×2×2×2 = 32 (state, mac_alive, volume, keep_on_error) cells.

    Each cell is asserted with both the explicit decision constant AND
    a "exactly one outcome" guarantee (via the asserted equality being
    True).
    """

    @pytest.mark.parametrize("mac_alive", [True, False])
    @pytest.mark.parametrize("volume_kind", ["persistent", "network"])
    @pytest.mark.parametrize("keep_on_error", [True, False])
    def test_cancelled_always_terminates_user_stop(
        self, mac_alive: bool, volume_kind: str, keep_on_error: bool,
    ) -> None:
        """8 cells: cancelled × any × any × any → TERMINATED_USER_STOP."""
        assert decide_terminal_outcome(
            terminal_state="cancelled",
            mac_alive=mac_alive,
            volume_kind=volume_kind,
            keep_on_error=keep_on_error,
        ) == PodTerminalOutcome.TERMINATED_USER_STOP

    @pytest.mark.parametrize("mac_alive", [True, False])
    @pytest.mark.parametrize("volume_kind", ["persistent", "network"])
    def test_failed_with_keep_on_error_always_kept_alive(
        self, mac_alive: bool, volume_kind: str,
    ) -> None:
        """4 cells: failed × any × any × keep=True → KEPT_ALIVE_FOR_DEBUG."""
        assert decide_terminal_outcome(
            terminal_state="failed",
            mac_alive=mac_alive,
            volume_kind=volume_kind,
            keep_on_error=True,
        ) == PodTerminalOutcome.KEPT_ALIVE_FOR_DEBUG

    @pytest.mark.parametrize("mac_alive", [True, False])
    def test_failed_network_volume_terminates_safety(
        self, mac_alive: bool,
    ) -> None:
        """2 cells: failed × any × network × keep=False → TERMINATED_SAFETY.

        Network volume can't be paused, so terminate is the only option
        regardless of Mac liveness.
        """
        assert decide_terminal_outcome(
            terminal_state="failed",
            mac_alive=mac_alive,
            volume_kind="network",
            keep_on_error=False,
        ) == PodTerminalOutcome.TERMINATED_SAFETY

    def test_failed_mac_alive_persistent_uses_diagnostic_grace(self) -> None:
        """1 cell — the PR-C change. Was TERMINATED_SAFETY (race),
        now TERMINATED_AFTER_DIAGNOSTIC_GRACE (graceful)."""
        assert decide_terminal_outcome(
            terminal_state="failed",
            mac_alive=True,
            volume_kind="persistent",
            keep_on_error=False,
        ) == PodTerminalOutcome.TERMINATED_AFTER_DIAGNOSTIC_GRACE

    def test_failed_mac_asleep_persistent_stops_for_resume(self) -> None:
        """1 cell — Mac asleep so no post-mortem race; pause for
        checkpoint accessibility."""
        assert decide_terminal_outcome(
            terminal_state="failed",
            mac_alive=False,
            volume_kind="persistent",
            keep_on_error=False,
        ) == PodTerminalOutcome.STOPPED_FOR_RESUME

    @pytest.mark.parametrize("mac_alive", [True, False])
    def test_completed_network_volume_terminates_safety(
        self, mac_alive: bool,
    ) -> None:
        """2 cells: completed × any × network × keep=any → TERMINATED_SAFETY."""
        for keep in (True, False):
            assert decide_terminal_outcome(
                terminal_state="completed",
                mac_alive=mac_alive,
                volume_kind="network",
                keep_on_error=keep,
            ) == PodTerminalOutcome.TERMINATED_SAFETY

    def test_completed_mac_alive_persistent_short_grace(self) -> None:
        """1 cell — natural completion + Mac alive: SHORT_GRACE for
        ModelRetriever. Both keep_on_error values."""
        for keep in (True, False):
            assert decide_terminal_outcome(
                terminal_state="completed",
                mac_alive=True,
                volume_kind="persistent",
                keep_on_error=keep,
            ) == PodTerminalOutcome.STOPPED_FOR_RESUME_SHORT_GRACE

    def test_completed_mac_asleep_persistent_stops(self) -> None:
        """1 cell — completion + Mac asleep: pause for resume."""
        for keep in (True, False):
            assert decide_terminal_outcome(
                terminal_state="completed",
                mac_alive=False,
                volume_kind="persistent",
                keep_on_error=keep,
            ) == PodTerminalOutcome.STOPPED_FOR_RESUME

    @pytest.mark.parametrize("mac_alive", [True, False])
    def test_unknown_state_persistent_kept_alive_for_debug(
        self, mac_alive: bool,
    ) -> None:
        """Unknown terminal_state on persistent volume: defensive
        KEPT_ALIVE_FOR_DEBUG so we don't destroy data we can't
        classify."""
        for keep in (True, False):
            assert decide_terminal_outcome(
                terminal_state="bogus_state",
                mac_alive=mac_alive,
                volume_kind="persistent",
                keep_on_error=keep,
            ) == PodTerminalOutcome.KEPT_ALIVE_FOR_DEBUG

    @pytest.mark.parametrize("mac_alive", [True, False])
    def test_unknown_state_network_volume_terminates_safety(
        self, mac_alive: bool,
    ) -> None:
        """Unknown terminal_state on a network volume: network volumes
        cannot be paused, so the only safe action is termination — the
        network-volume guard short-circuits before the unknown-state
        fallback. This is intentional asymmetry: persistent has a
        non-destructive option (do nothing), network does not."""
        for keep in (True, False):
            assert decide_terminal_outcome(
                terminal_state="bogus_state",
                mac_alive=mac_alive,
                volume_kind="network",
                keep_on_error=keep,
            ) == PodTerminalOutcome.TERMINATED_SAFETY


# ---------------------------------------------------------------------------
# 2. Dispatch — diagnostic grace fires THEN terminate
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestDiagnosticGraceDispatch:
    async def test_grace_runs_then_terminate_called(self) -> None:
        """Happy path: heartbeat alive throughout the grace window →
        full grace elapses, then provider.terminate fires."""
        client = _FakeLifecycleClient()
        terminator = _mk_terminator(
            client=client,
            diagnostic_grace_seconds=0.05,
            diagnostic_grace_tick_seconds=0.01,
        )
        spy = _PublisherSpy()

        await terminator.decide_and_act(
            terminal_state="failed",
            heartbeat=_AlwaysAliveHeartbeat(),
            bus_publish=spy,
        )

        # Decision was the new outcome
        decision_event = next(
            ev for ev in spy.events if ev[0] == "pod_terminal_decision"
        )
        assert decision_event[1]["decision"] == (
            PodTerminalOutcome.TERMINATED_AFTER_DIAGNOSTIC_GRACE
        )

        # Grace window opened and closed
        assert any(
            ev[0] == "pod_terminal_diagnostic_grace_started" for ev in spy.events
        )
        ended = next(
            ev for ev in spy.events
            if ev[0] == "pod_terminal_diagnostic_grace_ended"
        )
        assert ended[1]["reason"] == "max_budget_reached"

        # Provider.terminate was called exactly once
        assert client.calls == [("terminate", "pod-abc")]

    async def test_grace_aborts_early_when_heartbeat_dies(self) -> None:
        """Heartbeat dies after first grace tick → grace ends with
        ``heartbeat_lost`` reason and we still terminate (no point
        keeping the pod alive for an absent reader)."""
        client = _FakeLifecycleClient()
        terminator = _mk_terminator(
            client=client,
            diagnostic_grace_seconds=10.0,  # plenty of headroom
            diagnostic_grace_tick_seconds=0.01,
        )
        spy = _PublisherSpy()
        # First read (decision check): True. Second + third (grace ticks):
        # False → triggers early abort. Heartbeat retry is disabled
        # (HEARTBEAT_RETRY_ATTEMPTS=0) so the first False at the decision
        # check would short-circuit; we want the decision to be
        # ``failed + alive`` so we use a generous schedule.
        hb = _ScriptedHeartbeat([True, False, False, False])

        await terminator.decide_and_act(
            terminal_state="failed",
            heartbeat=hb,
            bus_publish=spy,
        )

        ended = next(
            ev for ev in spy.events
            if ev[0] == "pod_terminal_diagnostic_grace_ended"
        )
        assert ended[1]["reason"] == "heartbeat_lost"
        # Terminated despite grace abort
        assert client.calls == [("terminate", "pod-abc")]

    async def test_grace_payload_includes_elapsed_seconds(self) -> None:
        """For operator dashboards: the ``ended`` event carries
        ``elapsed_s`` so we can plot grace utilization (full vs aborted)."""
        client = _FakeLifecycleClient()
        terminator = _mk_terminator(
            client=client,
            diagnostic_grace_seconds=0.05,
            diagnostic_grace_tick_seconds=0.01,
        )
        spy = _PublisherSpy()

        await terminator.decide_and_act(
            terminal_state="failed",
            heartbeat=_AlwaysAliveHeartbeat(),
            bus_publish=spy,
        )

        ended = next(
            ev for ev in spy.events
            if ev[0] == "pod_terminal_diagnostic_grace_ended"
        )
        assert "elapsed_s" in ended[1]
        assert ended[1]["elapsed_s"] >= 0.05  # at least the configured budget

    async def test_kept_alive_for_debug_skips_diagnostic_grace(self) -> None:
        """Regression: ``failed + mac_alive + keep_on_error=True``
        must NOT enter the grace path — KEPT_ALIVE_FOR_DEBUG short
        circuits before we even consider grace. Otherwise we'd wait
        30 s for nothing on every keep_on_error failure."""
        client = _FakeLifecycleClient()
        terminator = PodTerminator(
            client=client,
            resource_id="pod-abc",
            volume_kind="persistent",
            keep_on_error=True,  # ← keep_on_error
            sleep=_AsyncNoOp(),
            grace_base_seconds=0.01,
            grace_max_seconds=0.1,
            grace_tick_seconds=0.005,
            heartbeat_retry_attempts=0,
            heartbeat_retry_tick_seconds=0.0,
            diagnostic_grace_seconds=10.0,  # would be expensive if hit
            diagnostic_grace_tick_seconds=0.01,
        )
        spy = _PublisherSpy()

        await terminator.decide_and_act(
            terminal_state="failed",
            heartbeat=_AlwaysAliveHeartbeat(),
            bus_publish=spy,
        )

        # Grace was never opened
        assert not any(
            ev[0] == "pod_terminal_diagnostic_grace_started" for ev in spy.events
        )
        # Provider was never called
        assert client.calls == []


# ---------------------------------------------------------------------------
# 3. Boundary — config knobs honoured
# ---------------------------------------------------------------------------


class TestConfigKnobs:
    def test_default_diagnostic_grace_is_thirty_seconds(self) -> None:
        """Class default protects production from accidental shrinkage —
        if someone bumps DIAGNOSTIC_GRACE_SECONDS down to ~5s, the test
        flags it. (5s is too short for slow Mac SSH connections.)"""
        assert PodTerminator.DIAGNOSTIC_GRACE_SECONDS == 30.0

    def test_diagnostic_grace_tick_default_two_seconds(self) -> None:
        """Tick should be small for snappy heartbeat-loss abort but
        not so small that we spam the bus with grace events."""
        assert PodTerminator.DIAGNOSTIC_GRACE_TICK_SECONDS == 2.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _AsyncNoOp:
    """Awaitable no-op suitable as the ``sleep`` injector."""

    async def __call__(self, _delay: float) -> None:
        return
