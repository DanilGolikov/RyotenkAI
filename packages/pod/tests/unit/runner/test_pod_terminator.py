"""Phase 11.B / 14.B — :class:`PodTerminator` + :func:`decide_terminal_outcome` contract.

Phase 14.B § 1.5 extracted the RunPod GraphQL transport into
:mod:`src.providers.runpod.runtime.lifecycle_client`. After
extraction, this file tests:

* :func:`decide_terminal_outcome` — pure function, exhaustively
  pinned (8 rows of the matrix in § 11.1).
* :class:`PodTerminator` dispatch — invokes the
  :class:`IPodLifecycleClient` Protocol via a fake stub. NO HTTP,
  NO GraphQL, NO env reads. RunPod-specific HTTP shape lives in
  :mod:`src.tests.unit.providers.runpod.runtime.test_lifecycle_client`.

7-category coverage:

1. **Positive** — happy paths for each decision branch.
2. **Negative** — KEPT_ALIVE_FOR_DEBUG short-circuits without client call.
3. **Boundary** — decision matrix corner cases (network volume,
   unknown terminal_state, KEEP_ON_ERROR honoured only on failed,
   invalid volume_kind clamped to persistent).
4. **Invariants** — ``run_terminal_hook`` never propagates errors;
   action dispatcher is idempotent (already-gone → ALREADY_*).
5. **Dependency errors** — client returns FAILED outcome → propagated.
6. **Regressions** — Phase 9.A ``"outcome"`` event field still
   present alongside the new ``decision`` / ``action`` fields.
   Phase 14.B adds ``provider`` + ``attempts_made``.
7. **Logic-specific** — grace loop exits on heartbeat-lost OR
   max-budget; pause fires after grace.
"""

from __future__ import annotations

from typing import Any

import pytest

from ryotenkai_shared.constants import PROVIDER_RUNPOD
from ryotenkai_pod.runner.pod_terminator import (
    PodTerminalOutcome,
    PodTerminator,
    decide_terminal_outcome,
    run_terminal_hook,
)
from ryotenkai_pod.runner.runtime.lifecycle_client import (
    IPodLifecycleClient,
    LifecycleActionResult,
)


pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _PublisherSpy:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    def __call__(self, kind: str, payload: dict[str, Any]) -> None:
        self.events.append((kind, dict(payload)))


class _AlwaysAliveHeartbeat:
    """Heartbeat stub that pretends Mac is always reachable."""

    def is_alive(self) -> bool:
        return True

    def age_seconds(self) -> float | None:
        return 5.0

    def mark_active(self) -> None:
        pass


class _AlwaysDeadHeartbeat:
    """Heartbeat stub that pretends Mac is always asleep."""

    def is_alive(self) -> bool:
        return False

    def age_seconds(self) -> float | None:
        return 600.0

    def mark_active(self) -> None:
        pass


class _ScriptedHeartbeat:
    """Heartbeat that reads is_alive() from a list, advancing per call."""

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
    """In-memory :class:`IPodLifecycleClient` for dispatch tests.

    Records each call + returns scripted outcomes. NO HTTP — replaces
    the pre-14.B ``httpx.MockTransport`` setup.
    """

    def __init__(
        self,
        *,
        terminate_outcome: str = PodTerminalOutcome.TERMINATED,
        pause_outcome: str = PodTerminalOutcome.STOPPED,
        resume_outcome: str = "resumed",
        attempts_made: int = 1,
    ) -> None:
        self._terminate_outcome = terminate_outcome
        self._pause_outcome = pause_outcome
        self._resume_outcome = resume_outcome
        self._attempts_made = attempts_made
        self.calls: list[tuple[str, str]] = []  # (method, resource_id)

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
            outcome=self._pause_outcome, attempts_made=self._attempts_made,
        )

    async def resume(self, *, resource_id: str) -> LifecycleActionResult:
        self.calls.append(("resume", resource_id))
        return LifecycleActionResult(
            outcome=self._resume_outcome, attempts_made=self._attempts_made,
        )


def _mk_terminator(
    *,
    client: IPodLifecycleClient | None = None,
    resource_id: str = "pod-abc",
    volume_kind: str = "persistent",
    keep_on_error: bool = False,
    sleep: Any = None,
    grace_base: float = 0.05,
    grace_max: float = 0.5,
    grace_tick: float = 0.01,
) -> PodTerminator:
    """Build a PodTerminator with deterministic defaults for tests."""

    async def _no_sleep(_: float) -> None:
        return

    return PodTerminator(
        client=client or _FakeLifecycleClient(),
        resource_id=resource_id,
        volume_kind=volume_kind,
        keep_on_error=keep_on_error,
        sleep=sleep or _no_sleep,
        grace_base_seconds=grace_base,
        grace_max_seconds=grace_max,
        grace_tick_seconds=grace_tick,
        # Disable retry latency in unit tests (Phase 11.E knobs).
        heartbeat_retry_attempts=0,
        heartbeat_retry_tick_seconds=0.0,
    )


# ---------------------------------------------------------------------------
# 1. Positive — pure decision function (8-row matrix)
# ---------------------------------------------------------------------------


class TestDecisionMatrixExhaustive:
    """All 8 rows of the § 11.1 table."""

    def test_cancelled_always_terminates(self) -> None:
        for mac_alive in (True, False):
            for volume_kind in ("persistent", "network"):
                for keep_on_error in (True, False):
                    out = decide_terminal_outcome(
                        terminal_state="cancelled",
                        mac_alive=mac_alive,
                        volume_kind=volume_kind,
                        keep_on_error=keep_on_error,
                    )
                    assert out == PodTerminalOutcome.TERMINATED_USER_STOP, (
                        f"cancelled+{mac_alive}+{volume_kind}+{keep_on_error}"
                    )

    def test_failed_with_keep_on_error_kept_alive(self) -> None:
        for mac_alive in (True, False):
            out = decide_terminal_outcome(
                terminal_state="failed",
                mac_alive=mac_alive,
                volume_kind="persistent",
                keep_on_error=True,
            )
            assert out == PodTerminalOutcome.KEPT_ALIVE_FOR_DEBUG

    def test_failed_mac_alive_persistent_terminates_after_diagnostic_grace(self) -> None:
        """PR-C — was TERMINATED_SAFETY (0 grace) pre-2026-05-02. Now we
        give Mac a brief window to SCP the post-mortem before tearing
        the pod down — closes the race that produced the
        ``<<MISSING>>`` postmortem in the 15-crash incident."""
        out = decide_terminal_outcome(
            terminal_state="failed",
            mac_alive=True,
            volume_kind="persistent",
            keep_on_error=False,
        )
        assert out == PodTerminalOutcome.TERMINATED_AFTER_DIAGNOSTIC_GRACE

    def test_failed_mac_asleep_persistent_stops(self) -> None:
        out = decide_terminal_outcome(
            terminal_state="failed",
            mac_alive=False,
            volume_kind="persistent",
            keep_on_error=False,
        )
        assert out == PodTerminalOutcome.STOPPED_FOR_RESUME

    def test_failed_network_volume_terminates(self) -> None:
        for mac_alive in (True, False):
            out = decide_terminal_outcome(
                terminal_state="failed",
                mac_alive=mac_alive,
                volume_kind="network",
                keep_on_error=False,
            )
            assert out == PodTerminalOutcome.TERMINATED_SAFETY

    def test_completed_mac_alive_persistent_short_grace(self) -> None:
        out = decide_terminal_outcome(
            terminal_state="completed",
            mac_alive=True,
            volume_kind="persistent",
            keep_on_error=False,
        )
        assert out == PodTerminalOutcome.STOPPED_FOR_RESUME_SHORT_GRACE

    def test_completed_mac_asleep_persistent_stops(self) -> None:
        out = decide_terminal_outcome(
            terminal_state="completed",
            mac_alive=False,
            volume_kind="persistent",
            keep_on_error=False,
        )
        assert out == PodTerminalOutcome.STOPPED_FOR_RESUME

    def test_completed_network_volume_terminates(self) -> None:
        for mac_alive in (True, False):
            out = decide_terminal_outcome(
                terminal_state="completed",
                mac_alive=mac_alive,
                volume_kind="network",
                keep_on_error=False,
            )
            assert out == PodTerminalOutcome.TERMINATED_SAFETY


# ---------------------------------------------------------------------------
# 2. Negative — KEPT_ALIVE_FOR_DEBUG short-circuits without client call
# ---------------------------------------------------------------------------


class TestNegative:
    async def test_kept_alive_skips_client_call(self) -> None:
        # ``failed`` + ``keep_on_error=True`` ⇒ KEPT_ALIVE_FOR_DEBUG
        # ⇒ NO client method invocation.
        publisher = _PublisherSpy()
        client = _FakeLifecycleClient()
        terminator = _mk_terminator(client=client, keep_on_error=True)

        result = await terminator.decide_and_act(
            terminal_state="failed",
            heartbeat=_AlwaysAliveHeartbeat(),
            bus_publish=publisher,
        )

        assert result["decision"] == PodTerminalOutcome.KEPT_ALIVE_FOR_DEBUG
        assert result["action"] is None
        assert client.calls == []  # No client methods called

    # Note: pre-14.B "missing API key → SKIPPED" / "missing pod_id →
    # SKIPPED" tests are removed. Phase 14.B § 1.7 moves credential
    # validation to lifespan boot — missing creds raise
    # :class:`BootstrapConfigError` and uvicorn refuses to start.
    # The runner code path that ran with bad creds no longer exists.


# ---------------------------------------------------------------------------
# 3. Boundary — corner cases
# ---------------------------------------------------------------------------


class TestBoundary:
    def test_unknown_terminal_state_kept_alive(self) -> None:
        # Defensive: unknown state ⇒ keep alive rather than guess.
        out = decide_terminal_outcome(
            terminal_state="unknown_value",
            mac_alive=True,
            volume_kind="persistent",
            keep_on_error=False,
        )
        assert out == PodTerminalOutcome.KEPT_ALIVE_FOR_DEBUG

    def test_keep_on_error_ignored_on_completed(self) -> None:
        # KEEP_ON_ERROR is for FAILED only — completed runs always
        # stop/terminate regardless.
        out = decide_terminal_outcome(
            terminal_state="completed",
            mac_alive=False,
            volume_kind="persistent",
            keep_on_error=True,
        )
        assert out == PodTerminalOutcome.STOPPED_FOR_RESUME

    def test_keep_on_error_ignored_on_cancelled(self) -> None:
        # User-stop overrides KEEP_ON_ERROR.
        out = decide_terminal_outcome(
            terminal_state="cancelled",
            mac_alive=False,
            volume_kind="persistent",
            keep_on_error=True,
        )
        assert out == PodTerminalOutcome.TERMINATED_USER_STOP

    async def test_unknown_volume_kind_clamped_to_persistent_in_constructor(
        self,
    ) -> None:
        # Phase 14.B § 1.4 — constructor clamps invalid volume_kind to
        # ``persistent`` (safety net mirroring
        # :func:`resolve_volume_kind_from_env`'s clamping at boot).
        publisher = _PublisherSpy()
        client = _FakeLifecycleClient()
        terminator = _mk_terminator(
            client=client, volume_kind="garbage_value",
        )
        result = await terminator.decide_and_act(
            terminal_state="completed",
            heartbeat=_AlwaysDeadHeartbeat(),
            bus_publish=publisher,
        )
        # Clamped → "persistent" → completed+asleep → STOPPED_FOR_RESUME
        assert result["decision"] == PodTerminalOutcome.STOPPED_FOR_RESUME
        assert ("pause", "pod-abc") in client.calls


# ---------------------------------------------------------------------------
# 4. Invariants — never raises, idempotency
# ---------------------------------------------------------------------------


class TestInvariants:
    async def test_run_terminal_hook_never_propagates_errors(self) -> None:
        # A raising decide_and_act should be swallowed.
        class _BoomTerminator:
            async def decide_and_act(self, **_kw: Any) -> dict[str, Any]:
                raise RuntimeError("internal explosion")

        publisher = _PublisherSpy()
        # Must not raise.
        await run_terminal_hook(
            _BoomTerminator(),  # type: ignore[arg-type]
            terminal_state="completed",
            heartbeat=_AlwaysAliveHeartbeat(),
            bus_publish=publisher,
        )

        # Error event published for forensics.
        kinds = [k for k, _ in publisher.events]
        assert "pod_stop_error" in kinds

    async def test_already_terminated_idempotent(self) -> None:
        # Client returns ALREADY_TERMINATED → propagated as action.
        client = _FakeLifecycleClient(
            terminate_outcome=PodTerminalOutcome.ALREADY_TERMINATED,
        )
        terminator = _mk_terminator(client=client)
        publisher = _PublisherSpy()
        result = await terminator.decide_and_act(
            terminal_state="cancelled",
            heartbeat=_AlwaysAliveHeartbeat(),
            bus_publish=publisher,
        )
        assert result["action"] == PodTerminalOutcome.ALREADY_TERMINATED

    async def test_already_stopped_idempotent(self) -> None:
        client = _FakeLifecycleClient(
            pause_outcome=PodTerminalOutcome.ALREADY_STOPPED,
        )
        terminator = _mk_terminator(client=client)
        publisher = _PublisherSpy()
        result = await terminator.decide_and_act(
            terminal_state="completed",
            heartbeat=_AlwaysDeadHeartbeat(),  # asleep → STOPPED_FOR_RESUME
            bus_publish=publisher,
        )
        assert result["action"] == PodTerminalOutcome.ALREADY_STOPPED


# ---------------------------------------------------------------------------
# 5. Dependency errors — client returns FAILED → propagated
# ---------------------------------------------------------------------------


class TestDependencyErrors:
    async def test_client_failed_propagates_to_action(self) -> None:
        # Pre-14.B simulated this with a 500-only mock transport; now
        # we just have the fake client return FAILED. HTTP-level retry
        # behaviour is tested in test_lifecycle_client.py.
        client = _FakeLifecycleClient(
            terminate_outcome=PodTerminalOutcome.FAILED,
            attempts_made=3,
        )
        terminator = _mk_terminator(client=client)
        publisher = _PublisherSpy()
        result = await terminator.decide_and_act(
            terminal_state="cancelled",
            heartbeat=_AlwaysAliveHeartbeat(),
            bus_publish=publisher,
        )
        assert result["action"] == PodTerminalOutcome.FAILED


# ---------------------------------------------------------------------------
# 6. Regressions — backwards-compat dashboard fields
# ---------------------------------------------------------------------------


class TestRegressions:
    async def test_pod_stop_attempt_event_carries_outcome_field(self) -> None:
        # Phase 9.A dashboards parsed ``payload["outcome"]``. Phase 11.B
        # introduces ``decision`` / ``action`` but keeps ``outcome``
        # populated with the action string for backwards compat.
        publisher = _PublisherSpy()
        terminator = _mk_terminator()
        await terminator.decide_and_act(
            terminal_state="cancelled",
            heartbeat=_AlwaysAliveHeartbeat(),
            bus_publish=publisher,
        )

        attempts = [
            payload for kind, payload in publisher.events
            if kind == "pod_stop_attempt"
        ]
        assert len(attempts) == 1
        # New fields:
        assert attempts[0]["decision"] == PodTerminalOutcome.TERMINATED_USER_STOP
        assert attempts[0]["action"] == PodTerminalOutcome.TERMINATED
        # Backwards-compat field:
        assert attempts[0]["outcome"] == PodTerminalOutcome.TERMINATED

    async def test_decision_event_includes_full_context(self) -> None:
        # Operator dashboards debug "why this outcome" by reading the
        # decision event payload — must carry the inputs that drove it.
        publisher = _PublisherSpy()
        terminator = _mk_terminator(keep_on_error=False)
        await terminator.decide_and_act(
            terminal_state="completed",
            heartbeat=_AlwaysDeadHeartbeat(),
            bus_publish=publisher,
        )

        decisions = [
            payload for kind, payload in publisher.events
            if kind == "pod_terminal_decision"
        ]
        assert len(decisions) == 1
        d = decisions[0]
        assert d["decision"] == PodTerminalOutcome.STOPPED_FOR_RESUME
        assert d["terminal_state"] == "completed"
        assert d["mac_alive"] is False
        assert d["volume_kind"] == "persistent"
        assert d["keep_on_error"] is False
        assert d["heartbeat_age_seconds"] == 600.0

    async def test_pod_stop_attempt_carries_phase_14b_fields(self) -> None:
        # Phase 14.B adds ``provider`` + ``attempts_made`` to the
        # ``pod_stop_attempt`` event so dashboards can disambiguate
        # outcomes (e.g. ``skipped`` from runpod-no-creds vs
        # ``skipped`` from single-node-no-op).
        publisher = _PublisherSpy()
        client = _FakeLifecycleClient(
            terminate_outcome=PodTerminalOutcome.TERMINATED, attempts_made=2,
        )
        terminator = _mk_terminator(client=client)
        await terminator.decide_and_act(
            terminal_state="cancelled",
            heartbeat=_AlwaysAliveHeartbeat(),
            bus_publish=publisher,
        )
        attempts = [
            payload for kind, payload in publisher.events
            if kind == "pod_stop_attempt"
        ]
        assert attempts[0]["provider"] == PROVIDER_RUNPOD
        assert attempts[0]["attempts_made"] == 2


# ---------------------------------------------------------------------------
# 7. Logic-specific — grace loop
# ---------------------------------------------------------------------------


class TestGraceLoop:
    async def test_grace_exits_on_heartbeat_lost_after_base(self) -> None:
        # Heartbeat alive for first 6 ticks (>= base 0.05s @ 0.01s tick),
        # then dies → loop exits.
        # base 0.05 / tick 0.01 = 5 ticks until base reached.
        # After base, the loop checks heartbeat at every tick.
        # Schedule: alive for 6 ticks, then dead.
        heartbeat = _ScriptedHeartbeat([True] * 6 + [False])

        publisher = _PublisherSpy()
        terminator = _mk_terminator(grace_base=0.05, grace_max=10.0, grace_tick=0.01)

        await terminator.decide_and_act(
            terminal_state="completed",
            heartbeat=heartbeat,
            bus_publish=publisher,
        )

        # Grace started + grace ended events emitted.
        kinds = [k for k, _ in publisher.events]
        assert "pod_terminal_grace_started" in kinds
        assert "pod_terminal_grace_ended" in kinds

        # End reason = heartbeat_lost
        ends = [
            payload for kind, payload in publisher.events
            if kind == "pod_terminal_grace_ended"
        ]
        assert ends[0]["reason"] == "heartbeat_lost"

    async def test_grace_exits_on_max_budget(self) -> None:
        # Heartbeat stays alive forever; max_budget kicks us out.
        publisher = _PublisherSpy()
        # tick=0.01s, max=0.05s → exits after 5 ticks regardless.
        terminator = _mk_terminator(grace_base=0.001, grace_max=0.05, grace_tick=0.01)

        await terminator.decide_and_act(
            terminal_state="completed",
            heartbeat=_AlwaysAliveHeartbeat(),
            bus_publish=publisher,
        )

        ends = [
            payload for kind, payload in publisher.events
            if kind == "pod_terminal_grace_ended"
        ]
        assert ends[0]["reason"] == "max_budget_exceeded"

    async def test_grace_followed_by_pause_call(self) -> None:
        # After grace exits (any reason), pause is called via the
        # client (not terminate).
        publisher = _PublisherSpy()
        client = _FakeLifecycleClient()
        terminator = _mk_terminator(
            client=client,
            grace_base=0.001, grace_max=0.05, grace_tick=0.01,
        )

        result = await terminator.decide_and_act(
            terminal_state="completed",
            heartbeat=_AlwaysAliveHeartbeat(),
            bus_publish=publisher,
        )

        assert result["decision"] == PodTerminalOutcome.STOPPED_FOR_RESUME_SHORT_GRACE
        assert result["action"] == PodTerminalOutcome.STOPPED
        # Exactly one pause call, no terminate call.
        method_names = [m for m, _ in client.calls]
        assert method_names == ["pause"]

    async def test_user_stop_uses_terminate_not_pause(self) -> None:
        publisher = _PublisherSpy()
        client = _FakeLifecycleClient()
        terminator = _mk_terminator(client=client)
        await terminator.decide_and_act(
            terminal_state="cancelled",
            heartbeat=_AlwaysAliveHeartbeat(),
            bus_publish=publisher,
        )
        method_names = [m for m, _ in client.calls]
        assert method_names == ["terminate"]


# ---------------------------------------------------------------------------
# 7b. Phase 14.B — resource_id wired through to client calls
# ---------------------------------------------------------------------------


class TestResourceIdPropagation:
    async def test_resource_id_passed_to_terminate(self) -> None:
        publisher = _PublisherSpy()
        client = _FakeLifecycleClient()
        terminator = _mk_terminator(client=client, resource_id="pod-zzz")
        await terminator.decide_and_act(
            terminal_state="cancelled",
            heartbeat=_AlwaysAliveHeartbeat(),
            bus_publish=publisher,
        )
        assert client.calls == [("terminate", "pod-zzz")]

    async def test_resource_id_passed_to_pause(self) -> None:
        publisher = _PublisherSpy()
        client = _FakeLifecycleClient()
        terminator = _mk_terminator(client=client, resource_id="pod-zzz")
        await terminator.decide_and_act(
            terminal_state="completed",
            heartbeat=_AlwaysDeadHeartbeat(),  # → STOPPED_FOR_RESUME
            bus_publish=publisher,
        )
        assert client.calls == [("pause", "pod-zzz")]
