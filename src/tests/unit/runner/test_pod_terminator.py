"""Phase 11.B — :class:`PodTerminator` + :func:`decide_terminal_outcome` contract.

Replaces Phase 9.A's ``test_pod_stopper.py``. The pure decision
function is exhaustively pinned (8 rows of the matrix in § 11.1);
the action dispatcher is tested with mock GraphQL transports so the
tests run instantly without hitting RunPod.

7-category coverage:

1. **Positive** — happy paths for each decision branch.
2. **Negative** — missing creds → SKIPPED; KEPT_ALIVE_FOR_DEBUG
   short-circuits without GraphQL.
3. **Boundary** — decision matrix corner cases (network volume,
   unknown terminal_state, KEEP_ON_ERROR honoured only on failed).
4. **Invariants** — ``run_terminal_hook`` never propagates errors;
   action dispatcher is idempotent (already-gone → ALREADY_*).
5. **Dependency errors** — flaky GraphQL → retries → FAILED.
6. **Regressions** — Phase 9.A ``"outcome"`` event field still
   present alongside the new ``decision`` / ``action`` fields.
7. **Logic-specific** — grace loop exits on heartbeat-lost OR
   max-budget; podStop fires after grace.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from src.runner.heartbeat import MacHeartbeat
from src.runner.pod_terminator import (
    DEFAULT_RUNPOD_GRAPHQL_URL,
    PodTerminalOutcome,
    PodTerminator,
    decide_terminal_outcome,
    run_terminal_hook,
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


def _mock_transport(handler):
    """httpx mock transport — handler returns httpx.Response."""
    return httpx.MockTransport(handler)


def _mk_terminator(
    *,
    transport_handler=None,
    sleep=None,
    grace_base: float = 0.05,
    grace_max: float = 0.5,
    grace_tick: float = 0.01,
) -> PodTerminator:
    """Build a PodTerminator with deterministic defaults for tests."""

    async def _no_sleep(_: float) -> None:
        return

    if transport_handler is None:
        # Default: every request returns 200 + success body for podStop AND podTerminate.
        def default_handler(req: httpx.Request) -> httpx.Response:
            body = req.read()
            text = body.decode()
            if "podTerminate" in text:
                return httpx.Response(200, text='{"data":{"podTerminate":null}}')
            if "podStop" in text:
                return httpx.Response(200, text='{"data":{"podStop":null}}')
            return httpx.Response(500, text='{"errors":[{"message":"unknown"}]}')

        transport_handler = default_handler

    transport = _mock_transport(transport_handler)

    def factory() -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=transport, base_url="http://test")

    return PodTerminator(
        graphql_url=DEFAULT_RUNPOD_GRAPHQL_URL,
        http_client_factory=factory,
        sleep=sleep or _no_sleep,
        grace_base_seconds=grace_base,
        grace_max_seconds=grace_max,
        grace_tick_seconds=grace_tick,
    )


def _env(**kw: str) -> dict[str, str]:
    """Build an env dict with the right keys."""
    base = {
        "RUNPOD_API_KEY": "rk-secret",
        "RUNPOD_POD_ID": "pod-abc",
    }
    base.update(kw)
    return base


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
                        f"failed for mac_alive={mac_alive} "
                        f"volume={volume_kind} keep={keep_on_error}"
                    )

    def test_failed_with_keep_on_error_kept_alive(self) -> None:
        out = decide_terminal_outcome(
            terminal_state="failed",
            mac_alive=False,
            volume_kind="persistent",
            keep_on_error=True,
        )
        assert out == PodTerminalOutcome.KEPT_ALIVE_FOR_DEBUG

    def test_failed_mac_alive_persistent_terminates(self) -> None:
        out = decide_terminal_outcome(
            terminal_state="failed",
            mac_alive=True,
            volume_kind="persistent",
            keep_on_error=False,
        )
        assert out == PodTerminalOutcome.TERMINATED_SAFETY

    def test_failed_mac_asleep_persistent_stops(self) -> None:
        out = decide_terminal_outcome(
            terminal_state="failed",
            mac_alive=False,
            volume_kind="persistent",
            keep_on_error=False,
        )
        assert out == PodTerminalOutcome.STOPPED_FOR_RESUME

    def test_failed_network_volume_terminates(self) -> None:
        out = decide_terminal_outcome(
            terminal_state="failed",
            mac_alive=False,
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
        out = decide_terminal_outcome(
            terminal_state="completed",
            mac_alive=True,
            volume_kind="network",
            keep_on_error=False,
        )
        assert out == PodTerminalOutcome.TERMINATED_SAFETY


# ---------------------------------------------------------------------------
# 2. Negative — missing creds, KEPT_ALIVE no-op
# ---------------------------------------------------------------------------


class TestNegative:
    async def test_kept_alive_skips_graphql(self) -> None:
        publisher = _PublisherSpy()
        called = {"count": 0}

        def transport(req: httpx.Request) -> httpx.Response:
            called["count"] += 1
            return httpx.Response(200, text='{"data":{}}')

        terminator = _mk_terminator(transport_handler=transport)
        result = await terminator.decide_and_act(
            terminal_state="failed",
            heartbeat=_AlwaysAliveHeartbeat(),
            bus_publish=publisher,
            env=_env(RUNPOD_KEEP_ON_ERROR="true"),
        )

        assert result["decision"] == PodTerminalOutcome.KEPT_ALIVE_FOR_DEBUG
        assert result["action"] is None
        assert called["count"] == 0  # No GraphQL call

    async def test_missing_api_key_yields_skipped(self) -> None:
        publisher = _PublisherSpy()
        terminator = _mk_terminator()
        env = {"RUNPOD_POD_ID": "pod-abc"}  # no API key
        result = await terminator.decide_and_act(
            terminal_state="cancelled",
            heartbeat=_AlwaysAliveHeartbeat(),
            bus_publish=publisher,
            env=env,
        )
        assert result["action"] == PodTerminalOutcome.SKIPPED

    async def test_missing_pod_id_yields_skipped(self) -> None:
        publisher = _PublisherSpy()
        terminator = _mk_terminator()
        env = {"RUNPOD_API_KEY": "rk-x"}  # no pod_id
        result = await terminator.decide_and_act(
            terminal_state="cancelled",
            heartbeat=_AlwaysAliveHeartbeat(),
            bus_publish=publisher,
            env=env,
        )
        assert result["action"] == PodTerminalOutcome.SKIPPED


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

    async def test_unknown_volume_kind_treated_as_persistent(self) -> None:
        # Env value other than "persistent"/"network" → fall back to
        # persistent (safer default).
        publisher = _PublisherSpy()
        terminator = _mk_terminator()
        result = await terminator.decide_and_act(
            terminal_state="completed",
            heartbeat=_AlwaysDeadHeartbeat(),
            bus_publish=publisher,
            env=_env(RUNPOD_VOLUME_KIND="garbage"),
        )
        assert result["decision"] == PodTerminalOutcome.STOPPED_FOR_RESUME


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
            env=_env(),
        )
        # Error event published for forensics.
        kinds = [k for k, _ in publisher.events]
        assert "pod_stop_error" in kinds

    async def test_already_terminated_idempotent(self) -> None:
        # GraphQL returns "already terminated" → ALREADY_TERMINATED.
        def handler(_req: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                text='{"errors":[{"message":"pod does not exist"}]}',
            )

        terminator = _mk_terminator(transport_handler=handler)
        publisher = _PublisherSpy()
        result = await terminator.decide_and_act(
            terminal_state="cancelled",
            heartbeat=_AlwaysAliveHeartbeat(),
            bus_publish=publisher,
            env=_env(),
        )
        assert result["action"] == PodTerminalOutcome.ALREADY_TERMINATED

    async def test_already_stopped_idempotent(self) -> None:
        def handler(_req: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                text='{"errors":[{"message":"pod is already stopped"}]}',
            )

        terminator = _mk_terminator(transport_handler=handler)
        publisher = _PublisherSpy()
        result = await terminator.decide_and_act(
            terminal_state="completed",
            heartbeat=_AlwaysDeadHeartbeat(),  # asleep → STOPPED_FOR_RESUME
            bus_publish=publisher,
            env=_env(),
        )
        assert result["action"] == PodTerminalOutcome.ALREADY_STOPPED


# ---------------------------------------------------------------------------
# 5. Dependency errors — flaky GraphQL
# ---------------------------------------------------------------------------


class TestDependencyErrors:
    async def test_retry_exhaustion_returns_failed(self) -> None:
        # All 3 attempts return 500 — should land on FAILED.
        def handler(_req: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text='{"errors":["server boom"]}')

        terminator = _mk_terminator(transport_handler=handler)
        publisher = _PublisherSpy()
        result = await terminator.decide_and_act(
            terminal_state="cancelled",
            heartbeat=_AlwaysAliveHeartbeat(),
            bus_publish=publisher,
            env=_env(),
        )
        assert result["action"] == PodTerminalOutcome.FAILED

    async def test_network_error_retried_then_failed(self) -> None:
        attempts = {"n": 0}

        def handler(_req: httpx.Request) -> httpx.Response:
            attempts["n"] += 1
            raise httpx.ConnectError("connection refused")

        terminator = _mk_terminator(transport_handler=handler)
        publisher = _PublisherSpy()
        result = await terminator.decide_and_act(
            terminal_state="cancelled",
            heartbeat=_AlwaysAliveHeartbeat(),
            bus_publish=publisher,
            env=_env(),
        )
        # All 3 retries hit the connect error → FAILED.
        assert result["action"] == PodTerminalOutcome.FAILED
        assert attempts["n"] == 3


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
            env=_env(),
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
        terminator = _mk_terminator()
        await terminator.decide_and_act(
            terminal_state="completed",
            heartbeat=_AlwaysDeadHeartbeat(),
            bus_publish=publisher,
            env=_env(RUNPOD_KEEP_ON_ERROR="false"),
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
            env=_env(),
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
            env=_env(),
        )

        ends = [
            payload for kind, payload in publisher.events
            if kind == "pod_terminal_grace_ended"
        ]
        assert ends[0]["reason"] == "max_budget_exceeded"

    async def test_grace_followed_by_pod_stop_call(self) -> None:
        # After grace exits (any reason), podStop is called.
        called = {"podStop": 0, "podTerminate": 0}

        def handler(req: httpx.Request) -> httpx.Response:
            text = req.read().decode()
            if "podStop" in text:
                called["podStop"] += 1
                return httpx.Response(200, text='{"data":{"podStop":null}}')
            if "podTerminate" in text:
                called["podTerminate"] += 1
                return httpx.Response(200, text='{"data":{"podTerminate":null}}')
            return httpx.Response(500)

        publisher = _PublisherSpy()
        terminator = _mk_terminator(
            transport_handler=handler,
            grace_base=0.001, grace_max=0.05, grace_tick=0.01,
        )

        result = await terminator.decide_and_act(
            terminal_state="completed",
            heartbeat=_AlwaysAliveHeartbeat(),
            bus_publish=publisher,
            env=_env(),
        )

        assert result["decision"] == PodTerminalOutcome.STOPPED_FOR_RESUME_SHORT_GRACE
        assert result["action"] == PodTerminalOutcome.STOPPED
        assert called["podStop"] == 1
        assert called["podTerminate"] == 0

    async def test_user_stop_uses_terminate_not_stop(self) -> None:
        # User-stop ⇒ podTerminate, never podStop. Sanity check that
        # the dispatcher routes correctly.
        called = {"podStop": 0, "podTerminate": 0}

        def handler(req: httpx.Request) -> httpx.Response:
            text = req.read().decode()
            if "podStop" in text:
                called["podStop"] += 1
                return httpx.Response(200, text='{"data":{"podStop":null}}')
            if "podTerminate" in text:
                called["podTerminate"] += 1
                return httpx.Response(200, text='{"data":{"podTerminate":null}}')
            return httpx.Response(500)

        publisher = _PublisherSpy()
        terminator = _mk_terminator(transport_handler=handler)

        await terminator.decide_and_act(
            terminal_state="cancelled",
            heartbeat=_AlwaysAliveHeartbeat(),
            bus_publish=publisher,
            env=_env(),
        )

        assert called["podTerminate"] == 1
        assert called["podStop"] == 0
