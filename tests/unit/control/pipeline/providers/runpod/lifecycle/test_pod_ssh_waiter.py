"""Tests for the canonical :class:`PodSshWaiter` primitive.

Inject fakes for every seam (``query``, ``clock``, ``sleep``,
``tcp_probe``) so the tests are deterministic and fast — no ``time.sleep``,
no socket calls.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import pytest

from ryotenkai_shared.utils.cancellation import PipelineCancelled
from ryotenkai_providers.runpod.lifecycle.pod_ssh_waiter import (
    PodQuery,
    PodSshWaiter,
)
from ryotenkai_providers.runpod.lifecycle.policy import WaitPolicy
from ryotenkai_providers.runpod.models import PodSnapshot, SshEndpoint
from ryotenkai_shared.utils.result import Err, Ok, ProviderError, Result

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


_SSH_OK = SshEndpoint(host="1.2.3.4", port=23828)


def _snap(
    *,
    status: str = "RUNNING",
    uptime: int = 0,
    ssh: SshEndpoint | None = None,
    port_count: int | None = None,
    pod_id: str = "pod-1",
) -> PodSnapshot:
    pc = port_count if port_count is not None else (1 if ssh else 0)
    return PodSnapshot(
        pod_id=pod_id, status=status, uptime_seconds=uptime,
        ssh_endpoint=ssh, port_count=pc,
    )


@dataclass
class FakeQuery:
    responses: list[Result[PodSnapshot, ProviderError]] = field(default_factory=list)
    calls: int = 0

    def query_pod_snapshot(self, pod_id: str) -> Result[PodSnapshot, ProviderError]:
        self.calls += 1
        if not self.responses:
            return Err(ProviderError(message="exhausted", code="STUB_NO_RESPONSE"))
        return self.responses.pop(0)


@dataclass
class FakeClock:
    """Monotonic clock fake. Each call advances by ``step`` seconds."""

    start: float = 0.0
    step: float = 1.0
    current: float = 0.0

    def __post_init__(self) -> None:
        self.current = self.start

    def now(self) -> float:
        v = self.current
        self.current += self.step
        return v


def _no_sleep(_: float) -> None:
    """Fake sleep — instant, no-op."""


def _always_open(_host: str, _port: int, _timeout: float) -> bool:
    return True


def _always_closed(_host: str, _port: int, _timeout: float) -> bool:
    return False


def _make_waiter(
    *,
    responses: list[Result[PodSnapshot, ProviderError]],
    policy: WaitPolicy | None = None,
    sleep: Callable[[float], None] = _no_sleep,
    clock: FakeClock | None = None,
    tcp_probe: Callable[[str, int, float], bool] = _always_open,
) -> tuple[PodSshWaiter, FakeQuery]:
    query = FakeQuery(responses=list(responses))
    pol = policy or WaitPolicy(
        total_timeout_s=300, poll_interval_s=5.0,
        no_exposed_tcp_grace_s=30,
    )
    cl = clock or FakeClock(step=1.0)
    waiter = PodSshWaiter(
        query=query, policy=pol, log=lambda _level, _msg: None,
        clock=cl.now, sleep=sleep, tcp_probe=tcp_probe,
    )
    return waiter, query


# ---------------------------------------------------------------------------
# 1. Positive
# ---------------------------------------------------------------------------


class TestPositive:
    def test_returns_ok_when_ready_and_tcp_open(self) -> None:
        waiter, query = _make_waiter(
            responses=[Ok(_snap(status="RUNNING", ssh=_SSH_OK))],
        )
        res = waiter.wait("pod-1")
        assert res.is_success()
        assert res.unwrap().is_ready
        assert query.calls == 1

    def test_returns_ok_only_when_both_status_and_tcp_ready(self) -> None:
        """RUNNING with SSH endpoint allocated but TCP refusing connections
        means sshd hasn't started yet — keep polling until it does."""
        waiter, query = _make_waiter(
            responses=[
                Ok(_snap(status="RUNNING", ssh=_SSH_OK)),  # tcp_probe says NO
                Ok(_snap(status="RUNNING", ssh=_SSH_OK)),  # tcp_probe says YES
            ],
            tcp_probe=_make_probe_sequence([False, True]),
        )
        res = waiter.wait("pod-1")
        assert res.is_success()
        assert query.calls == 2


def _make_probe_sequence(answers: list[bool]) -> Callable[[str, int, float], bool]:
    """Probe that returns the next answer from the list each call."""
    queue = list(answers)

    def probe(_host: str, _port: int, _timeout: float) -> bool:
        return queue.pop(0)

    return probe


# ---------------------------------------------------------------------------
# 2. Negative
# ---------------------------------------------------------------------------


class TestNegative:
    def test_terminal_status_returns_pod_failed(self) -> None:
        waiter, query = _make_waiter(
            responses=[Ok(_snap(status="FAILED"))],
        )
        res = waiter.wait("pod-1")
        assert res.is_failure()
        assert res.unwrap_err().code == "RUNPOD_POD_FAILED"
        assert query.calls == 1  # bailed on first poll, didn't loop

    def test_running_with_ports_no_ssh_endpoint_bails_after_grace(self) -> None:
        """RUNNING + ports>0 but no SSH endpoint → community-cloud limitation."""
        snap = _snap(status="RUNNING", ssh=None, port_count=2)
        # Many polls so we exceed the grace window.
        clock = FakeClock(step=10.0)
        waiter, _ = _make_waiter(
            responses=[Ok(snap)] * 50,
            clock=clock,
            policy=WaitPolicy(
                total_timeout_s=600, poll_interval_s=5.0,
                no_exposed_tcp_grace_s=30,
            ),
        )
        res = waiter.wait("pod-1")
        assert res.is_failure()
        assert res.unwrap_err().code == "RUNPOD_NO_EXPOSED_TCP"

    def test_running_with_zero_ports_waits_full_timeout(self) -> None:
        """RUNNING + ports==0 must NOT early-bail. The platform sometimes
        takes the full window to allocate ports; a mid-window cutoff
        forces retries on what would otherwise be a successful boot.

        Pre-refactor behaviour kept the loop polling until the deadline
        and then surfaced ``RUNPOD_POD_TIMEOUT`` — which the provider's
        ``_RECREATABLE_ERRORS`` filter handles as "fresh pod retry".
        """
        snap = _snap(status="RUNNING", ssh=None, port_count=0)
        clock = FakeClock(step=120.0)  # second poll lands past deadline
        waiter, _ = _make_waiter(
            responses=[Ok(snap)] * 5,
            clock=clock,
            policy=WaitPolicy(
                total_timeout_s=60, poll_interval_s=5.0,
                no_exposed_tcp_grace_s=30,
            ),
        )
        res = waiter.wait("pod-1")
        assert res.is_failure()
        assert res.unwrap_err().code == "RUNPOD_POD_TIMEOUT"

    def test_total_timeout_exceeded(self) -> None:
        clock = FakeClock(step=120.0)  # huge step — second poll past deadline
        waiter, _ = _make_waiter(
            responses=[Ok(_snap(status="STARTING"))] * 5,
            clock=clock,
            policy=WaitPolicy(
                total_timeout_s=60, poll_interval_s=5.0,
                no_exposed_tcp_grace_s=30,
            ),
        )
        res = waiter.wait("pod-1")
        assert res.is_failure()
        assert res.unwrap_err().code == "RUNPOD_POD_TIMEOUT"

    def test_terminal_query_code_aborts_immediately(self) -> None:
        """Pod doesn't exist on RunPod side → terminal query error,
        do not retry."""
        waiter, query = _make_waiter(
            responses=[
                Err(ProviderError(message="anything", code="RUNPOD_POD_DATA_MISSING")),
            ],
        )
        res = waiter.wait("pod-1")
        assert res.is_failure()
        assert res.unwrap_err().code == "RUNPOD_POD_DATA_MISSING"
        assert query.calls == 1


# ---------------------------------------------------------------------------
# 3. Boundary
# ---------------------------------------------------------------------------


class TestBoundary:
    def test_grace_window_resets_on_status_change(self) -> None:
        """RUNNING with ports>0 no SSH → status changes to PROVISIONING →
        back to RUNNING. The grace timer must reset; the second RUNNING
        stretch shouldn't count time from the first."""
        snap_running_no_ssh = _snap(status="RUNNING", ssh=None, port_count=2)
        snap_provisioning = _snap(status="PROVISIONING", port_count=0)
        snap_ready = _snap(status="RUNNING", ssh=_SSH_OK)
        clock = FakeClock(step=10.0)
        waiter, _ = _make_waiter(
            responses=[
                Ok(snap_running_no_ssh),
                Ok(snap_running_no_ssh),
                Ok(snap_provisioning),  # resets grace timer
                Ok(snap_running_no_ssh),
                Ok(snap_ready),
            ],
            clock=clock,
            policy=WaitPolicy(
                total_timeout_s=600, poll_interval_s=5.0,
                no_exposed_tcp_grace_s=30,
            ),
        )
        res = waiter.wait("pod-1")
        assert res.is_success()

    def test_tcp_probe_disabled_returns_on_status_alone(self) -> None:
        """``tcp_probe_enabled=False`` means status-only readiness check —
        RUNNING + SSH endpoint = ready, no socket call."""
        called: list[tuple[str, int, float]] = []

        def tracking_probe(host: str, port: int, t: float) -> bool:
            called.append((host, port, t))
            return False  # would block forever if invoked

        waiter, _ = _make_waiter(
            responses=[Ok(_snap(status="RUNNING", ssh=_SSH_OK))],
            tcp_probe=tracking_probe,
            policy=WaitPolicy(
                total_timeout_s=300, poll_interval_s=5.0,
                no_exposed_tcp_grace_s=30,
                tcp_probe_enabled=False,
            ),
        )
        res = waiter.wait("pod-1")
        assert res.is_success()
        assert called == []  # probe not invoked


# ---------------------------------------------------------------------------
# 4. Invariants
# ---------------------------------------------------------------------------


class TestInvariants:
    def test_two_wait_calls_have_independent_state(self) -> None:
        """Two ``wait()`` calls must not share the running_no_ports timer
        between invocations — fresh state each call."""
        snap_no_ports = _snap(status="RUNNING", ssh=None, port_count=0)
        clock = FakeClock(step=1.0)
        # First wait — short response, succeeds.
        waiter, query = _make_waiter(
            responses=[
                Ok(_snap(status="RUNNING", ssh=_SSH_OK)),
            ],
            clock=clock,
        )
        first = waiter.wait("pod-1")
        assert first.is_success()
        # Reset query for second wait.
        query.responses = [Ok(snap_no_ports)] * 10 + [Ok(_snap(status="RUNNING", ssh=_SSH_OK))]
        second = waiter.wait("pod-2")
        assert second.is_success()

    def test_transient_query_errors_do_not_abort(self) -> None:
        """Non-terminal query errors → keep polling."""
        waiter, query = _make_waiter(
            responses=[
                Err(ProviderError(message="net glitch", code="RUNPOD_TRANSIENT")),
                Err(ProviderError(message="another", code="RUNPOD_SDK_CALL_FAILED")),
                Ok(_snap(status="RUNNING", ssh=_SSH_OK)),
            ],
        )
        res = waiter.wait("pod-1")
        assert res.is_success()
        assert query.calls == 3


# ---------------------------------------------------------------------------
# 5. Cancellation — PipelineCancelled propagates
# ---------------------------------------------------------------------------


class TestCancellation:
    def test_pipeline_cancelled_in_sleep_propagates(self) -> None:
        """The waiter MUST NOT catch ``PipelineCancelled`` — provider's
        cleanup hook owns that responsibility."""

        def cancelled_sleep(_: float) -> None:
            raise PipelineCancelled()

        waiter, _ = _make_waiter(
            responses=[Ok(_snap(status="STARTING"))] * 5,
            sleep=cancelled_sleep,
        )
        with pytest.raises(PipelineCancelled):
            waiter.wait("pod-1")

    def test_pipeline_cancelled_during_query_propagates(self) -> None:
        @dataclass
        class CancellingQuery:
            calls: int = 0

            def query_pod_snapshot(
                self, pod_id: str
            ) -> Result[PodSnapshot, ProviderError]:
                self.calls += 1
                raise PipelineCancelled()

        cl = FakeClock(step=1.0)
        pol = WaitPolicy(total_timeout_s=300, poll_interval_s=5.0)
        waiter = PodSshWaiter(
            query=CancellingQuery(), policy=pol, log=lambda _l, _m: None,
            clock=cl.now, sleep=_no_sleep, tcp_probe=_always_open,
        )
        with pytest.raises(PipelineCancelled):
            waiter.wait("pod-1")


# ---------------------------------------------------------------------------
# 6. Logic-specific — combinatorial verdict matrix
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "status,port_count,ssh_ok,tcp_ok,expected",
    [
        ("RUNNING", 1, True, True, "ok"),
        ("RUNNING", 1, True, False, "tcp_no"),  # keeps polling, eventually times out
        ("FAILED", 0, False, None, "pod_failed"),
        ("EXITED", 0, False, None, "pod_failed"),
        ("STARTING", 0, False, None, "polling"),
    ],
)
def test_combinatorial_status_matrix(
    status: str, port_count: int, ssh_ok: bool, tcp_ok: bool | None, expected: str
) -> None:
    ssh = _SSH_OK if ssh_ok else None
    snap = _snap(status=status, port_count=port_count, ssh=ssh)
    clock = FakeClock(step=1.0)
    if expected == "tcp_no":
        # Force timeout fast so we exit the loop deterministically.
        policy = WaitPolicy(total_timeout_s=2, poll_interval_s=1.0)
        clock = FakeClock(step=1.0)
    elif expected == "polling":
        policy = WaitPolicy(total_timeout_s=2, poll_interval_s=1.0)
    else:
        policy = WaitPolicy(total_timeout_s=300, poll_interval_s=5.0)
    waiter, _ = _make_waiter(
        responses=[Ok(snap)] * 100,
        clock=clock,
        tcp_probe=_always_open if tcp_ok else _always_closed,
        policy=policy,
    )
    res = waiter.wait("pod-1")
    if expected == "ok":
        assert res.is_success()
    elif expected == "pod_failed":
        assert res.unwrap_err().code == "RUNPOD_POD_FAILED"
    elif expected in ("tcp_no", "polling"):
        # Both should hit total-timeout because we don't reach a ready
        # state within the budget.
        assert res.unwrap_err().code == "RUNPOD_POD_TIMEOUT"
