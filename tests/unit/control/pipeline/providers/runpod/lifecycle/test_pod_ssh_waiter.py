"""Tests for the canonical :class:`PodSshWaiter` primitive.

Inject fakes for every seam (``query``, ``clock``, ``sleep``,
``tcp_probe``) so the tests are deterministic and fast — no ``time.sleep``,
no socket calls.

Phase A2 Batch 11 (2026-05-15): raise-based contract.
``query_pod_snapshot`` returns ``PodSnapshot`` and raises typed
exceptions; ``wait`` returns ``PodSnapshot`` and raises typed
exceptions on terminal / stuck / timeout.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import pytest

from ryotenkai_shared.errors import ProviderUnavailableError
from ryotenkai_shared.utils.cancellation import PipelineCancelled
from ryotenkai_providers.runpod.lifecycle.pod_ssh_waiter import (
    PodQuery,
    PodSshWaiter,
)
from ryotenkai_providers.runpod.lifecycle.policy import WaitPolicy
from ryotenkai_providers.runpod.models import PodSnapshot, SshEndpoint

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
    """Raise-based fake for ``PodQuery``. Each action is a PodSnapshot
    (returned) or Exception (raised)."""

    actions: list[object] = field(default_factory=list)
    calls: int = 0

    def query_pod_snapshot(self, pod_id: str) -> PodSnapshot:
        self.calls += 1
        if not self.actions:
            raise ProviderUnavailableError(
                detail="exhausted", context={"code": "STUB_NO_RESPONSE"}
            )
        action = self.actions.pop(0)
        if isinstance(action, BaseException):
            raise action
        assert isinstance(action, PodSnapshot)
        return action


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
    actions: list[object],
    policy: WaitPolicy | None = None,
    sleep: Callable[[float], None] = _no_sleep,
    clock: FakeClock | None = None,
    tcp_probe: Callable[[str, int, float], bool] = _always_open,
) -> tuple[PodSshWaiter, FakeQuery]:
    query = FakeQuery(actions=list(actions))
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
    def test_returns_snapshot_when_ready_and_tcp_open(self) -> None:
        waiter, query = _make_waiter(
            actions=[_snap(status="RUNNING", ssh=_SSH_OK)],
        )
        out = waiter.wait("pod-1")
        assert out.is_ready
        assert query.calls == 1

    def test_returns_ok_only_when_both_status_and_tcp_ready(self) -> None:
        """RUNNING with SSH endpoint allocated but TCP refusing connections
        means sshd hasn't started yet — keep polling until it does."""
        waiter, query = _make_waiter(
            actions=[
                _snap(status="RUNNING", ssh=_SSH_OK),
                _snap(status="RUNNING", ssh=_SSH_OK),
            ],
            tcp_probe=_make_probe_sequence([False, True]),
        )
        out = waiter.wait("pod-1")
        assert out.is_ready
        assert query.calls == 2


def _make_probe_sequence(answers: list[bool]) -> Callable[[str, int, float], bool]:
    queue = list(answers)

    def probe(_host: str, _port: int, _timeout: float) -> bool:
        return queue.pop(0)

    return probe


# ---------------------------------------------------------------------------
# 2. Negative
# ---------------------------------------------------------------------------


class TestNegative:
    def test_terminal_status_raises_pod_failed(self) -> None:
        waiter, query = _make_waiter(actions=[_snap(status="FAILED")])
        with pytest.raises(ProviderUnavailableError) as ei:
            waiter.wait("pod-1")
        assert ei.value.context["code"] == "RUNPOD_POD_FAILED"
        assert query.calls == 1

    def test_running_with_ports_no_ssh_endpoint_bails_after_grace(self) -> None:
        snap = _snap(status="RUNNING", ssh=None, port_count=2)
        clock = FakeClock(step=10.0)
        waiter, _ = _make_waiter(
            actions=[snap] * 50,
            clock=clock,
            policy=WaitPolicy(
                total_timeout_s=600, poll_interval_s=5.0,
                no_exposed_tcp_grace_s=30,
            ),
        )
        with pytest.raises(ProviderUnavailableError) as ei:
            waiter.wait("pod-1")
        assert ei.value.context["code"] == "RUNPOD_NO_EXPOSED_TCP"

    def test_running_with_zero_ports_waits_full_timeout(self) -> None:
        snap = _snap(status="RUNNING", ssh=None, port_count=0)
        clock = FakeClock(step=120.0)
        waiter, _ = _make_waiter(
            actions=[snap] * 5,
            clock=clock,
            policy=WaitPolicy(
                total_timeout_s=60, poll_interval_s=5.0,
                no_exposed_tcp_grace_s=30,
            ),
        )
        with pytest.raises(ProviderUnavailableError) as ei:
            waiter.wait("pod-1")
        assert ei.value.context["code"] == "RUNPOD_POD_TIMEOUT"

    def test_total_timeout_exceeded(self) -> None:
        clock = FakeClock(step=120.0)
        waiter, _ = _make_waiter(
            actions=[_snap(status="STARTING")] * 5,
            clock=clock,
            policy=WaitPolicy(
                total_timeout_s=60, poll_interval_s=5.0,
                no_exposed_tcp_grace_s=30,
            ),
        )
        with pytest.raises(ProviderUnavailableError) as ei:
            waiter.wait("pod-1")
        assert ei.value.context["code"] == "RUNPOD_POD_TIMEOUT"

    def test_terminal_query_code_aborts_immediately(self) -> None:
        """Pod doesn't exist on RunPod side → terminal query error,
        do not retry."""
        waiter, query = _make_waiter(
            actions=[
                ProviderUnavailableError(
                    detail="anything", context={"code": "RUNPOD_POD_DATA_MISSING"}
                ),
            ],
        )
        with pytest.raises(ProviderUnavailableError) as ei:
            waiter.wait("pod-1")
        assert ei.value.context["code"] == "RUNPOD_POD_DATA_MISSING"
        assert query.calls == 1


# ---------------------------------------------------------------------------
# 3. Boundary
# ---------------------------------------------------------------------------


class TestBoundary:
    def test_grace_window_resets_on_status_change(self) -> None:
        snap_running_no_ssh = _snap(status="RUNNING", ssh=None, port_count=2)
        snap_provisioning = _snap(status="PROVISIONING", port_count=0)
        snap_ready = _snap(status="RUNNING", ssh=_SSH_OK)
        clock = FakeClock(step=10.0)
        waiter, _ = _make_waiter(
            actions=[
                snap_running_no_ssh,
                snap_running_no_ssh,
                snap_provisioning,  # resets grace timer
                snap_running_no_ssh,
                snap_ready,
            ],
            clock=clock,
            policy=WaitPolicy(
                total_timeout_s=600, poll_interval_s=5.0,
                no_exposed_tcp_grace_s=30,
            ),
        )
        out = waiter.wait("pod-1")
        assert out.is_ready

    def test_tcp_probe_disabled_returns_on_status_alone(self) -> None:
        called: list[tuple[str, int, float]] = []

        def tracking_probe(host: str, port: int, t: float) -> bool:
            called.append((host, port, t))
            return False

        waiter, _ = _make_waiter(
            actions=[_snap(status="RUNNING", ssh=_SSH_OK)],
            tcp_probe=tracking_probe,
            policy=WaitPolicy(
                total_timeout_s=300, poll_interval_s=5.0,
                no_exposed_tcp_grace_s=30,
                tcp_probe_enabled=False,
            ),
        )
        out = waiter.wait("pod-1")
        assert out.is_ready
        assert called == []  # probe not invoked


# ---------------------------------------------------------------------------
# 4. Invariants
# ---------------------------------------------------------------------------


class TestInvariants:
    def test_two_wait_calls_have_independent_state(self) -> None:
        snap_no_ports = _snap(status="RUNNING", ssh=None, port_count=0)
        clock = FakeClock(step=1.0)
        waiter, query = _make_waiter(
            actions=[_snap(status="RUNNING", ssh=_SSH_OK)],
            clock=clock,
        )
        first = waiter.wait("pod-1")
        assert first.is_ready
        query.actions = (
            [snap_no_ports] * 10 + [_snap(status="RUNNING", ssh=_SSH_OK)]
        )
        second = waiter.wait("pod-2")
        assert second.is_ready

    def test_transient_query_errors_do_not_abort(self) -> None:
        waiter, query = _make_waiter(
            actions=[
                ProviderUnavailableError(detail="net glitch", context={"code": "RUNPOD_TRANSIENT"}),
                ProviderUnavailableError(detail="another", context={"code": "RUNPOD_SDK_CALL_FAILED"}),
                _snap(status="RUNNING", ssh=_SSH_OK),
            ],
        )
        out = waiter.wait("pod-1")
        assert out.is_ready
        assert query.calls == 3


# ---------------------------------------------------------------------------
# 5. Cancellation — PipelineCancelled propagates
# ---------------------------------------------------------------------------


class TestCancellation:
    def test_pipeline_cancelled_in_sleep_propagates(self) -> None:
        def cancelled_sleep(_: float) -> None:
            raise PipelineCancelled()

        waiter, _ = _make_waiter(
            actions=[_snap(status="STARTING")] * 5,
            sleep=cancelled_sleep,
        )
        with pytest.raises(PipelineCancelled):
            waiter.wait("pod-1")

    def test_pipeline_cancelled_during_query_propagates(self) -> None:
        @dataclass
        class CancellingQuery:
            calls: int = 0

            def query_pod_snapshot(self, pod_id: str) -> PodSnapshot:
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
        ("RUNNING", 1, True, False, "tcp_no"),
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
        policy = WaitPolicy(total_timeout_s=2, poll_interval_s=1.0)
        clock = FakeClock(step=1.0)
    elif expected == "polling":
        policy = WaitPolicy(total_timeout_s=2, poll_interval_s=1.0)
    else:
        policy = WaitPolicy(total_timeout_s=300, poll_interval_s=5.0)
    waiter, _ = _make_waiter(
        actions=[snap] * 100,
        clock=clock,
        tcp_probe=_always_open if tcp_ok else _always_closed,
        policy=policy,
    )
    if expected == "ok":
        out = waiter.wait("pod-1")
        assert out.is_ready
    elif expected == "pod_failed":
        with pytest.raises(ProviderUnavailableError) as ei:
            waiter.wait("pod-1")
        assert ei.value.context["code"] == "RUNPOD_POD_FAILED"
    else:
        with pytest.raises(ProviderUnavailableError) as ei:
            waiter.wait("pod-1")
        assert ei.value.context["code"] == "RUNPOD_POD_TIMEOUT"
