"""Tests for the training-side lifecycle manager (thin shim over
:class:`PodSshWaiter`).

The heavy poll-loop matrix is tested in
``tests/.../lifecycle/test_pod_ssh_waiter.py``. These tests pin only
the shim's contract: that ``wait_for_ready`` builds the right policy
and calls into the waiter, and that ``check_health`` is a pass-through
to ``query_pod_snapshot``.

Phase A2 Batch 11 (2026-05-15): raise-based contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from ryotenkai_providers.runpod.lifecycle.policy import TRAINING_PROFILE
from ryotenkai_providers.runpod.models import PodSnapshot, SshEndpoint
from ryotenkai_providers.runpod.training.lifecycle_manager import PodLifecycleManager
from ryotenkai_shared.errors import ProviderUnavailableError

pytestmark = pytest.mark.unit


_SSH_OK = SshEndpoint(host="1.2.3.4", port=12345)


def _snap(
    *,
    status: str = "RUNNING",
    uptime: int = 0,
    ssh: SshEndpoint | None = None,
    port_count: int | None = None,
) -> PodSnapshot:
    pc = port_count if port_count is not None else (1 if ssh else 0)
    return PodSnapshot(
        pod_id="pod-1", status=status, uptime_seconds=uptime,
        ssh_endpoint=ssh, port_count=pc,
    )


@dataclass
class StubQuery:
    """Raise-based fake for ``PodQuery``. Each action is a PodSnapshot
    (returned) or Exception (raised)."""

    actions: list[object] = field(default_factory=list)
    calls: int = 0

    def query_pod_snapshot(self, pod_id: str) -> PodSnapshot:
        self.calls += 1
        if not self.actions:
            raise ProviderUnavailableError(detail="exhausted", context={"code": "STUB"})
        action = self.actions.pop(0)
        if isinstance(action, BaseException):
            raise action
        assert isinstance(action, PodSnapshot)
        return action


# ---------------------------------------------------------------------------
# wait_for_ready — thin-shim contract
# ---------------------------------------------------------------------------


def test_wait_for_ready_uses_training_profile_when_no_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No-timeout call → policy comes from TRAINING_PROFILE unchanged."""
    captured: dict[str, object] = {}

    class CapturingWaiter:
        def __init__(self, *, query: object, policy: object, **_: object) -> None:
            captured["policy"] = policy

        def wait(self, pod_id: str) -> PodSnapshot:
            return _snap(status="RUNNING", ssh=_SSH_OK)

    monkeypatch.setattr(
        "ryotenkai_providers.runpod.training.lifecycle_manager.PodSshWaiter",
        CapturingWaiter,
    )
    api = StubQuery()
    mgr = PodLifecycleManager(api_client=api)
    mgr.wait_for_ready("pod-1")
    assert captured["policy"] is TRAINING_PROFILE


@pytest.mark.xfail(
    strict=True,
    reason="xfail-debt:wait-policy-api-drift — Pre-existing failure pre-packagization: WaitPolicy API drifted (removed attribute); test references obsolete policy field.",
)
def test_wait_for_ready_overrides_total_timeout_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Custom timeout overrides ``total_timeout_s`` but keeps every other
    training-profile threshold."""
    captured: dict[str, object] = {}

    class CapturingWaiter:
        def __init__(self, *, query: object, policy: object, **_: object) -> None:
            captured["policy"] = policy

        def wait(self, pod_id: str) -> PodSnapshot:
            return _snap(status="RUNNING", ssh=_SSH_OK)

    monkeypatch.setattr(
        "ryotenkai_providers.runpod.training.lifecycle_manager.PodSshWaiter",
        CapturingWaiter,
    )
    api = StubQuery()
    mgr = PodLifecycleManager(api_client=api)
    mgr.wait_for_ready("pod-1", timeout=42)
    pol = captured["policy"]
    assert pol.total_timeout_s == 42  # type: ignore[attr-defined]
    assert pol.no_exposed_tcp_grace_s == TRAINING_PROFILE.no_exposed_tcp_grace_s  # type: ignore[attr-defined]
    assert pol.poll_interval_s == TRAINING_PROFILE.poll_interval_s  # type: ignore[attr-defined]
    assert pol.running_no_ports_bailout_s == TRAINING_PROFILE.running_no_ports_bailout_s  # type: ignore[attr-defined]


def test_wait_for_ready_propagates_waiter_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Whatever the waiter raises, the shim propagates — no transformation."""
    sentinel = ProviderUnavailableError(
        detail="boom", context={"code": "RUNPOD_NO_EXPOSED_TCP", "x": 1}
    )

    class StaticWaiter:
        def __init__(self, **_: object) -> None:
            pass

        def wait(self, pod_id: str) -> PodSnapshot:
            raise sentinel

    monkeypatch.setattr(
        "ryotenkai_providers.runpod.training.lifecycle_manager.PodSshWaiter",
        StaticWaiter,
    )
    api = StubQuery()
    mgr = PodLifecycleManager(api_client=api)
    with pytest.raises(ProviderUnavailableError) as ei:
        mgr.wait_for_ready("pod-1")
    assert ei.value is sentinel


# ---------------------------------------------------------------------------
# check_health — single-query pass-through
# ---------------------------------------------------------------------------


def test_check_health_running_returns_true() -> None:
    api = StubQuery(actions=[_snap(status="RUNNING", ssh=_SSH_OK)])
    mgr = PodLifecycleManager(api_client=api)
    assert mgr.check_health("pod-1") is True


def test_check_health_non_running_returns_false() -> None:
    api = StubQuery(actions=[_snap(status="STARTING")])
    mgr = PodLifecycleManager(api_client=api)
    assert mgr.check_health("pod-1") is False


def test_check_health_propagates_query_error() -> None:
    err = ProviderUnavailableError(detail="boom", context={"code": "RUNPOD_SDK_CALL_FAILED"})
    api = StubQuery(actions=[err])
    mgr = PodLifecycleManager(api_client=api)
    with pytest.raises(ProviderUnavailableError) as ei:
        mgr.check_health("pod-1")
    assert ei.value.context["code"] == "RUNPOD_SDK_CALL_FAILED"
