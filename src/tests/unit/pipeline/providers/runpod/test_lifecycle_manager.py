from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import pytest

import src.providers.runpod.training.lifecycle_manager as lm
from src.providers.runpod.models import PodSnapshot, SshEndpoint
from src.providers.runpod.training.lifecycle_manager import PodLifecycleManager
from src.utils.result import Err, Ok, ProviderError, Result


def _snap(
    *,
    status: str = "RUNNING",
    uptime: int = 0,
    ssh: SshEndpoint | None = None,
    port_count: int | None = None,
) -> PodSnapshot:
    """Build a PodSnapshot for testing."""
    pc = port_count if port_count is not None else (1 if ssh else 0)
    return PodSnapshot(pod_id="pod-1", status=status, uptime_seconds=uptime, ssh_endpoint=ssh, port_count=pc)


_SSH_OK = SshEndpoint(host="1.2.3.4", port=12345)


@dataclass
class StubAPI:
    responses: list[Result[PodSnapshot, ProviderError]] = field(default_factory=list)
    calls: int = 0

    def query_pod_snapshot(self, pod_id: str) -> Result[PodSnapshot, ProviderError]:
        self.calls += 1
        if self.responses:
            return self.responses.pop(0)
        return Err(ProviderError(message="no more responses", code="STUB_NO_RESPONSE"))


def _fake_time(start: float = 0.0, step: float = 1.0) -> Callable[[], float]:
    t = {"v": start}

    def now() -> float:
        t["v"] += step
        return t["v"]

    return now


def test_wait_single_attempt_success_when_running_and_ssh_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    api = StubAPI(responses=[Ok(_snap(status="RUNNING", uptime=1, ssh=_SSH_OK))])
    mgr = PodLifecycleManager(api_client=api)

    monkeypatch.setattr(lm.time, "sleep", lambda s: None)
    monkeypatch.setattr(lm.time, "time", _fake_time())

    res = mgr._wait_single_attempt("pod-1", timeout=10)
    assert res.is_success()
    assert res.unwrap().is_ready
    assert api.calls == 1


def test_wait_single_attempt_running_without_ssh_then_ssh_appears(monkeypatch: pytest.MonkeyPatch) -> None:
    api = StubAPI(
        responses=[
            Ok(_snap(status="RUNNING", uptime=1)),
            Ok(_snap(status="RUNNING", uptime=2, ssh=_SSH_OK)),
        ]
    )
    mgr = PodLifecycleManager(api_client=api)

    monkeypatch.setattr(lm.time, "sleep", lambda s: None)
    monkeypatch.setattr(lm.time, "time", _fake_time())

    res = mgr._wait_single_attempt("pod-1", timeout=10)
    assert res.is_success()
    assert api.calls == 2


def test_wait_single_attempt_running_with_non_ssh_ports_keeps_waiting(monkeypatch: pytest.MonkeyPatch) -> None:
    api = StubAPI(
        responses=[
            Ok(_snap(status="RUNNING", uptime=1, port_count=1)),
            Ok(_snap(status="RUNNING", uptime=2, ssh=_SSH_OK)),
        ]
    )
    mgr = PodLifecycleManager(api_client=api)

    monkeypatch.setattr(lm.time, "sleep", lambda s: None)
    monkeypatch.setattr(lm.time, "time", _fake_time())

    res = mgr._wait_single_attempt("pod-1", timeout=10)
    assert res.is_success()
    assert api.calls == 2


def test_wait_single_attempt_failed_state(monkeypatch: pytest.MonkeyPatch) -> None:
    api = StubAPI(responses=[Ok(_snap(status="FAILED"))])
    mgr = PodLifecycleManager(api_client=api)

    monkeypatch.setattr(lm.time, "sleep", lambda s: None)
    monkeypatch.setattr(lm.time, "time", _fake_time())

    res = mgr._wait_single_attempt("pod-1", timeout=10)
    assert res.is_failure()
    assert "FAILED" in str(res.unwrap_err())


def test_wait_single_attempt_stuck_detection(monkeypatch: pytest.MonkeyPatch) -> None:
    api = StubAPI(responses=[Ok(_snap(status="STARTING"))] * 50)
    mgr = PodLifecycleManager(api_client=api)

    monkeypatch.setattr(lm.time, "sleep", lambda s: None)
    monkeypatch.setattr(lm.time, "time", _fake_time())

    res = mgr._wait_single_attempt("pod-1", timeout=300)
    assert res.is_failure()
    assert "timeout" in str(res.unwrap_err()).lower()


def test_wait_single_attempt_no_exposed_tcp_fails_after_grace(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pod with ports but no SSH exposed TCP should fail after 30s grace period."""
    snap_no_ssh = _snap(status="RUNNING", uptime=30, port_count=2, ssh=None)
    api = StubAPI(responses=[Ok(snap_no_ssh)] * 50)
    mgr = PodLifecycleManager(api_client=api)

    monkeypatch.setattr(lm.time, "sleep", lambda s: None)
    clock = _fake_time(step=10)
    monkeypatch.setattr(lm.time, "time", clock)

    res = mgr._wait_single_attempt("pod-1", timeout=600)
    assert res.is_failure()
    err = res.unwrap_err()
    assert err.code == "RUNPOD_NO_EXPOSED_TCP"


def test_wait_for_ready_delegates_to_single_attempt(monkeypatch: pytest.MonkeyPatch) -> None:
    """wait_for_ready no longer retries; it delegates to _wait_single_attempt once."""
    api = StubAPI()
    mgr = PodLifecycleManager(api_client=api)

    monkeypatch.setattr(lm.time, "sleep", lambda s: None)

    ready_snap = _snap(status="RUNNING", uptime=5, ssh=_SSH_OK)
    monkeypatch.setattr(mgr, "_wait_single_attempt", lambda pod_id, timeout: Ok(ready_snap))

    res = mgr.wait_for_ready("pod-1", timeout=1)
    assert res.is_success()
    assert res.unwrap() == ready_snap


def test_wait_for_ready_returns_pod_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    expected = _snap(status="RUNNING", uptime=42, ssh=_SSH_OK)
    api = StubAPI(responses=[Ok(expected)])
    mgr = PodLifecycleManager(api_client=api)

    monkeypatch.setattr(lm.time, "sleep", lambda s: None)
    monkeypatch.setattr(lm.time, "time", _fake_time())

    res = mgr.wait_for_ready("pod-1", timeout=10)
    assert res.is_success()
    snap = res.unwrap()
    assert isinstance(snap, PodSnapshot)
    assert snap.ssh_endpoint == _SSH_OK


def test_check_health_returns_true_for_running_pod(monkeypatch: pytest.MonkeyPatch) -> None:
    api = StubAPI(responses=[Ok(_snap(status="RUNNING", uptime=5))])
    mgr = PodLifecycleManager(api_client=api)
    res = mgr.check_health("pod-1")
    assert res.is_success()
    assert res.unwrap() is True


def test_check_health_returns_false_for_starting_pod(monkeypatch: pytest.MonkeyPatch) -> None:
    api = StubAPI(responses=[Ok(_snap(status="STARTING"))])
    mgr = PodLifecycleManager(api_client=api)
    res = mgr.check_health("pod-1")
    assert res.is_success()
    assert res.unwrap() is False
