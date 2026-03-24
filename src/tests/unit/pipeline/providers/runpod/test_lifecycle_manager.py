from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import pytest

import src.providers.runpod.training.lifecycle_manager as lm
from src.providers.runpod.training.lifecycle_manager import PodLifecycleManager
from src.utils.result import Err, Ok, Result
from src.utils.result import ProviderError


@dataclass
class StubAPI:
    responses: list[Result[dict[str, Any] | None, ProviderError]] = field(default_factory=list)
    calls: int = 0

    def query_pod(self, pod_id: str) -> Result[dict[str, Any] | None, ProviderError]:
        self.calls += 1
        if self.responses:
            return self.responses.pop(0)
        return Err(ProviderError(message="no more responses", code="STUB_NO_RESPONSE"))


@dataclass
class StubCleanup:
    cleaned: list[str] = field(default_factory=list)

    def cleanup_pod(self, pod_id: str) -> Result[None, ProviderError]:
        self.cleaned.append(pod_id)
        return Ok(None)


def _fake_time(start: float = 0.0) -> Callable[[], float]:
    t = {"v": start}

    def now() -> float:
        t["v"] += 1.0
        return t["v"]

    return now


def test_wait_single_attempt_success_when_running_and_ports(monkeypatch: pytest.MonkeyPatch) -> None:
    api = StubAPI(
        responses=[
            Ok(
                {
                    "desiredStatus": "RUNNING",
                    "runtime": {"uptimeInSeconds": 1, "ports": [{"ip": "x"}]},
                }
            )
        ]
    )
    cleanup = StubCleanup()
    mgr = PodLifecycleManager(api_client=api, cleanup_manager=cleanup)

    monkeypatch.setattr(lm.time, "sleep", lambda s: None)
    monkeypatch.setattr(lm.time, "time", _fake_time())

    res = mgr._wait_single_attempt("pod-1", timeout=10)
    assert res.is_success()
    assert api.calls == 1


def test_wait_single_attempt_running_without_ports_then_ports(monkeypatch: pytest.MonkeyPatch) -> None:
    api = StubAPI(
        responses=[
            Ok({"desiredStatus": "RUNNING", "runtime": {"uptimeInSeconds": 1, "ports": []}}),
            Ok({"desiredStatus": "RUNNING", "runtime": {"uptimeInSeconds": 2, "ports": [{"ip": "x"}]}}),
        ]
    )
    mgr = PodLifecycleManager(api_client=api, cleanup_manager=StubCleanup())

    monkeypatch.setattr(lm.time, "sleep", lambda s: None)
    monkeypatch.setattr(lm.time, "time", _fake_time())

    res = mgr._wait_single_attempt("pod-1", timeout=10)
    assert res.is_success()
    assert api.calls == 2


def test_wait_single_attempt_failed_state(monkeypatch: pytest.MonkeyPatch) -> None:
    api = StubAPI(responses=[Ok({"desiredStatus": "FAILED", "runtime": {}})])
    mgr = PodLifecycleManager(api_client=api, cleanup_manager=StubCleanup())

    monkeypatch.setattr(lm.time, "sleep", lambda s: None)
    monkeypatch.setattr(lm.time, "time", _fake_time())

    res = mgr._wait_single_attempt("pod-1", timeout=10)
    assert res.is_failure()
    assert "FAILED" in str(res.unwrap_err())


def test_wait_single_attempt_stuck_detection(monkeypatch: pytest.MonkeyPatch) -> None:
    api = StubAPI(responses=[Ok({"desiredStatus": "STARTING", "runtime": {"uptimeInSeconds": 0, "ports": []}})] * 50)
    mgr = PodLifecycleManager(api_client=api, cleanup_manager=StubCleanup())

    monkeypatch.setattr(lm.time, "sleep", lambda s: None)
    monkeypatch.setattr(lm.time, "time", _fake_time())

    res = mgr._wait_single_attempt("pod-1", timeout=300)
    assert res.is_failure()
    assert "timeout" in str(res.unwrap_err()).lower()


def test_wait_for_ready_retries_without_cleanup(monkeypatch: pytest.MonkeyPatch) -> None:
    api = StubAPI()
    cleanup = StubCleanup()
    mgr = PodLifecycleManager(api_client=api, cleanup_manager=cleanup)

    monkeypatch.setattr(lm.time, "sleep", lambda s: None)

    # First attempt fails, second succeeds
    seq = iter([Err("timeout"), Ok({"desiredStatus": "RUNNING"})])
    monkeypatch.setattr(mgr, "_wait_single_attempt", lambda pod_id, timeout: next(seq))

    res = mgr.wait_for_ready("pod-1", timeout=1, max_retries=2)
    assert res.is_success()
    assert cleanup.cleaned == []
