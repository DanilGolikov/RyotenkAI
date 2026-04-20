from __future__ import annotations

from dataclasses import dataclass

import pytest

from src.providers.runpod.training.cleanup_manager import RunPodCleanupManager, create_cleanup_manager
from src.utils.result import Err, Ok, ProviderError, Result


@dataclass
class StubAPI:
    terminated: list[str]
    ok: bool = True

    def terminate_pod(self, pod_id: str) -> Result[None, ProviderError]:
        self.terminated.append(pod_id)
        return Ok(None) if self.ok else Err(ProviderError(message="fail", code="TEST_TERMINATE_FAILED"))


def test_cleanup_pod_terminates_pod() -> None:
    api = StubAPI(terminated=[], ok=True)
    mgr = RunPodCleanupManager(api_client=api)

    res = mgr.cleanup_pod("pod-1")
    assert res.is_success()
    assert api.terminated == ["pod-1"]


def test_cleanup_pod_propagates_failure() -> None:
    api = StubAPI(terminated=[], ok=False)
    mgr = RunPodCleanupManager(api_client=api)

    res = mgr.cleanup_pod("pod-1")
    assert res.is_failure()
    assert api.terminated == ["pod-1"]


def test_create_cleanup_manager_uses_api_client(monkeypatch: pytest.MonkeyPatch) -> None:
    created = {}

    class FakeClient:
        def __init__(self, api_base_url: str, api_key: str):
            created["api_base_url"] = api_base_url
            created["api_key"] = api_key

    import src.providers.runpod.training.api_client as api_mod

    monkeypatch.setattr(api_mod, "RunPodAPIClient", FakeClient)

    mgr = create_cleanup_manager(api_base="https://api", api_key="rk")
    assert isinstance(mgr, RunPodCleanupManager)
    assert created == {"api_base_url": "https://api", "api_key": "rk"}
