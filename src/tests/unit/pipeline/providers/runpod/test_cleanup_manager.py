from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

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


def test_register_and_unregister_pod(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    api = StubAPI(terminated=[])
    mgr = RunPodCleanupManager(api_client=api)
    mgr.registry_file = tmp_path / "pods.json"

    monkeypatch.setattr("time.time", lambda: 123.0)

    mgr.register_pod("pod-1", api_base="https://api")
    data = json.loads(mgr.registry_file.read_text(encoding="utf-8"))
    assert "pod-1" in data
    assert data["pod-1"]["api_base"] == "https://api"

    mgr.unregister_pod("pod-1")
    data2 = json.loads(mgr.registry_file.read_text(encoding="utf-8"))
    assert "pod-1" not in data2


def test_register_pod_tolerates_corrupted_registry(tmp_path: Path) -> None:
    api = StubAPI(terminated=[])
    mgr = RunPodCleanupManager(api_client=api)
    mgr.registry_file = tmp_path / "pods.json"
    mgr.registry_file.write_text("{not json", encoding="utf-8")

    mgr.register_pod("pod-2", api_base="https://api")
    data = json.loads(mgr.registry_file.read_text(encoding="utf-8"))
    assert "pod-2" in data


def test_list_registered_pods_handles_missing_file(tmp_path: Path) -> None:
    api = StubAPI(terminated=[])
    mgr = RunPodCleanupManager(api_client=api)
    mgr.registry_file = tmp_path / "pods.json"

    assert mgr.list_registered_pods() == []


def test_cleanup_pod_unregisters_on_success(tmp_path: Path) -> None:
    api = StubAPI(terminated=[], ok=True)
    mgr = RunPodCleanupManager(api_client=api)
    mgr.registry_file = tmp_path / "pods.json"
    mgr.register_pod("pod-1", api_base="https://api")

    res = mgr.cleanup_pod("pod-1")
    assert res.is_success()
    assert api.terminated == ["pod-1"]
    assert mgr.list_registered_pods() == []


def test_cleanup_pod_keeps_registry_on_failure(tmp_path: Path) -> None:
    api = StubAPI(terminated=[], ok=False)
    mgr = RunPodCleanupManager(api_client=api)
    mgr.registry_file = tmp_path / "pods.json"
    mgr.register_pod("pod-1", api_base="https://api")

    res = mgr.cleanup_pod("pod-1")
    assert res.is_failure()
    assert mgr.list_registered_pods() == ["pod-1"]


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
