from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from src.providers.runpod.models import PodSnapshot, SshEndpoint
from src.providers.runpod.pod_control import (
    RunPodInferencePodControl,
    RunPodTrainingPodControl,
)
from src.utils.result import Err, Ok, ProviderError, Result

pytestmark = pytest.mark.unit


@dataclass
class FakeRunpodctl:
    remove_result: Result[None, ProviderError] = Ok(None)  # type: ignore[call-arg]
    start_result: Result[None, ProviderError] = Ok(None)  # type: ignore[call-arg]
    stop_result: Result[None, ProviderError] = Ok(None)  # type: ignore[call-arg]

    def remove_pod(self, pod_id: str):
        return self.remove_result

    def start_pod(self, pod_id: str):
        return self.start_result

    def stop_pod(self, pod_id: str):
        return self.stop_result


@dataclass
class FakeTrainingApi:
    create_result: Result[dict[str, Any], ProviderError] = Ok({"pod_id": "pod-api"})  # type: ignore[call-arg]
    query_result: Result[dict[str, Any], ProviderError] = Ok(  # type: ignore[call-arg]
        {
            "id": "pod-api",
            "desiredStatus": "RUNNING",
            "runtime": {
                "uptimeInSeconds": 10,
                "ports": [{"ip": "5.6.7.8", "privatePort": 22, "publicPort": 2222, "isIpPublic": True}],
            },
        }
    )
    terminate_result: Result[None, ProviderError] = Ok(None)  # type: ignore[call-arg]
    ssh_result: Result[dict[str, Any], ProviderError] = Ok({"host": "5.6.7.8", "port": 2222})  # type: ignore[call-arg]

    def create_pod(self, config, *, pod_name: str | None = None):
        return self.create_result

    def query_pod(self, pod_id: str):
        return self.query_result

    def terminate_pod(self, pod_id: str):
        return self.terminate_result

    def get_ssh_info(self, pod_id: str):
        return self.ssh_result

    def extract_exposed_ssh_info(self, pod_data: dict[str, Any] | None, *, pod_id: str | None = None):
        if not pod_data or not pod_data.get("runtime"):
            return Err(ProviderError(message="no runtime", code="NO_RUNTIME"))
        snapshot = PodSnapshot.from_graphql(pod_data)
        if snapshot.ssh_endpoint:
            return Ok({"host": snapshot.ssh_endpoint.host, "port": snapshot.ssh_endpoint.port})
        return Err(ProviderError(message="no ssh", code="NO_SSH"))


@dataclass
class FakeInferenceApi:
    start_result: Result[None, ProviderError] = Ok(None)  # type: ignore[call-arg]
    stop_result: Result[None, ProviderError] = Ok(None)  # type: ignore[call-arg]
    delete_result: Result[None, ProviderError] = Ok(None)  # type: ignore[call-arg]
    get_result: Result[dict[str, Any], ProviderError] = Ok({"id": "pod-1"})  # type: ignore[call-arg]

    def start_pod(self, *, pod_id: str):
        return self.start_result

    def stop_pod(self, *, pod_id: str):
        return self.stop_result

    def delete_pod(self, *, pod_id: str):
        return self.delete_result

    def get_pod(self, *, pod_id: str):
        return self.get_result


# ---------------------------------------------------------------------------
# Training pod control (pure GraphQL API)
# ---------------------------------------------------------------------------


def test_create_delegates_to_api() -> None:
    api = FakeTrainingApi(create_result=Ok({"pod_id": "pod-api"}))  # type: ignore[call-arg]
    control = RunPodTrainingPodControl(api=api)
    res = control.create_pod(config=None, pod_name="pod-name")  # type: ignore[arg-type]
    assert res.is_success()
    assert res.unwrap()["pod_id"] == "pod-api"


def test_query_pod_delegates_to_api() -> None:
    api = FakeTrainingApi()
    control = RunPodTrainingPodControl(api=api)
    res = control.query_pod("pod-1")
    assert res.is_success()
    assert res.unwrap()["desiredStatus"] == "RUNNING"


def test_query_pod_snapshot_returns_typed_snapshot() -> None:
    api = FakeTrainingApi(
        query_result=Ok(  # type: ignore[call-arg]
            {
                "id": "pod-1",
                "desiredStatus": "RUNNING",
                "runtime": {
                    "uptimeInSeconds": 42,
                    "ports": [{"ip": "1.2.3.4", "privatePort": 22, "publicPort": 12345}],
                },
            }
        )
    )
    control = RunPodTrainingPodControl(api=api)
    res = control.query_pod_snapshot("pod-1")
    assert res.is_success()
    snap = res.unwrap()
    assert isinstance(snap, PodSnapshot)
    assert snap.ssh_endpoint == SshEndpoint(host="1.2.3.4", port=12345)


def test_query_pod_snapshot_propagates_api_error() -> None:
    api = FakeTrainingApi(
        query_result=Err(ProviderError(message="No pod data received", code="RUNPOD_POD_DATA_MISSING"))
    )
    control = RunPodTrainingPodControl(api=api)
    res = control.query_pod_snapshot("pod-1")
    assert res.is_failure()


def test_get_ssh_info_from_snapshot() -> None:
    api = FakeTrainingApi(
        query_result=Ok(  # type: ignore[call-arg]
            {
                "id": "pod-1",
                "desiredStatus": "RUNNING",
                "runtime": {
                    "uptimeInSeconds": 10,
                    "ports": [{"ip": "9.9.9.9", "privatePort": 22, "publicPort": 3456}],
                },
            }
        )
    )
    control = RunPodTrainingPodControl(api=api)
    res = control.get_ssh_info("pod-1")
    assert res.is_success()
    assert res.unwrap() == {"host": "9.9.9.9", "port": 3456}


def test_get_ssh_info_falls_back_to_api_when_no_ssh_endpoint() -> None:
    api = FakeTrainingApi(
        query_result=Ok(  # type: ignore[call-arg]
            {
                "id": "pod-1",
                "desiredStatus": "RUNNING",
                "runtime": {"uptimeInSeconds": 0, "ports": []},
            }
        ),
        ssh_result=Ok({"host": "fallback.host", "port": 4444}),  # type: ignore[call-arg]
    )
    control = RunPodTrainingPodControl(api=api)
    res = control.get_ssh_info("pod-1")
    assert res.is_success()
    assert res.unwrap() == {"host": "fallback.host", "port": 4444}


def test_terminate_delegates_to_api() -> None:
    api = FakeTrainingApi()
    control = RunPodTrainingPodControl(api=api)
    res = control.terminate_pod("pod-1")
    assert res.is_success()


# ---------------------------------------------------------------------------
# Training pod control - create_pod retry logic
# ---------------------------------------------------------------------------


class _CountingTrainingApi(FakeTrainingApi):
    """FakeTrainingApi that returns different results per call."""

    def __init__(self, results: list[Result[dict[str, Any], ProviderError]]):
        super().__init__()
        self._results = list(results)
        self.call_count = 0

    def create_pod(self, config, *, pod_name: str | None = None):
        idx = min(self.call_count, len(self._results) - 1)
        self.call_count += 1
        return self._results[idx]


def test_create_retries_on_transient_capacity_error(monkeypatch) -> None:
    monkeypatch.setattr("src.providers.runpod.pod_control.time.sleep", lambda _: None)
    transient = Err(ProviderError(
        message="There are no longer any instances available with the requested specifications. Please refresh and try again.",
        code="RUNPOD_GRAPHQL_ERROR",
    ))
    success = Ok({"pod_id": "pod-ok"})  # type: ignore[call-arg]
    api = _CountingTrainingApi(results=[transient, transient, success])
    control = RunPodTrainingPodControl(api=api)

    res = control.create_pod(config=None, pod_name="test")  # type: ignore[arg-type]
    assert res.is_success()
    assert res.unwrap()["pod_id"] == "pod-ok"
    assert api.call_count == 3


def test_create_does_not_retry_non_transient_error(monkeypatch) -> None:
    monkeypatch.setattr("src.providers.runpod.pod_control.time.sleep", lambda _: None)
    permanent = Err(ProviderError(
        message="Invalid API key",
        code="RUNPOD_GRAPHQL_ERROR",
    ))
    api = _CountingTrainingApi(results=[permanent])
    control = RunPodTrainingPodControl(api=api)

    res = control.create_pod(config=None, pod_name="test")  # type: ignore[arg-type]
    assert res.is_failure()
    assert api.call_count == 1


def test_create_exhausts_retries_on_persistent_transient_error(monkeypatch) -> None:
    monkeypatch.setattr("src.providers.runpod.pod_control.time.sleep", lambda _: None)
    transient = Err(ProviderError(
        message="no instances available, try again",
        code="RUNPOD_GRAPHQL_ERROR",
    ))
    api = _CountingTrainingApi(results=[transient, transient, transient])
    control = RunPodTrainingPodControl(api=api)

    res = control.create_pod(config=None, pod_name="test")  # type: ignore[arg-type]
    assert res.is_failure()
    assert api.call_count == 3


# ---------------------------------------------------------------------------
# Inference pod control (runpodctl-first with REST fallback)
# ---------------------------------------------------------------------------


def test_inference_control_falls_back_for_all_ops() -> None:
    ctl = FakeRunpodctl(
        start_result=Err(ProviderError(message="no cli", code="RUNPODCTL_NOT_AVAILABLE")),
        stop_result=Err(ProviderError(message="no cli", code="RUNPODCTL_NOT_AVAILABLE")),
        remove_result=Err(ProviderError(message="no cli", code="RUNPODCTL_NOT_AVAILABLE")),
    )
    control = RunPodInferencePodControl(runpodctl=ctl, api=FakeInferenceApi())
    assert control.start_pod(pod_id="pod-1").is_success()
    assert control.stop_pod(pod_id="pod-1").is_success()
    assert control.delete_pod(pod_id="pod-1").is_success()


def test_inference_control_uses_runpodctl_when_successful() -> None:
    control = RunPodInferencePodControl(runpodctl=FakeRunpodctl(), api=FakeInferenceApi())
    assert control.start_pod(pod_id="pod-1").is_success()
    assert control.stop_pod(pod_id="pod-1").is_success()
    assert control.delete_pod(pod_id="pod-1").is_success()


def test_inference_control_get_pod_passthrough() -> None:
    api = FakeInferenceApi(get_result=Ok({"id": "pod-xyz", "desiredStatus": "RUNNING"}))  # type: ignore[call-arg]
    control = RunPodInferencePodControl(runpodctl=FakeRunpodctl(), api=api)
    res = control.get_pod(pod_id="pod-xyz")
    assert res.is_success()
    assert res.unwrap()["id"] == "pod-xyz"
