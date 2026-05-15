"""Tests for the raise-based :class:`RunPodTrainingPodControl` and
:class:`RunPodInferencePodControl` (Phase A2 Batch 11)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from ryotenkai_providers.runpod.models import PodSnapshot, SshEndpoint
from ryotenkai_providers.runpod.pod_control import (
    RunPodInferencePodControl,
    RunPodTrainingPodControl,
)
from ryotenkai_shared.errors import ProviderUnavailableError

pytestmark = pytest.mark.unit


@dataclass
class FakeTrainingApi:
    """Canonical raise-based fake for the training API.

    Each ``*_value`` is the value to return; ``*_exc`` raises instead.
    """

    create_value: dict[str, Any] | None = field(
        default_factory=lambda: {"pod_id": "pod-api"}
    )
    create_exc: BaseException | None = None
    query_value: dict[str, Any] | None = field(
        default_factory=lambda: {
            "id": "pod-api",
            "desiredStatus": "RUNNING",
            "runtime": {
                "uptimeInSeconds": 10,
                "ports": [{"ip": "5.6.7.8", "privatePort": 22, "publicPort": 2222, "isIpPublic": True}],
            },
        }
    )
    query_exc: BaseException | None = None
    terminate_exc: BaseException | None = None
    ssh_value: dict[str, Any] | None = field(
        default_factory=lambda: {"host": "5.6.7.8", "port": 2222}
    )
    ssh_exc: BaseException | None = None

    def create_pod(self, config, *, pod_name: str | None = None):
        if self.create_exc is not None:
            raise self.create_exc
        assert self.create_value is not None
        return self.create_value

    def query_pod(self, pod_id: str):
        if self.query_exc is not None:
            raise self.query_exc
        assert self.query_value is not None
        return self.query_value

    def terminate_pod(self, pod_id: str) -> None:
        if self.terminate_exc is not None:
            raise self.terminate_exc

    def stop_pod(self, pod_id: str) -> None:
        pass

    def resume_pod(self, pod_id: str) -> None:
        pass

    def get_ssh_info(self, pod_id: str):
        if self.ssh_exc is not None:
            raise self.ssh_exc
        assert self.ssh_value is not None
        return self.ssh_value

    def extract_exposed_ssh_info(self, pod_data: dict[str, Any] | None, *, pod_id: str | None = None):
        if not pod_data or not pod_data.get("runtime"):
            raise ProviderUnavailableError(detail="no runtime", context={"code": "NO_RUNTIME"})
        snapshot = PodSnapshot.from_graphql(pod_data)
        if snapshot.ssh_endpoint:
            return {"host": snapshot.ssh_endpoint.host, "port": snapshot.ssh_endpoint.port}
        raise ProviderUnavailableError(detail="no ssh", context={"code": "NO_SSH"})


@dataclass
class FakeInferenceApi:
    start_exc: BaseException | None = None
    stop_exc: BaseException | None = None
    delete_exc: BaseException | None = None
    get_value: dict[str, Any] | None = field(default_factory=lambda: {"id": "pod-1"})
    get_exc: BaseException | None = None

    def start_pod(self, *, pod_id: str) -> None:
        if self.start_exc is not None:
            raise self.start_exc

    def stop_pod(self, *, pod_id: str) -> None:
        if self.stop_exc is not None:
            raise self.stop_exc

    def delete_pod(self, *, pod_id: str) -> None:
        if self.delete_exc is not None:
            raise self.delete_exc

    def get_pod(self, *, pod_id: str):
        if self.get_exc is not None:
            raise self.get_exc
        assert self.get_value is not None
        return self.get_value


# ---------------------------------------------------------------------------
# Training pod control (pure GraphQL API)
# ---------------------------------------------------------------------------


def test_create_delegates_to_api() -> None:
    api = FakeTrainingApi(create_value={"pod_id": "pod-api"})
    control = RunPodTrainingPodControl(api=api)
    out = control.create_pod(config=None, pod_name="pod-name")  # type: ignore[arg-type]
    assert out["pod_id"] == "pod-api"


def test_query_pod_delegates_to_api() -> None:
    api = FakeTrainingApi()
    control = RunPodTrainingPodControl(api=api)
    out = control.query_pod("pod-1")
    assert out["desiredStatus"] == "RUNNING"


def test_query_pod_snapshot_returns_typed_snapshot() -> None:
    api = FakeTrainingApi(
        query_value={
            "id": "pod-1",
            "desiredStatus": "RUNNING",
            "runtime": {
                "uptimeInSeconds": 42,
                "ports": [{"ip": "1.2.3.4", "privatePort": 22, "publicPort": 12345}],
            },
        }
    )
    control = RunPodTrainingPodControl(api=api)
    snap = control.query_pod_snapshot("pod-1")
    assert isinstance(snap, PodSnapshot)
    assert snap.ssh_endpoint == SshEndpoint(host="1.2.3.4", port=12345)


def test_query_pod_snapshot_propagates_api_error() -> None:
    api = FakeTrainingApi(
        query_exc=ProviderUnavailableError(
            detail="No pod data received", context={"code": "RUNPOD_POD_DATA_MISSING"}
        )
    )
    control = RunPodTrainingPodControl(api=api)
    with pytest.raises(ProviderUnavailableError) as ei:
        control.query_pod_snapshot("pod-1")
    assert ei.value.context["code"] == "RUNPOD_POD_DATA_MISSING"


def test_get_ssh_info_from_snapshot() -> None:
    api = FakeTrainingApi(
        query_value={
            "id": "pod-1",
            "desiredStatus": "RUNNING",
            "runtime": {
                "uptimeInSeconds": 10,
                "ports": [{"ip": "9.9.9.9", "privatePort": 22, "publicPort": 3456}],
            },
        }
    )
    control = RunPodTrainingPodControl(api=api)
    out = control.get_ssh_info("pod-1")
    assert out == {"host": "9.9.9.9", "port": 3456}


def test_get_ssh_info_falls_back_to_api_when_no_ssh_endpoint() -> None:
    api = FakeTrainingApi(
        query_value={
            "id": "pod-1",
            "desiredStatus": "RUNNING",
            "runtime": {"uptimeInSeconds": 0, "ports": []},
        },
        ssh_value={"host": "fallback.host", "port": 4444},
    )
    control = RunPodTrainingPodControl(api=api)
    out = control.get_ssh_info("pod-1")
    assert out == {"host": "fallback.host", "port": 4444}


def test_terminate_delegates_to_api() -> None:
    api = FakeTrainingApi()
    control = RunPodTrainingPodControl(api=api)
    control.terminate_pod("pod-1")  # no exception


# ---------------------------------------------------------------------------
# Training pod control - create_pod retry logic
# ---------------------------------------------------------------------------


class _CountingTrainingApi(FakeTrainingApi):
    """FakeTrainingApi that returns different results per call.

    Each ``actions`` entry is either a dict (returned) or an
    ``Exception`` (raised).
    """

    def __init__(self, actions: list[Any]):
        super().__init__()
        self._actions = list(actions)
        self.call_count = 0

    def create_pod(self, config, *, pod_name: str | None = None):
        idx = min(self.call_count, len(self._actions) - 1)
        self.call_count += 1
        action = self._actions[idx]
        if isinstance(action, BaseException):
            raise action
        return action


def test_create_retries_on_transient_capacity_error(monkeypatch) -> None:
    monkeypatch.setattr("ryotenkai_providers.runpod.pod_control.time.sleep", lambda _: None)
    transient = ProviderUnavailableError(
        detail="There are no longer any instances available with the requested specifications. Please refresh and try again.",
        context={"code": "RUNPOD_GRAPHQL_ERROR"},
    )
    success = {"pod_id": "pod-ok"}
    api = _CountingTrainingApi(actions=[transient, transient, success])
    control = RunPodTrainingPodControl(api=api)

    out = control.create_pod(config=None, pod_name="test")  # type: ignore[arg-type]
    assert out["pod_id"] == "pod-ok"
    assert api.call_count == 3


def test_create_does_not_retry_non_transient_error(monkeypatch) -> None:
    monkeypatch.setattr("ryotenkai_providers.runpod.pod_control.time.sleep", lambda _: None)
    permanent = ProviderUnavailableError(detail="Invalid API key", context={"code": "RUNPOD_GRAPHQL_ERROR"})
    api = _CountingTrainingApi(actions=[permanent])
    control = RunPodTrainingPodControl(api=api)

    with pytest.raises(ProviderUnavailableError):
        control.create_pod(config=None, pod_name="test")  # type: ignore[arg-type]
    assert api.call_count == 1


def test_create_exhausts_retries_on_persistent_transient_error(monkeypatch) -> None:
    monkeypatch.setattr("ryotenkai_providers.runpod.pod_control.time.sleep", lambda _: None)
    transient = ProviderUnavailableError(
        detail="no instances available, try again", context={"code": "RUNPOD_GRAPHQL_ERROR"}
    )
    api = _CountingTrainingApi(actions=[transient, transient, transient])
    control = RunPodTrainingPodControl(api=api)

    with pytest.raises(ProviderUnavailableError):
        control.create_pod(config=None, pod_name="test")  # type: ignore[arg-type]
    assert api.call_count == 3


@pytest.mark.parametrize(
    "message",
    [
        "Rate Limit exceeded by upstream",
        "Gateway returned 503 for create pod",
        "Gateway returned 502 for create pod",
        "Operation TIMEOUT while creating pod",
    ],
)
def test_create_retries_on_additional_transient_markers(monkeypatch, message: str) -> None:
    monkeypatch.setattr("ryotenkai_providers.runpod.pod_control.time.sleep", lambda _: None)
    transient = ProviderUnavailableError(detail=message, context={"code": "RUNPOD_GRAPHQL_ERROR"})
    success = {"pod_id": "pod-ok"}
    api = _CountingTrainingApi(actions=[transient, success])
    control = RunPodTrainingPodControl(api=api)

    out = control.create_pod(config=None, pod_name="test")  # type: ignore[arg-type]
    assert out["pod_id"] == "pod-ok"
    assert api.call_count == 2


def test_create_raises_last_error_instance_after_retry_exhaustion(monkeypatch) -> None:
    monkeypatch.setattr("ryotenkai_providers.runpod.pod_control.time.sleep", lambda _: None)
    first = ProviderUnavailableError(detail="timeout", context={"code": "RUNPOD_GRAPHQL_ERROR"})
    second = ProviderUnavailableError(detail="503 unavailable", context={"code": "RUNPOD_GRAPHQL_ERROR"})
    third = ProviderUnavailableError(detail="rate limit", context={"code": "RUNPOD_GRAPHQL_ERROR"})
    api = _CountingTrainingApi(actions=[first, second, third])
    control = RunPodTrainingPodControl(api=api)

    with pytest.raises(ProviderUnavailableError) as ei:
        control.create_pod(config=None, pod_name="test")  # type: ignore[arg-type]
    assert ei.value is third


# ---------------------------------------------------------------------------
# Inference pod control (SDK-backed Pod API)
# ---------------------------------------------------------------------------


def test_inference_control_delegates_to_api() -> None:
    control = RunPodInferencePodControl(api=FakeInferenceApi())
    control.start_pod(pod_id="pod-1")  # no exception
    control.stop_pod(pod_id="pod-1")
    control.delete_pod(pod_id="pod-1")


def test_inference_control_get_pod_passthrough() -> None:
    api = FakeInferenceApi(get_value={"id": "pod-xyz", "desiredStatus": "RUNNING"})
    control = RunPodInferencePodControl(api=api)
    out = control.get_pod(pod_id="pod-xyz")
    assert out["id"] == "pod-xyz"


def test_inference_control_propagates_api_errors() -> None:
    api = FakeInferenceApi(
        start_exc=ProviderUnavailableError(detail="start failed", context={"code": "RUNPOD_SDK_CALL_FAILED"}),
        stop_exc=ProviderUnavailableError(detail="stop failed", context={"code": "RUNPOD_SDK_CALL_FAILED"}),
        delete_exc=ProviderUnavailableError(detail="delete failed", context={"code": "RUNPOD_SDK_CALL_FAILED"}),
        get_exc=ProviderUnavailableError(detail="get failed", context={"code": "RUNPOD_SDK_CALL_FAILED"}),
    )
    control = RunPodInferencePodControl(api=api)

    with pytest.raises(ProviderUnavailableError) as ei:
        control.start_pod(pod_id="pod-1")
    assert ei.value.detail == "start failed"
    with pytest.raises(ProviderUnavailableError) as ei:
        control.stop_pod(pod_id="pod-1")
    assert ei.value.detail == "stop failed"
    with pytest.raises(ProviderUnavailableError) as ei:
        control.delete_pod(pod_id="pod-1")
    assert ei.value.detail == "delete failed"
    with pytest.raises(ProviderUnavailableError) as ei:
        control.get_pod(pod_id="pod-1")
    assert ei.value.detail == "get failed"


def test_inference_control_query_pod_snapshot_returns_typed_snapshot() -> None:
    """Inference control mirrors training's ``query_pod_snapshot`` so a
    single ``PodQuery`` Protocol can drive both surfaces."""
    api = FakeInferenceApi(
        get_value={
            "id": "pod-inf",
            "desiredStatus": "RUNNING",
            "runtime": {
                "uptimeInSeconds": 12,
                "ports": [{"ip": "1.2.3.4", "privatePort": 22, "publicPort": 12345, "isIpPublic": True}],
            },
        }
    )
    control = RunPodInferencePodControl(api=api)
    snap = control.query_pod_snapshot("pod-inf")
    assert isinstance(snap, PodSnapshot)
    assert snap.is_ready
    assert snap.ssh_endpoint == SshEndpoint(host="1.2.3.4", port=12345)


def test_inference_control_query_pod_snapshot_propagates_api_error() -> None:
    api = FakeInferenceApi(
        get_exc=ProviderUnavailableError(detail="boom", context={"code": "RUNPOD_SDK_CALL_FAILED"}),
    )
    control = RunPodInferencePodControl(api=api)
    with pytest.raises(ProviderUnavailableError) as ei:
        control.query_pod_snapshot("pod-1")
    assert ei.value.context["code"] == "RUNPOD_SDK_CALL_FAILED"


def test_inference_control_query_pod_snapshot_empty_dict_returns_data_missing() -> None:
    """Empty SDK response => typed terminal-class error code, matching
    training's behavior so the waiter's abort-vs-retry classification
    is identical for both surfaces."""
    api = FakeInferenceApi(get_value={})
    control = RunPodInferencePodControl(api=api)
    with pytest.raises(ProviderUnavailableError) as ei:
        control.query_pod_snapshot("pod-1")
    assert ei.value.context["code"] == "RUNPOD_POD_DATA_MISSING"
