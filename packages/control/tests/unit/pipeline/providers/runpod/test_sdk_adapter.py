from __future__ import annotations

from typing import Any

import pytest

from src.providers.runpod import sdk_adapter
from src.providers.runpod.sdk_adapter import RunPodSDKClient
from src.utils.result import Err, Ok, ProviderError


pytestmark = pytest.mark.unit


def test_call_sdk_returns_error_when_function_is_missing() -> None:
    client = RunPodSDKClient(api_key="rk")

    res = client._call_sdk("definitely_missing_fn")

    assert res.is_failure()
    err = res.unwrap_err()
    assert err.code == "RUNPOD_SDK_CALL_FAILED"
    assert err.details == {"function": "definitely_missing_fn"}


def test_call_sdk_maps_value_error_to_validation_error(monkeypatch: pytest.MonkeyPatch) -> None:
    client = RunPodSDKClient(api_key="rk")

    def fake_create_pod(**kwargs: Any) -> dict[str, Any]:
        _ = kwargs
        raise ValueError("bad payload")

    monkeypatch.setattr(sdk_adapter.runpod, "create_pod", fake_create_pod, raising=False)

    res = client._call_sdk("create_pod", name="pod")

    assert res.is_failure()
    err = res.unwrap_err()
    assert err.code == "RUNPOD_SDK_VALIDATION_ERROR"
    assert err.details == {"function": "create_pod"}


def test_get_pod_returns_unexpected_response_error_for_non_dict(monkeypatch: pytest.MonkeyPatch) -> None:
    client = RunPodSDKClient(api_key="rk")
    monkeypatch.setattr(sdk_adapter.runpod, "get_pod", lambda pod_id: ["not", "a", "dict"], raising=False)

    res = client.get_pod(pod_id="pod-1")

    assert res.is_failure()
    assert res.unwrap_err().code == "RUNPOD_SDK_UNEXPECTED_RESPONSE"


def test_list_pods_returns_unexpected_response_error_for_non_list(monkeypatch: pytest.MonkeyPatch) -> None:
    client = RunPodSDKClient(api_key="rk")
    monkeypatch.setattr(sdk_adapter.runpod, "get_pods", lambda: {"id": "pod-1"}, raising=False)

    res = client.list_pods()

    assert res.is_failure()
    assert res.unwrap_err().code == "RUNPOD_SDK_UNEXPECTED_RESPONSE"


def test_create_pod_returns_unexpected_response_error_for_non_dict(monkeypatch: pytest.MonkeyPatch) -> None:
    client = RunPodSDKClient(api_key="rk")
    monkeypatch.setattr(sdk_adapter.runpod, "create_pod", lambda **kwargs: ["bad"], raising=False)

    res = client.create_pod(name="pod")

    assert res.is_failure()
    assert res.unwrap_err().code == "RUNPOD_SDK_UNEXPECTED_RESPONSE"


def test_create_pod_from_payload_without_gpu_type_ids_normalizes_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    client = RunPodSDKClient(api_key="rk")
    captured: dict[str, Any] = {}

    def fake_create_pod(**kwargs: Any):
        captured.update(kwargs)
        return Ok({"id": "pod-1"})

    monkeypatch.setattr(client, "create_pod", fake_create_pod)

    res = client.create_pod_from_payload(
        payload={
            "name": "  pod-name  ",
            "imageName": " img ",
            "ports": ["22/tcp", " ", "8000/http"],
            "gpuCount": None,
            "volumeMountPath": " ",
            "dockerArgs": " --flag ",
            "supportPublicIp": False,
            "env": {"A": "B"},
        }
    )

    assert res.is_success()
    assert captured == {
        "name": "pod-name",
        "image_name": "img",
        "cloud_type": "ALL",
        "support_public_ip": False,
        "start_ssh": True,
        "gpu_count": 1,
        "volume_in_gb": 0,
        "min_vcpu_count": 1,
        "min_memory_in_gb": 1,
        "docker_args": "--flag",
        "ports": "22/tcp,8000/http",
        "volume_mount_path": "/runpod-volume",
        "env": {"A": "B"},
    }


def test_create_pod_from_payload_tries_next_gpu_type_on_capacity_error(monkeypatch: pytest.MonkeyPatch) -> None:
    client = RunPodSDKClient(api_key="rk")
    calls: list[dict[str, Any]] = []

    def fake_create_pod(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        if kwargs["gpu_type_id"] == "GPU-1":
            raise RuntimeError("There are no instances currently available")
        return {"id": "pod-1", "gpuTypeId": kwargs["gpu_type_id"]}

    monkeypatch.setattr(sdk_adapter.runpod, "create_pod", fake_create_pod, raising=False)

    res = client.create_pod_from_payload(
        payload={
            "name": "test",
            "imageName": "img",
            "gpuTypeIds": ["GPU-1", "GPU-2"],
            "gpuCount": 1,
            "ports": ["22/tcp", "8000/http"],
        }
    )

    assert res.is_success()
    assert res.unwrap()["gpuTypeId"] == "GPU-2"
    assert calls[0]["ports"] == "22/tcp,8000/http"
    assert calls[1]["gpu_type_id"] == "GPU-2"


def test_create_pod_from_payload_returns_last_capacity_error_when_all_gpu_types_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = RunPodSDKClient(api_key="rk")
    calls: list[str] = []

    def fake_create_pod(**kwargs: Any):
        calls.append(kwargs["gpu_type_id"])
        return Err(
            ProviderError(
                message=f"No available datacenter with requested resources for {kwargs['gpu_type_id']}",
                code="RUNPOD_SDK_CALL_FAILED",
            )
        )

    monkeypatch.setattr(client, "create_pod", fake_create_pod)

    res = client.create_pod_from_payload(
        payload={"name": "test", "imageName": "img", "gpuTypeIds": ["GPU-1", "GPU-2", "GPU-3"]}
    )

    assert res.is_failure()
    assert res.unwrap_err().message.endswith("GPU-3")
    assert calls == ["GPU-1", "GPU-2", "GPU-3"]


def test_create_pod_from_payload_does_not_retry_after_non_capacity_error(monkeypatch: pytest.MonkeyPatch) -> None:
    client = RunPodSDKClient(api_key="rk")
    calls: list[str] = []

    def fake_create_pod(**kwargs: Any):
        calls.append(kwargs["gpu_type_id"])
        return Err(ProviderError(message="invalid config", code="RUNPOD_SDK_VALIDATION_ERROR"))

    monkeypatch.setattr(client, "create_pod", fake_create_pod)

    res = client.create_pod_from_payload(
        payload={"name": "test", "imageName": "img", "gpuTypeIds": ["GPU-1", "GPU-2"]}
    )

    assert res.is_failure()
    assert res.unwrap_err().code == "RUNPOD_SDK_VALIDATION_ERROR"
    assert calls == ["GPU-1"]


def test_start_pod_uses_gpu_count_from_existing_pod(monkeypatch: pytest.MonkeyPatch) -> None:
    client = RunPodSDKClient(api_key="rk")
    calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def fake_get_pod(pod_id: str, api_key: str | None = None) -> dict[str, Any]:
        _ = api_key
        calls.append(("get_pod", (pod_id,), {}))
        return {"id": pod_id, "gpuCount": 3}

    def fake_resume_pod(pod_id: str, gpu_count: int) -> dict[str, Any]:
        calls.append(("resume_pod", (pod_id, gpu_count), {}))
        return {"id": pod_id}

    monkeypatch.setattr(sdk_adapter.runpod, "get_pod", fake_get_pod, raising=False)
    monkeypatch.setattr(sdk_adapter.runpod, "resume_pod", fake_resume_pod, raising=False)

    res = client.start_pod(pod_id="pod-1")

    assert res.is_success()
    assert calls[1] == ("resume_pod", ("pod-1", 3), {})


def test_start_pod_falls_back_to_one_gpu_when_gpu_count_is_invalid(monkeypatch: pytest.MonkeyPatch) -> None:
    client = RunPodSDKClient(api_key="rk")
    calls: list[tuple[str, tuple[Any, ...]]] = []

    def fake_get_pod(pod_id: str) -> dict[str, Any]:
        calls.append(("get_pod", (pod_id,)))
        return {"id": pod_id, "gpuCount": "not-an-int"}

    def fake_resume_pod(pod_id: str, gpu_count: int) -> dict[str, Any]:
        calls.append(("resume_pod", (pod_id, gpu_count)))
        return {"id": pod_id}

    monkeypatch.setattr(sdk_adapter.runpod, "get_pod", fake_get_pod, raising=False)
    monkeypatch.setattr(sdk_adapter.runpod, "resume_pod", fake_resume_pod, raising=False)

    res = client.start_pod(pod_id="pod-1")

    assert res.is_success()
    assert calls[-1] == ("resume_pod", ("pod-1", 1))


def test_start_pod_short_circuits_when_get_pod_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    client = RunPodSDKClient(api_key="rk")

    def fake_get_pod(pod_id: str) -> dict[str, Any]:
        _ = pod_id
        raise RuntimeError("sdk down")

    resume_called = False

    def fake_resume_pod(pod_id: str, gpu_count: int) -> dict[str, Any]:
        nonlocal resume_called
        _ = (pod_id, gpu_count)
        resume_called = True
        return {"id": pod_id}

    monkeypatch.setattr(sdk_adapter.runpod, "get_pod", fake_get_pod, raising=False)
    monkeypatch.setattr(sdk_adapter.runpod, "resume_pod", fake_resume_pod, raising=False)

    res = client.start_pod(pod_id="pod-1")

    assert res.is_failure()
    assert res.unwrap_err().code == "RUNPOD_SDK_CALL_FAILED"
    assert resume_called is False


def test_list_pods_filters_by_params(monkeypatch: pytest.MonkeyPatch) -> None:
    client = RunPodSDKClient(api_key="rk")

    monkeypatch.setattr(
        sdk_adapter.runpod,
        "get_pods",
        lambda: [
            {"id": "p1", "name": "wanted", "networkVolumeId": "vol-1", "desiredStatus": "RUNNING"},
            {"id": "p2", "name": "other", "networkVolumeId": "vol-2", "desiredStatus": "RUNNING"},
        ],
        raising=False,
    )

    res = client.list_pods(params={"name": "wanted", "networkVolumeId": "vol-1"})

    assert res.is_success()
    assert [pod["id"] for pod in res.unwrap()] == ["p1"]


def test_list_pods_treats_missing_compute_type_as_non_blocking(monkeypatch: pytest.MonkeyPatch) -> None:
    client = RunPodSDKClient(api_key="rk")

    monkeypatch.setattr(
        sdk_adapter.runpod,
        "get_pods",
        lambda: [
            {"id": "p1", "name": "wanted"},
            {"id": "p2", "name": "wanted", "computeType": "CPU"},
        ],
        raising=False,
    )

    res = client.list_pods(params={"name": "wanted", "computeType": "GPU"})

    assert res.is_success()
    assert [pod["id"] for pod in res.unwrap()] == ["p1"]


def test_list_pods_ignores_none_expected_filters(monkeypatch: pytest.MonkeyPatch) -> None:
    client = RunPodSDKClient(api_key="rk")

    monkeypatch.setattr(
        sdk_adapter.runpod,
        "get_pods",
        lambda: [{"id": "p1", "name": "wanted", "desiredStatus": "RUNNING"}],
        raising=False,
    )

    res = client.list_pods(params={"name": "wanted", "networkVolumeId": None})

    assert res.is_success()
    assert [pod["id"] for pod in res.unwrap()] == ["p1"]


@pytest.mark.parametrize(
    ("raw_ports", "expected_ports"),
    [
        (" 22/tcp ", "22/tcp"),
        (["22/tcp", "", "  ", "8000/http"], "22/tcp,8000/http"),
        ([], None),
        (1234, None),
    ],
)
def test_create_pod_from_payload_coerces_ports(
    monkeypatch: pytest.MonkeyPatch,
    raw_ports: Any,
    expected_ports: str | None,
) -> None:
    client = RunPodSDKClient(api_key="rk")
    captured: dict[str, Any] = {}

    def fake_create_pod(**kwargs: Any):
        captured.update(kwargs)
        return Ok({"id": "pod-1"})

    monkeypatch.setattr(client, "create_pod", fake_create_pod)

    res = client.create_pod_from_payload(payload={"name": "test", "imageName": "img", "ports": raw_ports})

    assert res.is_success()
    if expected_ports is None:
        assert "ports" not in captured
    else:
        assert captured["ports"] == expected_ports
