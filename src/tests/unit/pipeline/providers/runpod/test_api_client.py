from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from src.config.providers.runpod import RunPodProviderConfig
from src.providers.runpod.training.api_client import (
    RunPodAPIClient,
    build_pod_launch_kwargs,
    build_ssh_bootstrap_cmd,
)
from src.utils.result import Err, Ok, ProviderError


@dataclass
class FakeSDK:
    create_result: Any = None
    get_result: Any = None
    delete_result: Any = None

    def create_pod(self, **kwargs: Any):
        return self.create_result

    def get_pod(self, *, pod_id: str):
        return self.get_result

    def delete_pod(self, *, pod_id: str):
        return self.delete_result


def test_create_pod_success(monkeypatch: pytest.MonkeyPatch) -> None:
    client = RunPodAPIClient(api_base_url="https://api.runpod.io", api_key="rk")
    pod_payload = {
        "id": "pod-1",
        "desiredStatus": "RUNNING",
        "imageName": "img",
        "gpuCount": 1,
        "vcpuCount": 8,
        "memoryInGb": 32,
        "costPerHr": 0.79,
        "machine": {"podHostId": "host-1"},
    }
    # create_pod now enriches with a follow-up get_pod call; give the fake
    # SDK the same payload so the enriched branch doesn't crash on ``None``.
    fake_sdk = FakeSDK(create_result=Ok(pod_payload), get_result=Ok(pod_payload))
    monkeypatch.setattr(client, "_sdk", fake_sdk)

    cfg = RunPodProviderConfig(
        connect={"ssh": {"key_path": "/k"}},
        cleanup={},
        training={"image_name": "img", "gpu_type": "NVIDIA A40"},
        inference={},
    )
    res = client.create_pod(cfg)

    assert res.is_success()
    data = res.unwrap()
    assert data["pod_id"] == "pod-1"
    assert data["machine"] == "host-1"
    assert data["cost_per_hr"] == 0.79


def test_create_pod_graphql_error(monkeypatch: pytest.MonkeyPatch) -> None:
    client = RunPodAPIClient(api_base_url="https://api.runpod.io", api_key="rk")
    monkeypatch.setattr(
        client,
        "_sdk",
        FakeSDK(create_result=Err(ProviderError(message="no capacity", code="RUNPOD_SDK_CALL_FAILED"))),
    )

    res = client.create_pod(
        RunPodProviderConfig(connect={"ssh": {"key_path": "/k"}}, cleanup={}, training={"image_name": "img", "gpu_type": "NVIDIA A40"}, inference={})
    )
    assert res.is_failure()
    assert "no capacity" in str(res.unwrap_err())


def test_create_pod_missing_id(monkeypatch: pytest.MonkeyPatch) -> None:
    client = RunPodAPIClient(api_base_url="https://api.runpod.io", api_key="rk")
    monkeypatch.setattr(client, "_sdk", FakeSDK(create_result=Ok({})))

    res = client.create_pod(
        RunPodProviderConfig(connect={"ssh": {"key_path": "/k"}}, cleanup={}, training={"image_name": "img", "gpu_type": "NVIDIA A40"}, inference={})
    )
    assert res.is_failure()


def test_create_pod_sdk_error(monkeypatch: pytest.MonkeyPatch) -> None:
    client = RunPodAPIClient(api_base_url="https://api.runpod.io", api_key="rk")
    monkeypatch.setattr(
        client,
        "_sdk",
        FakeSDK(create_result=Err(ProviderError(message="boom", code="RUNPOD_SDK_CALL_FAILED"))),
    )

    res = client.create_pod(
        RunPodProviderConfig(connect={"ssh": {"key_path": "/k"}}, cleanup={}, training={"image_name": "img", "gpu_type": "NVIDIA A40"}, inference={})
    )
    assert res.is_failure()


def test_query_pod_success(monkeypatch: pytest.MonkeyPatch) -> None:
    client = RunPodAPIClient(api_base_url="https://api.runpod.io", api_key="rk")
    monkeypatch.setattr(
        client,
        "_sdk",
        FakeSDK(get_result=Ok({"id": "pod-1", "desiredStatus": "RUNNING", "runtime": {"uptimeInSeconds": 1}})),
    )

    res = client.query_pod("pod-1")
    assert res.is_success()
    assert res.unwrap()["id"] == "pod-1"


def test_query_pod_sdk_error(monkeypatch: pytest.MonkeyPatch) -> None:
    client = RunPodAPIClient(api_base_url="https://api.runpod.io", api_key="rk")
    monkeypatch.setattr(
        client,
        "_sdk",
        FakeSDK(get_result=Err(ProviderError(message="net down", code="RUNPOD_SDK_CALL_FAILED"))),
    )

    res = client.query_pod("pod-1")
    assert res.is_failure()


def test_query_pod_empty_payload_returns_data_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    client = RunPodAPIClient(api_base_url="https://api.runpod.io", api_key="rk")
    monkeypatch.setattr(client, "_sdk", FakeSDK(get_result=Ok({})))

    res = client.query_pod("pod-1")

    assert res.is_failure()
    err = res.unwrap_err()
    assert err.code == "RUNPOD_POD_DATA_MISSING"
    assert err.details == {"pod_id": "pod-1"}


def test_terminate_pod_request_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    client = RunPodAPIClient(api_base_url="https://api.runpod.io", api_key="rk")
    monkeypatch.setattr(
        client,
        "_sdk",
        FakeSDK(delete_result=Err(ProviderError(message="net down", code="RUNPOD_SDK_CALL_FAILED"))),
    )

    res = client.terminate_pod("pod-1")
    assert res.is_failure()


def test_terminate_pod_wraps_error_with_pod_details(monkeypatch: pytest.MonkeyPatch) -> None:
    client = RunPodAPIClient(api_base_url="https://api.runpod.io", api_key="rk")
    monkeypatch.setattr(
        client,
        "_sdk",
        FakeSDK(delete_result=Err(ProviderError(message="permission denied", code="RUNPOD_SDK_CALL_FAILED"))),
    )

    res = client.terminate_pod("pod-9")

    assert res.is_failure()
    err = res.unwrap_err()
    assert err.message == "Failed to terminate pod: permission denied"
    assert err.code == "RUNPOD_SDK_CALL_FAILED"
    assert err.details == {"pod_id": "pod-9"}


def test_get_ssh_info_success(monkeypatch: pytest.MonkeyPatch) -> None:
    client = RunPodAPIClient(api_base_url="https://api.runpod.io", api_key="rk")
    monkeypatch.setattr(
        client,
        "_sdk",
        FakeSDK(
            get_result=Ok(
                {
                    "id": "pod-1",
                    "runtime": {
                        "ports": [
                            {"ip": "1.2.3.4", "privatePort": 123, "publicPort": 111},
                            {"ip": "5.6.7.8", "privatePort": 22, "publicPort": 2222, "isIpPublic": True},
                        ]
                    },
                }
            )
        ),
    )

    res = client.get_ssh_info("pod-1")
    assert res.is_success()
    assert res.unwrap() == {"host": "5.6.7.8", "port": 2222}


def test_get_ssh_info_no_ssh_port(monkeypatch: pytest.MonkeyPatch) -> None:
    client = RunPodAPIClient(api_base_url="https://api.runpod.io", api_key="rk")
    monkeypatch.setattr(
        client,
        "_sdk",
        FakeSDK(get_result=Ok({"id": "pod-1", "runtime": {"ports": [{"privatePort": 80}]}})),
    )

    res = client.get_ssh_info("pod-1")
    assert res.is_failure()
    assert "SSH over exposed TCP" in str(res.unwrap_err())


def test_get_ssh_info_propagates_query_error(monkeypatch: pytest.MonkeyPatch) -> None:
    client = RunPodAPIClient(api_base_url="https://api.runpod.io", api_key="rk")
    monkeypatch.setattr(
        client,
        "_sdk",
        FakeSDK(get_result=Err(ProviderError(message="sdk unavailable", code="RUNPOD_SDK_CALL_FAILED"))),
    )

    res = client.get_ssh_info("pod-1")

    assert res.is_failure()
    assert res.unwrap_err().details == {"pod_id": "pod-1"}


def test_get_ssh_info_ignores_non_public_or_invalid_ssh_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
    client = RunPodAPIClient(api_base_url="https://api.runpod.io", api_key="rk")
    monkeypatch.setattr(
        client,
        "_sdk",
        FakeSDK(
            get_result=Ok(
                {
                    "id": "pod-1",
                    "runtime": {
                        "ports": [
                            {"ip": "10.0.0.1", "privatePort": 22, "publicPort": 2222, "isIpPublic": False},
                            {"ip": "", "privatePort": 22, "publicPort": 3333, "isIpPublic": True},
                        ]
                    },
                }
            )
        ),
    )

    res = client.get_ssh_info("pod-1")
    assert res.is_failure()
    assert "SSH over exposed TCP" in str(res.unwrap_err())


def test_extract_exposed_ssh_info_returns_runtime_not_available_for_missing_runtime() -> None:
    res = RunPodAPIClient.extract_exposed_ssh_info({"id": "pod-1"}, pod_id="pod-1")

    assert res.is_failure()
    err = res.unwrap_err()
    assert err.code == "RUNPOD_RUNTIME_NOT_AVAILABLE"
    assert err.details == {"pod_id": "pod-1"}


# ---------------------------------------------------------------------------
# Extracted pure functions
# ---------------------------------------------------------------------------


def test_build_ssh_bootstrap_cmd_contains_sshd_setup() -> None:
    cmd = build_ssh_bootstrap_cmd()
    assert "sshd" in cmd
    assert "authorized_keys" in cmd
    assert "PUBLIC_KEY" in cmd
    assert cmd.startswith("bash -c")


def test_build_pod_launch_kwargs_contains_config_values() -> None:
    cfg = RunPodProviderConfig(
        connect={"ssh": {"key_path": "/k"}},
        cleanup={},
        training={"image_name": "myimg:v1", "gpu_type": "NVIDIA A40", "ports": "8888/http,22/tcp"},
        inference={},
    )
    kwargs = build_pod_launch_kwargs(cfg, "test-pod", "ssh-ed25519 AAAA...")
    assert kwargs["gpu_type_id"] == "NVIDIA A40"
    assert kwargs["image_name"] == "myimg:v1"
    assert kwargs["name"] == "test-pod"
    assert kwargs["env"]["PUBLIC_KEY"] == "ssh-ed25519 AAAA..."
    assert kwargs["ports"] == "8888/http,22/tcp"
    assert kwargs["volume_mount_path"] == "/workspace"


def test_build_pod_launch_kwargs_without_public_key() -> None:
    cfg = RunPodProviderConfig(
        connect={"ssh": {"key_path": "/k"}},
        cleanup={},
        training={"image_name": "img", "gpu_type": "NVIDIA A40"},
        inference={},
    )
    kwargs = build_pod_launch_kwargs(cfg, None, None)
    assert kwargs["env"] is None


def test_build_pod_launch_kwargs_truncates_long_name() -> None:
    cfg = RunPodProviderConfig(
        connect={"ssh": {"key_path": "/k"}},
        cleanup={},
        training={"image_name": "img", "gpu_type": "NVIDIA A40"},
        inference={},
    )
    long_name = "x" * 200
    kwargs = build_pod_launch_kwargs(cfg, long_name, None)
    assert kwargs["name"] == "x" * 80
