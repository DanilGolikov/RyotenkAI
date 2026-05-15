"""Tests for the raise-based :class:`RunPodAPIClient` (Phase A2 Batch 11)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from ryotenkai_shared.config.providers.runpod import RunPodProviderConfig
from ryotenkai_providers.runpod.training.api_client import (
    RUNPOD_DOCKER_ARGS,
    RunPodAPIClient,
    build_pod_launch_kwargs,
)
from ryotenkai_shared.constants import RUNTIME_IMAGE
from ryotenkai_shared.errors import ProviderUnavailableError


@dataclass
class FakeSDK:
    """Canonical fake for ``RunPodSDKClient`` — raise-based contract.

    Each ``*_value`` field holds the return value to deliver on success;
    each ``*_exc`` field, when set, is raised instead. Either-or per call.
    """

    create_value: dict[str, Any] | None = None
    create_exc: BaseException | None = None
    get_value: dict[str, Any] | None = None
    get_exc: BaseException | None = None
    delete_exc: BaseException | None = None
    stop_exc: BaseException | None = None
    start_exc: BaseException | None = None

    def create_pod(self, **kwargs: Any) -> dict[str, Any]:
        if self.create_exc is not None:
            raise self.create_exc
        assert self.create_value is not None
        return self.create_value

    def get_pod(self, *, pod_id: str) -> dict[str, Any]:
        if self.get_exc is not None:
            raise self.get_exc
        assert self.get_value is not None
        return self.get_value

    def delete_pod(self, *, pod_id: str) -> None:
        if self.delete_exc is not None:
            raise self.delete_exc

    def stop_pod(self, *, pod_id: str) -> None:
        if self.stop_exc is not None:
            raise self.stop_exc

    def start_pod(self, *, pod_id: str) -> None:
        if self.start_exc is not None:
            raise self.start_exc


def _mk_cfg(*, training_overrides: dict[str, Any] | None = None) -> RunPodProviderConfig:
    """Build a minimal valid RunPodProviderConfig for tests."""
    training: dict[str, Any] = {"gpu_type": "NVIDIA A40"}
    if training_overrides:
        training.update(training_overrides)
    return RunPodProviderConfig(
        connect={"ssh": {"key_path": "/k"}},
        cleanup={},
        training=training,
        inference={},
    )


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
    fake_sdk = FakeSDK(create_value=pod_payload, get_value=pod_payload)
    monkeypatch.setattr(client, "_sdk", fake_sdk)

    data = client.create_pod(_mk_cfg())

    assert data["pod_id"] == "pod-1"
    assert data["machine"] == "host-1"
    assert data["cost_per_hr"] == 0.79


def test_create_pod_propagates_sdk_error(monkeypatch: pytest.MonkeyPatch) -> None:
    client = RunPodAPIClient(api_base_url="https://api.runpod.io", api_key="rk")
    monkeypatch.setattr(
        client,
        "_sdk",
        FakeSDK(create_exc=ProviderUnavailableError(detail="no capacity", context={"code": "RUNPOD_SDK_CALL_FAILED"})),
    )

    with pytest.raises(ProviderUnavailableError) as ei:
        client.create_pod(_mk_cfg())
    assert "no capacity" in (ei.value.detail or "")


def test_create_pod_missing_id_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    client = RunPodAPIClient(api_base_url="https://api.runpod.io", api_key="rk")
    monkeypatch.setattr(client, "_sdk", FakeSDK(create_value={}))

    with pytest.raises(ProviderUnavailableError) as ei:
        client.create_pod(_mk_cfg())
    assert ei.value.context.get("code") == "RUNPOD_POD_DATA_MISSING"


def test_query_pod_success(monkeypatch: pytest.MonkeyPatch) -> None:
    client = RunPodAPIClient(api_base_url="https://api.runpod.io", api_key="rk")
    monkeypatch.setattr(
        client,
        "_sdk",
        FakeSDK(get_value={"id": "pod-1", "desiredStatus": "RUNNING", "runtime": {"uptimeInSeconds": 1}}),
    )

    pod = client.query_pod("pod-1")
    assert pod["id"] == "pod-1"


def test_query_pod_sdk_error_attaches_pod_id(monkeypatch: pytest.MonkeyPatch) -> None:
    client = RunPodAPIClient(api_base_url="https://api.runpod.io", api_key="rk")
    monkeypatch.setattr(
        client,
        "_sdk",
        FakeSDK(get_exc=ProviderUnavailableError(detail="net down", context={"code": "RUNPOD_SDK_CALL_FAILED"})),
    )

    with pytest.raises(ProviderUnavailableError) as ei:
        client.query_pod("pod-1")
    assert ei.value.context.get("pod_id") == "pod-1"


def test_query_pod_empty_payload_raises_data_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    client = RunPodAPIClient(api_base_url="https://api.runpod.io", api_key="rk")
    monkeypatch.setattr(client, "_sdk", FakeSDK(get_value={}))

    with pytest.raises(ProviderUnavailableError) as ei:
        client.query_pod("pod-1")
    assert ei.value.context["code"] == "RUNPOD_POD_DATA_MISSING"
    assert ei.value.context["pod_id"] == "pod-1"


def test_terminate_pod_request_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    client = RunPodAPIClient(api_base_url="https://api.runpod.io", api_key="rk")
    monkeypatch.setattr(
        client,
        "_sdk",
        FakeSDK(delete_exc=ProviderUnavailableError(detail="net down", context={"code": "RUNPOD_SDK_CALL_FAILED"})),
    )

    with pytest.raises(ProviderUnavailableError):
        client.terminate_pod("pod-1")


def test_terminate_pod_wraps_error_with_pod_details(monkeypatch: pytest.MonkeyPatch) -> None:
    client = RunPodAPIClient(api_base_url="https://api.runpod.io", api_key="rk")
    monkeypatch.setattr(
        client,
        "_sdk",
        FakeSDK(
            delete_exc=ProviderUnavailableError(
                detail="permission denied", context={"code": "RUNPOD_SDK_CALL_FAILED"}
            )
        ),
    )

    with pytest.raises(ProviderUnavailableError) as ei:
        client.terminate_pod("pod-9")
    assert "permission denied" in (ei.value.detail or "")
    assert ei.value.context.get("code") == "RUNPOD_SDK_CALL_FAILED"
    assert ei.value.context.get("pod_id") == "pod-9"


def test_get_ssh_info_success(monkeypatch: pytest.MonkeyPatch) -> None:
    client = RunPodAPIClient(api_base_url="https://api.runpod.io", api_key="rk")
    monkeypatch.setattr(
        client,
        "_sdk",
        FakeSDK(
            get_value={
                "id": "pod-1",
                "runtime": {
                    "ports": [
                        {"ip": "1.2.3.4", "privatePort": 123, "publicPort": 111},
                        {"ip": "5.6.7.8", "privatePort": 22, "publicPort": 2222, "isIpPublic": True},
                    ]
                },
            }
        ),
    )

    out = client.get_ssh_info("pod-1")
    assert out == {"host": "5.6.7.8", "port": 2222}


def test_get_ssh_info_no_ssh_port_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    client = RunPodAPIClient(api_base_url="https://api.runpod.io", api_key="rk")
    monkeypatch.setattr(
        client,
        "_sdk",
        FakeSDK(get_value={"id": "pod-1", "runtime": {"ports": [{"privatePort": 80}]}}),
    )

    with pytest.raises(ProviderUnavailableError) as ei:
        client.get_ssh_info("pod-1")
    assert ei.value.context.get("code") == "RUNPOD_SSH_PORT_UNAVAILABLE"


def test_get_ssh_info_propagates_query_error(monkeypatch: pytest.MonkeyPatch) -> None:
    client = RunPodAPIClient(api_base_url="https://api.runpod.io", api_key="rk")
    monkeypatch.setattr(
        client,
        "_sdk",
        FakeSDK(get_exc=ProviderUnavailableError(detail="sdk unavailable", context={"code": "RUNPOD_SDK_CALL_FAILED"})),
    )

    with pytest.raises(ProviderUnavailableError) as ei:
        client.get_ssh_info("pod-1")
    assert ei.value.context.get("pod_id") == "pod-1"


def test_get_ssh_info_ignores_non_public_or_invalid_ssh_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
    client = RunPodAPIClient(api_base_url="https://api.runpod.io", api_key="rk")
    monkeypatch.setattr(
        client,
        "_sdk",
        FakeSDK(
            get_value={
                "id": "pod-1",
                "runtime": {
                    "ports": [
                        {"ip": "10.0.0.1", "privatePort": 22, "publicPort": 2222, "isIpPublic": False},
                        {"ip": "", "privatePort": 22, "publicPort": 3333, "isIpPublic": True},
                    ]
                },
            }
        ),
    )

    with pytest.raises(ProviderUnavailableError) as ei:
        client.get_ssh_info("pod-1")
    assert ei.value.context.get("code") == "RUNPOD_SSH_PORT_UNAVAILABLE"


def test_extract_exposed_ssh_info_raises_runtime_not_available_for_missing_runtime() -> None:
    with pytest.raises(ProviderUnavailableError) as ei:
        RunPodAPIClient.extract_exposed_ssh_info({"id": "pod-1"}, pod_id="pod-1")
    assert ei.value.context["code"] == "RUNPOD_RUNTIME_NOT_AVAILABLE"
    assert ei.value.context["pod_id"] == "pod-1"


# ---------------------------------------------------------------------------
# Extracted pure functions
# ---------------------------------------------------------------------------


def test_runpod_docker_args_is_empty_string() -> None:
    """Pod is intentionally INERT at boot — entrypoint.sh handles
    PUBLIC_KEY and sshd, then sleeps. The Mac launches uvicorn via
    SSH-exec from runner_launcher, NOT through ``docker_args``.
    """
    assert RUNPOD_DOCKER_ARGS == ""


def test_build_pod_launch_kwargs_contains_config_values() -> None:
    cfg = _mk_cfg(training_overrides={"ports": "8888/http,22/tcp"})
    kwargs = build_pod_launch_kwargs(cfg, "test-pod", "ssh-ed25519 AAAA...")
    assert kwargs["gpu_type_id"] == "NVIDIA A40"
    assert kwargs["image_name"] == RUNTIME_IMAGE
    assert kwargs["name"] == "test-pod"
    assert kwargs["env"]["PUBLIC_KEY"] == "ssh-ed25519 AAAA..."
    assert kwargs["ports"] == "8888/http,22/tcp"
    assert kwargs["volume_mount_path"] == "/workspace"


def test_build_pod_launch_kwargs_without_public_key() -> None:
    cfg = _mk_cfg()
    kwargs = build_pod_launch_kwargs(cfg, None, None)
    assert kwargs["env"] is None


def test_build_pod_launch_kwargs_truncates_long_name() -> None:
    cfg = _mk_cfg()
    long_name = "x" * 200
    kwargs = build_pod_launch_kwargs(cfg, long_name, None)
    assert kwargs["name"] == "x" * 80


def test_stop_pod_propagates_typed_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression guard: ``stop_pod`` re-raises with ``pod_id`` context."""
    client = RunPodAPIClient(api_base_url="https://api.runpod.io", api_key="rk")
    monkeypatch.setattr(
        client,
        "_sdk",
        FakeSDK(stop_exc=ProviderUnavailableError(detail="x", context={"code": "RUNPOD_SDK_CALL_FAILED"})),
    )

    with pytest.raises(ProviderUnavailableError) as ei:
        client.stop_pod("pod-stop-1")
    assert ei.value.context.get("pod_id") == "pod-stop-1"


def test_resume_pod_propagates_typed_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression guard: ``resume_pod`` re-raises with ``pod_id`` context."""
    client = RunPodAPIClient(api_base_url="https://api.runpod.io", api_key="rk")
    monkeypatch.setattr(
        client,
        "_sdk",
        FakeSDK(start_exc=ProviderUnavailableError(detail="x", context={"code": "RUNPOD_SDK_CALL_FAILED"})),
    )

    with pytest.raises(ProviderUnavailableError) as ei:
        client.resume_pod("pod-r-1")
    assert ei.value.context.get("pod_id") == "pod-r-1"
