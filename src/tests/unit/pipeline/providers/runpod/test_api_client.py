from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
import requests

from src.config.providers.runpod import RunPodProviderConfig
from src.providers.runpod.training.api_client import (
    RunPodAPIClient,
    build_pod_launch_mutation,
    build_ssh_bootstrap_cmd,
)


@dataclass
class FakeResponse:
    status_code: int = 200
    payload: dict[str, Any] | None = None
    text: str = ""
    raise_exc: Exception | None = None

    def raise_for_status(self) -> None:
        if self.raise_exc is not None:
            raise self.raise_exc

    def json(self) -> dict[str, Any]:
        assert self.payload is not None
        return self.payload


def test_create_pod_success(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_post(url: str, headers: dict[str, str], json: dict[str, Any], timeout: int):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        captured["timeout"] = timeout
        return FakeResponse(
            status_code=200,
            payload={
                "data": {
                    "podFindAndDeployOnDemand": {
                        "id": "pod-1",
                        "desiredStatus": "RUNNING",
                        "imageName": "img",
                        "gpuCount": 1,
                        "vcpuCount": 8,
                        "memoryInGb": 32,
                        "costPerHr": 0.79,
                        "machine": {"podHostId": "host-1"},
                    }
                }
            },
        )

    client = RunPodAPIClient(api_base_url="https://api.runpod.io", api_key="rk")
    monkeypatch.setattr(client.session, "post", fake_post)

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
    def fake_post(*a, **k):
        return FakeResponse(
            status_code=200,
            payload={"errors": [{"message": "no capacity"}]},
        )

    client = RunPodAPIClient(api_base_url="https://api.runpod.io", api_key="rk")
    monkeypatch.setattr(client.session, "post", fake_post)

    res = client.create_pod(
        RunPodProviderConfig(connect={"ssh": {"key_path": "/k"}}, cleanup={}, training={"image_name": "img", "gpu_type": "NVIDIA A40"}, inference={})
    )
    assert res.is_failure()
    assert "no capacity" in str(res.unwrap_err())


def test_create_pod_missing_id(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_post(*a, **k):
        return FakeResponse(status_code=200, payload={"data": {"podFindAndDeployOnDemand": {}}})

    client = RunPodAPIClient(api_base_url="https://api.runpod.io", api_key="rk")
    monkeypatch.setattr(client.session, "post", fake_post)

    res = client.create_pod(
        RunPodProviderConfig(connect={"ssh": {"key_path": "/k"}}, cleanup={}, training={"image_name": "img", "gpu_type": "NVIDIA A40"}, inference={})
    )
    assert res.is_failure()


def test_create_pod_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_post(*a, **k):
        return FakeResponse(
            status_code=500,
            text="server error",
            payload={},
            raise_exc=requests.HTTPError("boom"),
        )

    client = RunPodAPIClient(api_base_url="https://api.runpod.io", api_key="rk")
    monkeypatch.setattr(client.session, "post", fake_post)

    res = client.create_pod(
        RunPodProviderConfig(connect={"ssh": {"key_path": "/k"}}, cleanup={}, training={"image_name": "img", "gpu_type": "NVIDIA A40"}, inference={})
    )
    assert res.is_failure()


def test_query_pod_success(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_post(*a, **k):
        return FakeResponse(
            status_code=200,
            payload={"data": {"pod": {"id": "pod-1", "desiredStatus": "RUNNING", "runtime": {"uptimeInSeconds": 1}}}},
        )

    client = RunPodAPIClient(api_base_url="https://api.runpod.io", api_key="rk")
    monkeypatch.setattr(client.session, "post", fake_post)

    res = client.query_pod("pod-1")
    assert res.is_success()
    assert res.unwrap()["id"] == "pod-1"


def test_terminate_pod_request_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_post(*a, **k):
        raise requests.RequestException("net down")

    client = RunPodAPIClient(api_base_url="https://api.runpod.io", api_key="rk")
    monkeypatch.setattr(client.session, "post", fake_post)

    res = client.terminate_pod("pod-1")
    assert res.is_failure()


def test_get_ssh_info_success(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_post(*a, **k):
        return FakeResponse(
            status_code=200,
            payload={
                "data": {
                    "pod": {
                        "id": "pod-1",
                        "runtime": {
                            "ports": [
                                {"ip": "1.2.3.4", "privatePort": 123, "publicPort": 111},
                                {"ip": "5.6.7.8", "privatePort": 22, "publicPort": 2222, "isIpPublic": True},
                            ]
                        },
                    }
                }
            },
        )

    client = RunPodAPIClient(api_base_url="https://api.runpod.io", api_key="rk")
    monkeypatch.setattr(client.session, "post", fake_post)

    res = client.get_ssh_info("pod-1")
    assert res.is_success()
    assert res.unwrap() == {"host": "5.6.7.8", "port": 2222}


def test_get_ssh_info_no_ssh_port(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_post(*a, **k):
        return FakeResponse(
            status_code=200,
            payload={"data": {"pod": {"id": "pod-1", "runtime": {"ports": [{"privatePort": 80}]}}}},
        )

    client = RunPodAPIClient(api_base_url="https://api.runpod.io", api_key="rk")
    monkeypatch.setattr(client.session, "post", fake_post)

    res = client.get_ssh_info("pod-1")
    assert res.is_failure()
    assert "SSH over exposed TCP" in str(res.unwrap_err())


def test_get_ssh_info_ignores_non_public_or_invalid_ssh_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_post(*a, **k):
        return FakeResponse(
            status_code=200,
            payload={
                "data": {
                    "pod": {
                        "id": "pod-1",
                        "runtime": {
                            "ports": [
                                {"ip": "10.0.0.1", "privatePort": 22, "publicPort": 2222, "isIpPublic": False},
                                {"ip": "", "privatePort": 22, "publicPort": 3333, "isIpPublic": True},
                            ]
                        },
                    }
                }
            },
        )

    client = RunPodAPIClient(api_base_url="https://api.runpod.io", api_key="rk")
    monkeypatch.setattr(client.session, "post", fake_post)

    res = client.get_ssh_info("pod-1")
    assert res.is_failure()
    assert "SSH over exposed TCP" in str(res.unwrap_err())


# ---------------------------------------------------------------------------
# Extracted pure functions
# ---------------------------------------------------------------------------


def test_build_ssh_bootstrap_cmd_contains_sshd_setup() -> None:
    cmd = build_ssh_bootstrap_cmd()
    assert "sshd" in cmd
    assert "authorized_keys" in cmd
    assert "PUBLIC_KEY" in cmd
    assert cmd.startswith("bash -c")


def test_build_pod_launch_mutation_contains_config_values() -> None:
    cfg = RunPodProviderConfig(
        connect={"ssh": {"key_path": "/k"}},
        cleanup={},
        training={"image_name": "myimg:v1", "gpu_type": "NVIDIA A40", "ports": "8888/http,22/tcp"},
        inference={},
    )
    mutation = build_pod_launch_mutation(cfg, "test-pod", "ssh-ed25519 AAAA...")
    assert "NVIDIA A40" in mutation
    assert "myimg:v1" in mutation
    assert "test-pod" in mutation
    assert "PUBLIC_KEY" in mutation
    assert "8888/http,22/tcp" in mutation
    assert "podFindAndDeployOnDemand" in mutation


def test_build_pod_launch_mutation_without_public_key() -> None:
    cfg = RunPodProviderConfig(
        connect={"ssh": {"key_path": "/k"}},
        cleanup={},
        training={"image_name": "img", "gpu_type": "NVIDIA A40"},
        inference={},
    )
    mutation = build_pod_launch_mutation(cfg, None, None)
    assert "env: []" in mutation


def test_build_pod_launch_mutation_truncates_long_name() -> None:
    cfg = RunPodProviderConfig(
        connect={"ssh": {"key_path": "/k"}},
        cleanup={},
        training={"image_name": "img", "gpu_type": "NVIDIA A40"},
        inference={},
    )
    long_name = "x" * 200
    mutation = build_pod_launch_mutation(cfg, long_name, None)
    assert "x" * 80 in mutation
    assert "x" * 81 not in mutation
