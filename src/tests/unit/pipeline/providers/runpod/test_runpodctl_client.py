from __future__ import annotations

import os
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from src.providers.runpod.runpodctl_client import RunPodCtlClient, resolve_runpodctl_binary
from src.utils.result import Ok

pytestmark = pytest.mark.unit


def test_resolve_runpodctl_binary_prefers_explicit_arg() -> None:
    assert resolve_runpodctl_binary("/tmp/custom-runpodctl") == "/tmp/custom-runpodctl"


def test_resolve_runpodctl_binary_prefers_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RYOTENKAI_RUNPODCTL_PATH", "/tmp/from-env")
    assert resolve_runpodctl_binary() == "/tmp/from-env"


def test_resolve_runpodctl_binary_prefers_repo_local_binary(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RYOTENKAI_RUNPODCTL_PATH", raising=False)
    assert resolve_runpodctl_binary().endswith("/runpodctl")


def test_is_available_uses_resolved_binary(monkeypatch: pytest.MonkeyPatch) -> None:
    client = RunPodCtlClient(api_key="rk", binary="/tmp/runpodctl")
    monkeypatch.setattr("shutil.which", lambda value: value if value == "/tmp/runpodctl" else None)
    assert client.is_available() is True


def test_run_returns_not_available_when_binary_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    client = RunPodCtlClient(api_key="rk", binary="/tmp/missing-runpodctl")
    monkeypatch.setattr("shutil.which", lambda value: None)
    res = client._run(["version"], timeout=10)
    assert res.is_failure()
    assert res.unwrap_err().code == "RUNPODCTL_NOT_AVAILABLE"


def test_run_injects_runpod_env(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}
    client = RunPodCtlClient(api_key="rk", api_url="https://api.example", binary="/tmp/runpodctl")
    monkeypatch.setattr("shutil.which", lambda value: "/tmp/runpodctl")

    def fake_run(cmd, capture_output, text, timeout, env):
        captured["cmd"] = cmd
        captured["env"] = env
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    res = client._run(["version"], timeout=10)
    assert res.is_success()
    assert captured["cmd"][0] == "/tmp/runpodctl"
    assert captured["env"]["RUNPOD_API_KEY"] == "rk"
    assert captured["env"]["RUNPOD_API_URL"] == "https://api.example"


def test_run_handles_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    client = RunPodCtlClient(api_key="rk", binary="/tmp/runpodctl")
    monkeypatch.setattr("shutil.which", lambda value: "/tmp/runpodctl")

    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="runpodctl", timeout=10)

    monkeypatch.setattr(subprocess, "run", fake_run)
    res = client._run(["version"], timeout=10)
    assert res.is_failure()
    assert res.unwrap_err().code == "RUNPODCTL_COMMAND_FAILED"


def test_run_handles_oserror(monkeypatch: pytest.MonkeyPatch) -> None:
    client = RunPodCtlClient(api_key="rk", binary="/tmp/runpodctl")
    monkeypatch.setattr("shutil.which", lambda value: "/tmp/runpodctl")

    def fake_run(*args, **kwargs):
        raise OSError("boom")

    monkeypatch.setattr(subprocess, "run", fake_run)
    res = client._run(["version"], timeout=10)
    assert res.is_failure()
    assert res.unwrap_err().code == "RUNPODCTL_COMMAND_FAILED"


def test_run_handles_nonzero_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    client = RunPodCtlClient(api_key="rk", binary="/tmp/runpodctl")
    monkeypatch.setattr("shutil.which", lambda value: "/tmp/runpodctl")
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=2, stdout="", stderr="bad flags"),
    )
    res = client._run(["bad"], timeout=10)
    assert res.is_failure()
    assert res.unwrap_err().code == "RUNPODCTL_COMMAND_FAILED"


def test_run_json_parses_success(monkeypatch: pytest.MonkeyPatch) -> None:
    client = RunPodCtlClient(api_key="rk", binary="/tmp/runpodctl")
    monkeypatch.setattr(client, "_run", lambda args, timeout: Ok('{"id":"pod-1"}'))  # type: ignore[call-arg]
    res = client._run_json(["get", "pod"], timeout=10)
    assert res.is_success()
    assert res.unwrap()["id"] == "pod-1"


def test_run_json_rejects_invalid_json(monkeypatch: pytest.MonkeyPatch) -> None:
    client = RunPodCtlClient(api_key="rk", binary="/tmp/runpodctl")
    monkeypatch.setattr(client, "_run", lambda args, timeout: Ok("not-json"))  # type: ignore[call-arg]
    res = client._run_json(["get", "pod"], timeout=10)
    assert res.is_failure()
    assert res.unwrap_err().code == "RUNPODCTL_OUTPUT_INVALID"


def test_run_json_rejects_non_dict(monkeypatch: pytest.MonkeyPatch) -> None:
    client = RunPodCtlClient(api_key="rk", binary="/tmp/runpodctl")
    monkeypatch.setattr(client, "_run", lambda args, timeout: Ok('["x"]'))  # type: ignore[call-arg]
    res = client._run_json(["get", "pod"], timeout=10)
    assert res.is_failure()
    assert res.unwrap_err().code == "RUNPODCTL_OUTPUT_INVALID"


def test_create_training_pod_uses_new_cli_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    client = RunPodCtlClient(api_key="rk", binary="/tmp/runpodctl")
    captured: dict[str, Any] = {}

    def fake_run_json(args, timeout):
        captured["args"] = args
        return Ok({"id": "pod-1"})  # type: ignore[call-arg]

    monkeypatch.setattr(client, "_run_json", fake_run_json)
    res = client.create_training_pod(
        gpu_type="NVIDIA A40",
        image_name="img",
        pod_name="pod",
        cloud_type="SECURE",
        container_disk_gb=50,
        volume_disk_gb=20,
        ports="22/tcp",
        public_key="ssh-ed25519 AAA",
    )
    assert res.is_success()
    assert captured["args"][:2] == ["pod", "create"]
    assert "--cloud-type" in captured["args"]
    assert "SECURE" in captured["args"]
    assert "--gpu-id" in captured["args"]
    assert "--image" in captured["args"]
    assert "--container-disk-in-gb" in captured["args"]
    assert "--volume-in-gb" in captured["args"]
    assert "--volume-mount-path" in captured["args"]
    assert "--ssh" in captured["args"]
    assert "--env" in captured["args"]
    env_idx = captured["args"].index("--env") + 1
    assert captured["args"][env_idx] == '{"PUBLIC_KEY": "ssh-ed25519 AAA"}'


def test_read_public_key_returns_none_on_missing_file(tmp_path: Path) -> None:
    assert RunPodCtlClient.read_public_key(str(tmp_path / "id_ed25519")) is None


def test_read_public_key_reads_sibling_pub_file(tmp_path: Path) -> None:
    key_path = tmp_path / "id_ed25519"
    pub_path = tmp_path / "id_ed25519.pub"
    key_path.write_text("PRIVATE")
    pub_path.write_text("ssh-ed25519 AAA test")
    assert RunPodCtlClient.read_public_key(str(key_path)) == "ssh-ed25519 AAA test"
