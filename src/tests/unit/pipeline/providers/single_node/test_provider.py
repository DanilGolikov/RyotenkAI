from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import pytest

import src.providers.single_node.training.provider as sp
from src.pipeline.domain import RunContext
from src.providers.single_node.training.health_check import HealthCheckResult
from src.providers.single_node.training.provider import SingleNodeProvider
from src.providers.training.interfaces import GPUInfo, ProviderStatus
from src.utils.config import Secrets


@dataclass
class FakeSSHClient:
    host: str
    port: int
    username: str | None = None
    key_path: str = "/k"
    connect_timeout: int = 10
    dirs: set[str] = field(default_factory=set)
    commands: list[str] = field(default_factory=list)
    connection_ok: bool = True

    def test_connection(self, max_retries: int, retry_delay: int):
        return self.connection_ok, "" if self.connection_ok else "no"

    def directory_exists(self, remote_path: str) -> bool:
        return remote_path in self.dirs

    def create_directory(self, remote_path: str) -> tuple[bool, str]:
        self.dirs.add(remote_path)
        return True, ""

    def exec_command(self, command: str, timeout: int = 30, silent: bool = False):
        self.commands.append(command)
        # Disconnect cleanup verification
        if "echo EXISTS || echo DELETED" in command:
            return True, "DELETED\n", ""
        return True, "", ""


class FakeHealthCheck:
    def __init__(self, ssh_client: FakeSSHClient):
        self.ssh = ssh_client

    def run_all_checks(self, workspace_path: str):
        return HealthCheckResult(
            passed=True,
            gpu_info=GPUInfo(
                name="GPU",
                vram_total_mb=10000,
                vram_free_mb=9000,
                cuda_version="12.1",
                driver_version="x",
            ),
        )

    def check_gpu(self):
        return sp.Ok(  # type: ignore[attr-defined]
            GPUInfo(name="GPU", vram_total_mb=1, vram_free_mb=1, cuda_version="x", driver_version="x")
        )


def _deep_update(dst: dict[str, Any], src: dict[str, Any]) -> dict[str, Any]:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)  # type: ignore[index]
        else:
            dst[k] = v
    return dst


def _mk_provider(*, cfg_overrides: dict[str, Any] | None = None) -> SingleNodeProvider:
    # New provider schema (v3): connect / training / inference
    cfg: dict[str, Any] = {
        "connect": {"ssh": {"alias": "pc"}},
        "training": {"workspace_path": "/workspace", "docker_image": "test/runtime:latest"},
    }
    if cfg_overrides:
        _deep_update(cfg, cfg_overrides)
    secrets = Secrets(HF_TOKEN="hf_test")
    return SingleNodeProvider(config=cfg, secrets=secrets)


def _mk_run() -> RunContext:
    return RunContext(
        name="run_20260120_123456_abc12",
        created_at_utc=datetime(2026, 1, 20, 12, 34, 56, tzinfo=timezone.utc),
    )


def test_connect_success_creates_run_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sp, "SSHClient", FakeSSHClient)
    monkeypatch.setattr(sp, "SingleNodeHealthCheck", FakeHealthCheck)

    p = _mk_provider()
    run = _mk_run()

    # Pretend base workspace already exists
    ssh: FakeSSHClient = FakeSSHClient(host="pc", port=22)
    ssh.dirs.add("/workspace")
    monkeypatch.setattr(sp, "SSHClient", lambda *a, **k: ssh)

    res = p.connect(run=run)
    assert res.is_success()
    info = res.unwrap()
    assert info.workspace_path.endswith(f"/runs/{run.name}")
    assert p.get_status() == ProviderStatus.CONNECTED


def test_init_explicit_mode_and_provider_properties(monkeypatch: pytest.MonkeyPatch) -> None:
    # Explicit mode: cover __init__ non-alias logging branch + properties
    p = _mk_provider(
        cfg_overrides={"connect": {"ssh": {"host": "1.2.3.4", "user": "u", "port": 2222}}},
    )
    assert p.provider_name == "single_node"
    assert p.provider_type == "local"
    assert "SingleNodeProvider" in repr(p)


def test_connect_already_connected_returns_cached_info(monkeypatch: pytest.MonkeyPatch) -> None:
    ssh = FakeSSHClient(host="pc", port=22)
    monkeypatch.setattr(sp, "SSHClient", lambda *a, **k: ssh)
    monkeypatch.setattr(sp, "SingleNodeHealthCheck", FakeHealthCheck)

    p = _mk_provider()
    run = _mk_run()
    ssh.dirs.add("/workspace")

    first = p.connect(run=run)
    assert first.is_success()
    second = p.connect(run=run)
    assert second.is_success()
    assert second.unwrap() == first.unwrap()


def test_connect_fails_when_ssh_unreachable(monkeypatch: pytest.MonkeyPatch) -> None:
    ssh = FakeSSHClient(host="pc", port=22, connection_ok=False)
    monkeypatch.setattr(sp, "SSHClient", lambda *a, **k: ssh)
    monkeypatch.setattr(sp, "SingleNodeHealthCheck", FakeHealthCheck)

    p = _mk_provider()
    res = p.connect(run=_mk_run())
    assert res.is_failure()
    assert p.get_status() == ProviderStatus.ERROR


def test_connect_fails_when_health_checks_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    ssh = FakeSSHClient(host="pc", port=22)
    monkeypatch.setattr(sp, "SSHClient", lambda *a, **k: ssh)

    class _BadHealth:
        def __init__(self, ssh_client: FakeSSHClient):
            self.ssh = ssh_client

        def run_all_checks(self, workspace_path: str):
            return HealthCheckResult(passed=False, errors=["boom"])

    monkeypatch.setattr(sp, "SingleNodeHealthCheck", _BadHealth)

    p = _mk_provider()
    res = p.connect(run=_mk_run())
    assert res.is_failure()
    assert p.get_status() == ProviderStatus.ERROR


def test_connect_creates_base_workspace_and_handles_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    # Base workspace missing -> create ok, but run dir creation fails
    ssh = FakeSSHClient(host="pc", port=22)
    monkeypatch.setattr(sp, "SSHClient", lambda *a, **k: ssh)
    monkeypatch.setattr(sp, "SingleNodeHealthCheck", FakeHealthCheck)

    created: list[str] = []
    run = _mk_run()

    def create_directory(remote_path: str) -> tuple[bool, str]:
        created.append(remote_path)
        if remote_path.endswith(f"/{run.name}"):
            return False, "nope"
        ssh.dirs.add(remote_path)
        return True, ""

    ssh.create_directory = create_directory  # type: ignore[method-assign]

    p = _mk_provider()
    # workspace does not exist initially
    res = p.connect(run=run)
    assert res.is_failure()
    assert any(r == "/workspace" for r in created)
    assert p.get_status() == ProviderStatus.ERROR


def test_disconnect_cleanup_uses_docker_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    ssh = FakeSSHClient(host="pc", port=22)
    monkeypatch.setattr(sp, "SSHClient", lambda *a, **k: ssh)
    monkeypatch.setattr(sp, "SingleNodeHealthCheck", FakeHealthCheck)

    p = _mk_provider(cfg_overrides={"cleanup": {"cleanup_workspace": True, "keep_on_error": True}})
    run = _mk_run()
    ssh.dirs.add("/workspace")

    assert p.connect(run=run).is_success()
    assert p.disconnect().is_success()
    assert p.get_status() == ProviderStatus.AVAILABLE
    assert any("docker run --rm" in c for c in ssh.commands)


def test_disconnect_not_connected_is_noop() -> None:
    p = _mk_provider()
    assert p.disconnect().is_success()


def test_disconnect_docker_cleanup_failure_falls_back_to_rm(monkeypatch: pytest.MonkeyPatch) -> None:
    ssh = FakeSSHClient(host="pc", port=22)

    def exec_side_effect(command: str, timeout: int = 30, silent: bool = False):
        ssh.commands.append(command)
        if command.startswith("docker run --rm"):
            return False, "", "docker failed"
        if command.startswith("test -d"):
            return True, "EXISTS\n", ""
        return True, "", ""

    ssh.exec_command = exec_side_effect  # type: ignore[method-assign]

    monkeypatch.setattr(sp, "SSHClient", lambda *a, **k: ssh)
    monkeypatch.setattr(sp, "SingleNodeHealthCheck", FakeHealthCheck)

    p = _mk_provider(cfg_overrides={"cleanup": {"cleanup_workspace": True, "keep_on_error": False}})
    run = _mk_run()
    ssh.dirs.add("/workspace")
    assert p.connect(run=run).is_success()

    ssh.commands.clear()
    assert p.disconnect().is_success()
    assert any(c == f"rm -rf /workspace/runs/{run.name}" for c in ssh.commands)


def test_disconnect_keeps_workspace_on_error(monkeypatch: pytest.MonkeyPatch) -> None:
    ssh = FakeSSHClient(host="pc", port=22)
    monkeypatch.setattr(sp, "SSHClient", lambda *a, **k: ssh)
    monkeypatch.setattr(sp, "SingleNodeHealthCheck", FakeHealthCheck)

    p = _mk_provider(cfg_overrides={"cleanup": {"cleanup_workspace": True, "keep_on_error": True}})
    run = _mk_run()
    ssh.dirs.add("/workspace")
    assert p.connect(run=run).is_success()

    p.mark_error()
    ssh.commands.clear()

    assert p.disconnect().is_success()
    assert all("docker run --rm" not in c for c in ssh.commands)
