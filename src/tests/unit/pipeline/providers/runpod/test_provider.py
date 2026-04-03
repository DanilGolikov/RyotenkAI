from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import pytest

import src.providers.runpod.training.provider as rp
from src.pipeline.domain import RunContext
from src.providers.runpod.models import PodSnapshot, SshEndpoint
from src.providers.runpod.training.provider import RunPodProvider
from src.providers.training.interfaces import ProviderStatus
from src.utils.config import Secrets
from src.utils.result import Err, Ok, ProviderError, Result

_SSH_OK = SshEndpoint(host="1.2.3.4", port=2222)


def _ready_snapshot(*, ssh: SshEndpoint | None = _SSH_OK) -> PodSnapshot:
    return PodSnapshot(pod_id="pod-1", status="RUNNING", uptime_seconds=10, ssh_endpoint=ssh, port_count=1 if ssh else 0)


@dataclass
class StubAPI:
    create_result: Result[dict[str, Any], ProviderError] = Ok({"pod_id": "pod-1", "machine": "host-1"})  # type: ignore[call-arg]

    def create_pod(self, config, *, pod_name: str | None = None):
        _ = config
        _ = pod_name
        return self.create_result


@dataclass
class StubCleanup:
    registered: list[tuple[str, str]] = field(default_factory=list)
    cleaned: list[str] = field(default_factory=list)

    def register_pod(self, pod_id: str, *, api_base: str) -> None:
        self.registered.append((pod_id, api_base))

    def cleanup_pod(self, pod_id: str) -> Result[None, ProviderError]:
        self.cleaned.append(pod_id)
        return Ok(None)


@dataclass
class StubLifecycle:
    result: Result[PodSnapshot, ProviderError]

    def wait_for_ready(self, pod_id: str, timeout: int = 300, max_retries: int = 3) -> Result[PodSnapshot, ProviderError]:
        return self.result


class FakeSSHClient:
    def __init__(self, host: str, port: int, username: str | None = None, key_path: str | None = None):
        self.host = host
        self.port = port
        self.username = username
        self.key_path = key_path or "/k"

    def test_connection(self, max_retries: int = 12, retry_delay: int = 10):
        return True, ""

    def create_directory(self, remote_path: str) -> tuple[bool, str]:
        _ = remote_path
        return True, ""

    def exec_command(self, command: str, timeout: int = 30, silent: bool = False):
        return True, "GPU, 10000, 9000, 555.0", ""


def _mk_provider(*, cfg_overrides: dict[str, Any] | None = None) -> RunPodProvider:
    cfg: dict[str, Any] = {
        "connect": {"ssh": {"key_path": __file__}},
        "cleanup": {},
        "training": {
            "gpu_type": "NVIDIA A40",
            "image_name": "test/training:latest",
        },
        "inference": {},
    }
    if cfg_overrides:
        cfg.update(cfg_overrides)
    secrets = Secrets(HF_TOKEN="hf_test", RUNPOD_API_KEY="rk")
    p = RunPodProvider(config=cfg, secrets=secrets)
    return p


def _mk_run() -> RunContext:
    return RunContext(
        name="run_20260120_123456_abc12",
        created_at_utc=datetime(2026, 1, 20, 12, 34, 56, tzinfo=timezone.utc),
    )


def test_connect_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rp, "SSHClient", FakeSSHClient)

    p = _mk_provider()
    p._api_client = StubAPI()
    p._cleanup_manager = StubCleanup()
    p._lifecycle = StubLifecycle(result=Ok(_ready_snapshot()))

    res = p.connect(run=_mk_run())
    assert res.is_success()
    ssh = res.unwrap()
    assert ssh.host == "1.2.3.4"
    assert ssh.port == 2222
    assert ssh.user == "root"
    assert ssh.key_path == __file__
    assert p.get_status() == ProviderStatus.CONNECTED

    cleanup: StubCleanup = p._cleanup_manager
    assert cleanup.registered
    assert cleanup.registered[0][0] == "pod-1"
    assert cleanup.registered[0][1] == "https://api.runpod.io"


def test_disconnect_keeps_pod_when_marked_error_and_keep_pod_on_error_true() -> None:
    p = _mk_provider(cfg_overrides={"cleanup": {"auto_delete_pod": True, "keep_pod_on_error": True}})
    cleanup = StubCleanup()
    p._cleanup_manager = cleanup

    p._pod_id = "pod-1"
    p._status = ProviderStatus.CONNECTED

    p.mark_error()
    res = p.disconnect()

    assert res.is_success()
    assert cleanup.cleaned == []


def test_connect_uses_hardcoded_workspace_for_run_workspace(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rp, "SSHClient", FakeSSHClient)

    p = _mk_provider()
    p._api_client = StubAPI()
    p._cleanup_manager = StubCleanup()
    p._lifecycle = StubLifecycle(result=Ok(_ready_snapshot()))

    res = p.connect(run=_mk_run())
    assert res.is_success()
    ssh = res.unwrap()
    assert ssh.workspace_path.startswith("/workspace/runs/")


def test_connect_fails_when_snapshot_has_no_ssh_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pod reported ready by lifecycle but with no SSH endpoint -> fail and cleanup."""
    monkeypatch.setattr(rp, "SSHClient", FakeSSHClient)

    p = _mk_provider()
    p._api_client = StubAPI()
    cleanup = StubCleanup()
    p._cleanup_manager = cleanup
    p._lifecycle = StubLifecycle(result=Ok(_ready_snapshot(ssh=None)))

    res = p.connect(run=_mk_run())
    assert res.is_failure()
    assert cleanup.cleaned == ["pod-1"]
    assert "SSH endpoint is missing" in str(res.unwrap_err())


def test_provider_properties_and_repr() -> None:
    p = _mk_provider()
    assert p.provider_name == "runpod"
    assert p.provider_type == "cloud"
    assert "RunPodProvider" in repr(p)


def test_connect_already_connected_returns_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rp, "SSHClient", FakeSSHClient)

    p = _mk_provider()
    p._ssh_connection_info = rp.SSHConnectionInfo(
        host="1.2.3.4", port=2222, user="root", key_path="/k", workspace_path="/workspace", resource_id="pod-1"
    )
    p._status = ProviderStatus.CONNECTED

    res = p.connect(run=_mk_run())
    assert res.is_success()
    assert res.unwrap().host == "1.2.3.4"


def test_connect_create_pod_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rp, "SSHClient", FakeSSHClient)

    p = _mk_provider()
    p._api_client = StubAPI(create_result=Err("no capacity"))
    p._cleanup_manager = StubCleanup()
    p._lifecycle = StubLifecycle(result=Ok(_ready_snapshot()))

    res = p.connect(run=_mk_run())
    assert res.is_failure()
    assert p.get_status() == ProviderStatus.ERROR
    cleanup: StubCleanup = p._cleanup_manager
    assert cleanup.cleaned == []


def test_connect_invalid_pod_info_and_missing_machine(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rp, "SSHClient", FakeSSHClient)

    p = _mk_provider()
    p._api_client = StubAPI(create_result=Ok({"x": 1}))
    p._cleanup_manager = StubCleanup()
    p._lifecycle = StubLifecycle(result=Ok(_ready_snapshot()))
    assert p.connect(run=_mk_run()).is_failure()

    p = _mk_provider()
    p._api_client = StubAPI(create_result=Ok({"pod_id": "pod-1"}))
    p._cleanup_manager = StubCleanup()
    p._lifecycle = StubLifecycle(result=Ok(_ready_snapshot()))
    assert p.connect(run=_mk_run()).is_success()

    p = _mk_provider()
    p._api_client = StubAPI(create_result=Ok({"pod_id": "pod-1"}))
    cleanup = StubCleanup()
    p._cleanup_manager = cleanup
    p._lifecycle = StubLifecycle(result=Ok(_ready_snapshot(ssh=None)))
    assert p.connect(run=_mk_run()).is_failure()
    assert cleanup.cleaned == ["pod-1"]


def test_connect_ssh_test_connection_failure_triggers_cleanup(monkeypatch: pytest.MonkeyPatch) -> None:
    class _BadSSH(FakeSSHClient):
        def test_connection(self, max_retries: int = 12, retry_delay: int = 10):
            return False, "nope"

    monkeypatch.setattr(rp, "SSHClient", _BadSSH)

    p = _mk_provider()
    p._api_client = StubAPI()
    cleanup = StubCleanup()
    p._cleanup_manager = cleanup
    p._lifecycle = StubLifecycle(result=Ok(_ready_snapshot()))

    res = p.connect(run=_mk_run())
    assert res.is_failure()
    assert cleanup.cleaned == ["pod-1"]


def test_connect_health_check_failure_triggers_cleanup(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FailNvidiaSmiSSH(FakeSSHClient):
        def exec_command(self, command: str, timeout: int = 30, silent: bool = False):
            return False, "", "nvidia-smi not found"

    monkeypatch.setattr(rp, "SSHClient", _FailNvidiaSmiSSH)

    p = _mk_provider()
    p._api_client = StubAPI()
    cleanup = StubCleanup()
    p._cleanup_manager = cleanup
    p._lifecycle = StubLifecycle(result=Ok(_ready_snapshot()))

    res = p.connect(run=_mk_run())
    assert res.is_failure()
    assert cleanup.cleaned == ["pod-1"]


def test_connect_exception_triggers_cleanup(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rp, "SSHClient", FakeSSHClient)

    p = _mk_provider()
    p._api_client = StubAPI()
    cleanup = StubCleanup()
    p._cleanup_manager = cleanup

    class _BadLifecycle:
        def wait_for_ready(self, pod_id: str, timeout: int = 300, max_retries: int = 3):
            raise RuntimeError("boom")

    p._lifecycle = _BadLifecycle()

    res = p.connect(run=_mk_run())
    assert res.is_failure()
    assert cleanup.cleaned == ["pod-1"]


def test_connect_wait_for_ready_failure_triggers_cleanup(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rp, "SSHClient", FakeSSHClient)

    p = _mk_provider()
    p._api_client = StubAPI()
    p._cleanup_manager = StubCleanup()
    p._lifecycle = StubLifecycle(result=Err("timeout"))

    res = p.connect(run=_mk_run())
    assert res.is_failure()
    cleanup: StubCleanup = p._cleanup_manager
    assert cleanup.cleaned == ["pod-1"]


def test_disconnect_terminates_pod_when_configured() -> None:
    p = _mk_provider(cfg_overrides={"cleanup": {"auto_delete_pod": True}})
    cleanup = StubCleanup()
    p._cleanup_manager = cleanup

    p._pod_id = "pod-1"
    p._status = ProviderStatus.CONNECTED

    res = p.disconnect()
    assert res.is_success()
    assert cleanup.cleaned == ["pod-1"]
    assert p.get_status() == ProviderStatus.AVAILABLE


def test_disconnect_keeps_pod_when_auto_delete_false() -> None:
    p = _mk_provider(cfg_overrides={"cleanup": {"auto_delete_pod": False}})
    cleanup = StubCleanup()
    p._cleanup_manager = cleanup

    p._pod_id = "pod-1"
    p._status = ProviderStatus.CONNECTED

    res = p.disconnect()
    assert res.is_success()
    assert cleanup.cleaned == []


def test_disconnect_keep_pod_on_error_wins_over_auto_delete() -> None:
    p = _mk_provider(cfg_overrides={"cleanup": {"auto_delete_pod": True, "keep_pod_on_error": True}})
    cleanup = StubCleanup()
    p._cleanup_manager = cleanup

    p._pod_id = "pod-1"
    p._status = ProviderStatus.ERROR

    res = p.disconnect()
    assert res.is_success()
    assert cleanup.cleaned == []


def test_disconnect_not_connected_and_no_pod_id() -> None:
    p = _mk_provider()
    assert p.disconnect().is_success()

    p._status = ProviderStatus.CONNECTED
    p._pod_id = None
    assert p.disconnect().is_success()


def test_disconnect_while_connecting_terminates_pod_sigint_regression() -> None:
    p = _mk_provider(cfg_overrides={"cleanup": {"auto_delete_pod": True}})
    cleanup = StubCleanup()
    p._cleanup_manager = cleanup

    p._pod_id = "pod-sigint"
    p._status = ProviderStatus.CONNECTING

    res = p.disconnect()

    assert res.is_success()
    assert cleanup.cleaned == ["pod-sigint"], (
        "Pod must be terminated when status=CONNECTING but pod_id is already set "
        "(SIGINT race condition during wait_for_ready)"
    )
    assert p.get_status() == ProviderStatus.AVAILABLE


def test_check_gpu_parses_nvidia_smi(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rp, "SSHClient", FakeSSHClient)

    p = _mk_provider()
    p._status = ProviderStatus.CONNECTED
    p._ssh_connection_info = rp.SSHConnectionInfo(
        host="1.2.3.4",
        port=22,
        user="root",
        key_path="/k",
        workspace_path="/workspace",
        resource_id="pod-1",
    )

    res = p.check_gpu()
    assert res.is_success()
    gpu = res.unwrap()
    assert gpu.name == "GPU"


def test_check_gpu_errors_and_parse_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rp, "SSHClient", FakeSSHClient)
    p = _mk_provider()

    assert p.check_gpu().is_failure()

    class _FailSSH(FakeSSHClient):
        def exec_command(self, command: str, timeout: int = 30, silent: bool = False):
            return False, "", "err"

    monkeypatch.setattr(rp, "SSHClient", _FailSSH)
    p._status = ProviderStatus.CONNECTED
    p._ssh_connection_info = rp.SSHConnectionInfo(
        host="1.2.3.4", port=22, user="root", key_path="/k", workspace_path="/workspace", resource_id="pod-1"
    )
    assert p.check_gpu().is_failure()

    class _BadFmtSSH(FakeSSHClient):
        def exec_command(self, command: str, timeout: int = 30, silent: bool = False):
            return True, "only_one_field", ""

    monkeypatch.setattr(rp, "SSHClient", _BadFmtSSH)
    assert p.check_gpu().is_failure()

    class _ParseSSH(FakeSSHClient):
        def exec_command(self, command: str, timeout: int = 30, silent: bool = False):
            return True, "GPU, notint, 1, driver", ""

    monkeypatch.setattr(rp, "SSHClient", _ParseSSH)
    assert p.check_gpu().is_failure()


def test_get_capabilities_uses_detected_gpu_info() -> None:
    p = _mk_provider()
    p._gpu_info = rp.GPUInfo(
        name="A100",
        vram_total_mb=10000,
        vram_free_mb=9000,
        cuda_version="x",
        driver_version="x",
    )
    caps = p.get_capabilities()
    assert caps.gpu_name == "A100"
