from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import pytest

import ryotenkai_providers.runpod.training.provider as rp
from ryotenkai_shared.pipeline_context import RunContext
from ryotenkai_providers.runpod.models import PodSnapshot, SshEndpoint
from ryotenkai_providers.runpod.training.provider import RunPodProvider
from ryotenkai_providers.training.interfaces import ProviderStatus
from ryotenkai_shared.config import Secrets
from ryotenkai_shared.errors import ProviderUnavailableError
from ryotenkai_shared.utils.result import Err, Ok, ProviderError, Result

from tests._fakes.provider_context import attach_manifest_capabilities, make_provider_context

# Attach manifest-derived ClassVars once at module-import time so the
# tests don't depend on a prior ProviderRegistry instantiation having
# set them (which silently happens in the full lane but not in
# isolation).
attach_manifest_capabilities(
    RunPodProvider,
    provider_id="runpod",
    provider_name="runpod",
    provider_type="cloud",
)


_SSH_OK = SshEndpoint(host="1.2.3.4", port=2222)


def _ready_snapshot(*, ssh: SshEndpoint | None = _SSH_OK) -> PodSnapshot:
    return PodSnapshot(pod_id="pod-1", status="RUNNING", uptime_seconds=10, ssh_endpoint=ssh, port_count=1 if ssh else 0)


@dataclass
class StubAPI:
    """Raise-based stub: ``create_value`` returned, ``create_exc`` raised."""

    create_value: dict[str, Any] | None = field(
        default_factory=lambda: {"pod_id": "pod-1", "machine": "host-1"}
    )
    create_exc: BaseException | None = None

    def create_pod(self, config, *, pod_name: str | None = None):
        _ = config
        _ = pod_name
        if self.create_exc is not None:
            raise self.create_exc
        assert self.create_value is not None
        return self.create_value


@dataclass
class StubCleanup:
    """Raise-based stub: success when ``cleanup_pod`` returns None."""

    cleaned: list[str] = field(default_factory=list)

    def cleanup_pod(self, pod_id: str) -> None:
        self.cleaned.append(pod_id)
        return None


@dataclass
class StubLifecycle:
    """Raise-based stub: ``snapshot`` returned, ``exc`` raised."""

    snapshot: PodSnapshot | None = None
    exc: BaseException | None = None

    def wait_for_ready(self, pod_id: str, timeout: int = 300, max_retries: int = 3) -> PodSnapshot:
        if self.exc is not None:
            raise self.exc
        assert self.snapshot is not None
        return self.snapshot


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
        },
        "inference": {},
    }
    if cfg_overrides:
        cfg.update(cfg_overrides)
    secrets = Secrets(HF_TOKEN="hf_test", RUNPOD_API_KEY="rk")
    ctx = make_provider_context(provider_id="runpod", config=cfg, secrets=secrets)
    p = RunPodProvider(ctx)
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
    p._lifecycle = StubLifecycle(snapshot=_ready_snapshot())

    res = p.connect(run=_mk_run())
    assert res.is_success()
    ssh = res.unwrap()
    assert ssh.host == "1.2.3.4"
    assert ssh.port == 2222
    assert ssh.user == "root"
    assert ssh.key_path == __file__
    assert p.get_status() == ProviderStatus.CONNECTED


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
    p._lifecycle = StubLifecycle(snapshot=_ready_snapshot())

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
    p._lifecycle = StubLifecycle(snapshot=_ready_snapshot(ssh=None))

    res = p.connect(run=_mk_run())
    assert res.is_failure()
    assert cleanup.cleaned == ["pod-1"]
    assert "SSH endpoint is missing" in str(res.unwrap_err())


def test_provider_properties_and_repr() -> None:
    p = _mk_provider()
    # provider_name is manifest's [provider].name ("RunPod" in
    # provider.toml). In isolation the test-fixture
    # attach_manifest_capabilities() seeds it as "runpod"; when the
    # real ProviderRegistry loads the manifest (any test that touches
    # the registry) it overrides the ClassVar to "RunPod".  Accept
    # both case-insensitively so the test is stable under both
    # orderings.
    assert p.provider_name.lower() == "runpod"
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
    p._api_client = StubAPI(create_exc=ProviderUnavailableError(detail="no capacity", context={"code": "NO_CAP"}))
    p._cleanup_manager = StubCleanup()
    p._lifecycle = StubLifecycle(snapshot=_ready_snapshot())

    res = p.connect(run=_mk_run())
    assert res.is_failure()
    assert p.get_status() == ProviderStatus.ERROR
    cleanup: StubCleanup = p._cleanup_manager
    assert cleanup.cleaned == []


def test_connect_invalid_pod_info_and_missing_machine(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rp, "SSHClient", FakeSSHClient)

    p = _mk_provider()
    p._api_client = StubAPI(create_value={"x": 1})
    p._cleanup_manager = StubCleanup()
    p._lifecycle = StubLifecycle(snapshot=_ready_snapshot())
    assert p.connect(run=_mk_run()).is_failure()

    p = _mk_provider()
    p._api_client = StubAPI(create_value={"pod_id": "pod-1"})
    p._cleanup_manager = StubCleanup()
    p._lifecycle = StubLifecycle(snapshot=_ready_snapshot())
    assert p.connect(run=_mk_run()).is_success()

    p = _mk_provider()
    p._api_client = StubAPI(create_value={"pod_id": "pod-1"})
    cleanup = StubCleanup()
    p._cleanup_manager = cleanup
    p._lifecycle = StubLifecycle(snapshot=_ready_snapshot(ssh=None))
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
    p._lifecycle = StubLifecycle(snapshot=_ready_snapshot())

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
    p._lifecycle = StubLifecycle(snapshot=_ready_snapshot())

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
    # Non-recreatable error code → exits the recreate loop after one cleanup.
    p._lifecycle = StubLifecycle(exc=ProviderUnavailableError(detail="other", context={"code": "RUNPOD_OTHER"}))

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


# ---------------------------------------------------------------------------
# prepare_training_script_hooks — watchdog / auto-stop integration
# ---------------------------------------------------------------------------


class _RecordingSSH:
    """Minimal SSH client that records exec_command calls and returns success."""

    def __init__(self) -> None:
        self.commands: list[str] = []

    def exec_command(
        self,
        command: str,
        timeout: int = 30,
        silent: bool = False,
        background: bool = False,
    ):
        _ = timeout, silent, background
        self.commands.append(command)
        return True, "", ""


def test_prepare_hooks_disabled_when_auto_stop_off() -> None:
    p = _mk_provider(cfg_overrides={"cleanup": {"auto_stop_after_training": False}})
    p._pod_id = "pod-xyz"
    ssh = _RecordingSSH()

    result = p.prepare_training_script_hooks(ssh, context={"resource_id": "pod-xyz"})  # type: ignore[arg-type]

    assert result.is_success()
    hooks = result.unwrap()
    assert hooks.env_vars == {}
    # Phase 6.5: pre_python / post_python fields removed.
    assert ssh.commands == []  # nothing uploaded


def test_prepare_hooks_skipped_without_resource_id() -> None:
    p = _mk_provider(cfg_overrides={"cleanup": {"auto_stop_after_training": True}})
    p._pod_id = None
    ssh = _RecordingSSH()

    result = p.prepare_training_script_hooks(ssh, context={})  # type: ignore[arg-type]

    assert result.is_success()
    assert result.unwrap().env_vars == {}
    assert ssh.commands == []


def test_prepare_hooks_skipped_without_api_key() -> None:
    p = _mk_provider(cfg_overrides={"cleanup": {"auto_stop_after_training": True}})
    p._pod_id = "pod-xyz"
    p._api_key = ""  # simulate missing key
    ssh = _RecordingSSH()

    result = p.prepare_training_script_hooks(ssh, context={"resource_id": "pod-xyz"})  # type: ignore[arg-type]

    assert result.is_success()
    assert result.unwrap().env_vars == {}


def test_prepare_hooks_returns_runpod_env_vars() -> None:
    """Phase 6.5: hooks now only contribute env vars; the bash
    snippets and resource uploads are gone (replaced by IdleDetector
    + PodTerminator inside the in-pod runner).

    Phase 11.B: ``RUNPOD_AUTO_STOP`` env var removed (no toggle —
    PodTerminator's decision matrix runs unconditionally on terminal
    hooks). ``RUNPOD_KEEP_ON_ERROR`` retained for failed-run debug
    forensics (Phase 9.A carry-over).
    """
    p = _mk_provider(
        cfg_overrides={"cleanup": {"auto_stop_after_training": True, "keep_pod_on_error": True}}
    )
    p._pod_id = "pod-abc"
    p._api_key = "rk-secret"
    ssh = _RecordingSSH()

    result = p.prepare_training_script_hooks(ssh, context={"resource_id": "pod-abc"})  # type: ignore[arg-type]

    assert result.is_success()
    hooks = result.unwrap()

    # Env vars are forwarded to the trainer subprocess so the runner's
    # PodTerminator (Phase 11.B) can call the right GraphQL mutation
    # on terminal — podStop or podTerminate per the decision matrix.
    assert hooks.env_vars["RUNPOD_API_KEY"] == "rk-secret"
    assert hooks.env_vars["RUNPOD_POD_ID"] == "pod-abc"
    assert hooks.env_vars["RUNPOD_KEEP_ON_ERROR"] == "true"

    # Phase 11.B regression: ``RUNPOD_AUTO_STOP`` env is removed
    # entirely. No toggle disabling the PodTerminator decision matrix.
    assert "RUNPOD_AUTO_STOP" not in hooks.env_vars

    # Nothing uploaded over SSH any more — IdleDetector + PodTerminator
    # already live inside the runner image.
    assert ssh.commands == []


def test_prepare_hooks_keep_on_error_false_flag() -> None:
    p = _mk_provider(
        cfg_overrides={"cleanup": {"auto_stop_after_training": True, "keep_pod_on_error": False}}
    )
    p._pod_id = "pod-abc"
    p._api_key = "rk-secret"
    ssh = _RecordingSSH()

    hooks = p.prepare_training_script_hooks(ssh, context={"resource_id": "pod-abc"}).unwrap()  # type: ignore[arg-type]
    assert hooks.env_vars["RUNPOD_KEEP_ON_ERROR"] == "false"
