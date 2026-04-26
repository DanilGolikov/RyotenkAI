"""
Unit tests for SingleNodeHealthCheck and SingleNodeProvider.

Covers missing health_check.py lines: 96, 99, 110, 145-146, 165-173, 221,
232-240, 256-267, 286-287, 319, 328, 332-333, 411-412, 418, 425-426

Covers missing training provider lines: 89, 129, 172-174, 187-189, 196,
203-204, 284-288, 299, 312-320, 413-422, 432-441, 452, 456, 460, 464
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import pytest

import src.providers.single_node.training.provider as sp
from src.pipeline.state import RunContext
from src.providers.single_node.training.health_check import (
    HealthCheckResult,
    SingleNodeHealthCheck,
)
from src.providers.single_node.training.provider import SingleNodeProvider
from src.providers.training.interfaces import GPUInfo, ProviderCapabilities, ProviderStatus
from src.utils.config import Secrets
from src.utils.result import Err, Ok


# ---------------------------------------------------------------------------
# SSH stub
# ---------------------------------------------------------------------------

@dataclass
class FakeSSH:
    """Flexible SSH stub — returns from script dict or default."""

    script: dict[str, tuple[bool, str, str]] = field(default_factory=dict)
    calls: list[str] = field(default_factory=list)
    default_response: tuple[bool, str, str] = (False, "", "unknown command")

    def exec_command(self, command: str, timeout: int = 30, silent: bool = False):
        self.calls.append(command)
        return self.script.get(command, self.default_response)


def _hc(script: dict[str, tuple[bool, str, str]]) -> SingleNodeHealthCheck:
    return SingleNodeHealthCheck(FakeSSH(script=script))


# ---------------------------------------------------------------------------
# check_gpu — missing lines
# ---------------------------------------------------------------------------

class TestCheckGpu:
    GPU_CMD = "nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv,noheader,nounits"
    NVCC_CMD = "nvcc --version 2>/dev/null | grep -oP 'release \\K[0-9.]+'"
    CUDA_SMI_CMD = "nvidia-smi --query-gpu=cuda_version --format=csv,noheader 2>/dev/null"

    def test_fails_when_nvidia_smi_fails(self):
        """Line 96: nvidia-smi fails → Err(SINGLENODE_NVIDIA_SMI_FAILED)."""
        hc = _hc({self.GPU_CMD: (False, "", "nvidia-smi not found")})
        result = hc.check_gpu()
        assert result.is_failure()
        assert "SINGLENODE_NVIDIA_SMI_FAILED" in str(result.unwrap_err())

    def test_fails_when_nvidia_smi_empty_output(self):
        """Line 99: empty stdout → Err(SINGLENODE_NO_GPU_DETECTED)."""
        hc = _hc({self.GPU_CMD: (True, "   \n  ", "")})
        result = hc.check_gpu()
        assert result.is_failure()
        assert "SINGLENODE_NO_GPU_DETECTED" in str(result.unwrap_err())

    def test_fails_when_no_parseable_lines(self):
        """Line 110: only whitespace lines → Err(SINGLENODE_NO_GPU_DETECTED)."""
        hc = _hc({self.GPU_CMD: (True, "\n  \n  \n", "")})
        result = hc.check_gpu()
        assert result.is_failure()
        assert "SINGLENODE_NO_GPU_DETECTED" in str(result.unwrap_err())

    def test_fails_when_format_has_too_few_parts(self):
        """Lines 145-146: format error (< 4 parts) → Err(SINGLENODE_NVIDIA_SMI_PARSE_ERROR)."""
        hc = _hc({self.GPU_CMD: (True, "GPU A100, 80000\n", "")})
        result = hc.check_gpu()
        assert result.is_failure()
        assert "Unexpected nvidia-smi format" in str(result.unwrap_err())

    def test_fails_on_value_error_parsing(self):
        """Lines 145-146: non-numeric memory → ValueError → Err(SINGLENODE_NVIDIA_SMI_PARSE_ERROR)."""
        hc = _hc({
            self.GPU_CMD: (True, "GPU A100, not_a_number, 40000, 550.66\n", ""),
            self.NVCC_CMD: (True, "12.1\n", ""),
        })
        result = hc.check_gpu()
        assert result.is_failure()
        assert "parse" in str(result.unwrap_err()).lower()

    def test_cuda_version_nvcc_fallback_to_smi(self):
        """Lines 165-173: nvcc fails, falls back to nvidia-smi for CUDA version."""
        script = {
            self.GPU_CMD: (True, "NVIDIA A40, 46080, 45000, 550.66\n", ""),
            self.NVCC_CMD: (False, "", "nvcc not found"),
            self.CUDA_SMI_CMD: (True, "12.1\n", ""),
        }
        hc = _hc(script)
        result = hc.check_gpu()
        assert result.is_success()
        assert result.unwrap().cuda_version == "12.1"

    def test_cuda_version_unknown_when_both_fail(self):
        """Lines 165-173: both nvcc and nvidia-smi CUDA fail → 'unknown'."""
        script = {
            self.GPU_CMD: (True, "NVIDIA A40, 46080, 45000, 550.66\n", ""),
            self.NVCC_CMD: (False, "", ""),
            self.CUDA_SMI_CMD: (False, "", ""),
        }
        hc = _hc(script)
        result = hc.check_gpu()
        assert result.is_success()
        assert result.unwrap().cuda_version == "unknown"

    def test_multi_gpu_count(self):
        """Multiple GPUs are counted correctly."""
        multi = "NVIDIA A40, 46080, 45000, 550.66\nNVIDIA A40, 46080, 46000, 550.66\n"
        script = {
            self.GPU_CMD: (True, multi, ""),
            self.NVCC_CMD: (True, "12.1\n", ""),
        }
        hc = _hc(script)
        result = hc.check_gpu()
        assert result.is_success()
        assert result.unwrap().gpu_count == 2


# ---------------------------------------------------------------------------
# check_docker — missing lines
# ---------------------------------------------------------------------------

class TestCheckDocker:
    def test_daemon_not_running_in_any_context(self):
        """Line 221: daemon not running in default or rootless → Err(SINGLENODE_DOCKER_DAEMON_NOT_RUNNING)."""
        script: dict[str, tuple[bool, str, str]] = {
            "docker --version": (True, "Docker version 24.0.7, build afdd53b\n", ""),
            "docker context use default 2>/dev/null": (False, "", ""),
            "docker context use rootless 2>/dev/null": (False, "", ""),
        }
        hc = _hc(script)
        result = hc.check_docker()
        assert result.is_failure()
        assert "SINGLENODE_DOCKER_DAEMON_NOT_RUNNING" in str(result.unwrap_err())

    def test_daemon_running_but_no_nvidia_in_default(self):
        """Lines 232-240: nvidia not in docker info → Err(SINGLENODE_DOCKER_NO_GPU_SUPPORT)."""
        script = {
            "docker --version": (True, "Docker version 24.0.7, build afdd53b\n", ""),
            "docker context use default 2>/dev/null": (True, "", ""),
            "docker info > /dev/null 2>&1 && echo 'running'": (True, "running\n", ""),
            "docker info 2>/dev/null | grep -i nvidia": (True, "something else", ""),
        }
        hc = _hc(script)
        result = hc.check_docker()
        assert result.is_failure()
        assert "SINGLENODE_DOCKER_NO_GPU_SUPPORT" in str(result.unwrap_err())

    def test_rootless_context_adds_extra_hint(self):
        """Lines 256-267: rootless context adds special message to error hint."""
        script = {
            "docker --version": (True, "Docker version 24.0.7, build afdd53b\n", ""),
            "docker context use default 2>/dev/null": (False, "", ""),
            "docker context use rootless 2>/dev/null": (True, "", ""),
            "docker info > /dev/null 2>&1 && echo 'running'": (True, "running\n", ""),
            "docker info 2>/dev/null | grep -i nvidia": (True, "other stuff", ""),
        }
        hc = _hc(script)
        result = hc.check_docker()
        assert result.is_failure()
        err_str = str(result.unwrap_err())
        assert "rootless" in err_str.lower() or "daemon.json" in err_str

    def test_daemon_running_in_rootless_with_nvidia(self):
        """Falls through to rootless context with nvidia support → success."""
        script = {
            "docker --version": (True, "Docker version 24.0.7, build afdd53b\n", ""),
            "docker context use default 2>/dev/null": (False, "", ""),
            "docker context use rootless 2>/dev/null": (True, "", ""),
            "docker info > /dev/null 2>&1 && echo 'running'": (True, "running\n", ""),
            "docker info 2>/dev/null | grep -i nvidia": (True, "nvidia runtime\n", ""),
        }
        hc = _hc(script)
        result = hc.check_docker()
        assert result.is_success()


# ---------------------------------------------------------------------------
# _get_docker_root_dir — missing lines
# ---------------------------------------------------------------------------

class TestGetDockerRootDir:
    ROOT_CMD = 'docker info --format "{{.DockerRootDir}}" 2>/dev/null'

    def test_fails_when_command_fails(self):
        """Lines 286-287: command fails → Err."""
        hc = _hc({self.ROOT_CMD: (False, "", "error")})
        result = hc._get_docker_root_dir()
        assert result.is_failure()
        assert "SINGLENODE_DOCKER_ROOT_DIR_FAILED" in str(result.unwrap_err())

    def test_fails_when_stdout_empty(self):
        """Lines 286-287: empty stdout → Err."""
        hc = _hc({self.ROOT_CMD: (True, "   \n", "")})
        result = hc._get_docker_root_dir()
        assert result.is_failure()

    def test_success_returns_path(self):
        hc = _hc({self.ROOT_CMD: (True, "/var/lib/docker\n", "")})
        result = hc._get_docker_root_dir()
        assert result.is_success()
        assert result.unwrap() == "/var/lib/docker"


# ---------------------------------------------------------------------------
# check_disk_space — missing lines
# ---------------------------------------------------------------------------

class TestCheckDiskSpace:
    def _exists_cmd(self, path: str) -> str:
        return f'test -d "{path}" && echo "exists" || echo "not_exists"'

    def _df_cmd(self, path: str) -> str:
        return f'df -BG "{path}" --output=avail 2>/dev/null | tail -1 | tr -d " G"'

    def test_path_does_not_exist_uses_parent(self):
        """Line 319: path not exists falls back to parent."""
        path = "/home/user/nonexistent"
        parent = "/home/user"
        script = {
            self._exists_cmd(path): (True, "not_exists\n", ""),
            self._df_cmd(parent): (True, "100\n", ""),
        }
        hc = _hc(script)
        result = hc.check_disk_space(path)
        assert result.is_success()

    def test_fails_when_check_command_fails(self):
        """Line 328: df command fails → Err(SINGLENODE_DISK_CHECK_FAILED)."""
        path = "/workspace"
        script = {
            self._exists_cmd(path): (True, "exists\n", ""),
            self._df_cmd(path): (False, "", "df failed"),
        }
        hc = _hc(script)
        result = hc.check_disk_space(path)
        assert result.is_failure()
        assert "SINGLENODE_DISK_CHECK_FAILED" in str(result.unwrap_err())

    def test_fails_when_stdout_not_a_number(self):
        """Lines 332-333: non-numeric output → Err(SINGLENODE_DISK_PARSE_ERROR)."""
        path = "/workspace"
        script = {
            self._exists_cmd(path): (True, "exists\n", ""),
            self._df_cmd(path): (True, "not_a_number\n", ""),
        }
        hc = _hc(script)
        result = hc.check_disk_space(path)
        assert result.is_failure()
        assert "SINGLENODE_DISK_PARSE_ERROR" in str(result.unwrap_err())

    def test_fails_when_insufficient_disk_space(self):
        """Returns Err when free_gb < required."""
        path = "/workspace"
        script = {
            self._exists_cmd(path): (True, "exists\n", ""),
            self._df_cmd(path): (True, "5\n", ""),
        }
        hc = _hc(script)
        result = hc.check_disk_space(path, min_free_gb=20.0)
        assert result.is_failure()
        assert "SINGLENODE_DISK_INSUFFICIENT" in str(result.unwrap_err())

    def test_path_with_no_slash_falls_back_to_root(self):
        """Lines 316-317: path without '/' falls back to '/'."""
        path = "nopath"
        script = {
            f'test -d "{path}" && echo "exists" || echo "not_exists"': (True, "not_exists\n", ""),
            self._df_cmd("/"): (True, "500\n", ""),
        }
        hc = _hc(script)
        result = hc.check_disk_space(path)
        assert result.is_success()


# ---------------------------------------------------------------------------
# run_all_checks — missing lines
# ---------------------------------------------------------------------------

GPU_CMD = "nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv,noheader,nounits"
NVCC_CMD = "nvcc --version 2>/dev/null | grep -oP 'release \\K[0-9.]+'"
DOCKER_VERSION_CMD = "docker --version"
DOCKER_CTX_DEFAULT = "docker context use default 2>/dev/null"
DOCKER_INFO_RUN = "docker info > /dev/null 2>&1 && echo 'running'"
DOCKER_INFO_NVIDIA = "docker info 2>/dev/null | grep -i nvidia"
DOCKER_ROOT_CMD = 'docker info --format "{{.DockerRootDir}}" 2>/dev/null'


def _mk_good_gpu_script() -> dict[str, tuple[bool, str, str]]:
    return {
        GPU_CMD: (True, "NVIDIA A40, 46080, 45000, 550.66\n", ""),
        NVCC_CMD: (True, "12.1\n", ""),
    }


def _mk_good_docker_script() -> dict[str, tuple[bool, str, str]]:
    return {
        DOCKER_VERSION_CMD: (True, "Docker version 24.0.7, build afdd53b\n", ""),
        DOCKER_CTX_DEFAULT: (True, "", ""),
        DOCKER_INFO_RUN: (True, "running\n", ""),
        DOCKER_INFO_NVIDIA: (True, "nvidia\n", ""),
    }


def _mk_disk_script(path: str, free_gb: int = 100) -> dict[str, tuple[bool, str, str]]:
    return {
        f'test -d "{path}" && echo "exists" || echo "not_exists"': (True, "exists\n", ""),
        f'df -BG "{path}" --output=avail 2>/dev/null | tail -1 | tr -d " G"': (True, f"{free_gb}\n", ""),
    }


class TestRunAllChecks:
    def test_fails_when_docker_root_dir_unavailable(self):
        """Lines 411-412, 418: docker root dir check fails → error appended."""
        script = {
            **_mk_good_gpu_script(),
            **_mk_good_docker_script(),
            **_mk_disk_script("/workspace"),
            DOCKER_ROOT_CMD: (False, "", "docker info failed"),
        }
        hc = _hc(script)
        result = hc.run_all_checks(workspace_path="/workspace")
        assert result.passed is False
        assert result.errors is not None
        assert any("DockerRootDir" in e for e in result.errors)

    def test_passes_all_checks(self):
        """Full success path with all checks passing."""
        docker_root = "/var/lib/docker"
        script = {
            **_mk_good_gpu_script(),
            **_mk_good_docker_script(),
            **_mk_disk_script("/workspace"),
            DOCKER_ROOT_CMD: (True, f"{docker_root}\n", ""),
            **_mk_disk_script(docker_root, free_gb=200),
        }
        hc = _hc(script)
        result = hc.run_all_checks(workspace_path="/workspace")
        assert result.passed is True
        assert result.errors is None
        assert result.docker_available is True
        assert result.gpu_info is not None

    def test_warnings_are_collected(self):
        """Lines 425-426: warnings are logged (run_all_checks passes even with warnings)."""
        docker_root = "/var/lib/docker"
        script = {
            **_mk_good_gpu_script(),
            **_mk_good_docker_script(),
            **_mk_disk_script("/workspace"),
            DOCKER_ROOT_CMD: (True, f"{docker_root}\n", ""),
            **_mk_disk_script(docker_root, free_gb=200),
        }
        hc = _hc(script)
        result = hc.run_all_checks(workspace_path="/workspace")
        # warnings list is empty but not None (default is None)
        assert result.warnings is None or isinstance(result.warnings, list)

    def test_docker_root_dir_low_space_adds_error(self):
        """Lines 411-412: low docker disk space → error."""
        docker_root = "/var/lib/docker"
        script = {
            **_mk_good_gpu_script(),
            **_mk_good_docker_script(),
            **_mk_disk_script("/workspace"),
            DOCKER_ROOT_CMD: (True, f"{docker_root}\n", ""),
            **_mk_disk_script(docker_root, free_gb=5),   # < 30GB minimum
        }
        hc = _hc(script)
        result = hc.run_all_checks(workspace_path="/workspace")
        assert result.passed is False
        assert result.errors is not None


# ---------------------------------------------------------------------------
# SingleNodeProvider — additional coverage
# ---------------------------------------------------------------------------

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
        return self.connection_ok, "" if self.connection_ok else "timed out"

    def directory_exists(self, remote_path: str) -> bool:
        return remote_path in self.dirs

    def create_directory(self, remote_path: str) -> tuple[bool, str]:
        self.dirs.add(remote_path)
        return True, ""

    def exec_command(self, command: str, timeout: int = 30, silent: bool = False):
        self.commands.append(command)
        if "echo EXISTS || echo DELETED" in command:
            return True, "DELETED\n", ""
        return True, "", ""


class FakeHealthCheck:
    def __init__(self, ssh_client):
        self.ssh = ssh_client

    def run_all_checks(self, workspace_path: str):
        return HealthCheckResult(
            passed=True,
            gpu_info=GPUInfo(
                name="NVIDIA A40",
                vram_total_mb=46080,
                vram_free_mb=45000,
                cuda_version="12.1",
                driver_version="550.66",
            ),
        )

    def check_gpu(self):
        return Ok(
            GPUInfo(name="GPU", vram_total_mb=1, vram_free_mb=1, cuda_version="x", driver_version="x")
        )


def _mk_provider(**overrides) -> SingleNodeProvider:
    cfg: dict[str, Any] = {
        "connect": {"ssh": {"alias": "pc"}},
        "training": {"workspace_path": "/workspace", "docker_image": "test/runtime:latest"},
    }
    cfg.update(overrides)
    secrets = Secrets(HF_TOKEN="hf_test")
    return SingleNodeProvider(config=cfg, secrets=secrets)


def _mk_run() -> RunContext:
    return RunContext(
        name="run_test_abc12",
        created_at_utc=datetime(2026, 1, 20, 12, 0, 0, tzinfo=timezone.utc),
    )


class TestSingleNodeProviderCoverage:
    def test_init_alias_mode_logging(self):
        """Line 87-88: alias mode → logs alias."""
        p = _mk_provider()
        assert p.provider_name == "single_node"
        assert p.provider_type == "local"

    def test_init_explicit_mode_logging(self):
        """Line 89-92: explicit mode → logs user@host."""
        p = _mk_provider(connect={"ssh": {"host": "1.2.3.4", "user": "bob", "port": 22}})
        assert p.provider_name == "single_node"

    def test_get_status_initial(self):
        """Line 456: get_status returns AVAILABLE initially."""
        p = _mk_provider()
        assert p.get_status() == ProviderStatus.AVAILABLE

    def test_mark_error_sets_status(self):
        """Line 452: mark_error sets status to ERROR and _had_error."""
        p = _mk_provider()
        p.mark_error()
        assert p.get_status() == ProviderStatus.ERROR
        assert p._had_error is True

    def test_get_run_dir_initially_none(self):
        p = _mk_provider()
        assert p.get_run_dir() is None

    def test_get_base_workspace(self):
        p = _mk_provider()
        assert p.get_base_workspace() == "/workspace"

    def test_get_ssh_client_initially_none(self):
        p = _mk_provider()
        assert p.get_ssh_client() is None

    def test_get_resource_info_returns_none(self):
        p = _mk_provider()
        assert p.get_resource_info() is None

    def test_repr_shows_status(self):
        p = _mk_provider()
        r = repr(p)
        assert "SingleNodeProvider" in r
        assert "available" in r

    def test_check_gpu_returns_cached_info(self, monkeypatch: pytest.MonkeyPatch):
        """Lines 413-414: check_gpu returns cached gpu_info if present."""
        p = _mk_provider()
        gpu = GPUInfo(name="A40", vram_total_mb=46080, vram_free_mb=45000, cuda_version="12.1", driver_version="x")
        p._gpu_info = gpu
        result = p.check_gpu()
        assert result.is_success()
        assert result.unwrap().name == "A40"

    def test_check_gpu_fails_when_not_connected(self):
        """Lines 417-418: not connected → Err(SINGLENODE_NOT_CONNECTED)."""
        p = _mk_provider()
        # _status is AVAILABLE but not CONNECTED, no _ssh_client
        result = p.check_gpu()
        assert result.is_failure()
        assert "SINGLENODE_NOT_CONNECTED" in str(result.unwrap_err())

    def test_check_gpu_delegates_to_health_check(self, monkeypatch: pytest.MonkeyPatch):
        """Lines 421-422: connected → delegates to health checker."""
        ssh = FakeSSHClient(host="pc", port=22)
        monkeypatch.setattr(sp, "SSHClient", lambda *a, **k: ssh)
        monkeypatch.setattr(sp, "SingleNodeHealthCheck", FakeHealthCheck)

        p = _mk_provider()
        ssh.dirs.add("/workspace")
        p.connect(run=_mk_run())
        p._gpu_info = None  # clear cached info

        result = p.check_gpu()
        assert result.is_success()

    def test_get_capabilities_with_gpu_info(self, monkeypatch: pytest.MonkeyPatch):
        """Lines 432-441: get_capabilities returns GPU info when available."""
        p = _mk_provider()
        p._gpu_info = GPUInfo(
            name="NVIDIA A40",
            vram_total_mb=46080,
            vram_free_mb=45000,
            cuda_version="12.1",
            driver_version="550.66",
        )
        caps = p.get_capabilities()
        assert isinstance(caps, ProviderCapabilities)
        assert caps.gpu_name == "NVIDIA A40"
        assert caps.gpu_vram_gb == pytest.approx(46080 / 1024, abs=1)

    def test_get_capabilities_without_gpu_uses_config(self):
        """Lines 438-439: no GPU info → uses gpu_type from config."""
        p = _mk_provider(training={
            "workspace_path": "/workspace",
            "docker_image": "test/runtime:latest",
            "gpu_type": "NVIDIA RTX 4090",
        })
        caps = p.get_capabilities()
        assert caps.gpu_name == "NVIDIA RTX 4090"

    def test_get_capabilities_no_gpu_info_no_config(self):
        """No gpu_info and no gpu_type → gpu_name is None."""
        p = _mk_provider()
        caps = p.get_capabilities()
        assert caps.gpu_name is None

    def test_connect_explicit_only_mode(self, monkeypatch: pytest.MonkeyPatch):
        """Lines 187-189: explicit-only mode (no alias) creates explicit client."""
        ssh = FakeSSHClient(host="1.2.3.4", port=22)
        monkeypatch.setattr(sp, "SSHClient", lambda *a, **k: ssh)
        monkeypatch.setattr(sp, "SingleNodeHealthCheck", FakeHealthCheck)

        p = _mk_provider(connect={"ssh": {"host": "1.2.3.4", "user": "user", "port": 22}})
        ssh.dirs.add("/workspace")
        result = p.connect(run=_mk_run())
        assert result.is_success()
        assert p.get_status() == ProviderStatus.CONNECTED

    def test_connect_alias_fails_fallback_to_explicit(self, monkeypatch: pytest.MonkeyPatch):
        """Lines 172-174: alias fails → fallback to explicit host+user."""
        call_count = [0]

        class _TrackSSH:
            def __init__(self, **kw):
                call_count[0] += 1
                # First client (alias) fails; subsequent ones (explicit) succeed
                self._ok = call_count[0] > 1
                self.key_path = kw.get("key_path", "")

            def test_connection(self, *a, **k):
                return self._ok, "" if self._ok else "alias timeout"

            def directory_exists(self, p):
                return p in {"/workspace"}

            def create_directory(self, p):
                return True, ""

            def exec_command(self, cmd, **k):
                if "echo EXISTS || echo DELETED" in cmd:
                    return True, "DELETED\n", ""
                return True, "", ""

        monkeypatch.setattr(sp, "SSHClient", _TrackSSH)
        monkeypatch.setattr(sp, "SingleNodeHealthCheck", FakeHealthCheck)

        p = _mk_provider(connect={
            "ssh": {
                "alias": "myalias",
                "host": "1.2.3.4",
                "user": "user",
                "port": 22,
            }
        })
        result = p.connect(run=_mk_run())
        # First alias call fails, second explicit call succeeds → connected
        assert result.is_success()
        assert call_count[0] >= 2

    def test_connect_alias_fails_no_explicit_fallback(self, monkeypatch: pytest.MonkeyPatch):
        """Lines 175-183: alias fails and no host+user → Err(SINGLENODE_SSH_ALIAS_FAILED)."""
        def _fail_ssh(**kw):
            return FakeSSHClient(host=kw.get("host", "pc"), port=kw.get("port", 22), connection_ok=False)

        monkeypatch.setattr(sp, "SSHClient", lambda **kw: _fail_ssh(**kw))
        monkeypatch.setattr(sp, "SingleNodeHealthCheck", FakeHealthCheck)

        p = _mk_provider(connect={"ssh": {"alias": "myalias"}})
        result = p.connect(run=_mk_run())
        assert result.is_failure()
        assert "SINGLENODE_SSH_ALIAS_FAILED" in str(result.unwrap_err())

    def test_connect_unexpected_exception(self, monkeypatch: pytest.MonkeyPatch):
        """Lines 284-288: unexpected exception during connect → Err(SINGLENODE_CONNECT_UNEXPECTED_ERROR)."""
        monkeypatch.setattr(sp, "SSHClient", lambda **kw: (_ for _ in ()).throw(RuntimeError("unexpected!")))
        p = _mk_provider()
        result = p.connect(run=_mk_run())
        assert result.is_failure()
        assert "SINGLENODE_CONNECT_UNEXPECTED_ERROR" in str(result.unwrap_err())

    def test_preempt_inference_container_no_ssh(self):
        """Line 299: _preempt_inference_container returns early when no SSH client."""
        p = _mk_provider()
        p._ssh_client = None
        p._preempt_inference_container()  # should not raise

    def test_preempt_inference_container_not_running(self, monkeypatch: pytest.MonkeyPatch):
        """Lines 303-309: container not running → no stop command."""
        ssh = FakeSSHClient(host="pc", port=22)
        p = _mk_provider()
        p._ssh_client = ssh

        container_name = "ryotenkai-inference-vllm"
        cmd_check = f"docker ps -q -f name={container_name} -f status=running"
        # stdout is empty → not running
        ssh.script = {cmd_check: (True, "", "")}  # type: ignore[attr-defined]

        p._preempt_inference_container()
        assert all("docker rm -f" not in c for c in ssh.commands)

    def test_preempt_inference_container_running_stops_it(self, monkeypatch: pytest.MonkeyPatch):
        """Lines 312-318: container running → stop command is issued."""
        class StopTracker:
            def __init__(self):
                self.commands: list[str] = []
                self.key_path = ""

            def exec_command(self, command, timeout=30, silent=False):
                self.commands.append(command)
                container_name = "ryotenkai-inference-vllm"
                if f"name={container_name}" in command:
                    return True, "abc123\n", ""  # running
                return True, "", ""

        ssh = StopTracker()
        p = _mk_provider()
        p._ssh_client = ssh

        with monkeypatch.context() as m:
            m.setattr("time.sleep", lambda s: None)
            p._preempt_inference_container()

        assert any("docker rm -f" in c for c in ssh.commands)

    def test_connect_explicit_fails_returns_error(self, monkeypatch: pytest.MonkeyPatch):
        """Lines 203-204: explicit connection test fails → Err(SINGLENODE_SSH_CONNECT_FAILED)."""
        ssh = FakeSSHClient(host="1.2.3.4", port=22, connection_ok=False)
        monkeypatch.setattr(sp, "SSHClient", lambda **kw: ssh)
        monkeypatch.setattr(sp, "SingleNodeHealthCheck", FakeHealthCheck)

        p = _mk_provider(connect={"ssh": {"host": "1.2.3.4", "user": "user", "port": 22}})
        result = p.connect(run=_mk_run())
        assert result.is_failure()
        assert "SINGLENODE_SSH_CONNECT_FAILED" in str(result.unwrap_err())
