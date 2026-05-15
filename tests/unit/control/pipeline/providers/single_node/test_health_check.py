from __future__ import annotations

from dataclasses import dataclass

import pytest

from ryotenkai_providers.single_node.training.health_check import SingleNodeHealthCheck


@dataclass
class FakeSSH:
    """Minimal SSH stub for health checks."""

    script: dict[str, tuple[bool, str, str]]
    calls: list[str]

    def exec_command(self, command: str, timeout: int = 30, silent: bool = False):
        self.calls.append(command)
        return self.script.get(command, (False, "", "unknown command"))


def test_check_gpu_success_parses_nvidia_smi() -> None:
    script = {
        "nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv,noheader,nounits": (
            True,
            "NVIDIA A40, 46080, 45000, 550.66\n",
            "",
        ),
        "nvcc --version 2>/dev/null | grep -oP 'release \\K[0-9.]+'": (True, "12.1\n", ""),
    }
    ssh = FakeSSH(script=script, calls=[])
    hc = SingleNodeHealthCheck(ssh)

    res = hc.check_gpu()
    # (Phase A2 Batch 12: success implies no raise)
    gpu = res
    assert gpu.name == "NVIDIA A40"
    assert gpu.vram_total_mb == 46080
    assert gpu.vram_free_mb == 45000
    assert gpu.driver_version == "550.66"
    assert gpu.cuda_version == "12.1"


def test_check_gpu_invalid_format() -> None:
    from ryotenkai_shared.errors import ProviderUnavailableError

    script = {
        "nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv,noheader,nounits": (
            True,
            "broken\n",
            "",
        ),
        "nvcc --version 2>/dev/null | grep -oP 'release \\K[0-9.]+'": (True, "12.1\n", ""),
    }
    ssh = FakeSSH(script=script, calls=[])
    hc = SingleNodeHealthCheck(ssh)

    with pytest.raises(ProviderUnavailableError) as exc_info:
        hc.check_gpu()
    assert "Unexpected" in str(exc_info.value.detail or exc_info.value)


def test_check_docker_failure_when_missing() -> None:
    from ryotenkai_shared.errors import ProviderUnavailableError

    # docker --version fails
    script = {
        "docker --version": (False, "", "not found"),
    }
    ssh = FakeSSH(script=script, calls=[])
    hc = SingleNodeHealthCheck(ssh)
    with pytest.raises(ProviderUnavailableError):
        hc.check_docker()


def test_check_docker_success_with_gpu_support() -> None:
    script = {
        "docker --version": (True, "Docker version 24.0.7, build afdd53b\n", ""),
        "docker context use default 2>/dev/null": (True, "", ""),
        "docker info > /dev/null 2>&1 && echo 'running'": (True, "running\n", ""),
        "docker info 2>/dev/null | grep -i nvidia": (True, "nvidia\n", ""),
    }
    ssh = FakeSSH(script=script, calls=[])
    hc = SingleNodeHealthCheck(ssh)
    # (Phase A2 Batch 12: success implies no raise)
    assert hc.check_docker() == "24.0.7"


def test_check_disk_space_falls_back_to_parent_when_path_missing() -> None:
    path = "/home/user/workspace/run_1"
    parent = "/home/user/workspace"
    script = {
        f'test -d "{path}" && echo "exists" || echo "not_exists"': (True, "not_exists\n", ""),
        f'df -BG "{parent}" --output=avail 2>/dev/null | tail -1 | tr -d " G"': (True, "50\n", ""),
    }
    ssh = FakeSSH(script=script, calls=[])
    hc = SingleNodeHealthCheck(ssh)
    # (Phase A2 Batch 12: success implies no raise)
    assert hc.check_disk_space(path) == 50.0


def test_run_all_checks_fails_when_docker_root_dir_low_space() -> None:
    gpu_cmd = "nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv,noheader,nounits"
    nvcc_cmd = "nvcc --version 2>/dev/null | grep -oP 'release \\K[0-9.]+'"
    ws_exists = 'test -d "/workspace" && echo "exists" || echo "not_exists"'
    ws_df = 'df -BG "/workspace" --output=avail 2>/dev/null | tail -1 | tr -d " G"'
    docker_root = 'docker info --format "{{.DockerRootDir}}" 2>/dev/null'
    docker_root_exists = 'test -d "/var/lib/docker" && echo "exists" || echo "not_exists"'
    docker_root_df = 'df -BG "/var/lib/docker" --output=avail 2>/dev/null | tail -1 | tr -d " G"'
    script = {
        gpu_cmd: (True, "GPU, 100, 90, 1\n", ""),
        nvcc_cmd: (True, "12.1\n", ""),
        "docker --version": (True, "Docker version 24.0.7, build afdd53b\n", ""),
        "docker context use default 2>/dev/null": (True, "", ""),
        "docker info > /dev/null 2>&1 && echo 'running'": (True, "running\n", ""),
        "docker info 2>/dev/null | grep -i nvidia": (True, "nvidia\n", ""),
        ws_exists: (True, "exists\n", ""),
        ws_df: (True, "100\n", ""),
        docker_root: (True, "/var/lib/docker\n", ""),
        docker_root_exists: (True, "exists\n", ""),
        docker_root_df: (True, "5\n", ""),  # below MIN_DOCKER_DISK_FREE_GB (30)
    }
    ssh = FakeSSH(script=script, calls=[])
    hc = SingleNodeHealthCheck(ssh)
    result = hc.run_all_checks(workspace_path="/workspace")
    assert result.passed is False
    assert result.errors is not None
    assert any("DockerRootDir" in e for e in result.errors)


def test_run_all_checks_turns_failures_into_errors() -> None:
    from ryotenkai_shared.errors import ProviderUnavailableError

    hc = SingleNodeHealthCheck(FakeSSH(script={}, calls=[]))
    # Patch individual checks via monkeypatching methods — they raise
    # (Phase A2 Batch 12 contract) and ``run_all_checks`` translates
    # exceptions into entries in ``result.errors``.
    def _raise_gpu():
        raise ProviderUnavailableError(detail="gpu", context={})
    def _raise_docker():
        raise ProviderUnavailableError(detail="docker", context={})
    def _raise_disk(*a, **k):
        raise ProviderUnavailableError(detail="disk", context={})
    hc.check_gpu = _raise_gpu  # type: ignore[method-assign]
    hc.check_docker = _raise_docker  # type: ignore[method-assign]
    hc.check_disk_space = _raise_disk  # type: ignore[method-assign]

    result = hc.run_all_checks(workspace_path="/")
    assert result.passed is False
    assert result.errors is not None
