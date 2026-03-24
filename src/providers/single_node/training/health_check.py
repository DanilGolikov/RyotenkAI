"""
SingleNode Health Check - validates server is ready for training.

Checks:
    - GPU availability (nvidia-smi)
    - Docker availability (daemon + NVIDIA runtime)
    - Disk space (workspace path + DockerRootDir)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.providers.training.interfaces import GPUInfo
from src.utils.result import Err, Ok, ProviderError, Result

if TYPE_CHECKING:
    from src.utils.ssh_client import SSHClient

logger = logging.getLogger("ryotenkai")

_HEALTH_CMD_TIMEOUT = 15
_HEALTH_STDERR_TRUNCATE = 200


@dataclass
class HealthCheckResult:
    """Result of health check."""

    passed: bool
    gpu_info: GPUInfo | None = None
    docker_version: str | None = None
    docker_root_dir: str | None = None
    disk_free_gb: float | None = None
    docker_disk_free_gb: float | None = None
    docker_available: bool = False
    errors: list[str] | None = None
    warnings: list[str] | None = None


class SingleNodeHealthCheck:
    """
    Health checker for SingleNode (local PC) provider.

    Performs pre-training validation:
        - GPU detection via nvidia-smi
        - Docker availability (daemon + nvidia runtime)
        - Disk space checks (workspace + DockerRootDir)

    Example:
        checker = SingleNodeHealthCheck(ssh_client)
        result = checker.run_all_checks()

        if not result.passed:
            for error in result.errors:
                logger.error(f"Health check failed: {error}")
    """

    # Minimum requirements
    MIN_DISK_FREE_GB = 20  # Minimum free disk space
    MIN_DOCKER_DISK_FREE_GB = 30  # Docker images can be large (layers pulled into DockerRootDir)

    def __init__(self, ssh_client: SSHClient):
        """
        Initialize health checker.

        Args:
            ssh_client: Connected SSH client
        """
        self.ssh = ssh_client

    def check_gpu(self) -> Result[GPUInfo, ProviderError]:
        """
        Check GPU availability via nvidia-smi.

        Parses nvidia-smi output to get:
            - GPU name
            - VRAM total/free
            - CUDA version
            - Driver version

        Returns:
            Ok(GPUInfo): GPU information
            Err(ProviderError): Error
        """
        logger.debug("🔍 Checking GPU via nvidia-smi...")

        # Run nvidia-smi with query format
        cmd = "nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv,noheader,nounits"
        success, stdout, stderr = self.ssh.exec_command(cmd, timeout=_HEALTH_CMD_TIMEOUT, silent=True)

        if not success:
            return Err(ProviderError(message=f"nvidia-smi failed: {stderr}", code="SINGLENODE_NVIDIA_SMI_FAILED"))

        if not stdout.strip():
            return Err(
                ProviderError(
                    message="nvidia-smi returned empty output - no GPU detected",
                    code="SINGLENODE_NO_GPU_DETECTED",
                )
            )

        # Parse output (format: "name, total_mb, free_mb, driver")
        try:
            lines = [line.strip() for line in stdout.strip().split("\n") if line.strip()]
            if not lines:
                return Err(
                    ProviderError(message="No GPU found in nvidia-smi output", code="SINGLENODE_NO_GPU_DETECTED")
                )

            # Take first GPU
            parts = [p.strip() for p in lines[0].split(",")]
            if len(parts) < 4:
                return Err(
                    ProviderError(
                        message=f"Unexpected nvidia-smi format: {lines[0]}",
                        code="SINGLENODE_NVIDIA_SMI_PARSE_ERROR",
                    )
                )

            gpu_name = parts[0]
            vram_total = int(float(parts[1]))
            vram_free = int(float(parts[2]))
            driver_version = parts[3]

            # Get CUDA version
            cuda_version = self._get_cuda_version()

            gpu_info = GPUInfo(
                name=gpu_name,
                vram_total_mb=vram_total,
                vram_free_mb=vram_free,
                cuda_version=cuda_version,
                driver_version=driver_version,
                gpu_count=len(lines),
            )

            logger.info(f"✅ GPU detected: {gpu_name} ({vram_total}MB total, {vram_free}MB free, CUDA {cuda_version})")

            return Ok(gpu_info)

        except (ValueError, IndexError) as e:
            return Err(
                ProviderError(
                    message=f"Failed to parse nvidia-smi output: {e}\nOutput: {stdout}",
                    code="SINGLENODE_NVIDIA_SMI_PARSE_ERROR",
                )
            )

    def _get_cuda_version(self) -> str:
        """Get CUDA version from nvcc or nvidia-smi."""
        # Try nvcc first
        success, stdout, _ = self.ssh.exec_command(
            "nvcc --version 2>/dev/null | grep -oP 'release \\K[0-9.]+'",
            timeout=10,
            silent=True,
        )
        if success and stdout.strip():
            return stdout.strip()

        # Fallback to nvidia-smi
        success, stdout, _ = self.ssh.exec_command(
            "nvidia-smi --query-gpu=cuda_version --format=csv,noheader 2>/dev/null",
            timeout=10,
            silent=True,
        )
        if success and stdout.strip():
            return stdout.strip().split()[0]

        return "unknown"

    def check_docker(self) -> Result[str, ProviderError]:
        """
        Check Docker availability and version.

        Docker is REQUIRED for single_node (docker-only execution).

        Returns:
            Ok(str): Docker version string (e.g., "24.0.7")
            Err(ProviderError): Error
        """
        logger.debug("🔍 Checking Docker...")

        # Prefer Docker CE (default context) when available, but allow rootless fallback.
        # Docker Desktop doesn't support GPU passthrough properly on Linux.
        # Rootless Docker can be acceptable if NVIDIA runtime is configured for the rootless daemon.

        # Check if docker command exists
        success, stdout, _ = self.ssh.exec_command(
            "docker --version",
            timeout=10,
            silent=True,
        )

        if not success:
            install_hint = (
                "Docker is required for training. Install with:\n"
                "   curl -fsSL https://get.docker.com | sh\n"
                "   sudo usermod -aG docker $USER"
            )
            return Err(
                ProviderError(message=f"Docker not installed. {install_hint}", code="SINGLENODE_DOCKER_NOT_INSTALLED")
            )

        # Parse version: "Docker version 24.0.7, build afdd53b"
        match = re.search(r"Docker version (\d+\.\d+\.\d+)", stdout)
        version = match.group(1) if match else "unknown"

        # Check if docker daemon is running (try default -> rootless)
        active_context = None
        for ctx in ("default", "rootless"):
            ok_ctx, _out, _err = self.ssh.exec_command(
                f"docker context use {ctx} 2>/dev/null",
                timeout=5,
                silent=True,
            )
            if not ok_ctx:
                continue
            ok_info, out_info, _err_info = self.ssh.exec_command(
                "docker info > /dev/null 2>&1 && echo 'running'",
                timeout=10,
                silent=True,
            )
            if ok_info and "running" in out_info:
                active_context = ctx
                break

        if active_context is None:
            hint = (
                "Start Docker service with: sudo systemctl start docker\n"
                "Or use rootless Docker:\n"
                "  dockerd-rootless-setuptool.sh install -f\n"
                "  nvidia-ctk runtime configure --runtime=docker --config=$HOME/.config/docker/daemon.json\n"
                "  systemctl --user restart docker.service\n"
                "  docker context use rootless"
            )
            return Err(
                ProviderError(
                    message=f"Docker {version} installed but daemon not running (default/rootless). {hint}",
                    code="SINGLENODE_DOCKER_DAEMON_NOT_RUNNING",
                )
            )

        # Check nvidia-docker (GPU support) via docker info
        # We check if nvidia runtime is configured, not by running a container
        success, stdout, _ = self.ssh.exec_command(
            "docker info 2>/dev/null | grep -i nvidia",
            timeout=_HEALTH_CMD_TIMEOUT,
            silent=True,
        )

        if not success or "nvidia" not in stdout.lower():
            logger.warning("⚠️ nvidia-docker (GPU support) not configured")
            hint = (
                "Install nvidia-container-toolkit: "
                "https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
            )
            extra = ""
            if active_context == "rootless":
                extra = (
                    " For rootless Docker, ensure NVIDIA runtime is configured in "
                    "$HOME/.config/docker/daemon.json and restart docker.service (user)."
                )
            return Err(
                ProviderError(
                    message=f"Docker {version} installed but GPU support (nvidia runtime) not available. {hint}{extra}",
                    code="SINGLENODE_DOCKER_NO_GPU_SUPPORT",
                )
            )

        logger.info(f"✅ Docker {version} with GPU support (context={active_context})")
        return Ok(version)

    def _get_docker_root_dir(self) -> Result[str, ProviderError]:
        """
        Get Docker root directory on the remote host (where images/layers are stored).

        NOTE: Docker pulls write temporary blobs under DockerRootDir/tmp (e.g., /var/lib/docker/tmp/...).
        """
        cmd = 'docker info --format "{{.DockerRootDir}}" 2>/dev/null'
        success, stdout, stderr = self.ssh.exec_command(cmd, timeout=10, silent=True)
        if not success or not stdout.strip():
            details = (stderr or stdout or "").strip()[:_HEALTH_STDERR_TRUNCATE]
            return Err(
                ProviderError(
                    message=f"Failed to detect Docker root dir (DockerRootDir). Details: {details or 'empty'}",
                    code="SINGLENODE_DOCKER_ROOT_DIR_FAILED",
                )
            )
        return Ok(stdout.strip())

    def check_disk_space(self, path: str = "/", *, min_free_gb: float | None = None) -> Result[float, ProviderError]:
        """
        Check available disk space.

        Args:
            path: Path to check disk space for (falls back to / if not exists)
            min_free_gb: Minimum required free space (GB). Defaults to MIN_DISK_FREE_GB.

        Returns:
            Ok(float): Free space in GB
            Err(ProviderError): Error
        """
        # If path doesn't exist, check the parent or fall back to root
        check_path = path

        # First, check if path exists
        exists_cmd = f'test -d "{path}" && echo "exists" || echo "not_exists"'
        success, stdout, _ = self.ssh.exec_command(exists_cmd, timeout=5, silent=True)

        if not success or "not_exists" in stdout:
            # Path doesn't exist yet, check parent directory
            if "/" in path:
                check_path = "/".join(path.rstrip("/").split("/")[:-1]) or "/"
            else:
                check_path = "/"

        logger.debug(f"🔍 Checking disk space at {check_path}...")

        # Use df to get available space in GB
        cmd = f'df -BG "{check_path}" --output=avail 2>/dev/null | tail -1 | tr -d " G"'
        success, stdout, stderr = self.ssh.exec_command(cmd, timeout=10, silent=True)

        if not success or not stdout.strip():
            return Err(ProviderError(message=f"Disk space check failed: {stderr}", code="SINGLENODE_DISK_CHECK_FAILED"))

        try:
            free_gb = float(stdout.strip())
        except ValueError:
            return Err(ProviderError(message=f"Cannot parse disk space: {stdout}", code="SINGLENODE_DISK_PARSE_ERROR"))

        required = float(min_free_gb) if min_free_gb is not None else float(self.MIN_DISK_FREE_GB)
        if free_gb < required:
            return Err(
                ProviderError(
                    message=f"Insufficient disk space: {free_gb:.1f}GB free, minimum {required:.0f}GB required",
                    code="SINGLENODE_DISK_INSUFFICIENT",
                )
            )

        logger.info(f"✅ Disk space: {free_gb:.1f}GB free at {check_path}")
        return Ok(free_gb)

    def run_all_checks(
        self,
        workspace_path: str = "/",
    ) -> HealthCheckResult:
        """
        Run all health checks.

        Args:
            workspace_path: Path to check disk space for

        Returns:
            HealthCheckResult with all check results
        """
        logger.info("🏥 Running SingleNode health checks...")

        errors: list[str] = []
        warnings: list[str] = []

        # GPU check
        gpu_result = self.check_gpu()
        gpu_info = gpu_result.unwrap() if gpu_result.is_success() else None
        if gpu_result.is_failure():
            err = gpu_result.unwrap_err()
            errors.append(str(err) if err is not None else "Unknown GPU error")

        # Docker check
        docker_version = None
        docker_available = False
        docker_result = self.check_docker()

        if docker_result.is_success():
            docker_version = docker_result.unwrap()
            docker_available = True
        else:
            err = docker_result.unwrap_err()
            errors.append(str(err) if err is not None else "Unknown Docker error")

        # Disk space check
        disk_result = self.check_disk_space(workspace_path, min_free_gb=self.MIN_DISK_FREE_GB)
        disk_free_gb = disk_result.unwrap() if disk_result.is_success() else None
        if disk_result.is_failure():
            err = disk_result.unwrap_err()
            errors.append(str(err) if err is not None else "Unknown disk space error")

        # Docker disk space check (critical for docker-only execution: images are stored under DockerRootDir)
        docker_root_dir = None
        docker_disk_free_gb = None
        if docker_available:
            root_res = self._get_docker_root_dir()
            if root_res.is_success():
                docker_root_dir = root_res.unwrap()
                docker_disk_res = self.check_disk_space(
                    docker_root_dir,
                    min_free_gb=self.MIN_DOCKER_DISK_FREE_GB,
                )
                docker_disk_free_gb = docker_disk_res.unwrap() if docker_disk_res.is_success() else None
                if docker_disk_res.is_failure():
                    derr = docker_disk_res.unwrap_err()
                    msg = (
                        f"Insufficient disk space for Docker storage (DockerRootDir={docker_root_dir}): "
                        f"{derr if derr is not None else 'unknown'}"
                    )
                    errors.append(msg)
            else:
                rerr = root_res.unwrap_err()
                errors.append(str(rerr) if rerr is not None else "Failed to detect DockerRootDir")

        # Determine if passed
        passed = len(errors) == 0

        if passed:
            logger.info("✅ All health checks passed!")
        else:
            logger.error(f"❌ Health checks failed with {len(errors)} error(s)")
            for error in errors:
                logger.error(f"   - {error}")

        if warnings:
            for warning in warnings:
                logger.warning(f"⚠️ {warning}")

        return HealthCheckResult(
            passed=passed,
            gpu_info=gpu_info,
            docker_version=docker_version,
            docker_root_dir=docker_root_dir,
            disk_free_gb=disk_free_gb,
            docker_disk_free_gb=docker_disk_free_gb,
            docker_available=docker_available,
            errors=errors if errors else None,
            warnings=warnings if warnings else None,
        )
