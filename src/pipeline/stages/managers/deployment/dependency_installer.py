"""Verify Python/Docker runtime dependencies on the remote host.

Owns the DEPENDENCIES concern: in docker-only mode there are no
host-level package installs. For ``single_node`` we run the runtime
contract checker inside the configured Docker image; for cloud
providers (RunPod) we run it in the current container that SSH lands
in. Either way, missing packages = hard failure (no fallback install).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from src.pipeline.stages.managers.deployment.provider_config import (
    get_single_node_training_cfg,
    is_single_node_provider,
)
from src.pipeline.stages.managers.deployment_constants import (
    DEPLOYMENT_DOCKER_PULL_TIMEOUT,
    DEPLOYMENT_DOCKER_VERIFY_TIMEOUT,
    DEPLOYMENT_ERROR_TRUNCATE,
    DEPLOYMENT_PYTHON_VERIFY_TIMEOUT,
    DEPLOYMENT_STDERR_TRUNCATE,
    DEPLOYMENT_STDOUT_LINES,
)
from src.utils.docker import ensure_docker_image
from src.utils.logger import logger
from src.utils.result import AppError, ConfigError, Err, Ok, ProviderError, Result

if TYPE_CHECKING:
    from src.utils.config import PipelineConfig, Secrets
    from src.utils.ssh_client import SSHClient


DEFAULT_WORKSPACE = "/workspace"


class DependencyInstaller:
    """Verify training-runtime dependencies (no host installs in docker-only mode)."""

    # Docker images known to ship with all required runtime packages.
    PREBUILT_IMAGES: ClassVar[list[str]] = [
        "ryotenkai/ryotenkai-training-runtime",
    ]

    def __init__(self, config: PipelineConfig, secrets: Secrets) -> None:
        self.config = config
        self.secrets = secrets
        self._workspace = DEFAULT_WORKSPACE

    @property
    def workspace(self) -> str:
        return self._workspace

    def set_workspace(self, workspace_path: str) -> None:
        self._workspace = workspace_path

    def install(self, ssh_client: SSHClient) -> Result[None, AppError]:
        """Verify dependencies on the remote target.

        - ``single_node``: pull + run runtime image on host via SSH.
        - cloud providers: SSH lands inside the pod's container, so we
          run the contract checker in-place. Missing packages → fail.
        """
        if is_single_node_provider(self.config):
            logger.info("🐳 single_node: docker-only deps (no host installs) — verifying runtime image...")
            return self._verify_single_node_docker_runtime(ssh_client)

        logger.info("☁️ cloud: docker-only deps — verifying runtime contract inside the current container...")
        verify_result = self.verify_prebuilt_dependencies(ssh_client)
        if verify_result.is_failure():
            orig_err = verify_result.unwrap_err()
            return Err(
                ProviderError(
                    message="Training runtime image missing required packages. Dependencies are docker-only (no fallback install).",
                    code="RUNTIME_DEPS_MISSING",
                    details=orig_err.to_log_dict(),
                )
            )

        return Ok(None)

    @staticmethod
    def verify_prebuilt_dependencies(ssh_client: SSHClient) -> Result[None, ProviderError]:
        """Verify dependencies in prebuilt Docker image (cloud mode).

        SSH connects INSIDE the pod's container, so we directly run the
        runtime contract checker. Only checks key packages availability;
        does NOT install anything.
        """
        logger.info("📦 Verifying prebuilt image dependencies (cloud mode)...")

        # Single source of truth: runtime image must contain the contract checker.
        # Python binary may be either `python3` or `python` depending on the image.
        verify_cmd = (
            "if command -v python3 >/dev/null 2>&1; then python3 /opt/helix/runtime_check.py; "
            "elif command -v python >/dev/null 2>&1; then python /opt/helix/runtime_check.py; "
            "elif [ -x /opt/conda/bin/python3 ]; then /opt/conda/bin/python3 /opt/helix/runtime_check.py; "
            "elif [ -x /opt/conda/bin/python ]; then /opt/conda/bin/python /opt/helix/runtime_check.py; "
            'else echo "PYTHON_NOT_FOUND"; exit 127; fi'
        )
        success, stdout, stderr = ssh_client.exec_command(
            command=verify_cmd, background=False, timeout=DEPLOYMENT_PYTHON_VERIFY_TIMEOUT
        )

        if not success or "OK" not in (stdout or ""):
            details = (stderr or stdout or "").strip()[:DEPLOYMENT_STDERR_TRUNCATE]
            logger.error(f"❌ Runtime contract check failed: {details if details else 'unknown'}")
            return Err(
                ProviderError(
                    message="Runtime contract check failed (missing deps or wrong image).",
                    code="RUNTIME_CONTRACT_CHECK_FAILED",
                    details={"output": details},
                )
            )

        logger.info("✅ Runtime dependencies verified:")
        for line in (stdout or "").strip().split("\n")[:DEPLOYMENT_STDOUT_LINES]:
            logger.info(f"   {line}")

        return Ok(None)

    def _ensure_docker_image_present(self, ssh_client: SSHClient, *, image: str) -> Result[None, ProviderError]:
        return ensure_docker_image(ssh=ssh_client, image=image, pull_timeout_seconds=DEPLOYMENT_DOCKER_PULL_TIMEOUT)

    def _verify_single_node_docker_runtime(self, ssh_client: SSHClient) -> Result[None, AppError]:
        """Verify dependencies inside single_node training Docker image.

        Runs on the host (via SSH) because in single_node docker-mode SSH
        is connected to the host, not inside a container.
        """
        cfg = get_single_node_training_cfg(self.config)
        image_val = cfg.get("docker_image")
        if not isinstance(image_val, str) or not image_val.strip():
            return Err(
                ConfigError(
                    message="providers.single_node.training.docker_image is required (no default in docker-only mode)",
                    code="DOCKER_IMAGE_NOT_CONFIGURED",
                )
            )
        image = image_val.strip()

        logger.info(f"🐳 Training runtime image: {image}")
        pull_result = self._ensure_docker_image_present(ssh_client, image=image)
        if pull_result.is_failure():
            return Err(pull_result.unwrap_err())  # type: ignore[union-attr]  # already ProviderError

        logger.info("📦 Verifying Docker runtime image dependencies (single_node)...")
        verify_cmd = f"docker run --rm --gpus all {image} python3 /opt/helix/runtime_check.py"
        logger.info(f"🔎 Runtime contract check: {verify_cmd}")
        success, stdout, stderr = ssh_client.exec_command(
            command=verify_cmd, background=False, timeout=DEPLOYMENT_DOCKER_VERIFY_TIMEOUT
        )
        if not success or "OK" not in (stdout or ""):
            return Err(
                ProviderError(
                    message=f"Training runtime image '{image}' missing required packages or failed to start.",
                    code="DOCKER_RUNTIME_CHECK_FAILED",
                    details={"image": image, "stderr": stderr[:DEPLOYMENT_ERROR_TRUNCATE] if stderr else "empty"},
                )
            )

        logger.info("✅ Docker runtime image dependencies verified")
        return Ok(None)


__all__ = ["DEFAULT_WORKSPACE", "DependencyInstaller"]
