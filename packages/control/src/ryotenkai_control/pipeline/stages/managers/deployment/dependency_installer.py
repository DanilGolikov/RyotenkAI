"""Verify Python/Docker runtime dependencies on the remote host.

Owns the DEPENDENCIES concern: in docker-only mode there are no
host-level package installs. For ``single_node`` we run the runtime
contract checker inside the configured Docker image; for cloud
providers (RunPod) we run it in the current container that SSH lands
in. Either way, missing packages = hard failure (no fallback install).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from ryotenkai_control.pipeline.stages.managers.deployment.provider_config import (
    is_single_node_provider,
)
from ryotenkai_control.pipeline.stages.managers.deployment_constants import (
    DEPLOYMENT_DOCKER_PULL_TIMEOUT,
    DEPLOYMENT_DOCKER_VERIFY_TIMEOUT,
    DEPLOYMENT_ERROR_TRUNCATE,
    DEPLOYMENT_PYTHON_VERIFY_TIMEOUT,
    DEPLOYMENT_STDERR_TRUNCATE,
    DEPLOYMENT_STDOUT_LINES,
)
from ryotenkai_shared.constants import RUNTIME_IMAGE
from ryotenkai_shared.errors import SSHExecFailedError
from ryotenkai_shared.utils.docker import ensure_docker_image
from ryotenkai_shared.utils.logger import logger

if TYPE_CHECKING:
    from ryotenkai_shared.config import PipelineConfig, Secrets
    from ryotenkai_shared.utils.ssh_client import SSHClient


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

    def install(self, ssh_client: SSHClient) -> None:
        """Verify dependencies on the remote target.

        - ``single_node``: pull + run runtime image on host via SSH.
        - cloud providers: SSH lands inside the pod's container, so we
          run the contract checker in-place. Missing packages → fail.

        Phase A2 Batch 9 (2026-05-15): migrated from
        ``Result[None, AppError]`` to raise-based. Returns ``None`` on
        success; propagates :class:`SSHExecFailedError` or
        :class:`ProviderUnavailableError` from the underlying helpers.
        """
        if is_single_node_provider(self.config):
            logger.info("🐳 single_node: docker-only deps (no host installs) — verifying runtime image...")
            self._verify_single_node_docker_runtime(ssh_client)
            return None

        logger.info("☁️ cloud: docker-only deps — verifying runtime contract inside the current container...")
        try:
            self.verify_prebuilt_dependencies(ssh_client)
        except SSHExecFailedError as exc:
            # Re-raise with the deps-missing framing so observability
            # stays distinct from a transport-level SSH exec failure.
            raise SSHExecFailedError(
                detail="Training runtime image missing required packages. Dependencies are docker-only (no fallback install).",
                context={
                    "reason": "RUNTIME_DEPS_MISSING",
                    "underlying_reason": exc.context.get("reason") if exc.context else None,
                    "output": exc.context.get("output") if exc.context else None,
                },
                cause=exc,
            ) from exc

    @staticmethod
    def verify_prebuilt_dependencies(ssh_client: SSHClient) -> None:
        """Verify dependencies in prebuilt Docker image (cloud mode).

        SSH connects INSIDE the pod's container, so we directly run the
        runtime contract checker. Only checks key packages availability;
        does NOT install anything.

        Phase A2 Batch 9: returns ``None`` on success; raises
        :class:`SSHExecFailedError` when the runtime contract check
        fails (missing deps or wrong image).
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
            raise SSHExecFailedError(
                detail="Runtime contract check failed (missing deps or wrong image).",
                context={
                    "reason": "RUNTIME_CONTRACT_CHECK_FAILED",
                    "output": details,
                },
            )

        logger.info("✅ Runtime dependencies verified:")
        for line in (stdout or "").strip().split("\n")[:DEPLOYMENT_STDOUT_LINES]:
            logger.info(f"   {line}")

        return None

    def _ensure_docker_image_present(self, ssh_client: SSHClient, *, image: str) -> None:
        """Pull ``image``; propagates :class:`ProviderUnavailableError`.

        Phase A2 Batch 9 (2026-05-15): the Result→ProviderError adapter
        is gone — ``ensure_docker_image`` already raises typed
        exceptions (introduced Phase A2 Batch 4), so we just pass them
        through.
        """
        ensure_docker_image(ssh=ssh_client, image=image, pull_timeout_seconds=DEPLOYMENT_DOCKER_PULL_TIMEOUT)

    def _verify_single_node_docker_runtime(self, ssh_client: SSHClient) -> None:
        """Verify dependencies inside single_node training Docker image.

        Runs on the host (via SSH) because in single_node docker-mode SSH
        is connected to the host, not inside a container.

        The image is pinned in :data:`src.runner.__about__.RUNTIME_IMAGE`
        as of Phase 6.6 — no longer a user config field.

        Phase A2 Batch 9: returns ``None``; raises
        :class:`ProviderUnavailableError` (from
        :meth:`_ensure_docker_image_present`) on pull failure, or
        :class:`SSHExecFailedError` when the contract check inside the
        image fails.
        """
        # Image is pinned in code (Phase 6.6); we no longer read
        # ``cfg["docker_image"]`` here. Future single_node-specific
        # docker tunables (shm_size, container_name_prefix) live on
        # the same ``get_single_node_training_cfg`` helper if needed.
        image = RUNTIME_IMAGE

        logger.info(f"🐳 Training runtime image: {image}")
        # Propagates ProviderUnavailableError directly.
        self._ensure_docker_image_present(ssh_client, image=image)

        logger.info("📦 Verifying Docker runtime image dependencies (single_node)...")
        verify_cmd = f"docker run --rm --gpus all {image} python3 /opt/helix/runtime_check.py"
        logger.info(f"🔎 Runtime contract check: {verify_cmd}")
        success, stdout, stderr = ssh_client.exec_command(
            command=verify_cmd, background=False, timeout=DEPLOYMENT_DOCKER_VERIFY_TIMEOUT
        )
        if not success or "OK" not in (stdout or ""):
            raise SSHExecFailedError(
                detail=f"Training runtime image '{image}' missing required packages or failed to start.",
                context={
                    "reason": "DOCKER_RUNTIME_CHECK_FAILED",
                    "image": image,
                    "stderr": stderr[:DEPLOYMENT_ERROR_TRUNCATE] if stderr else "empty",
                },
            )

        logger.info("✅ Docker runtime image dependencies verified")
        return None


__all__ = ["DEFAULT_WORKSPACE", "DependencyInstaller"]
