"""
Training Deployment Manager - Manages training deployment on RunPod.

Handles file uploads, dependency installation, and training execution.
Extracted from RunPodDeployer as part of SOLID refactoring (Phase 4/5).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.pipeline.stages.managers.deployment.code_syncer import CodeSyncer
from src.pipeline.stages.managers.deployment.dependency_installer import DependencyInstaller
from src.pipeline.stages.managers.deployment.file_uploader import FileUploader
from src.pipeline.stages.managers.deployment.provider_config import (
    get_active_provider_name,
    get_cloud_training_cfg,
    get_single_node_training_cfg,
    is_single_node_provider,
)
from src.pipeline.stages.managers.deployment.training_launcher import TrainingLauncher
from src.utils.logger import logger

if TYPE_CHECKING:
    from src.utils.config import PipelineConfig, Secrets
    from src.utils.result import AppError, ProviderError, Result
    from src.utils.ssh_client import SSHClient


class TrainingDeploymentManager:
    """
    Manages training deployment on GPU servers.

    Responsibilities:
    - Upload datasets and training scripts
    - Verify dependencies (prebuilt Docker images)
    - Start training process in background
    - File operations via SSH

    Environment:
    - All providers use Docker images with pre-installed packages
    - RunPod: Pod created with image
    - single_node: Docker container started via SSH

    Does NOT handle:
    - Pod creation (RunPodAPIClient)
    - Pod lifecycle management (PodLifecycleManager)
    - SSH connection creation (stays in GPUDeployer)
    """

    # Default workspace for cloud providers (RunPod)
    DEFAULT_WORKSPACE = "/workspace"

    # UV installation script
    UV_INSTALL_SCRIPT = "curl -LsSf https://astral.sh/uv/install.sh | sh"

    def __init__(self, config: PipelineConfig, secrets: Secrets):
        """
        Initialize training deployment manager.

        Args:
            config: Pipeline configuration
            secrets: Secrets (for HF_TOKEN, etc.)
        """
        self.config = config
        self.secrets = secrets
        self._workspace = self.DEFAULT_WORKSPACE
        self._code_syncer = CodeSyncer(config=config, secrets=secrets)
        self._file_uploader = FileUploader(config=config, secrets=secrets, code_syncer=self._code_syncer)
        self._deps_installer = DependencyInstaller(config=config, secrets=secrets)
        self._launcher = TrainingLauncher(config=config, secrets=secrets, deps_installer=self._deps_installer)
        for component in (self._code_syncer, self._file_uploader, self._deps_installer, self._launcher):
            component.set_workspace(self._workspace)
        logger.debug("🚀 TrainingDeploymentManager initialized")

    @property
    def workspace(self) -> str:
        """Remote workspace path where code/configs are deployed."""
        return self._workspace

    def set_workspace(
        self,
        workspace_path: str,
    ) -> None:
        """
        Set workspace path for deployment.

        Args:
            workspace_path: Run directory (e.g., /workspace or /home/user/run_xxx/)
        """
        self._workspace = workspace_path
        for component in (self._code_syncer, self._file_uploader, self._deps_installer, self._launcher):
            component.set_workspace(workspace_path)
        logger.debug(f"[DEPLOY] Workspace: {self._workspace}")

    # =========================================================================
    # SOURCE CODE SYNC — delegates to CodeSyncer
    # =========================================================================

    def _sync_source_code(self, ssh_client: SSHClient) -> Result[None, AppError]:
        """Proxy to :meth:`CodeSyncer.sync` — kept until callers migrate."""
        return self._code_syncer.sync(ssh_client)

    # =========================================================================
    # FILES DEPLOY — delegates to FileUploader
    # =========================================================================

    def deploy_files(self, ssh_client: SSHClient, context: dict[str, Any]) -> Result[None, AppError]:
        """Proxy to :meth:`FileUploader.deploy_files`."""
        return self._file_uploader.deploy_files(ssh_client, context)

    # =========================================================================
    # PROVIDER HELPERS — proxies onto deployment.provider_config functions
    # =========================================================================

    def _get_active_provider_name(self) -> str:
        return get_active_provider_name(self.config)

    def _is_single_node_provider(self) -> bool:
        return is_single_node_provider(self.config)

    def _get_single_node_training_cfg(self) -> dict[str, Any]:
        return get_single_node_training_cfg(self.config)

    def _get_cloud_training_cfg(self) -> dict[str, Any]:
        return get_cloud_training_cfg(self.config)

    # =========================================================================
    # DEPENDENCY INSTALLATION — delegates to DependencyInstaller
    # =========================================================================

    def install_dependencies(self, ssh_client: SSHClient) -> Result[None, AppError]:
        """Proxy to :meth:`DependencyInstaller.install`."""
        return self._deps_installer.install(ssh_client)

    @staticmethod
    def _verify_prebuilt_dependencies(ssh_client: SSHClient) -> Result[None, ProviderError]:
        """Proxy to :meth:`DependencyInstaller.verify_prebuilt_dependencies`."""
        return DependencyInstaller.verify_prebuilt_dependencies(ssh_client)

    def _verify_single_node_docker_runtime(self, ssh_client: SSHClient) -> Result[None, AppError]:
        """Proxy to :meth:`DependencyInstaller._verify_single_node_docker_runtime`."""
        return self._deps_installer._verify_single_node_docker_runtime(ssh_client)

    def _ensure_docker_image_present(
        self, ssh_client: SSHClient, *, image: str
    ) -> Result[None, ProviderError]:
        """Proxy to :meth:`DependencyInstaller._ensure_docker_image_present` — used by TrainingLauncher tests."""
        return self._deps_installer._ensure_docker_image_present(ssh_client, image=image)

    # =========================================================================
    # TRAINING EXECUTION — delegates to TrainingLauncher
    # =========================================================================

    def _create_env_file(
        self,
        ssh_client,
        context=None,
        extra_env_vars=None,
    ):
        """Proxy to :meth:`TrainingLauncher._create_env_file`."""
        return self._launcher._create_env_file(ssh_client, context, extra_env_vars)

    def start_training(
        self,
        ssh_client,
        context,
        provider=None,
    ):
        """Proxy to :meth:`TrainingLauncher.start_training`."""
        return self._launcher.start_training(ssh_client, context, provider)

    def _start_training_cloud(
        self,
        ssh_client,
        context,
        provider,
    ):
        """Proxy to :meth:`TrainingLauncher._start_training_cloud`."""
        return self._launcher._start_training_cloud(ssh_client, context, provider)

    def _start_training_docker(
        self, ssh_client, context
    ):
        """Proxy to :meth:`TrainingLauncher._start_training_docker`."""
        return self._launcher._start_training_docker(ssh_client, context)

    @staticmethod
    def _sanitize_docker_name(name: str) -> str:
        """Proxy to :meth:`TrainingLauncher._sanitize_docker_name`."""
        return TrainingLauncher._sanitize_docker_name(name)


__all__ = ["TrainingDeploymentManager"]
