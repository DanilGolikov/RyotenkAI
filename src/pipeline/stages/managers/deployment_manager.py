"""Facade composing the four deployment components.

After Wave 3 decomposition the GPU-training deployment concern lives in
four single-responsibility components under
:mod:`src.pipeline.stages.managers.deployment`:

* :class:`CodeSyncer` — rsync of project source modules.
* :class:`FileUploader` — config + dataset upload, chains code-sync.
* :class:`DependencyInstaller` — Docker runtime contract checks.
* :class:`TrainingLauncher` — env file + Docker / cloud spawn + probe.

This module exposes a thin facade ``TrainingDeploymentManager`` whose
public API (``deploy_files``, ``install_dependencies``,
``start_training``, ``set_workspace`` + ``workspace`` property) is
preserved bit-for-bit so external callers (``gpu_deployer``,
``test_stages_deployer`` mocks, integration / e2e tests) keep working
without changes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.pipeline.stages.managers.deployment.code_syncer import CodeSyncer
from src.pipeline.stages.managers.deployment.dependency_installer import DependencyInstaller
from src.pipeline.stages.managers.deployment.file_uploader import FileUploader
from src.pipeline.stages.managers.deployment.training_launcher import TrainingLauncher
from src.utils.logger import logger

if TYPE_CHECKING:
    from src.providers.training.interfaces import IGPUProvider
    from src.utils.config import PipelineConfig, Secrets
    from src.utils.result import AppError, Result
    from src.utils.ssh_client import SSHClient


class TrainingDeploymentManager:
    """Facade composing CodeSyncer, FileUploader, DependencyInstaller, TrainingLauncher.

    Public API (stable contract for ``gpu_deployer`` and tests):

    * ``__init__(config, secrets)``
    * ``set_workspace(workspace_path)`` / ``workspace`` property
    * ``deploy_files(ssh_client, context)``
    * ``install_dependencies(ssh_client)``
    * ``start_training(ssh_client, context, provider=None)``

    Components are reachable as ``_code_syncer``, ``_file_uploader``,
    ``_deps_installer``, ``_launcher`` for tests that need to patch
    component internals; not part of the external contract.
    """

    DEFAULT_WORKSPACE = "/workspace"

    def __init__(self, config: PipelineConfig, secrets: Secrets):
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

    def set_workspace(self, workspace_path: str) -> None:
        """Propagate the workspace path into every component."""
        self._workspace = workspace_path
        for component in (self._code_syncer, self._file_uploader, self._deps_installer, self._launcher):
            component.set_workspace(workspace_path)
        logger.debug(f"[DEPLOY] Workspace: {self._workspace}")

    def deploy_files(self, ssh_client: SSHClient, context: dict[str, Any]) -> Result[None, AppError]:
        """Upload config + datasets, then sync source modules. Delegates to FileUploader."""
        return self._file_uploader.deploy_files(ssh_client, context)

    def install_dependencies(self, ssh_client: SSHClient) -> Result[None, AppError]:
        """Verify training-runtime dependencies on the remote target. Delegates to DependencyInstaller."""
        return self._deps_installer.install(ssh_client)

    def start_training(
        self,
        ssh_client: SSHClient,
        context: dict[str, Any],
        provider: IGPUProvider | None = None,
    ) -> Result[dict[str, Any], AppError]:
        """Spawn the training process on the remote target. Delegates to TrainingLauncher."""
        return self._launcher.start_training(ssh_client, context, provider)


__all__ = ["TrainingDeploymentManager"]
