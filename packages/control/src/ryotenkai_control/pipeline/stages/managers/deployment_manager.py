"""Facade composing the four deployment components.

After Wave 3 decomposition + Phase 3 PR-3.3 (transport-unification-v2)
the deployment concern lives in four single-responsibility components:

* :class:`CodeSyncer`         — rsync of project source modules (SSH).
* :class:`FileUploader`       — config + dataset HTTP upload.
* :class:`DependencyInstaller` — Docker runtime contract checks (SSH).
* :class:`TrainingLauncher`   — runner uvicorn launch + tunnel +
                                  HTTP file upload + HTTP job submit.

Bootstrap flow (post Phase 2):

    gpu_deployer:
      1. provider.connect()          → ssh_info
      2. deploy_code(ssh)            → SSH rsync packages/        (pre-launch)
      3. install_dependencies(ssh)   → SSH runtime image verify   (pre-launch)
      4. start_training(ssh, ctx)    → orchestrates:
           - launch_runner (SSH nohup uvicorn)
           - open SSH tunnel + wait /healthz
           - upload_files via HTTP    (NEW — replaces SSH tar-pipe)
           - submit_job via HTTP

The public API change vs the legacy facade: ``deploy_files`` is split
into two methods — ``deploy_code`` (pre-launch SSH rsync) and the
HTTP file upload that now lives inside ``start_training``. Callers
update their step ordering accordingly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ryotenkai_control.pipeline.stages.managers.deployment.code_syncer import CodeSyncer
from ryotenkai_control.pipeline.stages.managers.deployment.dependency_installer import DependencyInstaller
from ryotenkai_control.pipeline.stages.managers.deployment.file_uploader import FileUploader
from ryotenkai_control.pipeline.stages.managers.deployment.training_launcher import TrainingLauncher
from ryotenkai_shared.errors import PipelineStageFailedError, RyotenkAIError
from ryotenkai_shared.utils.logger import logger

if TYPE_CHECKING:
    from ryotenkai_providers.training.interfaces import IGPUProvider
    from ryotenkai_shared.config import PipelineConfig, Secrets
    from ryotenkai_shared.utils.ssh_client import SSHClient


class TrainingDeploymentManager:
    """Facade composing CodeSyncer, FileUploader, DependencyInstaller, TrainingLauncher.

    Public API:
      * ``__init__(config, secrets)``
      * ``set_workspace(workspace_path)`` / ``workspace`` property
      * ``deploy_code(ssh_client)``         — SSH rsync (pre-launch).
      * ``install_dependencies(ssh_client)`` — SSH runtime verify.
      * ``start_training(ssh_client, context, provider=None)`` —
        runner launch + tunnel + HTTP file upload + HTTP job submit.

    Components are reachable as ``_code_syncer``, ``_file_uploader``,
    ``_deps_installer``, ``_launcher`` for tests; not part of the
    external contract.
    """

    DEFAULT_WORKSPACE = "/workspace"

    def __init__(self, config: PipelineConfig, secrets: Secrets):
        self.config = config
        self.secrets = secrets
        self._workspace = self.DEFAULT_WORKSPACE
        self._code_syncer = CodeSyncer(config=config, secrets=secrets)
        self._file_uploader = FileUploader(config=config, secrets=secrets)
        self._deps_installer = DependencyInstaller(config=config, secrets=secrets)
        # TrainingLauncher orchestrates HTTP file upload too — inject
        # the file uploader so the launcher can call ``upload_via_http``
        # between the /healthz wait and the POST /jobs.
        self._launcher = TrainingLauncher(
            config=config, secrets=secrets,
            deps_installer=self._deps_installer,
            file_uploader=self._file_uploader,
        )
        for component in (self._code_syncer, self._file_uploader, self._deps_installer, self._launcher):
            component.set_workspace(self._workspace)
        logger.debug("🚀 TrainingDeploymentManager initialized")

    @property
    def workspace(self) -> str:
        return self._workspace

    def set_workspace(self, workspace_path: str) -> None:
        self._workspace = workspace_path
        for component in (self._code_syncer, self._file_uploader, self._deps_installer, self._launcher):
            component.set_workspace(workspace_path)
        logger.debug(f"[DEPLOY] Workspace: {self._workspace}")

    def deploy_code(self, ssh_client: SSHClient) -> None:
        """Pre-launch SSH rsync of the four pod-relevant
        ``ryotenkai_*`` packages. Replaces the legacy ``deploy_files``
        step that used to chain config/dataset upload before code
        sync — files now upload via HTTP after uvicorn is up
        (``start_training`` orchestrates that).

        Phase A2 Batch 9 (2026-05-15): migrated from
        ``Result[None, AppError]`` to raise-based. Returns ``None`` on
        success; propagates :class:`SSHTransferFailedError` (or
        :class:`PipelineStageFailedError` for unexpected failures)
        from :meth:`CodeSyncer.sync`.
        """
        try:
            self._code_syncer.sync(ssh_client)
        except RyotenkAIError:
            raise
        except Exception as exc:
            raise PipelineStageFailedError(
                detail=f"code sync failed: {exc}",
                context={"reason": "CODE_SYNC_FAILED"},
                cause=exc,
            ) from exc

    def install_dependencies(self, ssh_client: SSHClient) -> None:
        """Verify training-runtime dependencies on the remote target.

        Phase A2 Batch 9: returns ``None`` on success; propagates
        typed exceptions from :meth:`DependencyInstaller.install`.
        """
        try:
            self._deps_installer.install(ssh_client)
        except RyotenkAIError:
            raise
        except Exception as exc:
            raise PipelineStageFailedError(
                detail=f"dependency install failed: {exc}",
                context={"reason": "DEPS_INSTALL_FAILED"},
                cause=exc,
            ) from exc

    def start_training(
        self,
        ssh_client: SSHClient,
        context: dict[str, Any],
        provider: IGPUProvider | None = None,
    ) -> dict[str, Any]:
        """Spawn the training process on the remote target.

        Now also performs HTTP file upload between the /healthz wait
        and POST /jobs (Phase 3 PR-3.3). Delegates to TrainingLauncher.

        Phase A2 Batch 9: returns the launcher's metadata dict on
        success; propagates typed exceptions from
        :meth:`TrainingLauncher.start_training` (see its docstring for
        the failure-mode → exception type table).
        """
        try:
            return self._launcher.start_training(ssh_client, context, provider)
        except RyotenkAIError:
            raise
        except Exception as exc:
            raise PipelineStageFailedError(
                detail=f"training start failed: {exc}",
                context={"reason": "TRAINING_START_FAILED"},
                cause=exc,
            ) from exc


__all__ = ["TrainingDeploymentManager"]
