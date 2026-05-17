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
from ryotenkai_shared.events import UNKNOWN_OFFSET
from ryotenkai_shared.events.types.control_gpu import (
    GPUCodeSyncedEvent,
    GPUCodeSyncedPayload,
)
from ryotenkai_shared.utils.logger import logger

# Source URI for envelopes the deployment manager emits. The manager is
# logically part of the gpu_deployer stage, so envelopes correlate
# under the same ``stage_id`` (set by the gpu_deployer's
# ``stage_scope``). Keeping a distinct ``source`` here lets reports
# attribute the rsync event to the deployment-manager component while
# still grouping under the stage.
_DEPLOYMENT_MANAGER_SOURCE = "control://orchestrator/deployment_manager"

if TYPE_CHECKING:
    from ryotenkai_providers.training.interfaces import IGPUProvider
    from ryotenkai_shared.config import PipelineConfig, Secrets
    from ryotenkai_shared.events import IEventEmitter
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

    def __init__(
        self,
        config: PipelineConfig,
        secrets: Secrets,
        *,
        emitter: IEventEmitter | None = None,
    ):
        self.config = config
        self.secrets = secrets
        self._workspace = self.DEFAULT_WORKSPACE
        # Phase 5 (event-system coverage gaps, 2026-05-16): the manager
        # emits ``ryotenkai.control.gpu.code_synced`` after a successful
        # ``deploy_code`` so reports / live dashboards have a typed
        # signal for the rsync phase that previously surfaced only as
        # ``logger.info`` lines. ``None`` is accepted for legacy /
        # test wiring; emit helpers are no-ops in that case.
        self._emitter: IEventEmitter | None = emitter
        # Cached run-id so per-method emit helpers don't need it
        # threaded through every call. ``set_run_id`` is the one-shot
        # mutator the caller (gpu_deployer) invokes after the
        # PipelineContext is resolved.
        self._cached_run_id: str = "unknown"
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

    def set_emitter(self, emitter: IEventEmitter) -> None:
        """Wire (or replace) the typed event emitter.

        Mirrors :meth:`PipelineStage.set_emitter` — the orchestrator
        constructs stages eagerly but the emitter is only ready once
        the canonical run directory is resolved.
        """
        self._emitter = emitter

    def set_run_id(self, run_id: str) -> None:
        """Pre-populate the run-id used in emitted envelopes.

        ``gpu_deployer.execute`` resolves the canonical run id from
        the :class:`PipelineContext` and calls this once before
        ``deploy_code`` / ``start_training`` so emit helpers don't
        have to re-resolve it from each call site.
        """
        if isinstance(run_id, str) and run_id:
            self._cached_run_id = run_id

    def _emit_code_synced(
        self,
        *,
        run_id: str,
        local_sha: str,
        remote_sha: str,
        bytes_transferred: int,
    ) -> None:
        if self._emitter is None:
            return
        # ``stage_scope`` is opened by the gpu_deployer stage above us;
        # we don't need to re-enter it. The emitter fills ``stage_id``
        # from the ContextVar.
        self._emitter.emit(
            GPUCodeSyncedEvent(
                source=_DEPLOYMENT_MANAGER_SOURCE,
                run_id=run_id,
                offset=UNKNOWN_OFFSET,
                payload=GPUCodeSyncedPayload(
                    local_sha=local_sha,
                    remote_sha=remote_sha,
                    bytes_transferred=bytes_transferred,
                ),
            ),
        )

    @staticmethod
    def _resolve_run_id_from_context(context: dict[str, Any] | None) -> str:
        """Best-effort run-id extraction.

        The deployment manager doesn't always have a context argument
        in scope (deploy_code is called without one). For deploy_code
        we pin to ``"unknown"`` because the rsync event correlates by
        ``stage_id`` set by the parent gpu_deployer. start_training
        receives a context dict where the run name lives under
        ``PipelineContextKeys.RUN``; passing it through avoids a
        cross-package import here.
        """
        if not context:
            return "unknown"
        run_obj = context.get("run")
        run_name = getattr(run_obj, "name", None)
        if isinstance(run_name, str) and run_name:
            return run_name
        return "unknown"

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

        Phase 5 (event-system coverage gaps, 2026-05-16): emits
        :class:`GPUCodeSyncedEvent` on success so reports surface the
        rsync as a typed envelope rather than a log line. ``local_sha``
        and ``remote_sha`` are emitted empty for now — the CodeSyncer
        does not compute a content hash today; Phase 6 can fold a
        manifest-derived sha in. ``bytes_transferred`` defaults to ``0``
        for the same reason; the rsync output is not parsed.
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

        # Successful path — emit the typed envelope. ``run_id`` is
        # filled at emit-time below; the parent ``gpu_deployer`` stage
        # already opened a ``stage_scope`` so ``stage_id`` is auto-
        # filled by the emitter.
        self._emit_code_synced(
            run_id=self._cached_run_id,
            local_sha="",
            remote_sha="",
            bytes_transferred=0,
        )

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
