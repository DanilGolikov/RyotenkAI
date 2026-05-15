"""
GPU Deployer - Universal deployment stage using provider abstraction.

Replaces the old RunPodDeployer with a provider-agnostic implementation.
Uses IGPUProvider interface to work with any GPU provider.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, cast, runtime_checkable

from ryotenkai_control.pipeline.stages.base import PipelineStage
from ryotenkai_control.pipeline.stages.constants import PipelineContextKeys, StageNames
from ryotenkai_control.pipeline.stages.managers import TrainingDeploymentManager
from ryotenkai_shared.errors import (
    InternalError,
    ProviderUnavailableError,
    RyotenkAIError,
    SSHTransferFailedError,
)
from ryotenkai_shared.pipeline_context import RunContext
from ryotenkai_shared.utils.logger import get_run_log_layout, logger
from ryotenkai_shared.utils.ssh_client import SSHClient

# Truncation length for the docker image SHA shown in pipeline logs.
GPU_DEPLOYER_IMAGE_SHA_TRUNCATE = 20

if TYPE_CHECKING:
    from collections.abc import Callable

    from ryotenkai_providers.training.interfaces import IGPUProvider, SSHConnectionInfo
    from ryotenkai_shared.config import PipelineConfig, Secrets
    from ryotenkai_shared.utils.pod_layout import PodLayout


# =============================================================================
# OPTIONAL PROVIDER CAPABILITIES (runtime-checked)
# =============================================================================


@runtime_checkable
class _SupportsErrorMarking(Protocol):
    def mark_error(self) -> None: ...


@runtime_checkable
class IEarlyReleasable(Protocol):
    """Optional capability: a stage can release its GPU resource before pipeline end.

    Implemented by GPUDeployer. Called by the orchestrator after ModelRetriever
    completes when terminate_after_retrieval=true, to free the training pod
    before InferenceDeployer / ModelEvaluator stages run.
    """

    def release(self) -> None: ...


# =============================================================================
# EVENT CALLBACKS
# =============================================================================


@dataclass
class GPUDeployerEventCallbacks:
    """
    Callbacks for GPUDeployer events (SOLID-compliant event collection).

    Used to integrate GPUDeployer with MLflow or other logging systems.
    """

    # Provider created event
    on_provider_created: Callable[[str, str], None] | None = None
    # Args: provider_name, provider_type

    # Connected event
    on_connected: Callable[[str, str, int], None] | None = None
    # Args: provider_name, host, port

    # Files uploaded event
    on_files_uploaded: Callable[[float], None] | None = None
    # Args: duration_seconds

    # Dependencies installed event
    on_deps_installed: Callable[[float], None] | None = None
    # Args: duration_seconds

    # Training started event
    on_training_started: Callable[[str], None] | None = None
    # Args: resource_id

    # Error event
    on_error: Callable[[str, str], None] | None = None
    # Args: stage, error_message

    # Cleanup event
    on_cleanup: Callable[[str], None] | None = None
    # Args: provider_name


class GPUDeployer(PipelineStage):
    """
    Universal GPU deployment stage.

    Uses provider abstraction to work with any GPU provider:
        - single_node: Local PC via SSH
        - runpod: RunPod cloud
        - (future): lambda, vast, etc.

    Workflow:
        1. Create provider from config
        2. Connect to GPU server
        3. Upload files and start training
        4. Return SSH info for monitoring

    Example:
        deployer = GPUDeployer(config, secrets)
        result = deployer.execute(context)
        # result contains SSH connection info
    """

    def __init__(
        self,
        config: PipelineConfig,
        secrets: Secrets,
        callbacks: GPUDeployerEventCallbacks | None = None,
    ):
        """
        Initialize GPU deployer.

        Args:
            config: Pipeline configuration
            secrets: Secrets with API keys
            callbacks: Optional event callbacks for MLflow integration
        """
        super().__init__(config, StageNames.GPU_DEPLOYER)
        self.secrets = secrets
        self._callbacks = callbacks or GPUDeployerEventCallbacks()

        # Get provider config
        self._provider_name = config.get_active_provider_name()
        self._provider_config = config.get_provider_config()

        # Provider instance (created on execute)
        self._provider: IGPUProvider | None = None

        # SSH client (saved for log download on error)
        self._ssh_client: SSHClient | None = None

        # Set to True after release() is called; prevents double-disconnect in cleanup()
        self._released: bool = False

        # Deployment manager (for file upload and training start)
        self.deployment = TrainingDeploymentManager(config=config, secrets=secrets)

        logger.info(f"[DEPLOYER:INIT] GPUDeployer initialized: provider={self._provider_name}")

    def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        """
        Deploy training job to GPU server.

        Steps:
            1. Create provider
            2. Connect (creates pod for cloud, SSH for local)
            3. Upload files
            4. Install dependencies
            5. Start training
            6. Return connection info

        Args:
            context: Pipeline context with dataset information

        Returns:
            Updated context dict with deployment info.

        Raises:
            ProviderUnavailableError: provider create / SSH info missing /
                generic connect failure.
            SSHConnectionFailedError: pod reached but SSH handshake failed.
            RyotenkAIError: typed exceptions from the deployment manager
                (code sync / runtime / start_training) propagate.
        """
        logger.info(f"Deploying training job to {self._provider_name}...")

        run = context.get(PipelineContextKeys.RUN)
        if not isinstance(run, RunContext):
            raise InternalError(
                detail="Missing run context: context['run'] must be RunContext (initialized by PipelineOrchestrator)",
                context={"legacy_code": "MISSING_RUN_CONTEXT"},
            )

        # Step 1: Create provider via the manifest-driven registry.
        # Replaces the legacy ``GPUProviderFactory.create(provider_name,
        # provider_config: dict, secrets)`` call — the registry resolves
        # the class via the provider's manifest entry point and hands
        # it a ``ProviderContext``. Same contract for any provider with
        # a ``provider.toml`` declaring ``training`` in its roles.
        from ryotenkai_providers.registry import ProviderContext, get_registry

        ctx = ProviderContext(
            provider_id=self._provider_name,
            pipeline_config=self.config,
            provider_block=self._provider_config,
            secrets=self.secrets,
        )
        try:
            self._provider = get_registry().create_training(self._provider_name, ctx)
        except RyotenkAIError as exc:
            if self._callbacks.on_error:
                self._callbacks.on_error("provider_create", exc.detail or str(exc))
            raise
        # Fire callback
        if self._callbacks.on_provider_created:
            self._callbacks.on_provider_created(self._provider_name, self._provider.provider_type)

        # Step 2: Connect to GPU server
        logger.info(f"Connecting to {self._provider_name}...")
        connect_start = time.time()
        try:
            ssh_info = cast("SSHConnectionInfo | None", self._provider.connect(run=run))
        except RyotenkAIError as exc:
            if self._callbacks.on_error:
                self._callbacks.on_error("connect", exc.detail or str(exc))
            raise
        connect_duration = time.time() - connect_start

        if ssh_info is None:
            msg = "SSH info is None"
            if self._callbacks.on_error:
                self._callbacks.on_error("connect", msg)
            self._handle_error_and_disconnect(msg)
            raise ProviderUnavailableError(
                detail=msg,
                context={
                    "legacy_code": "PROVIDER_SSH_INFO_MISSING",
                    "provider": self._provider_name,
                },
            )

        logger.info(f"Connected: {ssh_info}")

        # Fire callback
        if self._callbacks.on_connected:
            self._callbacks.on_connected(self._provider_name, ssh_info.host, ssh_info.port)

        # Step 3: Set workspace path for deployment manager (docker-only training)
        self.deployment.set_workspace(workspace_path=ssh_info.workspace_path)

        logger.info("🐳 Using Docker-only training mode")
        logger.info(f"   Workspace: {ssh_info.workspace_path}")

        # Step 4: Create SSH client for deployment
        # For alias mode, don't pass username (SSHClient will use alias from ~/.ssh/config)
        ssh_client = SSHClient(
            host=ssh_info.host,
            port=ssh_info.port,
            username=None if ssh_info.is_alias_mode else ssh_info.user,
            key_path=ssh_info.key_path if not ssh_info.is_alias_mode else "",
        )
        self._ssh_client = ssh_client  # Save for log download on error

        # Step 5: Sync source modules via rsync (pre-launch SSH).
        # Phase 3 PR-3.3 (transport-unification-v2): split from the
        # legacy ``deploy_files`` — config/dataset upload moved into
        # ``start_training`` (post-launch, HTTP). The rsync still
        # happens here because uvicorn cannot start without the
        # synced ``ryotenkai_*`` packages on disk.
        #
        # Phase A2 Batch 9 (2026-05-15): the deployment manager's
        # public surface raises typed exceptions; this caller still
        # exposes ``Result[dict, AppError]`` upward (Batch 10 will
        # finish the migration), so each call is wrapped in a typed
        # → AppError translator at the boundary.
        logger.info("Syncing source code (rsync)...")
        upload_start = time.time()
        try:
            self.deployment.deploy_code(ssh_client)
        except RyotenkAIError as exc:
            upload_duration = time.time() - upload_start
            logger.error("Code sync failed, disconnecting...")
            self._handle_error_and_disconnect("Code sync failed")
            if self._callbacks.on_error:
                self._callbacks.on_error("upload", exc.detail or str(exc))
            raise
        upload_duration = time.time() - upload_start

        logger.info(f"Code synced! ({upload_duration:.1f}s)")
        if self._callbacks.on_files_uploaded:
            self._callbacks.on_files_uploaded(upload_duration)

        # Step 6: Verify runtime (docker-only: no host pip/venv installs)
        logger.info("Verifying training runtime (docker-only)...")
        deps_start = time.time()
        try:
            self.deployment.install_dependencies(ssh_client)
        except RyotenkAIError as exc:
            deps_duration = time.time() - deps_start
            logger.error("Runtime verification failed, disconnecting...")
            self._handle_error_and_disconnect("Runtime verification failed")
            if self._callbacks.on_error:
                self._callbacks.on_error("deps", exc.detail or str(exc))
            raise
        deps_duration = time.time() - deps_start

        logger.info(f"Runtime verified! ({deps_duration:.1f}s)")
        if self._callbacks.on_deps_installed:
            self._callbacks.on_deps_installed(deps_duration)

        # Step 7: Start training.
        #
        # Stamp ``resource_id`` into the context BEFORE delegating —
        # the launcher needs it to assemble runner-side env vars
        # (RunPod's ``required_runtime_env_vars`` keys ``RUNPOD_POD_ID``
        # off this value, and the runner's lifespan refuses to boot
        # without it). Previously only ``upload_context`` (a copy)
        # carried the id; the main ``context`` was missing it through
        # ``start_training`` and the runner died with
        # ``BootstrapConfigError: ... requires RUNPOD_POD_ID``.
        if ssh_info.resource_id:
            context["resource_id"] = ssh_info.resource_id
        logger.info("Starting training...")
        try:
            training_metadata = self.deployment.start_training(
                ssh_client, context, provider=self._provider,
            )
        except RyotenkAIError as exc:
            logger.error("Training start failed, disconnecting...")
            self._handle_error_and_disconnect("Training start failed")
            if self._callbacks.on_error:
                self._callbacks.on_error("training_start", exc.detail or str(exc))
            raise
        logger.info(f"Training started! Mode: {training_metadata.get('mode', 'unknown')}")

        # Log Docker image SHA to MLflow for reproducibility (if available)
        if training_metadata and "image_sha" in training_metadata:
            image_sha = training_metadata["image_sha"]
            logger.info(f"📌 Docker image SHA: {image_sha[:GPU_DEPLOYER_IMAGE_SHA_TRUNCATE]}...")
            context["docker_image_sha"] = image_sha

        if self._callbacks.on_training_started:
            self._callbacks.on_training_started(ssh_info.resource_id or "unknown")

        # Fetch pod info for cost/GPU metadata (RunPod only; None for single_node)
        pod_info = self._provider.get_resource_info()

        # Return connection info for monitoring
        return self.update_context(
            context,
            {
                "run_name": run.name,
                # Provider info
                "provider_name": self._provider_name,
                "provider_type": self._provider.provider_type,  # "local" or "cloud"
                # Provider instance — TrainingMonitor uses it for the
                # capability-Protocol gated recovery loop (Phase 14.D+F:
                # ``isinstance(provider, IRecoveryProbeProvider)``).
                "provider": self._provider,
                "resource_id": ssh_info.resource_id,  # pod_id or run_dir
                # SSH connection info (for TrainingMonitor)
                "ssh_host": ssh_info.host,
                "ssh_port": ssh_info.port,
                "ssh_user": ssh_info.user,
                "ssh_key_path": ssh_info.key_path,
                "is_alias_mode": ssh_info.is_alias_mode,  # True if using SSH alias
                "workspace_path": ssh_info.workspace_path,
                # Timing
                "training_started_at": time.time(),
                "pod_startup_seconds": connect_duration,
                "upload_duration_seconds": upload_duration,
                "deps_duration_seconds": deps_duration,
                # Pod metadata (RunPod: PodResourceInfo; single_node: None)
                "cost_per_hr": getattr(pod_info, "cost_per_hr", None),
                "gpu_type": getattr(pod_info, "gpu_type", None),
                "gpu_count": getattr(pod_info, "gpu_count", None),
            },
        )

    def get_provider(self) -> IGPUProvider | None:
        """Get the active provider (for cleanup)."""
        return self._provider

    def release(self) -> None:
        """Early-release: terminate training pod right after ModelRetriever.

        Called by the orchestrator when terminate_after_retrieval=true.
        Idempotent — safe to call multiple times or after cleanup().
        """
        if self._released or self._provider is None:
            return
        logger.info("[DEPLOYER] Early release: terminating training pod after ModelRetriever.")
        # Phase 14.D+F — capability-driven dispatch (was
        # ``self._provider_name == PROVIDER_RUNPOD`` string check).
        # Cloud providers expose log-download via SCP/HTTP; local
        # providers have logs already on the host filesystem.
        if self._provider.get_capabilities().supports_log_download and self._ssh_client:
            try:
                self._download_remote_logs("early_release")
            except Exception as e:
                logger.debug(f"[DEPLOYER] Failed to download logs on early release: {e}")
        self._provider.disconnect()
        self._provider = None
        self._released = True

    def _handle_error_and_disconnect(self, reason: str) -> None:
        """
        Mark error state, download logs, and disconnect from provider.

        This ensures keep_on_error=True works correctly:
        - Downloads training logs (even on timeout/error)
        - Marks provider as having an error
        - Then disconnects (which checks _had_error flag)

        Args:
            reason: Error description for logging
        """
        logger.warning(f"[DEPLOYER:ERROR] {reason}")

        # Try to download training logs before disconnecting
        if self._ssh_client:
            self._download_remote_logs(reason)

        if self._provider:
            # Mark error state BEFORE disconnect (for keep_on_error logic)
            if isinstance(self._provider, _SupportsErrorMarking):
                self._provider.mark_error()
            self._provider.disconnect()

    def _download_remote_logs(self, reason: str) -> None:
        """Best-effort SCP pull of pod-side logs on deployment failure.

        Phase 3 PR-3.3 (transport-unification-v2): post-PR-2.3 the
        legacy ``LogManager`` is gone, but this on-failure path still
        needs raw SCP because it runs BEFORE uvicorn is confirmed
        green — JobClient (HTTP) wouldn't be reachable. Falls back
        to the same two PodLayout files (``runner.log`` and
        ``trainer.stdio.log``); the SCP API on :class:`SSHClient`
        does not call ``exec_command`` so it's outside the AST
        sentinel's scope and stays a legitimate bootstrap-failure
        diagnostic path.
        """
        if not self._ssh_client:
            return

        from pathlib import PurePosixPath
        from ryotenkai_shared.utils.pod_layout import PodLayout

        try:
            pod_layout = PodLayout.from_root(
                PurePosixPath(self.deployment.workspace),
            )
        except ValueError as exc:
            logger.warning(
                f"[DEPLOYER] Cannot build PodLayout (reason='{reason}'): {exc}",
            )
            return

        mac_layout = get_run_log_layout()
        # Channel 1 — runner.log (pre-trainer crashes).
        self._scp_log_best_effort(
            label="runner.log",
            remote=str(pod_layout.runner_log),
            local=mac_layout.remote_runner_log,
            reason=reason,
        )
        # Channel 2 — trainer.stdio.log (trainer crashes mid-run).
        self._scp_log_best_effort(
            label="trainer.stdio.log",
            remote=str(pod_layout.trainer_stdio_log),
            local=mac_layout.remote_trainer_stdio_log,
            reason=reason,
        )

    def _scp_log_best_effort(
        self,
        *,
        label: str,
        remote: str,
        local: Any,
        reason: str,
    ) -> None:
        """One SCP pull, all failures swallowed at debug level."""
        if not self._ssh_client:
            return
        try:
            self._ssh_client.download_file(
                remote_path=remote,
                local_path=str(local),
            )
            logger.info(f"[DEPLOYER] Downloaded {label} → {local}")
        except SSHTransferFailedError as exc:
            logger.debug(
                f"[DEPLOYER] {label} not retrieved (reason='{reason}'): "
                f"{exc.detail or exc}",
            )
        except Exception as exc:  # noqa: BLE001 — best-effort
            logger.debug(
                f"[DEPLOYER] {label} SCP raised (reason='{reason}'): "
                f"{type(exc).__name__}: {exc}",
            )

    def notify_pipeline_failure(self) -> None:
        """Inform the deployer that the pipeline ended with an error.

        Called by the orchestrator before cleanup() so that keep_pod_on_error
        is respected when the failure happened outside this stage (e.g. Training Monitor).
        """
        if self._provider and isinstance(self._provider, _SupportsErrorMarking):
            self._provider.mark_error()

    def cleanup(self) -> None:
        """Cleanup resources (disconnect from provider)."""
        if self._released:
            logger.debug("[DEPLOYER] cleanup() skipped: pod already released early.")
            return
        if self._provider:
            logger.info(f"Cleaning up {self._provider_name}...")
            if self._callbacks.on_cleanup:
                self._callbacks.on_cleanup(self._provider_name)

            # Phase 14.D+F — capability-driven dispatch (was
            # ``self._provider_name == PROVIDER_RUNPOD`` string check).
            if self._provider.get_capabilities().supports_log_download and self._ssh_client:
                try:
                    self._download_remote_logs("cleanup")
                except Exception as e:
                    logger.debug(f"[DEPLOYER] Failed to download logs during cleanup: {e}")

            self._provider.disconnect()
            self._provider = None

        if self._ssh_client:
            try:
                self._ssh_client.close_master()
            except Exception as e:
                logger.debug(f"[DEPLOYER] Failed to close SSH ControlMaster: {e}")
            finally:
                self._ssh_client = None
