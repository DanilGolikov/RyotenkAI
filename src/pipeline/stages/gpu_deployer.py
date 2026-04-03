"""
GPU Deployer - Universal deployment stage using provider abstraction.

Replaces the old RunPodDeployer with a provider-agnostic implementation.
Uses IGPUProvider interface to work with any GPU provider.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, cast, runtime_checkable

from src.constants import PROVIDER_RUNPOD
from src.pipeline.constants import GPU_DEPLOYER_IMAGE_SHA_TRUNCATE
from src.pipeline.domain import RunContext
from src.pipeline.stages.base import PipelineStage
from src.pipeline.stages.constants import PipelineContextKeys, StageNames
from src.pipeline.stages.managers import LogManager, TrainingDeploymentManager
from src.providers.training.factory import GPUProviderFactory
from src.utils.logger import logger
from src.utils.result import AppError, Err, Ok, ProviderError, Result
from src.utils.ssh_client import SSHClient

if TYPE_CHECKING:
    from collections.abc import Callable

    from src.providers.training.interfaces import IGPUProvider, SSHConnectionInfo
    from src.utils.config import PipelineConfig, Secrets


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

    def execute(self, context: dict[str, Any]) -> Result[dict[str, Any], AppError]:
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
            Result with deployment info or AppError
        """
        logger.info(f"Deploying training job to {self._provider_name}...")

        run = context.get(PipelineContextKeys.RUN)
        if not isinstance(run, RunContext):
            return Err(
                ProviderError(
                    message="Missing run context: context['run'] must be RunContext (initialized by PipelineOrchestrator)",
                    code="MISSING_RUN_CONTEXT",
                )
            )

        # Step 1: Create provider by name
        create_result = GPUProviderFactory.create(
            provider_name=self._provider_name,
            provider_config=self._provider_config,
            secrets=self.secrets,
        )
        if create_result.is_failure():
            provider_err = create_result.unwrap_err()
            if self._callbacks.on_error:
                self._callbacks.on_error("provider_create", str(provider_err))
            return Err(provider_err)
        self._provider = create_result.unwrap()
        # Fire callback
        if self._callbacks.on_provider_created:
            self._callbacks.on_provider_created(self._provider_name, self._provider.provider_type)

        # Step 2: Connect to GPU server
        logger.info(f"Connecting to {self._provider_name}...")
        connect_start = time.time()
        connect_result = self._provider.connect(run=run)
        connect_duration = time.time() - connect_start

        if connect_result.is_err():
            err = connect_result.unwrap_err()  # type: ignore[union-attr]
            if self._callbacks.on_error:
                self._callbacks.on_error("connect", str(err))
            return Err(ProviderError(message=f"Connection failed: {err}", code="PROVIDER_CONNECT_FAILED"))

        ssh_info = cast("SSHConnectionInfo | None", connect_result.unwrap())

        if ssh_info is None:
            msg = "SSH info is None"
            if self._callbacks.on_error:
                self._callbacks.on_error("connect", msg)
            self._handle_error_and_disconnect(msg)
            return Err(ProviderError(message=msg, code="PROVIDER_SSH_INFO_MISSING"))

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

        # Step 5: Upload files
        logger.info("Uploading files...")
        upload_start = time.time()
        upload_context = dict(context)
        upload_context["resource_id"] = ssh_info.resource_id
        upload_result = self.deployment.deploy_files(ssh_client, upload_context)
        upload_duration = time.time() - upload_start

        if upload_result.is_failure():
            logger.error("File upload failed, disconnecting...")
            self._handle_error_and_disconnect("File upload failed")
            upload_err = upload_result.unwrap_err()  # AppError from deployment_manager
            if self._callbacks.on_error:
                self._callbacks.on_error("upload", str(upload_err))
            return Err(upload_err)

        logger.info(f"Files uploaded! ({upload_duration:.1f}s)")
        if self._callbacks.on_files_uploaded:
            self._callbacks.on_files_uploaded(upload_duration)

        # Step 6: Verify runtime (docker-only: no host pip/venv installs)
        logger.info("Verifying training runtime (docker-only)...")
        deps_start = time.time()
        deps_result = self.deployment.install_dependencies(ssh_client)
        deps_duration = time.time() - deps_start

        if deps_result.is_failure():
            logger.error("Runtime verification failed, disconnecting...")
            self._handle_error_and_disconnect("Runtime verification failed")
            deps_err = deps_result.unwrap_err()  # AppError from deployment_manager
            if self._callbacks.on_error:
                self._callbacks.on_error("deps", str(deps_err))
            return Err(deps_err)

        logger.info(f"Runtime verified! ({deps_duration:.1f}s)")
        if self._callbacks.on_deps_installed:
            self._callbacks.on_deps_installed(deps_duration)

        # Step 7: Start training
        logger.info("Starting training...")
        training_result = self.deployment.start_training(ssh_client, context)

        if training_result.is_failure():
            logger.error("Training start failed, disconnecting...")
            self._handle_error_and_disconnect("Training start failed")
            training_err = training_result.unwrap_err()  # AppError from deployment_manager
            if self._callbacks.on_error:
                self._callbacks.on_error("training_start", str(training_err))
            return Err(training_err)

        training_metadata = training_result.unwrap()
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
        return Ok(
            self.update_context(
                context,
                {
                    "run_name": run.name,
                    # Provider info
                    "provider_name": self._provider_name,
                    "provider_type": self._provider.provider_type,  # "local" or "cloud"
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
        if self._provider_name == PROVIDER_RUNPOD and self._ssh_client:
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
        """
        Download training logs from remote server.

        Logs may be in:
        - {workspace}/training.log (docker-only training writes directly into run dir mount)
        - {workspace}/logs/{timestamp}/pipeline.log (optional, if training writes structured logs dir)
        - {workspace}/logs/training.log (legacy/simple path)

        Args:
            reason: Error context for logging
        """
        if not self._ssh_client:
            return

        try:
            workspace = self.deployment.workspace
            # 1) Primary path: docker-only training log in run dir
            primary_log_path = f"{workspace}/training.log"
            log_manager = LogManager(self._ssh_client, remote_path=primary_log_path)
            if log_manager.download(silent=False):
                return

            logs_base = f"{workspace}/logs"

            # Find the latest log directory (run_training creates timestamped dirs)
            success, output, _ = self._ssh_client.exec_command(
                f"ls -1t {logs_base}/ 2>/dev/null | head -1",
                silent=True,
            )

            if success and output.strip():
                log_subdir = output.strip()
                # Check if it's a directory with pipeline.log
                remote_log_path = f"{logs_base}/{log_subdir}/pipeline.log"

                success, content, _ = self._ssh_client.exec_command(
                    f"cat {remote_log_path} 2>/dev/null || echo 'LOG_NOT_FOUND'",
                    silent=True,
                    timeout=10,
                )

                if success and "LOG_NOT_FOUND" not in content:
                    # Save to local logs directory
                    from src.utils.logger import get_run_log_dir

                    local_path = get_run_log_dir() / "training.log"
                    local_path.write_text(content)
                    logger.info(f"📥 Downloaded training log: {local_path} ({len(content):,} bytes)")
                    return

            # Fallback: try simple path
            remote_log_path = f"{logs_base}/training.log"
            log_manager = LogManager(self._ssh_client, remote_path=remote_log_path)
            log_manager.download_on_error(error_context=reason)

        except Exception as e:
            logger.debug(f"[DEPLOYER] Failed to download logs on error: {e}")

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

            if self._provider_name == PROVIDER_RUNPOD and self._ssh_client:
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
