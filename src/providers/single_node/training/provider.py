"""
SingleNode Provider - local PC via SSH.

Implements IGPUProvider for direct SSH connection to local GPU servers.
No cloud API - server is always on and accessible.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from src.constants import PROVIDER_SINGLE_NODE
from src.providers.training.interfaces import (
    GPUInfo,
    IGPUProvider,
    ProviderCapabilities,
    ProviderStatus,
    SSHConnectionInfo,
    TrainingScriptHooks,
)
from src.utils.result import AppError, Err, Ok, ProviderError, Result
from src.utils.ssh_client import SSHClient

from .config import SingleNodeConfig
from .health_check import SingleNodeHealthCheck

_SSH_PORT_DEFAULT = 22
_CLEANUP_TIMEOUT = 1800

if TYPE_CHECKING:
    from src.pipeline.domain import RunContext
    from src.utils.config import Secrets

logger = logging.getLogger("ryotenkai")


class SingleNodeProvider(IGPUProvider):
    """
    GPU provider for local PC via SSH.

    Features:
        - Direct SSH connection (no cloud API)
        - Auto GPU detection via nvidia-smi
        - Health checks before training
        - MemoryManager integration for auto batch_size
        - Run isolation: workspace_path/runs/<run_name>/
        - Cleanup options (like RunPod)

    Lifecycle:
        1. __init__: Parse config
        2. connect(): Establish SSH, create run directory, run health checks
        3. check_gpu(): Get GPU info (optional, done in connect)
        4. disconnect(): Cleanup workspace (optional)

    Workspace structure:
        /home/user/ryotenkai_training/              <- workspace_path (base)
        ├── runs/
        │   ├── run_20260120_123456_abc12/      <- run directory (isolated)
        │   ├── data/
        │   ├── config/
        │   ├── src/
        │   └── output/
        └── runs/run_20260120_124010_k9p0z/     <- another run
    """

    def __init__(self, config: dict[str, Any], secrets: Secrets):
        """
        Initialize SingleNode provider.

        Args:
            config: Provider configuration dict
            secrets: Secrets with credentials
        """
        self._config = SingleNodeConfig.from_dict(config)
        self._secrets = secrets
        self._status = ProviderStatus.AVAILABLE
        self._ssh_client: SSHClient | None = None
        self._gpu_info: GPUInfo | None = None
        self._ssh_connection_info: SSHConnectionInfo | None = None

        # Run directory (created on connect)
        self._run_dir: str | None = None
        self._had_error: bool = False

        if self._config.is_alias_mode:
            logger.info(f"[PROVIDER:INIT] SingleNodeProvider (alias: {self._config.ssh.alias})")
        else:
            logger.info(
                f"[PROVIDER:INIT] SingleNodeProvider: "
                f"{self._config.ssh.user}@{self._config.ssh.host}:{self._config.ssh.port}"
            )

    @property
    def provider_name(self) -> str:
        """Human-readable provider name."""
        return PROVIDER_SINGLE_NODE

    @property
    def provider_type(self) -> str:
        """Provider type: local."""
        return "local"

    def connect(self, *, run: RunContext) -> Result[SSHConnectionInfo, ProviderError]:
        """
        Connect to local PC via SSH.

        Steps:
            1. Create SSH client
            2. Test connection
            3. Run health checks (GPU, Python, disk)
            4. Create isolated run directory
            5. Return connection info

        Returns:
            Ok(SSHConnectionInfo): Connection established
            Err(ProviderError): Error
        """
        if self._status == ProviderStatus.CONNECTED:
            logger.warning("[PROVIDER:CONNECT] Already connected")
            if self._ssh_connection_info:
                return Ok(self._ssh_connection_info)

        self._status = ProviderStatus.CONNECTING

        if self._config.is_alias_mode:
            logger.info(f"[PROVIDER:CONNECT] Connecting via SSH alias '{self._config.ssh.alias}'...")
        else:
            logger.info(f"[PROVIDER:CONNECT] Connecting to {self._config.ssh.host}:{self._config.ssh.port}...")

        try:
            ssh_cfg = self._config.ssh
            connect_settings = ssh_cfg.connect_settings

            # Create SSH client with alias-first fallback to explicit (if configured)
            self._ssh_client = None

            def _mk_alias_client() -> SSHClient:
                return SSHClient(
                    host=str(ssh_cfg.alias),
                    port=int(ssh_cfg.port or _SSH_PORT_DEFAULT),
                    username=None,
                    key_path="",
                    connect_timeout=int(connect_settings.timeout_seconds),
                )

            def _mk_explicit_client() -> SSHClient:
                key_path = self._config.resolve_ssh_key_path_for_client()
                return SSHClient(
                    host=str(ssh_cfg.host),
                    port=int(ssh_cfg.port or _SSH_PORT_DEFAULT),
                    username=str(ssh_cfg.user),
                    key_path=key_path,
                    connect_timeout=int(connect_settings.timeout_seconds),
                )

            used_alias_mode = False
            switched_from_alias_to_explicit = False
            success = False
            error = ""
            if ssh_cfg.alias:
                self._ssh_client = _mk_alias_client()
                used_alias_mode = True

                success, error = self._ssh_client.test_connection(
                    max_retries=connect_settings.max_retries,
                    retry_delay=connect_settings.retry_delay_seconds,
                )
                if not success:
                    # Alias failed, try explicit fallback if configured
                    if ssh_cfg.host and ssh_cfg.user:
                        self._ssh_client = _mk_explicit_client()
                        used_alias_mode = False
                        switched_from_alias_to_explicit = True
                    else:
                        self._ssh_client = None
                        self._status = ProviderStatus.ERROR
                        return Err(
                            ProviderError(
                                message="SSH alias connection failed and no explicit fallback (host+user) is configured",
                                code="SINGLENODE_SSH_ALIAS_FAILED",
                            )
                        )

            if self._ssh_client is None:
                # Explicit-only mode
                self._ssh_client = _mk_explicit_client()
                used_alias_mode = False
                success, error = self._ssh_client.test_connection(
                    max_retries=connect_settings.max_retries,
                    retry_delay=connect_settings.retry_delay_seconds,
                )

            # If we switched from alias to explicit fallback, test explicit now.
            if switched_from_alias_to_explicit and self._ssh_client is not None:
                success, error = self._ssh_client.test_connection(
                    max_retries=connect_settings.max_retries,
                    retry_delay=connect_settings.retry_delay_seconds,
                )

            # Test connection
            if not success:
                self._status = ProviderStatus.ERROR
                return Err(
                    ProviderError(message=f"SSH connection failed: {error}", code="SINGLENODE_SSH_CONNECT_FAILED")
                )

            # Run health checks (docker-only)
            health_checker = SingleNodeHealthCheck(self._ssh_client)
            health_result = health_checker.run_all_checks(
                workspace_path=self._config.workspace_path,
            )

            if not health_result.passed:
                self._status = ProviderStatus.ERROR
                errors = health_result.errors or ["Unknown error"]
                return Err(
                    ProviderError(
                        message=f"Health checks failed: {'; '.join(errors)}",
                        code="SINGLENODE_HEALTH_CHECK_FAILED",
                    )
                )

            # Store GPU info
            self._gpu_info = health_result.gpu_info

            # Preempt inference container to free GPU for training (single-GPU hosts).
            self._preempt_inference_container()

            # Ensure base workspace exists
            if not self._ssh_client.directory_exists(self._config.workspace_path):
                logger.info(f"Creating base workspace: {self._config.workspace_path}")
                success, error = self._ssh_client.create_directory(self._config.workspace_path)
                if not success:
                    self._status = ProviderStatus.ERROR
                    return Err(
                        ProviderError(
                            message=f"Failed to create workspace: {error}",
                            code="SINGLENODE_WORKSPACE_CREATE_FAILED",
                        )
                    )

            # Create isolated run directory
            base = self._config.workspace_path.rstrip("/")
            runs_root = f"{base}/runs"
            self._run_dir = f"{runs_root}/{run.name}"

            logger.info(f"📁 Creating run directory: {self._run_dir}")
            # Ensure /runs exists
            self._ssh_client.create_directory(runs_root)
            success, error = self._ssh_client.create_directory(self._run_dir)
            if not success:
                self._status = ProviderStatus.ERROR
                return Err(
                    ProviderError(
                        message=f"Failed to create run directory: {error}",
                        code="SINGLENODE_RUN_DIR_CREATE_FAILED",
                    )
                )

            # Create subdirectories
            for subdir in ["data", "config", "src", "output", "logs"]:
                subdir_path = f"{self._run_dir}/{subdir}"
                self._ssh_client.create_directory(subdir_path)

            # Build connection info (with run directory as workspace!)
            is_alias = used_alias_mode
            self._ssh_connection_info = SSHConnectionInfo(
                host=self._config.get_ssh_host_for_client(),
                port=self._config.get_ssh_port_for_client(),
                user=str(ssh_cfg.user or ""),  # Empty for alias mode
                key_path=self._ssh_client.key_path,
                workspace_path=self._run_dir,  # Run-specific workspace!
                resource_id=run.name,  # Canonical run name
                is_alias_mode=is_alias,
            )

            self._status = ProviderStatus.CONNECTED
            logger.info(f"[PROVIDER:CONNECTED] {self._ssh_connection_info}")
            logger.info(f"📂 Run directory: {self._run_dir}")

            return Ok(self._ssh_connection_info)

        except Exception as e:
            self._status = ProviderStatus.ERROR
            self._had_error = True
            logger.error(f"[PROVIDER:ERROR] Connection failed: {e}")
            return Err(ProviderError(message=str(e), code="SINGLENODE_CONNECT_UNEXPECTED_ERROR"))

    def _preempt_inference_container(self) -> None:
        """
        Provider-level pre-training hook.

        Stop the inference container to free GPU VRAM before training starts.
        This is intentionally implemented in the provider (not in deployment manager),
        because it is a host-specific policy decision.
        """
        if not self._ssh_client:
            return

        container_name = "ryotenkai-inference-vllm"
        try:
            ok, stdout, _stderr = self._ssh_client.exec_command(
                command=f"docker ps -q -f name={container_name} -f status=running",
                timeout=10,
                silent=True,
            )
            is_running = bool(ok and stdout.strip())
            if not is_running:
                return

            logger.warning(f"[PROVIDER:PREEMPT] Stopping inference container before training: {container_name}")
            self._ssh_client.exec_command(
                command=f"docker rm -f {container_name} >/dev/null 2>&1 || true",
                timeout=60,
                silent=False,
            )
            time.sleep(2)
        except Exception as e:
            logger.debug(f"[PROVIDER] Failed to inspect/stop inference container (best-effort): {e}")

    def disconnect(self) -> Result[None, ProviderError]:
        """
        Disconnect from local PC.

        Cleanup behavior (configurable):
            - cleanup_workspace=True: Delete run directory
            - keep_on_error=True: Keep workspace if there was an error

        Returns:
            Ok(None): Disconnected
            Err(ProviderError): Error during cleanup
        """
        if self._status not in (ProviderStatus.CONNECTED, ProviderStatus.ERROR):
            logger.debug("[PROVIDER:DISCONNECT] Not connected, nothing to do")
            return Ok(None)

        self._status = ProviderStatus.DISCONNECTING
        logger.info("[PROVIDER:DISCONNECT] Disconnecting...")

        # Determine if we should cleanup
        should_cleanup = self._config.cleanup_workspace

        if self._had_error and self._config.keep_on_error:
            should_cleanup = False
            logger.warning(f"[PROVIDER:DISCONNECT] Keeping run directory due to error: {self._run_dir}")

        # Cleanup run directory if configured
        # Use docker to remove files created by root inside container
        if should_cleanup and self._run_dir and self._ssh_client:
            logger.info(f"🧹 Cleaning up run directory: {self._run_dir}")

            # Step 1: Use docker to remove root-owned files (like .cache)
            # Mount parent dir and remove the run folder from inside
            parent_dir = "/".join(self._run_dir.rstrip("/").split("/")[:-1])
            run_name = self._run_dir.rstrip("/").split("/")[-1]

            docker_cleanup = f"docker run --rm -v {parent_dir}:/parent alpine rm -rf /parent/{run_name}"
            success, _, stderr = self._ssh_client.exec_command(docker_cleanup, timeout=_CLEANUP_TIMEOUT)

            if not success:
                logger.warning(f"Docker cleanup failed: {stderr[:100] if stderr else 'unknown'}")
                # Fallback: try regular rm (might work for user-owned files)
                self._ssh_client.exec_command(f"rm -rf {self._run_dir}", timeout=_CLEANUP_TIMEOUT)

            # Verify cleanup
            verify_cmd = f"test -d {self._run_dir} && echo EXISTS || echo DELETED"
            _, stdout, _ = self._ssh_client.exec_command(verify_cmd)

            if "DELETED" in stdout:
                logger.info("✅ Run directory cleaned up")
            else:
                logger.warning(f"⚠️ Run directory still exists: {self._run_dir}")
        elif self._run_dir:
            logger.info(f"📂 Keeping run directory: {self._run_dir}")

        # Clear references
        self._ssh_client = None
        self._run_dir = None
        self._had_error = False

        self._status = ProviderStatus.AVAILABLE
        logger.info("[PROVIDER:DISCONNECTED] Connection closed, server is still running")

        return Ok(None)

    def mark_error(self) -> None:
        """
        Mark that an error occurred during training.

        This affects cleanup behavior if keep_on_error=True.
        """
        self._had_error = True
        self._status = ProviderStatus.ERROR
        logger.warning("[PROVIDER:ERROR] Marked as error state")

    def get_status(self) -> ProviderStatus:
        """Get current provider status."""
        return self._status

    def check_gpu(self) -> Result[GPUInfo, ProviderError]:
        """
        Check GPU availability.

        If already connected, returns cached GPU info.
        Otherwise, attempts to connect first.

        Returns:
            Ok(GPUInfo): GPU information
            Err(ProviderError): Error
        """
        # Return cached info if available
        if self._gpu_info:
            return Ok(self._gpu_info)

        # Must be connected
        if self._status != ProviderStatus.CONNECTED or not self._ssh_client:
            return Err(ProviderError(message="Not connected. Call connect() first.", code="SINGLENODE_NOT_CONNECTED"))

        # Run GPU check
        health_checker = SingleNodeHealthCheck(self._ssh_client)
        return health_checker.check_gpu()

    def get_capabilities(self) -> ProviderCapabilities:
        """
        Get provider capabilities.

        Returns capabilities based on:
            - Static config (provider type)
            - Detected GPU info (if connected)
        """
        gpu_name = None
        gpu_vram_gb = None

        if self._gpu_info:
            gpu_name = self._gpu_info.name
            gpu_vram_gb = self._gpu_info.vram_total_gb
        elif self._config.gpu_type:
            gpu_name = self._config.gpu_type

        return ProviderCapabilities(
            provider_type="local",
            supports_multi_gpu=False,  # Single node = single GPU typically
            supports_spot_instances=False,  # Not cloud
            max_runtime_hours=None,  # Unlimited for local
            gpu_name=gpu_name,
            gpu_vram_gb=gpu_vram_gb,
        )

    def get_resource_info(self) -> None:
        """Local provider has no dynamic resource metadata."""
        return None

    def prepare_training_script_hooks(
        self,
        ssh_client: SSHClient,
        context: dict[str, Any],
    ) -> Result[TrainingScriptHooks, AppError]:
        """Single-node has no cloud lifecycle concerns — empty hooks."""
        del ssh_client, context
        return Ok(TrainingScriptHooks.empty())

    def get_ssh_client(self) -> SSHClient | None:
        """Get SSH client for direct operations."""
        return self._ssh_client

    def get_run_dir(self) -> str | None:
        """Get current run directory path."""
        return self._run_dir

    def get_base_workspace(self) -> str:
        """Get base workspace path (without run directory)."""
        return self._config.workspace_path

    def __repr__(self) -> str:
        status = self._status.value
        run_dir = self._run_dir or "no-run"
        return f"SingleNodeProvider({self._config.get_ssh_host_for_client()}, run={run_dir}, status={status})"
