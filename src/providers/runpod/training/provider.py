"""
RunPod Provider - cloud GPU via RunPod API.

Implements IGPUProvider for RunPod cloud instances.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.constants import PROVIDER_RUNPOD
from src.providers.training.interfaces import (
    GPUInfo,
    IGPUProvider,
    ProviderCapabilities,
    ProviderStatus,
    SSHConnectionInfo,
)
from src.utils.result import Err, Ok, ProviderError, Result
from src.utils.ssh_client import SSHClient

from ..pod_control import RunPodTrainingPodControl
from .api_client import RunPodAPIClient
from .cleanup_manager import RunPodCleanupManager
from .config import RunPodProviderConfig
from .constants import RUNPOD_API_BASE_URL
from .lifecycle_manager import PodLifecycleManager

_GPU_CHECK_TIMEOUT = 20
_SSH_RETRIES = 12
_SSH_RETRY_DELAY = 10
_GPU_CHECK_FAILED_CODE = "GPU_CHECK_FAILED"

if TYPE_CHECKING:
    from src.pipeline.domain import RunContext
    from src.utils.config import Secrets

logger = logging.getLogger("ryotenkai")


class RunPodProvider(IGPUProvider):
    """
    GPU provider for RunPod cloud.

    Features:
        - Creates pods via GraphQL API
        - Waits for pod to be ready
        - Provides SSH connection info
        - Automatic cleanup on disconnect

    Lifecycle:
        1. __init__: Parse config, initialize API client
        2. connect(): Create pod, wait for ready, return SSH info
        3. check_gpu(): Query pod GPU info
        4. disconnect(): Terminate pod (if cleanup.auto_delete_pod)
    """

    def __init__(self, config: dict[str, Any], secrets: Secrets):
        """
        Initialize RunPod provider.

        Args:
            config: Provider configuration dict
            secrets: Secrets with RunPod API key
        """
        self._config = RunPodProviderConfig.from_dict(config)
        self._secrets = secrets
        self._status = ProviderStatus.AVAILABLE
        self._pod_id: str | None = None
        self._ssh_connection_info: SSHConnectionInfo | None = None
        self._gpu_info: GPUInfo | None = None
        self._pod_info: dict[str, Any] | None = None
        self._had_error: bool = False

        api_key = secrets.runpod_api_key
        if not api_key:
            raise ValueError("secrets.runpod_api_key is required to use RunPodProvider")
        self._api_key: str = api_key

        self._graphql_api_client = RunPodAPIClient(
            api_base_url=RUNPOD_API_BASE_URL,
            api_key=self._api_key,
        )
        self._api_client = RunPodTrainingPodControl(api=self._graphql_api_client)

        self._cleanup_manager = RunPodCleanupManager(self._api_client)

        self._lifecycle = PodLifecycleManager(
            api_client=self._api_client,
            cleanup_manager=self._cleanup_manager,
        )

        logger.info(
            f"[PROVIDER:INIT] RunPodProvider initialized: "
            f"GPU={self._config.training.gpu_type}, image={self._config.training.image_name}"
        )

    @property
    def provider_name(self) -> str:
        """Human-readable provider name."""
        return PROVIDER_RUNPOD

    @property
    def provider_type(self) -> str:
        """Provider type: cloud."""
        return "cloud"

    def connect(self, *, run: RunContext) -> Result[SSHConnectionInfo, ProviderError]:
        """
        Connect to RunPod by creating a new pod.

        Steps:
            1. Create pod via API
            2. Wait for pod to be ready
            3. Wait for SSH to be ready
            4. Return connection info

        Returns:
            Ok(SSHConnectionInfo): Connection established
            Err(ProviderError): Structured provider error
        """
        if self._status == ProviderStatus.CONNECTED and self._ssh_connection_info:
            logger.warning("[PROVIDER:CONNECT] Already connected")
            return Ok(self._ssh_connection_info)

        self._status = ProviderStatus.CONNECTING
        self._had_error = False
        logger.info(f"[PROVIDER:CONNECT] Creating RunPod with {self._config.training.gpu_type}...")

        try:
            # Step 1: Create pod (using RunPodProviderConfig directly!)
            pod_result = self._api_client.create_pod(
                config=self._config,
                pod_name=f"ryotenkai-train-{run.name}",
            )

            if pod_result.is_failure():
                self._status = ProviderStatus.ERROR
                return Err(pod_result.unwrap_err())  # type: ignore[union-attr]

            self._pod_info = pod_result.unwrap()

            if self._pod_info is None or "pod_id" not in self._pod_info:
                self._status = ProviderStatus.ERROR
                return Err(ProviderError(message="Invalid pod info returned", code="RUNPOD_INVALID_POD_INFO"))

            pod_id_raw = self._pod_info.get("pod_id")
            if not isinstance(pod_id_raw, str) or not pod_id_raw:
                self._status = ProviderStatus.ERROR
                return Err(
                    ProviderError(message="Invalid pod info returned (pod_id missing)", code="RUNPOD_INVALID_POD_INFO")
                )

            pod_id = pod_id_raw
            self._pod_id = pod_id

            # Register for cleanup
            self._cleanup_manager.register_pod(pod_id=pod_id, api_base=RUNPOD_API_BASE_URL)

            logger.info(f"✅ Pod created: {pod_id}")

            # Step 2: Wait for pod to be ready (returns typed PodSnapshot with ssh_endpoint).
            ready_result = self._lifecycle.wait_for_ready(pod_id)
            if ready_result.is_failure():
                self._status = ProviderStatus.ERROR
                self._cleanup_manager.cleanup_pod(pod_id)
                return Err(ready_result.unwrap_err())  # type: ignore[union-attr]

            snapshot = ready_result.unwrap()
            logger.info("✅ Pod is ready!")

            # Step 3: Build SSH connection target from the typed snapshot.
            key_path = str(Path(self._config.connect.ssh.key_path).expanduser())
            if not Path(key_path).exists():
                self._status = ProviderStatus.ERROR
                self._cleanup_manager.cleanup_pod(pod_id)
                return Err(
                    ProviderError(
                        message=f"SSH key not found at: {key_path}",
                        code="SSH_KEY_NOT_FOUND",
                        details={"key_path": key_path},
                    )
                )

            ssh_ep = snapshot.ssh_endpoint
            if ssh_ep is None:
                self._status = ProviderStatus.ERROR
                self._cleanup_manager.cleanup_pod(pod_id)
                return Err(
                    ProviderError(
                        message="Pod reported ready but SSH endpoint is missing",
                        code="RUNPOD_SSH_INFO_INVALID",
                    )
                )

            ssh_client = SSHClient(host=ssh_ep.host, port=ssh_ep.port, username="root", key_path=key_path)

            # Step 4: Wait for SSH to be ready
            success, error = ssh_client.test_connection(max_retries=_SSH_RETRIES, retry_delay=_SSH_RETRY_DELAY)
            if not success:
                self._status = ProviderStatus.ERROR
                self._cleanup_manager.cleanup_pod(pod_id)
                return Err(ProviderError(message=f"SSH connection failed: {error}", code="SSH_CONNECTION_FAILED"))

            logger.info("✅ SSH is ready!")

            # Minimal training health check (fail-fast).
            gpu_check = self._check_gpu_via_ssh(ssh_client)
            if gpu_check.is_err():
                self._status = ProviderStatus.ERROR
                self._cleanup_manager.cleanup_pod(pod_id)
                return Err(gpu_check.unwrap_err())  # type: ignore[union-attr]
            self._gpu_info = gpu_check.unwrap()

            # Create run-scoped workspace inside the pod.
            # Canonical layout: /workspace/runs/<run_name>/
            base = "/workspace"
            runs_root = f"{base}/runs"
            run_workspace = f"{runs_root}/{run.name}"
            ok_root, err_root = ssh_client.create_directory(runs_root)
            if not ok_root:
                self._status = ProviderStatus.ERROR
                self._cleanup_manager.cleanup_pod(pod_id)
                return Err(
                    ProviderError(message=f"Failed to create runs root on pod: {err_root}", code="SSH_MKDIR_FAILED")
                )

            ok_run, err_run = ssh_client.create_directory(run_workspace)
            if not ok_run:
                self._status = ProviderStatus.ERROR
                self._cleanup_manager.cleanup_pod(pod_id)
                return Err(
                    ProviderError(message=f"Failed to create run workspace on pod: {err_run}", code="SSH_MKDIR_FAILED")
                )

            self._ssh_connection_info = SSHConnectionInfo(
                host=ssh_ep.host,
                port=ssh_ep.port,
                user="root",
                key_path=key_path,
                workspace_path=run_workspace,
                resource_id=pod_id,
            )

            self._status = ProviderStatus.CONNECTED
            logger.info(f"[PROVIDER:CONNECTED] {self._ssh_connection_info}")

            return Ok(self._ssh_connection_info)

        except KeyboardInterrupt:
            # SIGINT arrived mid-connect (most likely during create_pod HTTP request
            # or wait_for_ready polling). If a pod was already created, terminate it
            # immediately before re-raising so the pod is not left orphaned.
            self._status = ProviderStatus.ERROR
            if self._pod_id:
                logger.warning(f"[PROVIDER:CONNECT] SIGINT during connect — terminating pod {self._pod_id}")
                self._cleanup_manager.cleanup_pod(self._pod_id)
                self._pod_id = None
            else:
                logger.warning("[PROVIDER:CONNECT] SIGINT during connect — no pod to clean up")
            raise

        except Exception as e:
            self._status = ProviderStatus.ERROR
            self._had_error = True
            if self._pod_id:
                self._cleanup_manager.cleanup_pod(self._pod_id)
            logger.error(f"[PROVIDER:ERROR] Connection failed: {e}")
            return Err(
                ProviderError(
                    message=str(e), code="RUNPOD_CONNECT_UNEXPECTED_ERROR", details={"exception_type": type(e).__name__}
                )
            )

    def mark_error(self) -> None:
        """
        Mark that an error occurred during pipeline execution after connect().

        This is used by GPUDeployer to ensure provider-level keep-on-error policies
        (like keep_pod_on_error) are applied on disconnect.
        """
        self._had_error = True
        self._status = ProviderStatus.ERROR
        logger.warning("[PROVIDER:ERROR] Marked as error state")

    def disconnect(self) -> Result[None, ProviderError]:
        """
        Disconnect from RunPod by terminating the pod.

        Behavior depends on config:
            - cleanup.auto_delete_pod=True: Terminate pod
            - cleanup.auto_delete_pod=False: Keep pod running
            - cleanup.keep_pod_on_error=True: Keep pod if there was an error

        Returns:
            Ok(None): Disconnected successfully
            Err(ProviderError): Structured provider error
        """
        # Allow cleanup from CONNECTING state when pod_id is already set.
        # This handles SIGINT arriving mid-connect (e.g. during wait_for_ready):
        # status stays CONNECTING but the pod was already created and billed.
        connecting_with_pod = self._status == ProviderStatus.CONNECTING and bool(self._pod_id)
        if self._status not in (ProviderStatus.CONNECTED, ProviderStatus.ERROR) and not connecting_with_pod:
            logger.debug("[PROVIDER:DISCONNECT] Not connected, nothing to do")
            return Ok(None)

        was_error = self._had_error or self._status == ProviderStatus.ERROR
        self._status = ProviderStatus.DISCONNECTING

        if not self._pod_id:
            self._status = ProviderStatus.AVAILABLE
            return Ok(None)

        # Check if we should keep the pod
        should_terminate = self._config.cleanup.auto_delete_pod

        if was_error and self._config.cleanup.keep_pod_on_error:
            should_terminate = False
            logger.warning(f"[PROVIDER:DISCONNECT] Keeping pod {self._pod_id} due to error (keep_pod_on_error=True)")

        if should_terminate:
            logger.info(f"[PROVIDER:DISCONNECT] Terminating pod {self._pod_id}...")
            self._cleanup_manager.cleanup_pod(self._pod_id)
            logger.info("✅ Pod terminated")
        else:
            logger.info(
                f"[PROVIDER:DISCONNECT] Keeping pod {self._pod_id} running "
                f"(auto_delete_pod={self._config.cleanup.auto_delete_pod}, "
                f"keep_pod_on_error={self._config.cleanup.keep_pod_on_error})"
            )

        self._pod_id = None
        self._ssh_connection_info = None
        self._had_error = False
        self._status = ProviderStatus.AVAILABLE

        return Ok(None)

    def get_status(self) -> ProviderStatus:
        """Get current provider status."""
        return self._status

    @staticmethod
    def _check_gpu_via_ssh(ssh_client: SSHClient) -> Result[GPUInfo, ProviderError]:
        cmd = "nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv,noheader,nounits"
        success, stdout, stderr = ssh_client.exec_command(cmd, timeout=_GPU_CHECK_TIMEOUT, silent=True)
        if not success:
            stderr = (stderr or "").strip()
            stdout = (stdout or "").strip()
            details = stderr or stdout or "unknown error"
            return Err(
                ProviderError(
                    message=f"Training health check failed: nvidia-smi error: {details}", code=_GPU_CHECK_FAILED_CODE
                )
            )

        lines = [ln.strip() for ln in (stdout or "").splitlines() if ln.strip()]
        if not lines:
            return Err(
                ProviderError(
                    message="Training health check failed: nvidia-smi returned empty output",
                    code=_GPU_CHECK_FAILED_CODE,
                )
            )

        # Use the first GPU line for detailed metrics; also infer GPU count.
        first = lines[0]
        parts = [p.strip() for p in first.split(",")]
        if len(parts) < 4:
            return Err(
                ProviderError(
                    message=f"Training health check failed: unexpected nvidia-smi format: {first}",
                    code=_GPU_CHECK_FAILED_CODE,
                )
            )

        try:
            return Ok(
                GPUInfo(
                    name=parts[0],
                    vram_total_mb=int(float(parts[1])),
                    vram_free_mb=int(float(parts[2])),
                    cuda_version="unknown",
                    driver_version=parts[3],
                    gpu_count=len(lines),
                )
            )
        except (TypeError, ValueError) as e:
            return Err(
                ProviderError(
                    message=f"Training health check failed: failed to parse nvidia-smi output: {e}",
                    code=_GPU_CHECK_FAILED_CODE,
                )
            )

    def check_gpu(self) -> Result[GPUInfo, ProviderError]:
        """Check GPU availability on the pod."""
        if self._gpu_info:
            return Ok(self._gpu_info)

        if self._status != ProviderStatus.CONNECTED or not self._ssh_connection_info:
            return Err(ProviderError(message="Not connected. Call connect() first.", code="PROVIDER_NOT_CONNECTED"))

        ssh_client = SSHClient(
            host=self._ssh_connection_info.host,
            port=self._ssh_connection_info.port,
            username=self._ssh_connection_info.user,
        )
        gpu_check = self._check_gpu_via_ssh(ssh_client)
        if gpu_check.is_err():
            return Err(gpu_check.unwrap_err())  # type: ignore[union-attr]
        self._gpu_info = gpu_check.unwrap()
        return Ok(self._gpu_info)

    def get_capabilities(self) -> ProviderCapabilities:
        """Get provider capabilities."""
        gpu_name = self._config.training.gpu_type
        gpu_vram_gb = None

        if self._gpu_info:
            gpu_name = self._gpu_info.name
            gpu_vram_gb = self._gpu_info.vram_total_gb

        return ProviderCapabilities(
            provider_type="cloud",
            supports_multi_gpu=True,
            supports_spot_instances=True,
            max_runtime_hours=None,
            gpu_name=gpu_name,
            gpu_vram_gb=gpu_vram_gb,
        )

    def get_pod_id(self) -> str | None:
        """Get current pod ID."""
        return self._pod_id

    def get_resource_info(self) -> dict[str, Any] | None:
        """Return RunPod instance metadata (cost_per_hr, gpu_type, gpu_count) after connect()."""
        return self._pod_info

    def get_base_workspace(self) -> str:
        """Base workspace root inside pod (for shared venv/caches)."""
        return "/workspace"

    def __repr__(self) -> str:
        status = self._status.value
        pod_id = self._pod_id or "no-pod"
        return f"RunPodProvider({self._config.training.gpu_type}, pod={pod_id}, status={status})"
