"""
GPU Provider Interfaces - Protocol and data classes.

This module defines the contract that all GPU providers must implement.

Design principles:
    - Protocol-based (structural typing, no inheritance required)
    - Provider-agnostic data classes
    - Clear separation: cloud vs local providers
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from src.pipeline.domain import RunContext
    from src.providers.runpod.models import PodResourceInfo
    from src.utils.result import AppError, ProviderError, Result
    from src.utils.ssh_client import SSHClient


class ProviderStatus(Enum):
    """Provider lifecycle states."""

    AVAILABLE = "available"  # Ready to connect (local: online, cloud: can create)
    CONNECTING = "connecting"  # Establishing connection (cloud: creating instance)
    CONNECTED = "connected"  # SSH session active, ready for operations
    DISCONNECTING = "disconnecting"  # Cleaning up (cloud: terminating instance)
    UNAVAILABLE = "unavailable"  # Offline or error state
    ERROR = "error"  # Fatal error, requires manual intervention


@dataclass(frozen=True)
class SSHConnectionInfo:
    """
    SSH connection details - returned by provider.connect().

    Immutable to prevent accidental modification of connection state.
    """

    host: str
    port: int
    user: str
    key_path: str
    workspace_path: str  # Remote workspace: /workspace (cloud) or /home/user/xxx (local)
    resource_id: str = ""  # Provider-specific ID (pod_id for RunPod, run_dir for SingleNode)
    is_alias_mode: bool = False  # True if host is SSH alias (user/key from ~/.ssh/config)

    def __repr__(self) -> str:
        if self.is_alias_mode:
            return f"SSH(alias:{self.host} → {self.workspace_path})"
        return f"SSH({self.user}@{self.host}:{self.port} → {self.workspace_path})"


@dataclass
class GPUInfo:
    """
    GPU information from nvidia-smi.

    Used for:
        - Logging and monitoring
        - Validating GPU availability
        - MemoryManager auto-configuration
    """

    name: str  # "NVIDIA GeForce RTX 4060"
    vram_total_mb: int  # 8188
    vram_free_mb: int  # 7500
    cuda_version: str  # "12.9"
    driver_version: str  # "575.64.03"
    gpu_count: int = 1  # Number of GPUs
    compute_capability: str = ""  # "8.9" (for compatibility checks)

    @property
    def vram_total_gb(self) -> float:
        """Total VRAM in GB."""
        return self.vram_total_mb / 1024

    @property
    def vram_free_gb(self) -> float:
        """Free VRAM in GB."""
        return self.vram_free_mb / 1024

    @property
    def vram_used_percent(self) -> float:
        """VRAM usage percentage."""
        if self.vram_total_mb == 0:
            return 0.0
        return (1 - self.vram_free_mb / self.vram_total_mb) * 100


@dataclass
class ProviderCapabilities:
    """
    Provider capabilities and constraints.

    Used by orchestrator to validate training configs.
    """

    provider_type: str  # "local" or "cloud"
    supports_multi_gpu: bool = False
    supports_spot_instances: bool = False  # Cloud only
    max_runtime_hours: int | None = None  # None = unlimited

    # Constraints (from GPU info or config)
    gpu_name: str | None = None
    gpu_vram_gb: float | None = None


@dataclass(frozen=True)
class TrainingScriptHooks:
    """
    Provider-specific customizations for the training launch script.

    Returned by ``IGPUProvider.prepare_training_script_hooks`` so the generic
    ``TrainingDeploymentManager`` can stay provider-agnostic — it simply
    merges env vars into ``.env`` and splices bash snippets around the
    Python invocation inside the generated ``start_training.sh``.

    Attributes:
        env_vars: Extra environment variables to append to the pod's .env file
            (e.g., API keys, pod IDs). Merged AFTER generic env vars so
            provider values take precedence.
        pre_python: Bash code injected **before** the Python training process
            is launched. Typical use: spawn detached sidecar processes
            (watchdog, telemetry). Must not block — sidecars should be
            detached via ``setsid nohup ... & disown``.
        post_python: Bash code injected **after** ``exit_code=$?`` capture.
            Has access to ``$exit_code``. Typical use: graceful provider
            cleanup (stop pod, release resources).
    """

    env_vars: dict[str, str] = field(default_factory=dict)
    pre_python: str = ""
    post_python: str = ""

    @classmethod
    def empty(cls) -> TrainingScriptHooks:
        """Return hooks with no customizations — default for providers with nothing to inject."""
        return cls()


@runtime_checkable
class IGPUProvider(Protocol):
    """
    Unified interface for GPU providers.

    Supports both:
        - Cloud providers (RunPod): create/terminate instances dynamically
        - Local providers (single_node): always-on servers via SSH

    Lifecycle:
        1. connect() → SSHConnectionInfo
        2. check_gpu() → GPUInfo (optional, for validation)
        3. ... deploy training via SSH ...
        4. disconnect() → cleanup

    Example:
        provider = GPUProviderFactory.create(config, secrets)

        result = provider.connect()
        if result.is_err():
            logger.error(f"Connection failed: {result.unwrap_err()}")
            return

        ssh_info = result.unwrap()
        # Use ssh_info.host, ssh_info.port, etc.

        # When done:
        provider.disconnect()
    """

    @property
    def provider_name(self) -> str:
        """Human-readable provider name (e.g., 'single_node', 'runpod')."""
        ...

    @property
    def provider_type(self) -> str:
        """Provider type: 'local' or 'cloud'."""
        ...

    def connect(self, *, run: RunContext) -> Result[SSHConnectionInfo, ProviderError]:
        """
        Connect to GPU server.

        For cloud providers:
            - Creates instance via API
            - Waits for instance to be ready
            - Returns SSH connection info

        For local providers:
            - Verifies server is reachable
            - Returns SSH connection info

        Returns:
            Ok(SSHConnectionInfo): Connection established
            Err(ProviderError): Structured provider error
        """
        ...

    def disconnect(self) -> Result[None, ProviderError]:
        """
        Disconnect from GPU server.

        For cloud providers:
            - Terminates instance
            - Cleans up resources

        For local providers:
            - No-op (server stays on)
            - Just marks provider as disconnected

        Returns:
            Ok(None): Disconnected successfully
            Err(ProviderError): Structured provider error (non-fatal for local)
        """
        ...

    def get_status(self) -> ProviderStatus:
        """
        Get current provider status.

        Returns:
            ProviderStatus enum value
        """
        ...

    def check_gpu(self) -> Result[GPUInfo, ProviderError]:
        """
        Check GPU availability and specs via nvidia-smi.

        Should be called after connect() to validate GPU.

        Returns:
            Ok(GPUInfo): GPU information
            Err(ProviderError): Structured provider error (GPU not found, nvidia-smi failed, etc.)
        """
        ...

    def get_capabilities(self) -> ProviderCapabilities:
        """
        Get provider capabilities and constraints.

        Returns:
            ProviderCapabilities with provider info
        """
        ...

    def prepare_training_script_hooks(
        self,
        ssh_client: SSHClient,
        context: dict[str, Any],
    ) -> Result[TrainingScriptHooks, AppError]:
        """
        Prepare provider-specific customizations for the training launch script.

        Called by ``TrainingDeploymentManager`` after SSH is connected and
        before ``start_training.sh`` is generated. Providers may:
            - Upload auxiliary scripts to the pod (e.g., watchdog, stop helpers).
            - Return env vars to merge into the ``.env`` file.
            - Return bash snippets to inject before/after the Python invocation.

        Default behavior (for providers with nothing to contribute): return
        ``Ok(TrainingScriptHooks.empty())``.

        Args:
            ssh_client: Connected SSH client to the pod/server.
            context: Pipeline context dict (may include ``resource_id``,
                ``workspace``, etc.).

        Returns:
            Ok(TrainingScriptHooks): Customizations to apply.
            Err(AppError): Upload or configuration failure — deployment
                manager will abort training launch.
        """
        ...

    def get_resource_info(self) -> PodResourceInfo | None:
        """
        Return provider resource metadata if available.

        Cloud providers (RunPod) return a ``PodResourceInfo`` with cost_per_hr,
        gpu_type, gpu_count and other instance-level details populated after
        connect().

        Local providers (single_node) return None — no dynamic resource info.
        """
        ...


# Type alias for provider factory
ProviderFactory = type[IGPUProvider]
