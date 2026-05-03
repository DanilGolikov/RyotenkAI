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
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

if TYPE_CHECKING:
    from ryotenkai_shared.pipeline_context import RunContext
    from ryotenkai_providers.runpod.models import PodResourceInfo
    from ryotenkai_shared.utils.pod_layout import PodLayout
    from ryotenkai_shared.utils.result import AppError, ProviderError, Result
    from ryotenkai_shared.utils.ssh_client import SSHClient


class ProviderStatus(Enum):
    """Provider lifecycle states."""

    AVAILABLE = "available"  # Ready to connect (local: online, cloud: can create)
    CONNECTING = "connecting"  # Establishing connection (cloud: creating instance)
    CONNECTED = "connected"  # SSH session active, ready for operations
    DISCONNECTING = "disconnecting"  # Cleaning up (cloud: terminating instance)
    UNAVAILABLE = "unavailable"  # Offline or error state
    ERROR = "error"  # Fatal error, requires manual intervention


# ---------------------------------------------------------------------------
# Phase 14.A — provider capability abstraction
# ---------------------------------------------------------------------------


class VolumeKind(str, Enum):
    """Storage semantics for the provider's pod/host workspace.

    Phase 14.A introduces this enum so the runner's terminal-hook
    decision matrix (and downstream phases) can ask the provider
    "is your workspace stoppable?" without parsing strings.

    String values intentionally match the legacy ``RUNPOD_VOLUME_KIND``
    env var values so the env-boundary translation in Phase 14.D is
    simply ``VolumeKind(env_str)`` — no extra mapping table.
    """

    PERSISTENT = "persistent"
    """Cloud pod with persistent volume — stoppable, ``/workspace``
    survives podStop, recoverable via podResume. RunPod default."""

    NETWORK = "network"
    """Cloud pod with a network volume — terminate-only per RunPod
    constraint (network-volume pods cannot be stopped). The
    PodTerminator falls back to ``podTerminate`` regardless of
    natural-completion vs failure when this is the volume kind."""

    LOCAL_DISK = "local_disk"
    """Local host (single_node) — no cloud volume semantics. The
    provider's ``disconnect()`` handles workspace cleanup; the
    PodTerminator decision matrix returns SKIPPED for any cloud
    lifecycle action."""


@dataclass(frozen=True)
class AvailabilityVerdict:
    """Outcome of probing a provider for pod/host availability.

    Phase 14.A. Returned by :meth:`IGPUProvider.probe_availability`.
    Used by future :class:`LaunchResumeService` (Phase 14.C) to decide
    whether to wake a sleeping pod, restart-from-checkpoint a gone
    pod, or just continue (already running).

    Single_node always returns ``state="running"`` — the host is
    always reachable; if it isn't, the SSH connect step in the
    pipeline surfaces the real error.
    """

    state: Literal[
        "running",
        "sleeping_resumable",
        "gone",
        "probe_failed",
        "unknown",
    ]
    """Provider-agnostic availability bucket.

    * ``running`` — ready to accept work.
    * ``sleeping_resumable`` — paused but recoverable (RunPod
      ``EXITED`` / ``STOPPED`` / ``PAUSED``).
    * ``gone`` — terminated; needs fresh-pod resume.
    * ``probe_failed`` — transient probe error; caller decides.
    * ``unknown`` — provider doesn't track availability.
    """

    resource_id: str
    """Provider-specific resource identifier the verdict refers to.
    Empty string is acceptable (single_node has no resource_id)."""

    raw_status: str | None = None
    """Provider-native status string for logs (e.g. RunPod's
    ``desiredStatus``). ``None`` when the provider doesn't have one
    or the probe couldn't reach the backend."""

    message: str = ""
    """Human-readable hint surfaced to operators in CLI / Web UI."""


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

    Phase 14.A: defaults safe — keyword-only construction is the only
    used pattern across the codebase (verified via audit), so adding
    fields with defaults does NOT break existing callers.
    """

    provider_type: str  # "local" or "cloud"
    supports_multi_gpu: bool = False
    supports_spot_instances: bool = False  # Cloud only
    max_runtime_hours: int | None = None  # None = unlimited

    # Constraints (from GPU info or config)
    gpu_name: str | None = None
    gpu_vram_gb: float | None = None

    # ---- Phase 14.A — capability surface for the multi-provider refactor ----

    supports_lifecycle_actions: bool = False
    """True iff the provider implements :class:`ITerminalActionProvider`.

    Two-source-of-truth invariant (Phase 14.A): this flag MUST equal
    ``isinstance(provider, ITerminalActionProvider)``. Enforced by a
    factory-level runtime assertion at boot. Failing assert =
    blocker (provider author forgot to update one of the two)."""

    volume_kind: VolumeKind = VolumeKind.PERSISTENT
    """Storage semantics — drives the PodTerminator decision matrix
    (Phase 14.B) and the launcher env builder (Phase 14.D).
    Defaults to PERSISTENT because that's the RunPod default; provider
    impls override (single_node = LOCAL_DISK)."""

    has_pause_resume: bool = False
    """Subset of :attr:`supports_lifecycle_actions`: True iff the
    provider supports the FULL pause→resume cycle (not just terminate).
    Single_node = False, RunPod = True. A future provider with
    terminate-only semantics would have ``supports_lifecycle_actions=True``
    but ``has_pause_resume=False``."""

    runner_workspace_root: str = "/workspace"
    """What ``HELIX_WORKSPACE`` / ``PYTHONPATH`` resolve to inside the
    in-pod runner. Both RunPod and single_node currently use
    ``/workspace`` (same value, but coming from the provider instead
    of hardcoded in :func:`_build_job_env`). A future provider that
    mounts a different path (e.g. ``/data``) only needs to override
    this field — no edits to the launcher."""

    # ---- Phase 14.D+F — provider-leak elimination ----

    is_local: bool = False
    """True for providers that run on a local always-on host
    (single_node). False for cloud providers (RunPod). Replaces the
    pre-14.D :func:`is_single_node_provider` string-check helper —
    callers gate on this flag instead of comparing
    ``provider_name == "single_node"``."""

    supports_log_download: bool = False
    """True iff the provider exposes a structured log-download path
    (cloud providers SCP/HTTP-fetch; local hosts read directly).
    Replaces the pre-14.F ``provider == PROVIDER_RUNPOD`` checks in
    :class:`GPUDeployer`. Single_node = False (logs already on
    host filesystem); RunPod = True."""


@dataclass(frozen=True)
class TrainingScriptHooks:
    """Provider-specific env vars forwarded to the trainer subprocess.

    Phase 6.5 simplification: previously this dataclass also carried
    ``pre_python`` / ``post_python`` bash snippets that the legacy
    launcher spliced into a generated ``start_training.sh``. After
    Phase 6.3 the launcher is gone (the in-pod runner owns trainer
    spawn) and the bash injection points have no caller — so the
    fields are removed. Only ``env_vars`` survives, threading
    provider-supplied values (RunPod auto-stop credentials, etc.)
    into the JobSpec the launcher submits to the runner.

    Attributes:
        env_vars: Extra environment variables merged AFTER the
            generic launcher env so provider values take precedence.
    """

    env_vars: dict[str, str] = field(default_factory=dict)

    @classmethod
    def empty(cls) -> TrainingScriptHooks:
        """Return hooks with no env contribution."""
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

    # ---- Phase 14.A — capability methods ----

    def required_runtime_env_vars(
        self, *, resource_id: str | None,
    ) -> dict[str, str]:
        """Env vars the in-pod runner needs at boot.

        Phase 14.A introduces this method as the single source of
        truth for "what env vars must the launcher forward to the
        runner so it boots correctly?".

        ALWAYS includes ``RYOTENKAI_RUNTIME_PROVIDER`` keyed to
        :attr:`provider_name` so the runner's bootstrap registry
        (Phase 14.B) can pick the right :class:`IPodLifecycleClient`
        impl without having to guess from other env presence.

        Returns:
            A flat dict of env-var name → value. Single_node returns
            ``{RYOTENKAI_RUNTIME_PROVIDER: "single_node"}``. RunPod
            additionally returns ``RUNPOD_API_KEY``,
            ``RUNPOD_KEEP_ON_ERROR``, ``RUNPOD_VOLUME_KIND``, plus
            ``RUNPOD_POD_ID`` when ``resource_id`` is provided.

        Args:
            resource_id: The provider's resource identifier
                (RunPod pod_id, etc.). ``None`` when the launcher
                calls before :meth:`connect` has assigned one — the
                provider returns what it CAN (everything except
                resource-keyed values).

        Note:
            Phase 14.A keeps both this method AND
            :meth:`prepare_training_script_hooks` callable; both
            return the same data on RunPod. Phase 14.D explicitly
            collapses the legacy hooks API into this one. A FIXME in
            the RunPod provider flags the redundancy.
        """
        ...

    def probe_availability(self, resource_id: str) -> AvailabilityVerdict:
        """Query the provider for pod/host availability.

        Phase 14.A. Always defined for every provider so callers can
        ``provider.probe_availability(...)`` unconditionally instead
        of branching on provider name.

        * Single_node returns ``state="running"`` immediately,
          without any network round-trip — the host is always
          reachable; SSH connect step surfaces real errors later.
        * RunPod queries GraphQL via :class:`RunPodAPIClient.query_pod`
          and maps ``desiredStatus`` to the verdict's bucket.

        Returns:
            :class:`AvailabilityVerdict`. Never raises — transient
            probe errors map to ``state="probe_failed"``.

        Args:
            resource_id: Provider-specific identifier of the resource
                to probe. Empty string is acceptable for providers
                that don't track per-resource availability
                (single_node).
        """
        ...

    # ---- pod-side filesystem layout ----

    def pod_layout_for_run(self, run_id: str) -> PodLayout:
        """Pod-side filesystem layout for the given run.

        Provider-agnostic interface backing the canonical path tree
        defined in :class:`src.utils.pod_layout.PodLayout`. Each
        provider supplies its own ``root`` (where the workspace lives
        on its filesystem); the directory structure under ``root`` is
        identical across providers.

        Stateless by contract — depends only on ``run_id`` and the
        provider's static config. Callable BEFORE :meth:`connect` so
        that deployment-side components (CodeSyncer, FileUploader,
        runner_launcher) can plan paths without round-tripping the
        provider state.

        Returns:
            :class:`PodLayout` rooted at the provider-specific run
            directory. RunPod's root: ``/workspace/runs/<run_id>``.
            single_node's root: ``<config.workspace_path>/runs/<run_id>``.

        Args:
            run_id: Stable run identifier (typically ``run.name``).
                Must not be empty.
        """
        ...

    # ---- Phase 14.D+F — secrets validation ----

    def required_secrets(self) -> tuple[str, ...]:
        """Names of operator-environment secrets that MUST be
        present at startup.

        Phase 14.D+F replaces the hardcoded ``PROVIDER_RUNPOD``
        secret-presence check in
        :mod:`src.pipeline.bootstrap.startup_validator` with a
        provider-driven loop:

            for name in provider.required_secrets():
                if not has_secret(secrets, name):
                    raise StartupValidationError(...)

        RunPod returns ``("RUNPOD_API_KEY",)``; single_node returns
        ``()``. The startup validator iterates the tuple and checks
        each name against :class:`Secrets` — missing → fail-fast
        with the secret name in the error message.

        Returns:
            Sorted, immutable tuple of secret env-var names. Empty
            tuple for providers that don't need any secret
            (single_node host has none).
        """
        ...


@runtime_checkable
class ITerminalActionProvider(Protocol):
    """Capability-gated Protocol: provider can pause / resume / terminate.

    Phase 14.A introduces this as a SEPARATE Protocol from
    :class:`IGPUProvider` — providers that DON'T support cloud
    lifecycle actions (single_node) intentionally do NOT implement
    this. The type checker then prevents callers from accidentally
    invoking ``.terminate()`` / ``.pause()`` / ``.resume()`` on a
    single_node instance.

    Two-source-of-truth invariant:
        ``ProviderCapabilities.supports_lifecycle_actions``
        MUST equal ``isinstance(provider, ITerminalActionProvider)``.
        Verified at factory boot — failing assert is a blocker.

    Why a separate Protocol (not just a flag on the base):
        * Type-system enforcement — senior reviewers see at the call
          site that ``.resume()`` is only callable when the type
          says it's a :class:`ITerminalActionProvider`.
        * Adding a third provider becomes a typed conformance
          question, not a Liskov-violation review.

    Why ``runtime_checkable``:
        Callers need to ``isinstance(p, ITerminalActionProvider)``
        to gate optional capability paths (Phase 14.B PodTerminator,
        Phase 14.C LaunchResumeService).

    All methods are sync to match the rest of the :class:`IGPUProvider`
    surface; impls that need async transports (RunPod GraphQL) wrap
    via ``asyncio.run`` — same pattern as
    :meth:`RunPodAPIClient.query_pod`.
    """

    def terminate(
        self, *, resource_id: str, reason: str,
    ) -> Result[None, ProviderError]:
        """Permanently delete the resource (irreversible).

        Used by the runner's terminal-hook decision matrix (Phase
        14.B) when ``terminal_state="cancelled"`` (user-stop) or
        ``volume_kind=NETWORK`` (RunPod constraint).

        Idempotent — already-gone resources return ``Ok``.

        Args:
            resource_id: Identifier of the resource to terminate
                (e.g. RunPod ``pod_id``).
            reason: Operator-visible reason string for telemetry +
                logs (e.g. ``"user_stop"`` / ``"failed_safety"``).
        """
        ...

    def pause(
        self, *, resource_id: str,
    ) -> Result[None, ProviderError]:
        """Stop the resource preserving its workspace.

        Phase 14.A. RunPod calls ``podStop``; resources can be
        recovered via :meth:`resume`. For single_node this method
        is NEVER callable (single_node does not implement this
        Protocol).

        Idempotent — already-stopped resources return ``Ok``.
        """
        ...

    def resume(
        self, *, resource_id: str,
    ) -> Result[None, ProviderError]:
        """Wake a previously :meth:`pause`-d resource.

        Phase 14.A. RunPod calls ``podResume``. Caller is responsible
        for capacity-aware retry — this method does ONE attempt
        and returns its outcome. Phase 14.C's
        :class:`LaunchResumeService` orchestrates the budget loop.
        """
        ...


# Type alias for provider factory
ProviderFactory = type[IGPUProvider]


__all__ = [
    "AvailabilityVerdict",
    "GPUInfo",
    "IGPUProvider",
    "ITerminalActionProvider",
    "ProviderCapabilities",
    "ProviderFactory",
    "ProviderStatus",
    "SSHConnectionInfo",
    "TrainingScriptHooks",
    "VolumeKind",
]
