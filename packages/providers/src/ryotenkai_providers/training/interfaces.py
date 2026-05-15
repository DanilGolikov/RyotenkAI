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
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Protocol, runtime_checkable

if TYPE_CHECKING:
    from ryotenkai_shared.pipeline_context import RunContext
    from ryotenkai_providers.runpod.models import PodResourceInfo
    from ryotenkai_shared.utils.pod_layout import PodLayout
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

    supports_recovery_probe: bool = False
    """True iff the provider can probe + recover an existing resource
    after the runner connection was lost. Pairs 1:1 with
    :class:`IRecoveryProbeProvider` membership. Replaces the hardcoded
    ``self._provider_name != "runpod"`` skip in
    :mod:`ryotenkai_control.pipeline.stages.training_monitor` —
    callers gate on this flag instead of comparing names."""

    supports_capacity_error_detection: bool = False
    """True iff the provider can classify backend error messages as
    transient capacity exhaustion vs hard failures. Pairs 1:1 with
    :class:`ICapacityErrorClassifier` membership. Replaces the
    ``if metadata.provider == PROVIDER_RUNPOD: is_capacity_error_message``
    branch in :mod:`ryotenkai_control.pipeline.launch.resume_service`."""


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


class ProviderBase:
    """Concrete base class that holds manifest-derived identity + capabilities.

    Provider implementations (``RunPodProvider``, ``SingleNodeProvider``,
    inference-side equivalents) inherit this base for the **default
    implementation** of the four shared accessors — ``provider_id``,
    ``provider_name``, ``provider_type``, ``get_capabilities()`` — so the
    same identity+capability data does not have to be hand-coded in every
    provider class. The `ProviderRegistry` sets the four ``_manifest_*``
    ``ClassVar`` slots when it loads ``provider.toml``; the properties
    read from them.

    Why a base class (not a Protocol) for the shared bits:

    * The four accessors have an obvious **default implementation**
      (read the manifest-set ClassVar). A Protocol can only declare; a
      concrete base can also implement. Removing the duplicated stubs
      across :class:`IGPUProvider`, :class:`IInferenceProvider`,
      :class:`IPodLifecycleClient` is a DRY win — Phase B's audit found
      `provider_name` declared three times in three Protocols.

    * Protocols stay (see :class:`IGPUProvider` /
      :class:`IInferenceProvider` below) — they describe the role
      contract for ``isinstance`` checks and mypy structural typing.
      Provider classes inherit ``ProviderBase`` AND structurally
      conform to the Protocol of their role. The two responsibilities
      (default impl vs declarative interface) are kept separate.

    The four ``_manifest_*`` slots are ``ClassVar`` so a provider class
    has the right identity even before any instance is constructed — the
    registry validators need to compare ``cls.provider_id`` to manifest
    id without spinning up a real provider.

    Test fixtures may set the ClassVars directly to bypass the registry
    (mirrors the ``object.__new__()`` pattern in
    ``test_factory_capability_invariant.py``).
    """

    #: Canonical id from manifest's ``[provider].id``. Set by the
    #: registry at load time. Empty string = "not registered" (fail loud
    #: in :meth:`get_capabilities` if accessed in that state).
    _manifest_provider_id: ClassVar[str] = ""

    #: Display name from manifest's ``[provider].name``.
    _manifest_provider_name: ClassVar[str] = ""

    #: ``"local"`` or ``"cloud"`` from manifest's ``[capabilities].provider_type``.
    _manifest_provider_type: ClassVar[str] = ""

    #: Frozen ProviderCapabilities derived from the manifest's
    #: ``[capabilities]`` block, set by the registry. ``None`` means the
    #: provider was constructed outside the registry (test fixture, scaffold);
    #: :meth:`get_capabilities` raises with an actionable error in that case.
    _manifest_capabilities: ClassVar["ProviderCapabilities | None"] = None

    @property
    def provider_id(self) -> str:
        """Canonical id from manifest. Empty string = unregistered."""
        return type(self)._manifest_provider_id

    @property
    def provider_name(self) -> str:
        """Display name from manifest's ``[provider].name``."""
        return type(self)._manifest_provider_name

    @property
    def provider_type(self) -> str:
        """``"local"`` or ``"cloud"`` from manifest's ``[capabilities].provider_type``."""
        return type(self)._manifest_provider_type

    def get_capabilities(self) -> "ProviderCapabilities":
        """Return the manifest-derived capabilities snapshot.

        The registry attaches this at load time. Hand-overriding is
        forbidden by the invariant test (single source of truth =
        manifest, not Python). Test fixtures that bypass the registry
        must set ``_manifest_capabilities`` on the class explicitly.
        """
        caps = type(self)._manifest_capabilities
        if caps is None:
            raise RuntimeError(
                f"{type(self).__name__} has no manifest capabilities attached. "
                f"Provider classes must be registered through ProviderRegistry "
                f"before construction (it sets _manifest_capabilities from the "
                f"provider.toml). Test fixtures may set the ClassVar directly."
            )
        return caps


@runtime_checkable
class IRecoveryProbeProvider(Protocol):
    """Capability-gated Protocol: provider can probe + recover after runner-connection loss.

    Phase 14.D+F. Replaces the hardcoded ``self._provider_name != "runpod"``
    skip in
    :mod:`ryotenkai_control.pipeline.stages.training_monitor` (line 682
    pre-refactor). The training monitor gates the recovery loop on
    ``isinstance(provider, IRecoveryProbeProvider)`` instead of comparing
    names — a third cloud provider with similar GraphQL-probe semantics
    just inherits this Protocol and the recovery path lights up
    automatically.

    Two-source-of-truth invariant (mirrors :class:`ITerminalActionProvider`):
        :attr:`ProviderCapabilities.supports_recovery_probe` MUST equal
        ``isinstance(provider, IRecoveryProbeProvider)``. Verified by the
        check_manifests script and the pytest invariant suite.

    Recovery semantics: a successful ``attempt_recovery`` returns the
    fresh :class:`ProviderStatus` (typically ``CONNECTED`` after the pod
    woke up). Caller resumes the runner SSH session against the recovered
    resource.
    """

    def attempt_recovery(
        self, *, resource_id: str,
    ) -> ProviderStatus:
        """Probe the resource and try to bring it back online.

        Idempotent — already-running resources return
        :data:`ProviderStatus.CONNECTED` without side effects. Caller
        is responsible for the retry budget; this method is ONE
        attempt.

        Args:
            resource_id: Provider-specific resource identifier
                (e.g. RunPod ``pod_id``).

        Returns:
            Fresh :class:`ProviderStatus` after the recovery attempt
            (typically ``CONNECTED`` after the pod woke up).

        Raises:
            ProviderUnavailableError: probe or wake-up failed
                (transient / permanent backend error). Carries
                ``context["legacy_code"]`` in
                ``{"POD_PROBE_FAILED", "POD_TERMINAL", "POD_WAKE_FAILED"}``
                so callers can branch on the specific failure mode.
        """
        ...


@runtime_checkable
class ICapacityErrorClassifier(Protocol):
    """Capability-gated Protocol: provider can classify backend error messages.

    Phase 14.D+F. Replaces the
    ``if metadata.provider == PROVIDER_RUNPOD: is_capacity_error_message``
    branch in :mod:`ryotenkai_control.pipeline.launch.resume_service`
    (line 374 pre-refactor). The resume service asks the provider to
    classify a backend error message; the provider returns ``True`` if
    it's transient capacity exhaustion (retry-worthy) vs a hard failure.

    Two-source-of-truth invariant:
        :attr:`ProviderCapabilities.supports_capacity_error_detection`
        MUST equal ``isinstance(provider, ICapacityErrorClassifier)``.
        Verified by the same suite as the other capability Protocols.

    Capacity errors are a cloud-fleet concept; the manifest schema
    rejects ``supports_capacity_error_detection=true`` for
    ``provider_type="local"``. Local hosts never run out of "capacity"
    (their GPU is dedicated).
    """

    def is_capacity_error(self, message: str) -> bool:
        """Return True iff the message indicates transient capacity exhaustion.

        Caller (resume service) treats True as a retry-worthy signal
        and returns ``False`` from anything else (hard failure → stop).
        """
        ...


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

    Provider implementations inherit :class:`ProviderBase` for the
    default impl of identity/capability accessors (``provider_id``,
    ``provider_name``, ``provider_type``, ``get_capabilities``); they
    structurally conform to this Protocol via their role-specific
    methods (``connect``, ``check_gpu``, etc.).

    Example:
        provider = registry.create_training("runpod", ctx).unwrap()

        result = provider.connect(run=run_ctx)
        if result.is_err():
            logger.error(f"Connection failed: {result.unwrap_err()}")
            return

        ssh_info = result.unwrap()
        # Use ssh_info.host, ssh_info.port, etc.

        # When done:
        provider.disconnect()
    """

    # Identity + capability accessors — provided as default impl by
    # ``ProviderBase`` (which provider classes inherit). Listed here so
    # ``runtime_checkable`` ``isinstance(p, IGPUProvider)`` checks see
    # them on the protocol surface.

    @property
    def provider_id(self) -> str:
        """Canonical id from manifest (e.g. ``"runpod"``, ``"single_node"``)."""
        ...

    @property
    def provider_name(self) -> str:
        """Display name from manifest (e.g. ``"RunPod"``)."""
        ...

    @property
    def provider_type(self) -> str:
        """Provider type: 'local' or 'cloud'."""
        ...

    def connect(self, *, run: RunContext) -> SSHConnectionInfo:
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
            SSHConnectionInfo when the connection is established.

        Raises:
            ProviderUnavailableError: backend create / wait failed,
                SSH info missing, generic transport error.
            SSHConnectionFailedError: SSH handshake failed after pod
                creation succeeded.
        """
        ...

    def disconnect(self) -> None:
        """
        Disconnect from GPU server.

        For cloud providers:
            - Terminates instance
            - Cleans up resources

        For local providers:
            - No-op (server stays on)
            - Just marks provider as disconnected

        Raises:
            ProviderUnavailableError: cleanup failure (best-effort —
                callers typically swallow on cleanup paths).
        """
        ...

    def get_status(self) -> ProviderStatus:
        """
        Get current provider status.

        Returns:
            ProviderStatus enum value
        """
        ...

    def check_gpu(self) -> GPUInfo:
        """
        Check GPU availability and specs via nvidia-smi.

        Should be called after connect() to validate GPU.

        Returns:
            GPUInfo for the connected pod / host.

        Raises:
            ProviderUnavailableError: GPU not found, nvidia-smi failed,
                output parse failure, or provider is not connected.
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
    ) -> TrainingScriptHooks:
        """
        Prepare provider-specific customizations for the training launch script.

        Called by ``TrainingDeploymentManager`` after SSH is connected and
        before ``start_training.sh`` is generated. Providers may:
            - Upload auxiliary scripts to the pod (e.g., watchdog, stop helpers).
            - Return env vars to merge into the ``.env`` file.
            - Return bash snippets to inject before/after the Python invocation.

        Default behavior (for providers with nothing to contribute): return
        ``TrainingScriptHooks.empty()``.

        Args:
            ssh_client: Connected SSH client to the pod/server.
            context: Pipeline context dict (may include ``resource_id``,
                ``workspace``, etc.).

        Returns:
            TrainingScriptHooks customizations to apply.

        Raises:
            RyotenkAIError: upload or configuration failure — deployment
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
    ) -> None:
        """Permanently delete the resource (irreversible).

        Used by the runner's terminal-hook decision matrix (Phase
        14.B) when ``terminal_state="cancelled"`` (user-stop) or
        ``volume_kind=NETWORK`` (RunPod constraint).

        Idempotent — already-gone resources return cleanly.

        Args:
            resource_id: Identifier of the resource to terminate
                (e.g. RunPod ``pod_id``).
            reason: Operator-visible reason string for telemetry +
                logs (e.g. ``"user_stop"`` / ``"failed_safety"``).

        Raises:
            ProviderUnavailableError: backend rejected the terminate
                request (transient / permanent).
        """
        ...

    def pause(
        self, *, resource_id: str,
    ) -> None:
        """Stop the resource preserving its workspace.

        Phase 14.A. RunPod calls ``podStop``; resources can be
        recovered via :meth:`resume`. For single_node this method
        is NEVER callable (single_node does not implement this
        Protocol).

        Idempotent — already-stopped resources return cleanly.

        Raises:
            ProviderUnavailableError: backend rejected the pause.
        """
        ...

    def resume(
        self, *, resource_id: str,
    ) -> None:
        """Wake a previously :meth:`pause`-d resource.

        Phase 14.A. RunPod calls ``podResume``. Caller is responsible
        for capacity-aware retry — this method does ONE attempt.
        Phase 14.C's :class:`LaunchResumeService` orchestrates the
        budget loop.

        Raises:
            ProviderUnavailableError: capacity-exhausted / transport
                error / hard failure. The
                :class:`ICapacityErrorClassifier` Protocol (when
                implemented) lets the caller distinguish.
        """
        ...


# Type alias for provider factory
ProviderFactory = type[IGPUProvider]


__all__ = [
    "AvailabilityVerdict",
    "GPUInfo",
    "ICapacityErrorClassifier",
    "IGPUProvider",
    "IRecoveryProbeProvider",
    "ITerminalActionProvider",
    "ProviderBase",
    "ProviderCapabilities",
    "ProviderFactory",
    "ProviderStatus",
    "SSHConnectionInfo",
    "TrainingScriptHooks",
    "VolumeKind",
]
