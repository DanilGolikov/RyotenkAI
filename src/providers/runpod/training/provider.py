"""
RunPod Provider - cloud GPU via RunPod API.

Implements IGPUProvider for RunPod cloud instances.
"""

from __future__ import annotations

import logging
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

from src.constants import PROVIDER_RUNPOD, RUNTIME_PROVIDER_ENV_VAR
from src.utils.cancellation import PipelineCancelled
from src.providers.training.interfaces import (
    AvailabilityVerdict,
    GPUInfo,
    IGPUProvider,
    ITerminalActionProvider,
    ProviderCapabilities,
    ProviderStatus,
    SSHConnectionInfo,
    TrainingScriptHooks,
    VolumeKind,
)
from src.runner.__about__ import RUNTIME_IMAGE
from src.utils.pod_layout import PodLayout
from src.utils.result import AppError, Err, Ok, ProviderError, Result
from src.utils.ssh_client import SSHClient

from ..models import PodResourceInfo
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
_POD_CREATE_MAX_RETRIES = 3
# Error codes from PodSshWaiter that warrant a fresh-pod recreate
# rather than aborting the connect path. If this tuple goes out of
# sync with the waiter's error vocabulary, the provider falls back
# to "report and abort" — bisect the missing code here first.
#
# Note: the RUNNING-with-port_count==0 stuck state is intentionally
# NOT a separate code here. RunPod sometimes takes the full 300s
# window to allocate ports; an early bailout half-way through cuts
# off would-be successful boots. That state surfaces here as
# ``RUNPOD_POD_TIMEOUT`` after the full window.
_RECREATABLE_ERRORS = (
    "RUNPOD_NO_EXPOSED_TCP",
    "RUNPOD_POD_TIMEOUT",
    "RUNPOD_POD_FAILED",
)

# Phase 14.A — RunPod-specific "pod is gone" markers. Used by
# :meth:`RunPodProvider.probe_availability` to distinguish a
# permanently-terminated pod (operator should ``run restart``) from
# a transient API flap (operator should retry). Lower-cased on
# match so case variations don't break detection.
_GONE_ERROR_MARKERS: tuple[str, ...] = (
    "not found",
    "does not exist",
    "no such pod",
    "no pod with",
)

if TYPE_CHECKING:
    from src.pipeline.state import RunContext
    from src.providers.runpod.models import PodSnapshot
    from src.utils.config import Secrets

logger = logging.getLogger("ryotenkai")


# Phase 14.C — :func:`map_runpod_desired_status_to_availability` lives
# in :mod:`src.providers.runpod._status_mapper` (a sibling-of-package
# thin module) so importing the helper does NOT trigger this
# package's heavy ``training/__init__.py`` chain. We re-export it
# here for symmetry — callers who already import from the provider
# module keep working.
from src.providers.runpod._status_mapper import (
    map_runpod_desired_status_to_availability,
)


class RunPodProvider(IGPUProvider, ITerminalActionProvider):
    """
    GPU provider for RunPod cloud.

    Features:
        - Creates pods via GraphQL API
        - Waits for pod to be ready
        - Provides SSH connection info
        - Automatic cleanup on disconnect
        - Phase 14.A: implements :class:`ITerminalActionProvider`
          (terminate / pause / resume) — single_node does NOT.

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
        self._pod_info: PodResourceInfo | None = None
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

        self._lifecycle = PodLifecycleManager(api_client=self._api_client)

        logger.info(
            f"[PROVIDER:INIT] RunPodProvider initialized: "
            f"GPU={self._config.training.gpu_type}, image={RUNTIME_IMAGE}"
        )

    @classmethod
    def from_resume_metadata(cls, *, api_key: str) -> RunPodProvider:
        """Phase 14.C — minimal-construction factory for resume flow.

        Bypasses the heavy Pydantic config validator (which requires
        a full :class:`RunPodProviderConfig` with training, cleanup,
        connect, etc. sections). The resume flow only invokes
        :class:`ITerminalActionProvider` methods (terminate / pause /
        resume) and :meth:`probe_availability`, which read:

          * ``self._api_key``
          * ``self._graphql_api_client`` (RunPodAPIClient — used by
            ``probe_availability`` to call ``query_pod``)
          * ``self._api_client`` (:class:`RunPodTrainingPodControl`
            — used by ``terminate``/``pause``/``resume`` to call
            ``terminate_pod``/``stop_pod``/``start_pod``)
          * ``self._status``

        Construction via ``object.__new__`` is intentional — same
        pattern Phase 14.A and Phase 14.B test fixtures use. If
        ``__init__`` ever grows new attributes the lifecycle methods
        rely on, this factory needs an update; the factory test pins
        which attributes are required.
        """
        provider = object.__new__(cls)
        provider._api_key = api_key
        provider._secrets = None  # type: ignore[assignment]
        provider._status = ProviderStatus.AVAILABLE
        provider._pod_id = None
        provider._ssh_connection_info = None
        provider._gpu_info = None
        provider._pod_info = None
        provider._had_error = False
        provider._config = None  # type: ignore[assignment]
        # Mirror :meth:`__init__` wiring: GraphQL client for
        # query_pod, training-pod-control for terminate/stop/start.
        provider._graphql_api_client = RunPodAPIClient(
            api_base_url=RUNPOD_API_BASE_URL,
            api_key=api_key,
        )
        provider._api_client = RunPodTrainingPodControl(
            api=provider._graphql_api_client,
        )
        return provider

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
            # Step 1+2: Create pod and wait for ready (with pod recreation on failure).
            pod_name = f"ryotenkai-train-{run.name}"
            create_wait_result = self._create_and_wait_for_pod(pod_name)
            if create_wait_result.is_failure():
                self._status = ProviderStatus.ERROR
                return Err(create_wait_result.unwrap_err())  # type: ignore[union-attr]

            snapshot, resource_info = create_wait_result.unwrap()
            self._pod_info = resource_info
            logger.info("✅ Pod is ready!")

            pod_id = snapshot.pod_id

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

        except PipelineCancelled:
            # Cancel signal arrived mid-connect (most likely during
            # create_pod HTTP request or PodSshWaiter polling). If a pod
            # was already created, terminate it immediately before
            # re-raising so it isn't left orphaned and billing.
            #
            # Was ``except KeyboardInterrupt:`` until the worker
            # subprocess gained an explicit cancel handler — Python's
            # default SIGINT path no longer fires here, so we catch the
            # canonical ``PipelineCancelled`` instead.
            self._status = ProviderStatus.ERROR
            if self._pod_id:
                logger.warning(f"[PROVIDER:CONNECT] cancelled during connect — terminating pod {self._pod_id}")
                self._cleanup_manager.cleanup_pod(self._pod_id)
                self._pod_id = None
            else:
                logger.warning("[PROVIDER:CONNECT] cancelled during connect — no pod to clean up")
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

    def _create_and_wait_for_pod(self, pod_name: str) -> Result[tuple[PodSnapshot, PodResourceInfo], ProviderError]:
        """Create a pod and wait for it to become ready.

        If the pod fails to get SSH exposed TCP (community cloud limitation),
        terminates it and tries creating a new one on a different machine.

        Side-effect: ``self._pod_id`` is kept in sync with the current pod so
        that the SIGINT handler in ``connect()`` can always clean up.
        """
        last_err: ProviderError | None = None

        for attempt in range(1, _POD_CREATE_MAX_RETRIES + 1):
            if attempt > 1:
                logger.info(f"🔄 Pod creation attempt {attempt}/{_POD_CREATE_MAX_RETRIES}")

            pod_result = self._api_client.create_pod(config=self._config, pod_name=pod_name)
            if pod_result.is_failure():
                return Err(pod_result.unwrap_err())  # type: ignore[union-attr]

            raw_info = pod_result.unwrap()
            resource_info = PodResourceInfo.from_create_response(raw_info)

            if not resource_info.pod_id:
                return Err(ProviderError(message="Invalid pod info returned", code="RUNPOD_INVALID_POD_INFO"))

            pod_id = resource_info.pod_id
            self._pod_id = pod_id  # kept for SIGINT safety in connect()
            logger.info(f"✅ Pod created: {pod_id}")

            ready_result = self._lifecycle.wait_for_ready(pod_id)
            if ready_result.is_success():
                return Ok((ready_result.unwrap(), resource_info))

            last_err = ready_result.unwrap_err()

            if last_err.code not in _RECREATABLE_ERRORS or attempt >= _POD_CREATE_MAX_RETRIES:
                self._safe_cleanup_pod(pod_id)
                self._pod_id = None
                break

            logger.warning(
                f"[PROVIDER] Pod {pod_id} failed ({last_err.code}), "
                f"terminating and trying a new one ({attempt}/{_POD_CREATE_MAX_RETRIES})..."
            )
            self._safe_cleanup_pod(pod_id)
            self._pod_id = None

        return Err(
            last_err
            or ProviderError(
                message=f"Pod creation failed after {_POD_CREATE_MAX_RETRIES} attempts",
                code="RUNPOD_POD_NOT_READY",
            )
        )

    def _safe_cleanup_pod(self, pod_id: str) -> None:
        """Terminate and unregister a pod, logging on failure instead of raising."""
        result = self._cleanup_manager.cleanup_pod(pod_id)
        if result.is_failure():
            logger.warning(f"[PROVIDER] Failed to cleanup pod {pod_id}: {result.unwrap_err()}")

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
            # Previously the Result was discarded and "✅ Pod terminated" was
            # printed unconditionally. That masked real cleanup failures
            # (pod already gone after platform-side eviction, network blip,
            # auth expiry) — the operator saw a clean shutdown message even
            # when the API returned an error. Surface failures explicitly
            # so we don't leak orphan pods quietly.
            cleanup_result = self._cleanup_manager.cleanup_pod(self._pod_id)
            if cleanup_result.is_ok():
                logger.info("✅ Pod terminated")
            else:
                logger.warning(
                    f"[PROVIDER:DISCONNECT] Pod {self._pod_id} terminate call "
                    f"returned an error: {cleanup_result.unwrap_err()}. The pod "
                    f"may already be gone (platform-side eviction) or may need "
                    f"manual cleanup — verify in the RunPod console."
                )
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
        """Get provider capabilities.

        Phase 14.A: populates the capability surface
        (``supports_lifecycle_actions``, ``volume_kind``,
        ``has_pause_resume``, ``runner_workspace_root``) so callers
        can avoid string-checks like ``provider == "runpod"``.

        Two-source-of-truth invariant: ``supports_lifecycle_actions``
        ↔ ``isinstance(self, ITerminalActionProvider)``. RunPod
        sets True and DOES inherit from the Protocol.
        """
        gpu_name = self._config.training.gpu_type
        gpu_vram_gb = None

        if self._gpu_info:
            gpu_name = self._gpu_info.name
            gpu_vram_gb = self._gpu_info.vram_total_gb

        # Today RunPod training pods always use a persistent volume
        # (config has ``volume_disk_gb``, no network volume support).
        # If/when network-volume training lands, this should read the
        # provider config — keeping VolumeKind.PERSISTENT pinned for now.
        volume_kind = VolumeKind.PERSISTENT

        return ProviderCapabilities(
            provider_type="cloud",
            supports_multi_gpu=True,
            supports_spot_instances=True,
            max_runtime_hours=None,
            gpu_name=gpu_name,
            gpu_vram_gb=gpu_vram_gb,
            # Phase 14.A capability fields:
            supports_lifecycle_actions=True,
            volume_kind=volume_kind,
            has_pause_resume=True,  # podStop + podResume both supported
            runner_workspace_root="/workspace",  # RunPod pod mount path
            # Phase 14.D+F capability fields:
            is_local=False,  # cloud provider
            supports_log_download=True,  # SCP-based log fetch via RunPodAPIClient
        )

    def required_secrets(self) -> tuple[str, ...]:
        """Phase 14.D+F — RunPod requires the API key in the operator
        environment.

        Replaces the pre-14.D ``PROVIDER_RUNPOD`` secret-presence
        branch in :mod:`src.pipeline.bootstrap.startup_validator`.
        """
        return ("RUNPOD_API_KEY",)

    def pod_layout_for_run(self, run_id: str) -> PodLayout:
        """RunPod-rooted pod layout: ``/workspace/runs/<run_id>/...``.

        Matches the directory structure created in :meth:`connect`
        (lines 281-283 use the same ``/workspace/runs/{run.name}``
        formula).
        """
        if not run_id:
            raise ValueError("run_id must be non-empty")
        return PodLayout.from_root(PurePosixPath(f"/workspace/runs/{run_id}"))

    # ------------------------------------------------------------------
    # Phase 14.A — capability methods (IGPUProvider extension)
    # ------------------------------------------------------------------

    def required_runtime_env_vars(
        self,
        *,
        resource_id: str | None,
    ) -> dict[str, str]:
        """Env vars the in-pod runner needs.

        Phase 14.A. ALWAYS includes ``RYOTENKAI_RUNTIME_PROVIDER`` so
        the runner's bootstrap registry (Phase 14.B) can pick the
        right :class:`IPodLifecycleClient`.

        Without ``resource_id`` we still return what we know — the
        launcher calls AFTER :meth:`connect` so the empty-id case
        is purely defensive.

        FIXME(Phase 14.D): :meth:`prepare_training_script_hooks`
        returns the same dict (with extra ``cleanup.auto_stop_after_training``
        gating). Phase 14.D collapses both methods into this one.
        """
        env: dict[str, str] = {
            RUNTIME_PROVIDER_ENV_VAR: PROVIDER_RUNPOD,
            "RUNPOD_API_KEY": self._api_key,
            "RUNPOD_KEEP_ON_ERROR": ("true" if self._config.cleanup.keep_pod_on_error else "false"),
            "RUNPOD_VOLUME_KIND": VolumeKind.PERSISTENT.value,
        }
        if resource_id:
            env["RUNPOD_POD_ID"] = str(resource_id)
        return env

    def probe_availability(
        self,
        resource_id: str,
    ) -> AvailabilityVerdict:
        """Query RunPod for pod state and map to provider-agnostic verdict.

        Phase 14.A. Delegates to the existing
        :meth:`RunPodAPIClient.query_pod` for transport; reuses the
        :data:`_RUNPOD_STATUS_MAP` from
        :mod:`src.pipeline.launch.pod_availability` for parity with
        the legacy probe path. Phase 14.C will move that map into
        this module.

        Never raises — transient probe errors map to
        ``state="probe_failed"``. Empty ``resource_id`` (defensive)
        maps to ``state="unknown"`` so callers can branch on a
        clearly-disambiguated bucket.
        """
        if not resource_id:
            return AvailabilityVerdict(
                state="unknown",
                resource_id="",
                message="probe_availability called without resource_id",
            )

        try:
            result = self._graphql_api_client.query_pod(resource_id)
        except Exception as exc:
            return AvailabilityVerdict(
                state="probe_failed",
                resource_id=resource_id,
                message=f"query_pod raised: {exc!r}",
            )

        if result.is_failure():
            err = result.unwrap_err()
            err_msg = str(err).lower()
            # RunPod's GraphQL returns various flavors of "pod is
            # gone" — distinguish from transient "API flapping" so
            # the operator-side UX (Phase 14.C) can branch:
            # GONE → "use ``run restart`` to recreate from
            # checkpoint"; PROBE_FAILED → "transient, retry later".
            if any(marker in err_msg for marker in _GONE_ERROR_MARKERS):
                return AvailabilityVerdict(
                    state="gone",
                    resource_id=resource_id,
                    raw_status=None,
                    message="Pod terminated or does not exist",
                )
            return AvailabilityVerdict(
                state="probe_failed",
                resource_id=resource_id,
                message=str(err),
            )

        pod_data = result.unwrap()
        if not isinstance(pod_data, dict):  # type: ignore[unreachable]
            return AvailabilityVerdict(  # type: ignore[unreachable]
                state="probe_failed",
                resource_id=resource_id,
                message="query_pod returned non-dict payload",
            )

        # Phase 14.C — relocated map. Provider owns the RunPod GraphQL
        # vocabulary; the shared probe (:class:`PodAvailabilityProbe`)
        # now consumes ``map_runpod_desired_status_to_availability``
        # instead of importing the raw dict.
        from src.pipeline.launch.pod_availability import PodAvailability

        raw_status = str(pod_data.get("desiredStatus") or "").upper()
        if not raw_status:
            return AvailabilityVerdict(
                state="probe_failed",
                resource_id=resource_id,
                message="Pod data missing desiredStatus field",
            )
        mapped = map_runpod_desired_status_to_availability(raw_status)
        if mapped == PodAvailability.PROBE_FAILED:
            return AvailabilityVerdict(
                state="probe_failed",
                resource_id=resource_id,
                raw_status=raw_status,
                message=f"Unknown desiredStatus: {raw_status!r}",
            )

        # Map the legacy enum to our new state literal.
        bucket: AvailabilityVerdict = AvailabilityVerdict(
            state=(
                "running"
                if mapped == PodAvailability.RUNNING
                else "sleeping_resumable"
                if mapped == PodAvailability.SLEEPING_RESUMABLE
                else "gone"
                if mapped == PodAvailability.GONE
                else "probe_failed"
            ),
            resource_id=resource_id,
            raw_status=raw_status,
        )
        return bucket

    # ------------------------------------------------------------------
    # Phase 14.A — ITerminalActionProvider methods
    # ------------------------------------------------------------------

    def terminate(
        self,
        *,
        resource_id: str,
        reason: str,
    ) -> Result[None, ProviderError]:
        """Permanently delete the pod via ``podTerminate``.

        Phase 14.A. Delegates to the existing
        :meth:`RunPodTrainingPodControl.terminate_pod`. Idempotent —
        already-gone pods return ``Ok``.

        ``reason`` is currently logged but not forwarded to the
        GraphQL call (RunPod doesn't accept a reason field). Phase
        14.B's :class:`PodTerminator` will pipe it into telemetry.
        """
        logger.info(
            "[PROVIDER:TERMINATE] pod=%s reason=%s",
            resource_id,
            reason,
        )
        return self._api_client.terminate_pod(resource_id)

    def pause(
        self,
        *,
        resource_id: str,
    ) -> Result[None, ProviderError]:
        """Stop the pod via ``podStop`` (preserves /workspace).

        Phase 14.A. Delegates to the existing
        :meth:`RunPodTrainingPodControl.stop_pod`. Idempotent —
        already-stopped pods return ``Ok``.
        """
        logger.info("[PROVIDER:PAUSE] pod=%s", resource_id)
        return self._api_client.stop_pod(pod_id=resource_id)

    def resume(
        self,
        *,
        resource_id: str,
    ) -> Result[None, ProviderError]:
        """Wake a stopped pod via ``podResume``.

        Phase 14.A. Delegates to the existing
        :meth:`RunPodTrainingPodControl.start_pod` (RunPod's
        ``podResume`` mutation lives there). Single attempt — caller
        orchestrates capacity-aware retry (Phase 14.C
        :class:`LaunchResumeService`).
        """
        logger.info("[PROVIDER:RESUME] pod=%s", resource_id)
        return self._api_client.start_pod(pod_id=resource_id)

    def get_pod_id(self) -> str | None:
        """Get current pod ID."""
        return self._pod_id

    def get_resource_info(self) -> PodResourceInfo | None:
        """Return RunPod instance metadata (cost_per_hr, gpu_type, gpu_count) after connect()."""
        return self._pod_info

    def get_base_workspace(self) -> str:
        """Base workspace root inside pod (for shared venv/caches)."""
        return "/workspace"

    def prepare_training_script_hooks(
        self,
        ssh_client: SSHClient,
        context: dict[str, Any],
    ) -> Result[TrainingScriptHooks, AppError]:
        """Forward auto-stop credentials to the in-pod runner via env.

        Phase 6.5 simplification: the legacy hooks uploaded
        ``watchdog.sh`` (GPU idle detection) and ``runpod_stop_pod.sh``
        (GraphQL pod stop) plus injected ``pre_python`` /
        ``post_python`` bash snippets into the generated
        ``start_training.sh``. All three responsibilities now live in
        the in-pod runner:

        - ``watchdog.sh`` → :class:`src.runner.idle_detector.IdleDetector`
          (Phase 4.1)
        - ``runpod_stop_pod.sh`` → :class:`src.runner.pod_terminator.PodTerminator`
          (Phase 4.4 → Phase 11.B decision matrix)
        - ``start_training.sh`` itself → :class:`Supervisor` subprocess
          spawn (Phase 2)

        So this method is reduced to passing the same RUNPOD_* env
        vars the runner reads on startup. ``ssh_client`` is unused
        (no more file uploads); kept on the signature for protocol
        compatibility with :class:`IGPUProvider`.

        Phase 11.B: ``RUNPOD_AUTO_STOP`` env removed (no-toggle
        policy; PodTerminator always runs the decision matrix).
        ``cleanup.auto_stop_after_training`` retained as a way to
        skip wiring the RUNPOD_* creds entirely (e.g. for ephemeral
        smoke runs that don't need the auto-clean path), but the
        absence of creds simply maps to PodTerminalOutcome.SKIPPED;
        no env-flag silently disables the matrix.
        """
        del ssh_client  # no SSH side-effects post Phase 6.5

        cleanup = self._config.cleanup
        if not cleanup.auto_stop_after_training:
            logger.info("[PROVIDER:HOOKS] auto_stop_after_training disabled")
            return Ok(TrainingScriptHooks.empty())

        resource_id = context.get("resource_id") or self._pod_id
        if not resource_id:
            logger.warning("[PROVIDER:HOOKS] resource_id (pod_id) unknown — skipping auto-stop env")
            return Ok(TrainingScriptHooks.empty())

        if not self._api_key:
            logger.warning("[PROVIDER:HOOKS] RunPod API key missing — skipping auto-stop env")
            return Ok(TrainingScriptHooks.empty())

        env_vars = {
            "RUNPOD_API_KEY": self._api_key,
            "RUNPOD_POD_ID": str(resource_id),
            # Phase 11.B: ``RUNPOD_AUTO_STOP`` removed (always-on
            # decision matrix). ``RUNPOD_KEEP_ON_ERROR`` kept for
            # debug-forensics on failed runs (Phase 9.A carry-over).
            "RUNPOD_KEEP_ON_ERROR": "true" if cleanup.keep_pod_on_error else "false",
        }

        logger.info(f"[PROVIDER:HOOKS] Auto-stop env wired for pod {resource_id}")
        return Ok(TrainingScriptHooks(env_vars=env_vars))

    def __repr__(self) -> str:
        status = self._status.value
        pod_id = self._pod_id or "no-pod"
        return f"RunPodProvider({self._config.training.gpu_type}, pod={pod_id}, status={status})"
