"""Training monitor stage — Phase 6.3b rewrite.

The legacy monitor (982 LoC) opened an SSH connection to the pod and
polled marker files (``TRAINING_COMPLETE``, ``TRAINING_FAILED``,
``.pipeline_heartbeat``, ``.watchdog_heartbeat``, …) every 5
seconds, with separate code paths for laptop-sleep recovery,
``ps`` checks, ``docker ps`` checks, and post-mortem diagnostics.

After Phase 6.3a (TrainingLauncher rewrite) the launcher leaves an
open :class:`SSHTunnelManager` + :class:`JobClient` on the pipeline
context. Monitor's job is now to subscribe to the in-pod runner's
WebSocket event stream — terminal state arrives as a structured
event, no filesystem polling, no marker IPC.

Public surface preserved 1:1 so :mod:`src.pipeline.orchestrator`
keeps working without changes:
- :class:`TrainingMonitor` constructor signature
- :class:`TrainingMonitorEventCallbacks` dataclass
- ``execute(context) -> Result[dict, AppError]`` return shape

Async strategy: same as the launcher — sync facade wrapping a
``asyncio.run`` islet around the WebSocket consumer.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from dataclasses import dataclass
from datetime import timedelta
from typing import TYPE_CHECKING, Any

from ryotenkai_shared.utils.clients.job_client import (
    JobClientError,
    JobNotFoundError,
    ReplayTruncatedError,
)
from ryotenkai_shared.constants import (
    CONSOLE_LINE_WIDTH,
    LOG_DOWNLOAD_INTERVAL_DEFAULT,
    SSH_PORT_DEFAULT,
)
from ryotenkai_control.pipeline.stages.base import PipelineStage
from ryotenkai_control.pipeline.stages.constants import PipelineContextKeys, StageNames
from ryotenkai_control.pipeline.stages.managers.log_manager import LogManager
from ryotenkai_shared.utils.logger import get_run_log_layout, logger
from ryotenkai_shared.utils.result import AppError, Err, Ok, Result, TrainingError
from ryotenkai_shared.utils.ssh_client import SSHClient

# Final-flush deadline. Pod-side training.log can be hundreds of KB
# even for short runs; 60 s mirrors LogManager's per-command timeout.
_FINAL_LOG_FLUSH_TIMEOUT = 60.0

# Constants kept on the module so existing callers / tests that import
# them by name don't break (TRAINING_MONITOR_LINE_WIDTH is referenced
# by report rendering).
TRAINING_MONITOR_SSH_PORT = SSH_PORT_DEFAULT
TRAINING_MONITOR_START_TIMEOUT_DEFAULT = 30
TRAINING_MONITOR_LOG_DOWNLOAD_INTERVAL = LOG_DOWNLOAD_INTERVAL_DEFAULT
TRAINING_MONITOR_LOG_STATUS_INTERVAL = 15
TRAINING_MONITOR_LINE_WIDTH = CONSOLE_LINE_WIDTH

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from ryotenkai_shared.utils.clients.job_client import JobClient
    from ryotenkai_shared.utils.clients.ssh_tunnel import SSHTunnelManager
    from ryotenkai_shared.config.secrets.model import Secrets
    from ryotenkai_shared.config import PipelineConfig


# ---------------------------------------------------------------------------
# Public callback contract — preserved verbatim from legacy monitor
# ---------------------------------------------------------------------------


@dataclass
class TrainingMonitorEventCallbacks:
    """Hooks for external integrations (MLflow event log, dashboards).

    All callbacks are optional; the monitor calls only those that are
    set. Signatures match the legacy contract so MLflow event-logging
    code doesn't have to change.
    """

    on_training_started: Callable[[], None] | None = None
    on_training_completed: Callable[[float], None] | None = None
    """Args: ``duration_seconds``."""

    on_training_failed: Callable[[str, float], None] | None = None
    """Args: ``error_message``, ``duration_seconds``."""

    on_training_timeout: Callable[[int, float], None] | None = None
    """DEPRECATED — training has no wall-clock cap any more, but the
    field stays so MLflow callback registrations don't fail."""

    on_process_died: Callable[[float], None] | None = None
    """Args: ``duration_seconds``. Fires on FAILED with no error
    message (e.g. SIGKILL by IdleDetector)."""

    on_resource_check: Callable[[dict], None] | None = None
    """Args: ``resources`` dict. Fires on every ``health_snapshot``
    event from :class:`HealthReporter`."""


# ---------------------------------------------------------------------------
# Internal: terminal-state strings the runner emits via FSM
# ---------------------------------------------------------------------------

_TERMINAL_COMPLETED = "completed"
_TERMINAL_FAILED = "failed"
_TERMINAL_CANCELLED = "cancelled"
_TERMINAL_STATES = frozenset({_TERMINAL_COMPLETED, _TERMINAL_FAILED, _TERMINAL_CANCELLED})


# ---------------------------------------------------------------------------
# Monitor
# ---------------------------------------------------------------------------


class TrainingMonitor(PipelineStage):
    """Watch training progress over the in-pod runner's WebSocket.

    Runs in lockstep with :class:`TrainingLauncher`: the launcher
    stashes ``job_client`` + ``ssh_tunnel`` + ``job_id`` on the
    pipeline context; the monitor reads them and subscribes to the
    runner's event stream until the FSM lands in a terminal state.

    Terminal-state mapping (runner FSM → monitor outcome):
    - ``completed`` → ``Ok({status: "completed", duration_seconds})``
    - ``failed`` → ``Err(TrainingError("...", code="TRAINING_FAILED"))``
    - ``cancelled`` → ``Err(TrainingError("...", code="TRAINING_CANCELLED"))``

    Tunnel lifecycle: closed in ``finally`` regardless of outcome —
    leaving ``ssh -L`` alive after the monitor exits would leak a
    local port forever.
    """

    # Reserved for laptop-sleep recovery (consulted by
    # ``_ensure_pod_running``). Kept as class constants so the values
    # stay grouped with the related logic.
    _POD_READY_POLL_INTERVAL = 5
    _POD_READY_TIMEOUT = 300

    def __init__(
        self,
        config: PipelineConfig,
        secrets: Secrets | None = None,
        callbacks: TrainingMonitorEventCallbacks | None = None,
    ) -> None:
        super().__init__(config, StageNames.TRAINING_MONITOR)
        self._secrets = secrets
        self._callbacks = callbacks or TrainingMonitorEventCallbacks()
        self._training_start_time: float = 0.0
        # Rate-limit cursor for ``[MONITOR] ALIVE`` status lines. The
        # runner publishes ``health_snapshot`` every 30 s, but the
        # 15-second cadence is preserved as a maximum-rate guard so
        # any future tightening of the runner interval doesn't spam
        # the log.
        self._last_status_log_time: float = 0.0
        # Track the latest event offset we've consumed so reconnects
        # via :meth:`JobClient.subscribe_events` resume from the
        # right place.
        self._last_offset: int = 0
        # Phase 11.E — handles for pipeline-level teardown. Captured
        # from context on first execute(); torn down later in
        # :meth:`cleanup` (called by the orchestrator AFTER all
        # downstream stages, including ModelRetriever, have run).
        # Why pipeline-level instead of stage-level: ``ModelRetriever``
        # streams adapters via a separate SSH ``tar | ssh`` pipeline
        # that bypasses the runner's FastAPI; without the tunnel
        # staying up the in-pod heartbeat cannot be refreshed and
        # :class:`PodTerminator` would podStop the pod mid-download.
        self._tunnel: SSHTunnelManager | None = None
        self._client: JobClient | None = None
        self._heartbeat_service: Any | None = None
        # Periodic log download — captured here so :meth:`cleanup`
        # can close the raw pod-SSH connection alongside the tunnel.
        # Single-node / mock flows keep both as ``None``.
        self._ssh_client: SSHClient | None = None
        self._log_manager: LogManager | None = None
        # PR-B — second LogManager for runner.log (uvicorn / pre-import
        # crashes). Shares the SSH ControlMaster with ``_log_manager``.
        self._runner_log_manager: LogManager | None = None
        # Pod recovery (laptop-sleep): captured from deployer context
        # so :meth:`_recover_pod_if_needed` can call the provider's
        # IRecoveryProbeProvider impl. The instance is propagated by
        # GPUDeployer (Phase 14.D+F refactor) — replaces the previous
        # ``provider_name == "runpod"`` string-check + inline SDK access.
        self._provider: Any | None = None
        self._provider_name: str | None = None
        self._resource_id: str | None = None
        # Bound recovery attempts so a flapping pod can't trap the
        # monitor in an infinite reconnect loop.
        self._recovery_attempts: int = 0
        # Milestone flags — keep INFO-level logs to one line per
        # interesting transition, so a 60-second run still shows
        # "trainer started", a metric, and "trainer exited" at minimum.
        self._first_event_logged: bool = False
        self._trainer_started_logged: bool = False

    # --- public entry point ---------------------------------------------

    def execute(self, context: dict[str, Any]) -> Result[dict[str, Any], AppError]:
        """Subscribe to the runner's WebSocket; return on terminal state.

        The launcher contract guarantees ``context["job_client"]``,
        ``context["ssh_tunnel"]``, ``context["job_id"]`` are populated
        on success. If they're missing it means TrainingLauncher
        didn't run (or ran with the legacy flow), and we fail loud
        rather than silently mock-mode.
        """
        # ---- mock fast-path (kept for in-process integration tests)
        deployer_context = context.get(StageNames.GPU_DEPLOYER, {})
        provider_info = deployer_context.get("provider_info", {})
        if provider_info.get("mock"):
            logger.info("[MONITOR] MOCK MODE — simulating successful training")
            if self._callbacks.on_training_started:
                self._callbacks.on_training_started()
            if self._callbacks.on_training_completed:
                self._callbacks.on_training_completed(0.0)
            return Ok({"status": "completed", "duration_seconds": 0.0, "mock": True})

        # ---- real flow: pull the launcher-provided handles
        client: JobClient | None = context.get("job_client")
        tunnel: SSHTunnelManager | None = context.get("ssh_tunnel")
        job_id: str | None = context.get("job_id")
        # Phase 11.E — capture for pipeline-level teardown in cleanup().
        # See class docstring for why pipeline-level instead of stage.
        self._client = client
        self._tunnel = tunnel
        self._heartbeat_service = context.get("control_plane_heartbeat")
        if client is None or job_id is None:
            return Err(
                TrainingError(
                    message=(
                        "TrainingMonitor: context is missing job_client / "
                        "job_id — TrainingLauncher must have run first "
                        "(Phase 6.3a contract)."
                    ),
                    code="MONITOR_LAUNCHER_NOT_WIRED",
                ),
            )

        logger.info(f"[MONITOR] Subscribing to runner events for job {job_id!r}")
        self._training_start_time = time.time()
        if self._callbacks.on_training_started:
            self._callbacks.on_training_started()

        # Build pod-side log pullers. Cloud providers expose both the
        # trainer's ``trainer.stdio.log`` and the runner's ``runner.log``
        # over SCP via PodLayout; single_node / mock flows skip — the
        # files are already on the host filesystem. Captured on ``self``
        # so :meth:`cleanup` closes the shared SSH ControlMaster after
        # downstream stages finish.
        (
            self._log_manager,
            self._runner_log_manager,
            self._ssh_client,
        ) = self._build_log_manager_from_context(deployer_context)
        # Pod-recovery preconditions: record provider instance + name +
        # resource so a transient WS error can trigger
        # provider.attempt_recovery() through the IRecoveryProbeProvider
        # capability. The provider instance is set by GPUDeployer.
        self._provider = deployer_context.get("provider")
        self._provider_name = deployer_context.get("provider_name")
        self._resource_id = deployer_context.get("resource_id")
        self._recovery_attempts = 0

        # Phase 11.E — no try/finally for tunnel teardown anymore.
        # Resources captured on ``self`` are torn down in :meth:`cleanup`
        # at orchestrator finalize-time, AFTER ModelRetriever has run.
        # An exception escaping ``_watch`` propagates up the stack
        # cleanly — orchestrator's reverse-order cleanup() will still
        # tear down the captured handles.
        watch_result = asyncio.run(
            self._watch_and_download(client, job_id, self._log_manager, self._runner_log_manager),
        )

        # Phase 9.C / Phase 11.A — Mac-side reconciliation for both
        # terminal markers:
        #
        # * ``cancelled.marker`` (Phase 9.C) — written when the
        #   CancellationCallback's 5-second flush deadline fires.
        #   The runner SIGKILLed the trainer before HF MLflow
        #   callback could close the run; MLflow shows RUNNING.
        #   Reconcile to ``KILLED``.
        # * ``completion.marker`` (Phase 11.A) — written on every
        #   natural completion, regardless of flush outcome. If
        #   Mac was asleep when HF MLflow callback ran, ``end_run``
        #   timed out and the run is stuck in RUNNING. Reconcile
        #   to ``FINISHED``.
        #
        # If both markers exist (rare race), cancellation wins —
        # explicit user-stop overrides natural completion.
        #
        # Best-effort throughout: any failure logs and moves on.
        self._reconcile_terminal_marker_if_present(context)

        return watch_result

    def cleanup(self) -> None:
        """Phase 11.E — pipeline-level teardown.

        Called by the orchestrator's reverse-order cleanup AFTER all
        downstream stages (notably :class:`ModelRetriever`) have run.
        Tears down resources captured during :meth:`execute` in this
        order:

        1. Stop the :class:`ControlPlaneHeartbeat` service so it
           doesn't keep pinging a tunnel we're about to close. We
           stop it first so a stop-during-close race doesn't surface
           as a transient error in the heartbeat task's logs.
        2. Close the :class:`JobClient` HTTP pool.
        3. Close the SSH tunnel (port forward + ControlMaster socket).

        Idempotent — second call is a no-op. All errors logged at
        DEBUG and swallowed; teardown failure must NOT mask whatever
        the pipeline result is.
        """
        # 1. Heartbeat service.
        if self._heartbeat_service is not None:
            try:
                asyncio.run(self._heartbeat_service.stop())
            except Exception as exc:
                logger.debug(
                    f"[MONITOR] control-plane heartbeat stop failed: {exc}"
                )
            self._heartbeat_service = None

        # 2. JobClient HTTP pool.
        if self._client is not None:
            try:
                asyncio.run(self._client.aclose())
            except Exception as exc:
                logger.debug(f"[MONITOR] client.aclose failed: {exc}")
            self._client = None

        # 3. SSH tunnel.
        if self._tunnel is not None:
            try:
                asyncio.run(self._tunnel.close())
            except Exception as exc:
                logger.debug(f"[MONITOR] tunnel.close failed: {exc}")
            self._tunnel = None

        # 4. Raw pod-SSH ControlMaster (held for log downloads + the
        # post-mortem diagnostics block in :meth:`_handle_trainer_exited`).
        if self._ssh_client is not None:
            try:
                self._ssh_client.close_master()
            except Exception as exc:
                logger.debug(f"[MONITOR] ssh_client.close_master failed: {exc}")
            self._ssh_client = None
        self._log_manager = None
        self._runner_log_manager = None

    # --- Phase 9.C / Phase 11.A reconciliation --------------------------

    #: Mapping marker filename → (target MLflow status, reason label).
    #: Cancellation must come first: if both markers exist (rare race
    #: where trainer was cancelled mid-natural-completion), explicit
    #: user-stop wins.
    _TERMINAL_MARKER_PRIORITY: tuple[tuple[str, str, str], ...] = (
        ("cancelled.marker", "KILLED", "cancellation"),
        ("completion.marker", "FINISHED", "natural_completion"),
    )

    def _reconcile_terminal_marker_if_present(
        self,
        context: dict[str, Any],
    ) -> None:
        """Reconcile MLflow run status against terminal-marker files.

        Two markers exist:

        * ``cancelled.marker`` (Phase 9.C) — written when the
          cancellation flush exceeded its 5-second budget; trainer
          got SIGKILLed before HF could ``end_run("KILLED")``. Force
          ``KILLED``.
        * ``completion.marker`` (Phase 11.A) — written on every
          natural completion. If Mac was asleep when HF tried to
          ``end_run("FINISHED")``, the run is stuck in ``RUNNING``.
          Force ``FINISHED``.

        Behaviour:

        * Both markers present ⇒ cancellation wins (priority list).
        * Marker missing ⇒ silent skip.
        * Marker JSON unreadable ⇒ warn + skip (don't bring down
          the rest of the pipeline cleanup).
        * MLflow run already terminal ⇒ NO-OP — the upstream MLflow
          callback or earlier reconciliation already closed it.
        * MLflow client fails (network / auth) ⇒ warn + skip.
        """
        import json
        from pathlib import Path

        attempt_dir_str = context.get(PipelineContextKeys.ATTEMPT_DIRECTORY)
        if not isinstance(attempt_dir_str, str) or not attempt_dir_str:
            # No attempt dir in context — same defensive default as
            # ``_persist_job_submission`` in the launcher; tests that
            # bypass pipeline_bootstrap won't have it.
            return

        attempt_dir = Path(attempt_dir_str)

        # Walk markers in priority order; first existing one wins.
        for marker_name, target_status, reason in self._TERMINAL_MARKER_PRIORITY:
            marker_path = attempt_dir / marker_name
            if not marker_path.is_file():
                continue

            try:
                payload = json.loads(marker_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning(
                    "[MONITOR] %s exists at %s but is unreadable (%s); " "skipping reconciliation",
                    marker_name,
                    marker_path,
                    exc,
                )
                # Don't try the next marker — disk read failure
                # likely means the whole attempt dir is in a bad
                # shape; bail.
                return

            run_id = payload.get("run_id")
            if not run_id:
                logger.info(
                    "[MONITOR] %s at %s has no run_id; MLflow " "reconciliation skipped (no run to address)",
                    marker_name,
                    marker_path,
                )
                return

            self._force_mlflow_run_status(
                run_id=run_id,
                target_status=target_status,
                marker_name=marker_name,
                reason=reason,
                payload=payload,
            )
            # First-found wins; do not process the lower-priority
            # marker even if it's also present.
            return

    def _force_mlflow_run_status(
        self,
        *,
        run_id: str,
        target_status: str,
        marker_name: str,
        reason: str,
        payload: dict[str, Any],
    ) -> None:
        """Force MLflow run from RUNNING → ``target_status``. Best-effort."""
        try:
            from mlflow.tracking import MlflowClient

            client = MlflowClient()
            run = client.get_run(run_id)
            current_status = getattr(
                getattr(run, "info", None),
                "status",
                None,
            )

            if current_status != "RUNNING":
                # Already terminal (KILLED / FAILED / FINISHED) —
                # something else closed the run. No-op.
                logger.debug(
                    "[MONITOR] MLflow run %s already terminal " "(status=%s); %s reconciliation no-op",
                    run_id,
                    current_status,
                    reason,
                )
                return

            client.set_terminated(run_id=run_id, status=target_status)
            # Use warning level for cancellation (operator should
            # know SIGKILL fallback fired) and info for natural
            # completion (expected post-sleep recovery).
            log_fn = logger.warning if marker_name == "cancelled.marker" else logger.info
            extra_note = ""
            if marker_name == "completion.marker" and payload.get(
                "flush_timed_out",
            ):
                extra_note = " — flush_timed_out=true; some metrics may be " "missing"
            log_fn(
                "[MONITOR] reconciled MLflow run %s: RUNNING → %s " "(%s indicated %s)%s",
                run_id,
                target_status,
                marker_name,
                reason,
                extra_note,
            )
        except Exception as exc:
            logger.warning(
                "[MONITOR] %s reconciliation for run_id=%s failed: %s "
                "— operator may need to manually set the MLflow run "
                "status",
                marker_name,
                run_id,
                exc,
            )

    # --- pod-side log puller (sync; spun off into a thread) -------------

    def _build_log_manager_from_context(
        self,
        deployer_context: dict[str, Any],
    ) -> tuple[LogManager | None, LogManager | None, SSHClient | None]:
        """Construct SSH-backed :class:`LogManager` instances for both
        ``trainer.stdio.log`` and ``runner.log`` from gpu_deployer
        context, or ``(None, None, None)`` when the active flow doesn't
        need one (single_node / mock / missing handles).

        Returns ``(trainer_lm, runner_lm, ssh_client)``. Both managers
        share a single SSH ControlMaster connection — RP6 (5s polling)
        is cheap because each tick is just a ``stat -c%s`` round-trip
        on the existing connection.

        The legacy SSH-polling monitor did the periodic
        ``log_manager.download(silent=True)`` itself; restoring that
        keeps ``runs/<id>/attempts/<n>/logs/{trainer.stdio,runner}.log``
        populated as the run progresses, which the web UI's
        :class:`LogDock` already tails.
        """
        provider_type = deployer_context.get("provider_type")
        ssh_host = deployer_context.get("ssh_host")
        # Cloud-only: local providers ship the trainer on the same
        # host so the log file is already available; nothing to scp.
        if provider_type != "cloud" or not ssh_host:
            return None, None, None

        workspace_path = deployer_context.get("workspace_path")
        if not workspace_path:
            logger.debug(
                "[MONITOR] deployer_context missing workspace_path — "
                "periodic log download disabled",
            )
            return None, None, None

        try:
            ssh_port = int(deployer_context.get("ssh_port") or SSH_PORT_DEFAULT)
        except (TypeError, ValueError):
            ssh_port = SSH_PORT_DEFAULT

        is_alias_mode = bool(deployer_context.get("is_alias_mode"))
        ssh_user = deployer_context.get("ssh_user")
        ssh_key_path = deployer_context.get("ssh_key_path")
        try:
            ssh_client = SSHClient(
                host=ssh_host,
                port=ssh_port,
                username=None if is_alias_mode else ssh_user,
                key_path=None if is_alias_mode else ssh_key_path,
            )
        except Exception as exc:
            logger.debug(f"[MONITOR] SSH client construction failed: {exc}")
            return None, None, None

        # Build PodLayout from the deployer's workspace_path so we
        # pull the canonical per-run trainer.stdio.log. Pre-PodLayout
        # this code used a global LogManager.DEFAULT_REMOTE_PATH —
        # which silently downloaded from the wrong path and made
        # the periodic-tail feature a no-op. PodLayout closes that.
        from pathlib import PurePosixPath

        from ryotenkai_shared.utils.pod_layout import PodLayout

        try:
            pod_layout = PodLayout.from_root(PurePosixPath(workspace_path))
        except ValueError as exc:
            logger.debug(f"[MONITOR] Cannot build PodLayout: {exc}")
            with contextlib.suppress(Exception):
                ssh_client.close_master()
            return None, None, None

        try:
            mac_layout = get_run_log_layout()
            trainer_lm = LogManager(
                ssh_client,
                remote_path=str(pod_layout.trainer_stdio_log),
                local_path=mac_layout.remote_trainer_stdio_log,
            )
            # PR-B — second LogManager for runner.log. Without it,
            # uvicorn pre-import crashes (e.g. ImportError before the
            # FastAPI app even binds 8080) never hit Mac until the
            # postmortem fires — and by then the pod might be evicted.
            runner_lm = LogManager(
                ssh_client,
                remote_path=str(pod_layout.runner_log),
                local_path=mac_layout.remote_runner_log,
            )
        except Exception as exc:
            logger.debug(f"[MONITOR] LogManager init failed: {exc}")
            with contextlib.suppress(Exception):
                ssh_client.close_master()
            return None, None, None
        return trainer_lm, runner_lm, ssh_client

    async def _log_downloader_loop(
        self,
        trainer_log_manager: LogManager,
        runner_log_manager: LogManager | None = None,
    ) -> None:
        """Pull ``trainer.stdio.log`` (and optionally ``runner.log``)
        every :data:`TRAINING_MONITOR_LOG_DOWNLOAD_INTERVAL` seconds.

        Single debug line per tick per file —
        ``ok`` / ``no data`` / ``error``. Errors on either pull do not
        abort the loop or each other; both pulls are independent
        best-effort observability.
        """
        while True:
            try:
                await asyncio.sleep(TRAINING_MONITOR_LOG_DOWNLOAD_INTERVAL)
            except asyncio.CancelledError:
                return

            # Trainer stdio — ground truth for trainer subprocess output.
            try:
                ok_trainer = await asyncio.to_thread(
                    trainer_log_manager.download, silent=True,
                )
                logger.debug(
                    f"[MONITOR] trainer.stdio.log download "
                    f"{'ok' if ok_trainer else 'no data'}",
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.debug(f"[MONITOR] trainer.stdio.log download error: {exc}")

            # Runner stdout (uvicorn) — covers pre-import crashes the
            # trainer pump cannot capture.
            if runner_log_manager is not None:
                try:
                    ok_runner = await asyncio.to_thread(
                        runner_log_manager.download, silent=True,
                    )
                    logger.debug(
                        f"[MONITOR] runner.log download "
                        f"{'ok' if ok_runner else 'no data'}",
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.debug(f"[MONITOR] runner.log download error: {exc}")

    # --- pod resilience (laptop-sleep recovery) ------------------------

    _RECOVERY_ATTEMPT_CAP = 3

    def _recover_pod_if_needed(
        self,
        exc: JobClientError,
    ) -> Result[dict[str, Any], AppError] | None:
        """Best-effort RunPod SDK wake-up for a stopped pod.

        When the WS subscription dies with :class:`JobClientError`,
        the most likely cause on a Mac control plane is a laptop
        going to sleep — RunPod's idle terminator stops the pod a
        few minutes later. We probe the SDK to find out which case
        we're in and, when the pod is merely stopped, request a
        wake-up so the operator's restart attempt has a live pod
        to talk to.

        Returns:
            ``None`` — recovery did not apply (non-runpod / no
              resource id / no SDK key), caller should fall through
              to the original ``Err``.
            ``Err`` — pod is in a terminal state or wake-up failed;
              the caller should propagate this.
            ``Err(MONITOR_POD_RECOVERED)`` — wake-up succeeded; the
              monitor cannot itself rebuild the upstream tunnel +
              JobClient (those belong to TrainingLauncher), so we
              surface a clear actionable error instead of looping.

        Capped at :data:`_RECOVERY_ATTEMPT_CAP` per stage execution to
        prevent a flapping pod from trapping us in a tight loop.
        """
        # Capability-Protocol gate: only providers that declare
        # ``IRecoveryProbeProvider`` (and have ``supports_recovery_probe=
        # true`` in their manifest) participate in the recovery loop.
        # Replaces the hardcoded ``self._provider_name != "runpod"``
        # skip — a third cloud provider with similar GraphQL-probe
        # semantics inherits the Protocol and the recovery path lights
        # up automatically.
        from ryotenkai_providers.training.interfaces import (
            IRecoveryProbeProvider,
            ProviderStatus,
        )

        if self._provider is None or not self._resource_id:
            return None
        if not isinstance(self._provider, IRecoveryProbeProvider):
            return None
        if self._recovery_attempts >= self._RECOVERY_ATTEMPT_CAP:
            return Err(
                TrainingError(
                    message=(
                        f"runner connection lost ({exc}); "
                        "pod recovery exhausted after "
                        f"{self._RECOVERY_ATTEMPT_CAP} attempts"
                    ),
                    code="MONITOR_RECOVERY_EXHAUSTED",
                ),
            )
        self._recovery_attempts += 1

        logger.warning(
            "[MONITOR] runner connection lost (%s) — invoking provider "
            "recovery for resource %s",
            exc,
            self._resource_id,
        )
        recovery_result = self._provider.attempt_recovery(
            resource_id=self._resource_id,
        )
        if recovery_result.is_err():
            err = recovery_result.unwrap_err()
            # Map provider error codes to monitor codes for backwards
            # compatibility of operator-facing error vocabulary.
            code_map = {
                "POD_PROBE_FAILED": "MONITOR_POD_PROBE_FAILED",
                "POD_TERMINAL": "MONITOR_POD_TERMINATED",
                "POD_WAKE_FAILED": "MONITOR_POD_WAKE_FAILED",
            }
            return Err(
                TrainingError(
                    message=err.message,
                    code=code_map.get(err.code or "", "MONITOR_RECOVERY_FAILED"),
                ),
            )
        status = recovery_result.unwrap()
        if status == ProviderStatus.CONNECTED:
            # Pod was already running; WS failure unrelated. Caller
            # surfaces the original error.
            return None
        # Pod was stopped, woke back up — pipeline restart needed.
        return Err(
            TrainingError(
                message=(
                    f"pod {self._resource_id} was stopped (likely "
                    "laptop sleep); woke it back up — restart the "
                    "pipeline to resume training"
                ),
                code="MONITOR_POD_RECOVERED",
            ),
        )

    # --- post-mortem diagnostics ---------------------------------------

    def _collect_death_diagnostics(self) -> None:
        """Render a self-contained autopsy when the trainer dies
        non-zero. Three sources, in priority order:

        1. **Local ``trainer.stdio.log``** — pulled by LogManager
           periodically + on cleanup. Contains the full
           stdout/stderr of the trainer subprocess including Python
           tracebacks AND native faulthandler dumps (SEGV/ABRT/CUDA).
           This is the ground-truth for application-level death.
        2. **Local ``runner.log``** — pulled the same way. Captures
           uvicorn boot output: surfaces ImportError / SyntaxError
           that fired BEFORE the trainer ever spawned (when
           trainer.stdio.log is empty by definition).
        3. **Live SSH probes** — kernel signals (OOM, NVRM/Xid)
           and current GPU state. These cannot come from local files
           because they are environment-level, not subprocess output.

        Each source gets ``<<MISSING>>`` / ``<<EMPTY>>`` tokens when
        appropriate so an operator can distinguish "file does not
        exist" from "file exists but empty" from "file has data" —
        no more "0/6 probes empty" black-box.

        Skipped silently if no SSH client AND no log layout are wired
        (test / mock paths); per-probe failures are also swallowed so
        one broken command doesn't suppress the rest.
        """
        logger.info(
            "[MONITOR:POSTMORTEM] non-zero exit detected — collecting diagnostics",
        )

        # --- Source 1+2: local files pulled by LogManager ---
        try:
            log_layout = get_run_log_layout()
        except Exception:  # noqa: BLE001 — defensive (test paths)
            log_layout = None

        if log_layout is not None:
            self._dump_local_log_tail(
                label="trainer",
                path=log_layout.remote_trainer_stdio_log,
                tail_lines=30,
            )
            self._dump_local_log_tail(
                label="runner",
                path=log_layout.remote_runner_log,
                tail_lines=30,
            )

        # --- Source 3: live SSH probes (kernel + GPU) ---
        if self._ssh_client is None:
            logger.info(
                "[MONITOR:POSTMORTEM] (no SSH client wired — kernel/GPU probes skipped)",
            )
            return

        # Kernel signals + GPU state. These cannot come from local
        # log files — they are environment-level.
        probes: list[tuple[str, str, int]] = [
            (
                "dmesg_kernel_signals",
                "dmesg 2>/dev/null | grep -iE 'oom|kill|memory|nvrm|xid|nvidia' | tail -n 30",
                10,
            ),
            ("dmesg_tail", "dmesg 2>/dev/null | tail -n 30", 10),
            (
                "nvidia_smi",
                "nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total"
                " --format=csv,noheader",
                10,
            ),
        ]
        for label, command, timeout in probes:
            try:
                ok, stdout, stderr = self._ssh_client.exec_command(
                    command=command,
                    silent=True,
                    timeout=timeout,
                )
            except Exception as exc:  # noqa: BLE001 — defensive
                logger.debug(f"[MONITOR:POSTMORTEM] {label} raised: {exc}")
                continue
            text = (stdout or "").strip()
            if ok and text:
                for line in text.splitlines():
                    logger.info(f"[MONITOR:POSTMORTEM] {label}: {line}")
            elif ok and not text:
                logger.info(f"[MONITOR:POSTMORTEM] {label}: <<EMPTY>>")
            elif stderr:
                logger.debug(f"[MONITOR:POSTMORTEM] {label} stderr: {stderr.strip()}")

    def _dump_local_log_tail(
        self,
        *,
        label: str,
        path: "Path",
        tail_lines: int,
    ) -> None:
        """Read the tail of a local log file and emit it under the
        ``[MONITOR:POSTMORTEM:<label>]`` prefix.

        Distinguishes three states explicitly so the operator can
        tell missing-file from empty-file from has-content. The
        ``trainer.stdio.log`` / ``runner.log`` paths come from
        :class:`LogLayout` so the layout drift between writer (pod
        Supervisor / runner_launcher) and reader (this code) is
        physically impossible — both go through the same source of
        truth.

        Pod-side ``trainer.stdio.log`` is written by Supervisor with
        ``[OUT] ``/``[ERR] `` per-line prefixes (single-file merge of
        stdout+stderr for atomic ordering on disk). The prefix is an
        on-disk serialization detail, NOT operator-facing — strip it
        when emitting to ``pipeline.log``. Operators see clean trainer
        output without protocol noise.
        """
        try:
            if not path.exists():
                logger.info(f"[MONITOR:POSTMORTEM:{label}] <<MISSING>>")
                return
            size = path.stat().st_size
            if size == 0:
                logger.info(f"[MONITOR:POSTMORTEM:{label}] <<EMPTY>>")
                return
            with path.open(encoding="utf-8", errors="replace") as fh:
                lines = fh.readlines()
            tail = lines[-tail_lines:] if len(lines) > tail_lines else lines
            for raw_line in tail:
                line = raw_line.rstrip()
                # Strip Supervisor-internal channel prefix. Both 6-char
                # markers ("[OUT] " / "[ERR] ") map to plain operator
                # output — severity routing through logger.error would
                # alarm-spam on every trainer Python traceback, which
                # is "expected trainer-side error" not "control-plane
                # error". Keep everything at INFO.
                if line.startswith("[OUT] ") or line.startswith("[ERR] "):
                    line = line[6:]
                logger.info(f"[MONITOR:POSTMORTEM:{label}] {line}")
        except OSError as exc:  # noqa: BLE001 — defensive
            logger.warning(
                f"[MONITOR:POSTMORTEM:{label}] read failed: {exc}",
            )

    async def _watch_and_download(
        self,
        client: JobClient,
        job_id: str,
        log_manager: LogManager | None,
        runner_log_manager: LogManager | None = None,
    ) -> Result[dict[str, Any], AppError]:
        """Run :meth:`_watch` with a parallel periodic-download task.

        The downloader is spawned only when the trainer
        :class:`LogManager` is wired (cloud providers); for local / mock
        flows we still want the unchanged WS-watcher behaviour. PR-B
        adds an optional second ``runner_log_manager`` so uvicorn /
        pre-import crashes also flow to Mac as the run progresses, not
        just at postmortem time. The final ``silent=False`` flush of
        BOTH files guarantees on-disk artefact completeness even when
        the watcher exits between tick boundaries.
        """
        if log_manager is None:
            return await self._watch(client, job_id)

        download_task = asyncio.create_task(
            self._log_downloader_loop(log_manager, runner_log_manager),
            name="monitor.log_downloader",
        )
        try:
            return await self._watch(client, job_id)
        finally:
            download_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await download_task
            # Final flush trainer.stdio.log
            try:
                ok = await asyncio.wait_for(
                    asyncio.to_thread(log_manager.download, silent=False),
                    timeout=_FINAL_LOG_FLUSH_TIMEOUT,
                )
            except (TimeoutError, Exception) as exc:
                logger.debug(f"[MONITOR] final trainer.stdio.log flush error: {exc}")
            else:
                logger.debug(
                    f"[MONITOR] final trainer.stdio.log flush {'ok' if ok else 'no data'}",
                )
            # Final flush runner.log (PR-B)
            if runner_log_manager is not None:
                try:
                    ok = await asyncio.wait_for(
                        asyncio.to_thread(runner_log_manager.download, silent=False),
                        timeout=_FINAL_LOG_FLUSH_TIMEOUT,
                    )
                except (TimeoutError, Exception) as exc:
                    logger.debug(f"[MONITOR] final runner.log flush error: {exc}")
                else:
                    logger.debug(
                        f"[MONITOR] final runner.log flush {'ok' if ok else 'no data'}",
                    )

    # --- async core -----------------------------------------------------

    async def _watch(
        self,
        client: JobClient,
        job_id: str,
    ) -> Result[dict[str, Any], AppError]:
        """Iterate over WS events; dispatch callbacks; return on terminal.

        Reconnect / replay-truncation are handled by
        :meth:`JobClient.subscribe_events` itself; we only catch the
        exit conditions and translate them to the monitor's
        ``Result`` shape.
        """
        try:
            async for event in client.subscribe_events(
                job_id,
                since=self._last_offset,
            ):
                # Track offset so a manual restart of subscribe (e.g.
                # after ReplayTruncatedError) resumes correctly.
                offset = event.get("offset")
                if isinstance(offset, int):
                    self._last_offset = offset + 1

                terminal = self._dispatch_event(event)
                if terminal is not None:
                    return terminal
            # Stream ended cleanly without a terminal kind — treat as
            # a server-side bug; the runner is supposed to close only
            # on FSM terminal.
            return Err(
                TrainingError(
                    message="runner closed the event stream before reaching terminal state",
                    code="MONITOR_STREAM_EOF",
                ),
            )
        except JobNotFoundError:
            return Err(
                TrainingError(
                    message=f"runner reports unknown job {job_id!r} " "(pod restart wiped state?)",
                    code="MONITOR_JOB_NOT_FOUND",
                ),
            )
        except ReplayTruncatedError:
            # Buffer rolled past the offset we asked for. Refetch the
            # current snapshot, treat it as authoritative, and call
            # back the appropriate callback. We don't try to replay
            # the missed events — they're already gone.
            return await self._fallback_to_status(client, job_id)
        except JobClientError as exc:
            recovery = self._recover_pod_if_needed(exc)
            if recovery is not None:
                # Recovery attempted — return a structured Result the
                # caller (orchestrator) can use to decide whether to
                # restart the stage. We deliberately do NOT loop and
                # rebuild handles inline: that requires re-running the
                # launcher (tunnel+client+job_id), which is the
                # orchestrator's job.
                return recovery
            return Err(
                TrainingError(
                    message=f"runner client error: {exc}",
                    code="MONITOR_CLIENT_ERROR",
                ),
            )

    def _dispatch_event(
        self,
        event: dict[str, Any],
    ) -> Result[dict[str, Any], AppError] | None:
        """Fire callbacks; return a terminal Result or ``None`` to keep
        listening.

        Recognised event kinds (everything else is logged at debug
        and ignored):
        - first event (any kind) → "[MONITOR] WS stream open"
        - ``trainer_spawned`` → "[MONITOR] Trainer process started"
        - ``health_snapshot`` → ``on_resource_check`` + rate-limited ALIVE
        - ``trainer_exited`` → ``on_training_completed`` /
          ``on_training_failed`` / ``on_process_died`` based on
          payload, then return terminal Result
        - other kinds → log only at debug

        Note: ``trainer_log`` events were removed in the data-plane
        refactor — trainer stdout/stderr now lands in
        ``trainer.stdio.log`` on the pod (written by the Supervisor's
        pump) and is pulled to Mac via LogManager scp. The Web UI's
        LogDock reads ``trainer.stdio.log`` directly. The bus / WS
        stream carries only control + telemetry events.
        """
        kind = event.get("kind") or ""
        payload = event.get("payload") or {}

        if not self._first_event_logged:
            self._first_event_logged = True
            logger.info("[MONITOR] WS event stream open — runner is reachable")

        if kind == "trainer_spawned" and not self._trainer_started_logged:
            self._trainer_started_logged = True
            pid = payload.get("pid")
            logger.info(
                "[MONITOR] Trainer process started%s",
                f" (pid={pid})" if pid else "",
            )

        if kind == "health_snapshot":
            self._maybe_log_status(payload)
            if self._callbacks.on_resource_check:
                try:
                    self._callbacks.on_resource_check(dict(payload))
                except Exception as exc:
                    logger.debug(f"[MONITOR] on_resource_check raised: {exc}")
            return None

        if kind == "trainer_exited":
            return self._handle_trainer_exited(payload)

        # Catch-all for FSM-state transitions emitted alongside
        # trainer_exited. The runner publishes a structured event
        # for the transition itself; we use it as a backstop so a
        # missed ``trainer_exited`` (e.g. supervisor crash) still
        # surfaces a terminal state to the orchestrator.
        state = payload.get("state") if isinstance(payload, dict) else None
        if isinstance(state, str) and state in _TERMINAL_STATES:
            return self._terminal_from_state(state, payload)

        logger.debug(f"[MONITOR] event kind={kind!r} (no callback)")
        return None

    def _maybe_log_status(self, payload: dict[str, Any]) -> None:
        """Emit a rate-limited ``[MONITOR] ALIVE`` line for the operator.

        The legacy SSH-polling monitor printed a one-line status every
        15 s so the user could tell at a glance that training was
        progressing without tailing trainer stdout. We restore the
        same surface against the WS event stream — driven by the
        runner's ``health_snapshot`` events instead of an SSH probe.

        Missing fields render as ``—`` rather than ``0`` so the user
        can distinguish "GPU is genuinely idle" from "psutil/nvidia-smi
        couldn't read the value".
        """
        now = time.time()
        if now - self._last_status_log_time < TRAINING_MONITOR_LOG_STATUS_INTERVAL:
            return
        self._last_status_log_time = now

        elapsed = max(0.0, now - self._training_start_time)
        elapsed_str = str(timedelta(seconds=int(elapsed)))

        gpu_util = payload.get("gpu_util_percent")
        vram_pct = payload.get("gpu_memory_percent")
        vram_used = payload.get("vram_used_gb")
        vram_total = payload.get("vram_total_gb")
        gpu_temp = payload.get("gpu_temp_c")
        cpu = payload.get("cpu_percent")
        ram_used = payload.get("ram_used_gb")
        ram_total = payload.get("ram_total_gb")

        # VRAM: prefer absolute GB if both fields present (richer signal
        # for operator capacity planning), fall back to percent, else "—".
        if isinstance(vram_used, (int, float)) and isinstance(vram_total, (int, float)):
            vram_str = (
                f"{vram_used:.1f}/{vram_total:.0f} GB"
                + (f" ({vram_pct:.0f}%)" if isinstance(vram_pct, (int, float)) else "")
            )
        elif isinstance(vram_pct, (int, float)):
            vram_str = f"{vram_pct:.0f}%"
        else:
            vram_str = "—"

        ram_str = (
            f"{ram_used:.1f}/{ram_total:.0f} GB"
            if isinstance(ram_used, (int, float)) and isinstance(ram_total, (int, float))
            else "—"
        )

        # ``running`` matches the develop-branch convention; the legacy
        # ``ALIVE`` token confused operators ("alive vs what?") — here
        # we always log when the trainer is actively running so the
        # state name matches the FSM JobState.value.
        logger.info(
            "[MONITOR] running | %s | GPU: %s | VRAM: %s | Temp: %s | CPU: %s | RAM: %s",
            elapsed_str,
            f"{gpu_util:.0f}%" if isinstance(gpu_util, (int, float)) else "—",
            vram_str,
            f"{gpu_temp:.0f}C" if isinstance(gpu_temp, (int, float)) else "—",
            f"{cpu:.0f}%" if isinstance(cpu, (int, float)) else "—",
            ram_str,
        )

    def _handle_trainer_exited(
        self,
        payload: dict[str, Any],
    ) -> Result[dict[str, Any], AppError]:
        """Translate a ``trainer_exited`` event to a terminal Result.

        Payload shape (per :class:`Supervisor._reap`):
        ``{exit_code, signal, cancellation_requested}``. The trainer's
        last stdout/stderr lives on disk in ``trainer.stdio.log`` and
        is rendered by :meth:`_collect_death_diagnostics` from the
        SCP-pulled local copy — single source, no embedded WS tail.
        """
        duration = max(0.0, time.time() - self._training_start_time)
        exit_code = payload.get("exit_code")
        signal_name = payload.get("signal")
        cancelled = bool(payload.get("cancellation_requested"))

        # Pull post-mortem context from the pod BEFORE returning Err.
        # Cancellation is operator-initiated (``stop`` from CLI/web UI)
        # so there's no crash to investigate; we only run the probes
        # on a genuine non-zero / signal-killed exit. The pod is still
        # reachable here because :class:`PodTerminator` runs from the
        # runner's terminal hook, which fires AFTER the
        # ``trainer_exited`` event; deployer-side teardown happens
        # later via :meth:`cleanup`.
        if not cancelled and ((isinstance(exit_code, int) and exit_code != 0) or signal_name):
            self._collect_death_diagnostics()

        if exit_code == 0 and not cancelled:
            if self._callbacks.on_training_completed:
                self._callbacks.on_training_completed(duration)
            return Ok(
                {
                    "status": "completed",
                    "duration_seconds": duration,
                    "duration_human": str(timedelta(seconds=int(duration))),
                    # Phase 11.C-2 — hint for stage_execution_loop's
                    # _capture_pod_status_if_present. Maps to PodTerminator's
                    # default Phase 11.B outcome on natural completion:
                    # podStop ⇒ ``"stopped"``. ``PodAvailabilityProbe`` is
                    # the live source of truth; this just speeds up CLI /
                    # Web UI hints by avoiding a RunPod GraphQL query for
                    # the common case.
                    "pod_terminal_state": "stopped",
                }
            )

        if cancelled:
            return self._fail(
                f"training cancelled (exit_code={exit_code}, signal={signal_name})",
                duration,
                code="TRAINING_CANCELLED",
            )

        if exit_code is None and signal_name is None:
            # Process died without a parsed exit code — use
            # ``on_process_died`` per legacy contract.
            if self._callbacks.on_process_died:
                self._callbacks.on_process_died(duration)
            return self._fail("trainer process died (no exit code)", duration)

        return self._fail(
            f"trainer exited non-zero (exit_code={exit_code}, signal={signal_name})",
            duration,
        )

    def _terminal_from_state(
        self,
        state: str,
        payload: dict[str, Any],
    ) -> Result[dict[str, Any], AppError]:
        """Backstop: derive terminal Result from a bare FSM state event."""
        duration = max(0.0, time.time() - self._training_start_time)
        message = payload.get("message") if isinstance(payload, dict) else None

        if state == _TERMINAL_COMPLETED:
            if self._callbacks.on_training_completed:
                self._callbacks.on_training_completed(duration)
            return Ok({"status": "completed", "duration_seconds": duration})

        if state == _TERMINAL_CANCELLED:
            return self._fail(
                f"training cancelled ({message or 'no detail'})",
                duration,
                code="TRAINING_CANCELLED",
            )

        return self._fail(
            f"training failed ({message or 'no detail'})",
            duration,
        )

    def _fail(
        self,
        message: str,
        duration: float,
        *,
        code: str = "TRAINING_FAILED",
    ) -> Result[dict[str, Any], AppError]:
        """Common path for FAILED / CANCELLED → ``Err`` translation."""
        if self._callbacks.on_training_failed:
            try:
                self._callbacks.on_training_failed(message, duration)
            except Exception as exc:
                logger.debug(f"[MONITOR] on_training_failed raised: {exc}")
        return Err(TrainingError(message=message, code=code))

    async def _fallback_to_status(
        self,
        client: JobClient,
        job_id: str,
    ) -> Result[dict[str, Any], AppError]:
        """Buffer rolled past us — refetch :meth:`JobClient.get_status`
        and translate the snapshot to a terminal Result.

        We deliberately do NOT try to resume subscribing here —
        events between the truncation and ``get_status`` are gone,
        and re-subscribing would race the very same buffer rollover
        again. The orchestrator's restart logic re-runs the monitor
        if the job is somehow still active.
        """
        try:
            snap = await client.get_status(job_id)
        except JobClientError as exc:
            return Err(
                TrainingError(
                    message=("WebSocket replay buffer rolled over and " f"GET /jobs/{job_id} failed: {exc}"),
                    code="MONITOR_STATUS_FETCH_FAILED",
                ),
            )

        state = (snap.get("state") or "").lower()
        if state in _TERMINAL_STATES:
            return self._terminal_from_state(state, snap)
        return Err(
            TrainingError(
                message=(
                    f"WebSocket replay truncated and FSM is still "
                    f"non-terminal ({state!r}); cannot resume "
                    "monitoring without losing events"
                ),
                code="MONITOR_REPLAY_TRUNCATED",
            ),
        )
