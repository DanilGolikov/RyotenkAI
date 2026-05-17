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

Phase 4 (event-system unification, 2026-05-16): the legacy
``TrainingMonitorEventCallbacks`` dataclass was removed. The monitor
now takes an optional :class:`IEventEmitter` and emits typed
``ryotenkai.control.training.*`` envelopes
(``monitor_started`` on entry, ``monitor_timeout`` on
``ReplayTruncatedError`` / ``JobClientError``). Detailed
per-resource-check telemetry stays in :data:`logger` for now — the
Phase 5 coverage-gap pass will fold it back into typed events.

Return shape (post Phase A2 Batch 10): ``execute(context) -> dict``;
raises typed :class:`RyotenkAIError` (``TrainingFailedError`` /
``InternalError``) on failure.

Async strategy: same as the launcher — sync facade wrapping a
``asyncio.run`` islet around the WebSocket consumer.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from ryotenkai_control.pipeline.stages.base import PipelineStage
from ryotenkai_control.pipeline.stages.constants import PipelineContextKeys, StageNames
from ryotenkai_control.pipeline.stages.managers.log_fetcher import LogFetcher
from ryotenkai_control.pipeline.stages.training_monitor_pod_event_forwarder import (
    PodEventForwarder,
)
from ryotenkai_shared.constants import (
    CONSOLE_LINE_WIDTH,
    LOG_DOWNLOAD_INTERVAL_DEFAULT,
    SSH_PORT_DEFAULT,
)
from ryotenkai_shared.errors import (
    InternalError,
    RyotenkAIError,
    TrainingFailedError,
    TrainingOOMError,
)
from ryotenkai_shared.events import UNKNOWN_OFFSET
from ryotenkai_shared.events.types.control_training import (
    TrainingMonitorStartedEvent,
    TrainingMonitorStartedPayload,
    TrainingMonitorTimeoutEvent,
    TrainingMonitorTimeoutPayload,
)
from ryotenkai_shared.utils.clients.job_client import (
    JobClientError,
    JobNotFoundError,
    ReplayTruncatedError,
)
from ryotenkai_shared.utils.logger import get_run_log_layout, logger

# Phase 3 PR-3.2 (transport-unification-v2): SSHClient removed from
# this module. Training monitor's runtime path is HTTP-only after
# PR-2.3 migrated LogManager → LogFetcher; the third tuple slot of
# ``_build_log_manager_from_context`` is always None, so the
# ``self._ssh_client`` field below holds None too. Kept as a typed
# attribute for shape compatibility with downstream cleanup code
# until that branch is also removed in PR-3.3.

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
    from pathlib import Path

    from ryotenkai_shared.config import PipelineConfig
    from ryotenkai_shared.config.secrets.model import Secrets
    from ryotenkai_shared.events import IEventEmitter
    from ryotenkai_shared.utils.clients.job_client import JobClient
    from ryotenkai_shared.utils.clients.ssh_tunnel import SSHTunnelManager


# Source URI for envelopes emitted from this stage.
_STAGE_SOURCE = "control://orchestrator/training_monitor"


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
    - ``completed`` → returns ``{status: "completed", duration_seconds}``
    - ``failed`` → raises :class:`TrainingFailedError` (legacy code ``TRAINING_FAILED``)
    - ``cancelled`` → raises :class:`TrainingFailedError` (legacy code ``TRAINING_CANCELLED``)

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
        *,
        emitter: IEventEmitter | None = None,
    ) -> None:
        super().__init__(config, StageNames.TRAINING_MONITOR)
        self._secrets = secrets
        self._emitter = emitter
        self._training_start_time: float = 0.0
        # Tracks the wall-clock instant we last saw an event from the
        # pod. ``ReplayTruncated``/``JobClient`` errors fire a typed
        # ``monitor_timeout`` envelope that carries this value so the
        # consumer can render "no event in N seconds" without a side
        # channel.
        self._last_event_at: datetime = datetime.now(UTC)
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
        # ``object | None`` (not ``SSHClient | None``) because the
        # SSHClient import is gone from this module after Phase 3
        # PR-3.2. The slot is reserved for the deployer context's
        # third tuple element (legacy bootstrap SSH handle) — but
        # after PR-2.3 the third element is always None.
        self._ssh_client: object | None = None
        self._log_manager: LogFetcher | None = None
        # PR-B — second LogManager for runner.log (uvicorn / pre-import
        # crashes). Shares the SSH ControlMaster with ``_log_manager``.
        self._runner_log_manager: LogFetcher | None = None
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
        # Cached run id (resolved on execute() entry); used by the
        # async ``_watch`` helper which cannot reach the pipeline
        # context.
        self._run_id_cache: str | None = None

        # Pod-event forwarder — holds the dispatch / forward / status
        # logging / replay-fallback logic that used to live inline on
        # this class. Built lazily via :meth:`_get_forwarder` so unit
        # tests that bypass ``__init__`` via ``TrainingMonitor.__new__``
        # (the ``_make_monitor`` helper in
        # ``test_training_monitor_v2.py``) still see a working forwarder
        # the first time ``_dispatch_event`` / ``_maybe_log_status`` is
        # exercised. See ``training_monitor_pod_event_forwarder.py`` for
        # the extracted surface. The forwarder reads + mutates a small
        # slice of monitor state (the milestone flags,
        # ``_last_event_at``, ``_last_status_log_time``, ``_last_offset``,
        # ``_training_start_time``) via the ``MonitorState`` Protocol;
        # trainer-exit + reconciliation callbacks stay on this class.
        self._forwarder: PodEventForwarder | None = None

    def _get_forwarder(self) -> PodEventForwarder:
        """Return the lazily-constructed pod-event forwarder.

        Built on first access so test helpers that build a partially-
        initialised monitor via ``TrainingMonitor.__new__`` still get a
        functional forwarder when they invoke
        :meth:`_dispatch_event` / :meth:`_maybe_log_status` directly.
        Subsequent calls return the cached instance.

        ``getattr`` fallback handles the ``__new__``-only test path
        where the attribute is never assigned to ``None`` by ``__init__``.
        """
        if getattr(self, "_forwarder", None) is None:
            self._forwarder = PodEventForwarder(
                state=self,
                emitter=getattr(self, "_emitter", None),
                handle_trainer_exited=self._handle_trainer_exited,
                terminal_from_state=self._terminal_from_state,
                fallback_to_status=self._fallback_to_status,
                terminal_states=_TERMINAL_STATES,
                log_status_interval=TRAINING_MONITOR_LOG_STATUS_INTERVAL,
            )
        assert self._forwarder is not None  # narrows mypy / type checker
        return self._forwarder

    # --- emitter wiring -------------------------------------------------

    def set_emitter(self, emitter: IEventEmitter) -> None:
        """Inject an emitter after construction (lazy wiring)."""
        self._emitter = emitter
        # ``getattr`` fallback — when called on a test-fixture monitor
        # built via ``TrainingMonitor.__new__`` the attribute may not
        # exist yet (set lazily by :meth:`_get_forwarder`).
        forwarder = getattr(self, "_forwarder", None)
        if forwarder is not None:
            forwarder.set_emitter(emitter)

    @staticmethod
    def _resolve_run_id(context: dict[str, Any]) -> str:
        run_obj = context.get(PipelineContextKeys.RUN)
        run_name = getattr(run_obj, "name", None)
        if isinstance(run_name, str) and run_name:
            return run_name
        return "unknown"

    def _emit_monitor_started(
        self,
        run_id: str,
        *,
        pod_endpoint: str,
        poll_interval_s: float,
    ) -> None:
        if self._emitter is None:
            return
        self._emitter.emit(
            TrainingMonitorStartedEvent(
                source=_STAGE_SOURCE,
                run_id=run_id,
                offset=UNKNOWN_OFFSET,
                payload=TrainingMonitorStartedPayload(
                    pod_endpoint=pod_endpoint,
                    poll_interval_s=poll_interval_s,
                ),
            ),
        )

    def _monitor_run_id(self) -> str:
        """Return the cached run_id (set on :meth:`execute` entry).

        Falls back to ``"unknown"`` for paths that reach the async
        watcher without an explicit context (test helpers that call
        ``_watch`` directly).
        """
        return self._run_id_cache or "unknown"

    def _emit_monitor_timeout(
        self,
        run_id: str,
        *,
        last_event_at: datetime,
        timeout_s: float,
    ) -> None:
        if self._emitter is None:
            return
        self._emitter.emit(
            TrainingMonitorTimeoutEvent(
                source=_STAGE_SOURCE,
                run_id=run_id,
                offset=UNKNOWN_OFFSET,
                payload=TrainingMonitorTimeoutPayload(
                    last_event_at=last_event_at,
                    timeout_s=timeout_s,
                ),
            ),
        )

    # --- public entry point ---------------------------------------------

    def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        """Subscribe to the runner's WebSocket; return on terminal state.

        The launcher contract guarantees ``context["job_client"]``,
        ``context["ssh_tunnel"]``, ``context["job_id"]`` are populated
        on success. If they're missing it means TrainingLauncher
        didn't run (or ran with the legacy flow), and we fail loud
        rather than silently mock-mode.

        Returns:
            Terminal-state dict (``status="completed"``, ``duration_seconds``,
            etc.) on success.

        Raises:
            TrainingFailedError: any terminal non-success state from the runner.
            InternalError: launcher wiring missing from the pipeline context.
        """
        # ---- mock fast-path (kept for in-process integration tests)
        deployer_context = context.get(StageNames.GPU_DEPLOYER, {})
        provider_info = deployer_context.get("provider_info", {})
        if provider_info.get("mock"):
            logger.info("[MONITOR] MOCK MODE — simulating successful training")
            # Emit a started event for the mock path so consumers still
            # observe the lifecycle; the runner-facing pod_endpoint /
            # poll_interval are degenerate values in mock mode.
            self._emit_monitor_started(
                self._resolve_run_id(context),
                pod_endpoint="mock://noop",
                poll_interval_s=0.0,
            )
            return {"status": "completed", "duration_seconds": 0.0, "mock": True}

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
            raise InternalError(
                detail=(
                    "TrainingMonitor: context is missing job_client / "
                    "job_id — TrainingLauncher must have run first "
                    "(Phase 6.3a contract)."
                ),
                context={"legacy_code": "MONITOR_LAUNCHER_NOT_WIRED"},
            )

        logger.info(f"[MONITOR] Subscribing to runner events for job {job_id!r}")
        self._training_start_time = time.time()
        self._last_event_at = datetime.now(UTC)
        self._run_id_cache = self._resolve_run_id(context)
        # Best-effort pod endpoint hint; the launcher stores
        # ``pod_endpoint`` on the context when known, otherwise we fall
        # back to the job id so the typed event is still readable.
        self._emit_monitor_started(
            self._run_id_cache,
            pod_endpoint=str(context.get("pod_endpoint") or f"job://{job_id}"),
            poll_interval_s=0.0,
        )

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
        try:
            watch_result = asyncio.run(
                self._watch_and_download(client, job_id, self._log_manager, self._runner_log_manager),
            )
        finally:
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
    ) -> tuple[LogFetcher | None, LogFetcher | None, object | None]:
        """Construct HTTP-backed :class:`LogFetcher` instances for both
        ``trainer.stdio.log`` and ``runner.log``.

        Phase 2 PR-2.3 (transport-unification-v2): replaced the SSH
        ``stat -c %s`` + ``tail -c <delta>`` polling pair with a
        single :class:`JobClient.read_log` per tick. The legacy
        ``SSHClient`` ControlMaster is no longer constructed here —
        both fetchers share the existing :attr:`self._client`
        :class:`JobClient` that the stage already holds for jobs/
        events. PodLayout resolution moved pod-side: the LogName
        enum is mapped to a concrete file path inside the runner
        from ``app.state.pod_layout``.

        Return shape kept as ``(trainer, runner, ssh)`` for the
        orchestrator's existing teardown contract; the third slot
        is now always ``None`` and will be deleted in PR-3.2 when
        the SSHClient import contract lands.
        """
        from ryotenkai_shared.contracts.runner_api.logs import LogName

        provider_type = deployer_context.get("provider_type")
        # Cloud-only: local providers ship the trainer on the same
        # host so the log file is already available locally.
        if provider_type != "cloud":
            return None, None, None

        client = self._client
        if client is None:
            logger.debug(
                "[MONITOR] no JobClient on stage — periodic log download disabled",
            )
            return None, None, None

        try:
            mac_layout = get_run_log_layout()
            trainer_lm = LogFetcher(
                client,
                name=LogName.TRAINER_STDIO,
                local_path=mac_layout.remote_trainer_stdio_log,
            )
            runner_lm = LogFetcher(
                client,
                name=LogName.RUNNER,
                local_path=mac_layout.remote_runner_log,
            )
        except Exception as exc:
            logger.debug(f"[MONITOR] LogFetcher init failed: {exc}")
            return None, None, None
        return trainer_lm, runner_lm, None

    async def _log_downloader_loop(
        self,
        trainer_log_manager: LogFetcher,
        runner_log_manager: LogFetcher | None = None,
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
    ) -> RyotenkAIError | None:
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
              resource id / no SDK key); caller should raise its
              original error.
            :class:`RyotenkAIError` — pod was either terminal or
              recovered; the caller should raise this directly.

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
            return TrainingFailedError(
                detail=(
                    f"runner connection lost ({exc}); "
                    "pod recovery exhausted after "
                    f"{self._RECOVERY_ATTEMPT_CAP} attempts"
                ),
                context={"legacy_code": "MONITOR_RECOVERY_EXHAUSTED"},
            )
        self._recovery_attempts += 1

        logger.warning(
            "[MONITOR] runner connection lost (%s) — invoking provider "
            "recovery for resource %s",
            exc,
            self._resource_id,
        )
        try:
            status = self._provider.attempt_recovery(
                resource_id=self._resource_id,
            )
        except RyotenkAIError as recovery_err:
            # Map provider context.legacy_code values back to the
            # monitor's operator-facing error vocabulary for
            # back-compatibility.
            code_map = {
                "POD_PROBE_FAILED": "MONITOR_POD_PROBE_FAILED",
                "POD_TERMINAL": "MONITOR_POD_TERMINATED",
                "POD_WAKE_FAILED": "MONITOR_POD_WAKE_FAILED",
            }
            provider_code = (recovery_err.context or {}).get("legacy_code", "")
            return TrainingFailedError(
                detail=recovery_err.detail or str(recovery_err),
                context={
                    "legacy_code": code_map.get(provider_code, "MONITOR_RECOVERY_FAILED"),
                },
                cause=recovery_err,
            )
        if status == ProviderStatus.CONNECTED:
            # Pod was already running; WS failure unrelated. Caller
            # surfaces the original error.
            return None
        # Pod was stopped, woke back up — pipeline restart needed.
        return TrainingFailedError(
            detail=(
                f"pod {self._resource_id} was stopped (likely "
                "laptop sleep); woke it back up — restart the "
                "pipeline to resume training"
            ),
            context={"legacy_code": "MONITOR_POD_RECOVERED"},
        )

    # --- post-mortem diagnostics ---------------------------------------

    def _collect_death_diagnostics(self) -> None:
        """Render a compact autopsy when the trainer dies non-zero.

        Two sources, in priority order:

        1. **Local log file pointers** — ``trainer.stdio.log`` and
           ``runner.log`` are pulled by LogManager periodically + on
           cleanup. We do NOT re-stream their content into the
           pipeline log: each line in those files already has its own
           timestamp + module + level prefix from the trainer's
           logger, and re-prefixing it through control-plane logger
           creates double-timestamped, hard-to-read output. Operators
           open those files directly when diagnosing.
        2. **Live HTTP probes** — kernel signals (OOM, NVRM/Xid) and
           current GPU state. These cannot come from local files
           because they are environment-level, not subprocess output.

        Each pointer reports state explicitly (``<<MISSING>>``,
        ``<<EMPTY>>``, ``size=N bytes``) so an operator can decide
        whether to ``cat`` the file or skip it.

        Skipped silently if no log layout is wired (test paths); per-
        probe failures are swallowed so one broken command doesn't
        suppress the rest.
        """
        logger.info(
            "[MONITOR:POSTMORTEM] non-zero exit detected — collecting diagnostics",
        )

        # --- Source 1: log file pointers (no inline content dump) ---
        try:
            log_layout = get_run_log_layout()
        except Exception:
            log_layout = None

        if log_layout is not None:
            self._emit_log_pointer(
                label="trainer.stdio.log",
                path=log_layout.remote_trainer_stdio_log,
            )
            self._emit_log_pointer(
                label="runner.log",
                path=log_layout.remote_runner_log,
            )

        # --- Source 2: live HTTP diagnostics (kernel + GPU) ---
        # Phase 2 transport-unification-v2: replaced 3 SSH ``dmesg`` /
        # ``nvidia-smi`` probes with a single typed HTTP call. Per-block
        # failures (CAP_SYSLOG missing, ``nvidia-smi`` not installed)
        # surface inside ``response.<block>.error`` instead of silently
        # being swallowed by the old loop.
        self._dump_http_diagnostics()

    def _dump_http_diagnostics(self) -> None:
        """Postmortem diagnostics via ``GET /api/v1/diagnostics``.

        Best-effort — any transport failure or missing JobClient is
        logged at INFO and the postmortem continues without it.

        Implementation note: this method is synchronous but
        ``client.get_diagnostics()`` is an async coroutine. We get
        called from within the WS-driven ``_watch_and_download`` chain
        whose event loop is already running, so a naive
        ``asyncio.run(...)`` raises ``RuntimeError: asyncio.run() cannot
        be called from a running event loop`` and the diagnostics
        silently disappear. Bridge through a fresh worker thread that
        spins up its own loop with ``asyncio.run`` — works whether the
        caller is sync OR has a running loop.
        """
        client = getattr(self, "_client", None)
        if client is None:
            logger.info(
                "[MONITOR:POSTMORTEM] (no JobClient wired — HTTP diagnostics skipped)",
            )
            return

        import asyncio
        import threading

        from ryotenkai_shared.contracts.runner_api.diagnostics import (  # noqa: F401  -- imported for downstream rendering
            DiagnosticsBlockError,
        )

        response_holder: list = []
        error_holder: list = []

        def _run_in_fresh_loop() -> None:
            try:
                response_holder.append(asyncio.run(client.get_diagnostics()))
            except Exception as exc:
                error_holder.append(exc)

        worker = threading.Thread(target=_run_in_fresh_loop, daemon=True)
        worker.start()
        worker.join(timeout=15.0)
        if worker.is_alive():
            logger.info(
                "[MONITOR:POSTMORTEM] diagnostics HTTP call timed out after 15s",
            )
            return
        if error_holder:
            logger.info(
                f"[MONITOR:POSTMORTEM] diagnostics HTTP call failed: "
                f"{error_holder[0]!r}",
            )
            return
        if not response_holder:
            # Worker exited without populating either holder — defensive,
            # would only happen on threading-level corruption.
            return
        response = response_holder[0]

        def _emit_lines(label: str, lines: list[str]) -> None:
            if not lines:
                logger.info(f"[MONITOR:POSTMORTEM] {label}: <<EMPTY>>")
                return
            for line in lines:
                logger.info(f"[MONITOR:POSTMORTEM] {label}: {line}")

        if response.dmesg is not None:
            if response.dmesg.error is not None:
                logger.info(
                    f"[MONITOR:POSTMORTEM] dmesg: <<{response.dmesg.error.value.upper()}>>",
                )
            else:
                _emit_lines("dmesg_tail", response.dmesg.lines)
        if response.kernel_signals is not None:
            if response.kernel_signals.error is not None:
                logger.info(
                    f"[MONITOR:POSTMORTEM] dmesg_kernel_signals: "
                    f"<<{response.kernel_signals.error.value.upper()}>>",
                )
            else:
                _emit_lines("dmesg_kernel_signals", response.kernel_signals.matches)
        if response.gpu is not None:
            if response.gpu.error is not None:
                logger.info(
                    f"[MONITOR:POSTMORTEM] nvidia_smi: <<{response.gpu.error.value.upper()}>>",
                )
            elif not response.gpu.rows:
                logger.info("[MONITOR:POSTMORTEM] nvidia_smi: <<EMPTY>>")
            else:
                for row in response.gpu.rows:
                    logger.info(
                        f"[MONITOR:POSTMORTEM] nvidia_smi: {row.name}, "
                        f"{row.utilization_gpu_percent} %, "
                        f"{row.memory_used_mib} MiB, "
                        f"{row.memory_total_mib} MiB",
                    )

    def _emit_log_pointer(
        self,
        *,
        label: str,
        path: Path,
    ) -> None:
        """Emit a one-line pointer to a local log file (no content dump).

        Operators got annoyed by the previous behavior — re-streaming 30
        lines of trainer log through the control-plane logger produced
        double-timestamped, hard-to-read output (each trainer line already
        carried its own ``YYYY-MM-DD HH:MM:SS module:line LEVEL`` prefix,
        which got wrapped in another control-plane prefix). We now log
        a single line per file with size + path so the operator can
        ``cat`` or ``tail`` directly when needed.

        Three states surface explicitly so the operator knows whether
        the file is worth opening:
          * ``<<MISSING>>``  — file does not exist (most likely the
            corresponding subprocess never produced output before death)
          * ``<<EMPTY>>``    — file exists but has zero bytes
          * ``size=N bytes`` — file has content; open the path
        """
        try:
            if not path.exists():
                logger.info(f"[MONITOR:POSTMORTEM] {label}: <<MISSING>>")
                return
            size = path.stat().st_size
            if size == 0:
                logger.info(f"[MONITOR:POSTMORTEM] {label}: <<EMPTY>>")
                return
            logger.info(
                f"[MONITOR:POSTMORTEM] {label}: size={size:,} bytes — {path}"
            )
        except OSError as exc:
            logger.warning(
                f"[MONITOR:POSTMORTEM] {label}: stat failed: {exc}",
            )

    async def _watch_and_download(
        self,
        client: JobClient,
        job_id: str,
        log_manager: LogFetcher | None,
        runner_log_manager: LogFetcher | None = None,
    ) -> dict[str, Any]:
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
    ) -> dict[str, Any]:
        """Iterate over WS events; dispatch callbacks; return on terminal.

        Reconnect / replay-truncation are handled by
        :meth:`JobClient.subscribe_events` itself; we only catch the
        exit conditions and translate them to dict (success) or raise.
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

                terminal = self._get_forwarder().dispatch_event(event)
                if terminal is not None:
                    return terminal
            # Stream ended cleanly without a terminal kind — treat as
            # a server-side bug; the runner is supposed to close only
            # on FSM terminal.
            raise TrainingFailedError(
                detail="runner closed the event stream before reaching terminal state",
                context={"legacy_code": "MONITOR_STREAM_EOF"},
            )
        except JobNotFoundError as exc:
            raise TrainingFailedError(
                detail=(
                    f"runner reports unknown job {job_id!r} "
                    "(pod restart wiped state?)"
                ),
                context={"legacy_code": "MONITOR_JOB_NOT_FOUND"},
                cause=exc,
            ) from exc
        except ReplayTruncatedError:
            # Buffer (ring or disk-backed journal) rolled past the offset
            # we asked for. Phase 6.a added an HTTP replay endpoint
            # backed by the pod's on-disk journal — try that first so a
            # long Mac sleep doesn't drop the training timeline. If
            # replay also fails (transport, no journal, etc.) we fall
            # back to the legacy status-snapshot path which loses event
            # history but still terminates the run cleanly.
            return await self._get_forwarder().replay_then_resume_or_fallback(client, job_id)
        except JobClientError as exc:
            recovery = self._recover_pod_if_needed(exc)
            if recovery is not None:
                # Recovery produced a typed exception — raise it.
                # We deliberately do NOT loop and rebuild handles
                # inline: that requires re-running the launcher
                # (tunnel+client+job_id), which is the orchestrator's
                # job.
                raise recovery
            # Emit a typed timeout event with the wall-clock window
            # between the last observed event and now. The emitter is
            # the only side-channel where the duration is captured;
            # the raise below conveys the legacy error code only.
            now = datetime.now(UTC)
            self._emit_monitor_timeout(
                self._monitor_run_id(),
                last_event_at=self._last_event_at,
                timeout_s=max(0.0, (now - self._last_event_at).total_seconds()),
            )
            raise TrainingFailedError(
                detail=f"runner client error: {exc}",
                context={"legacy_code": "MONITOR_CLIENT_ERROR"},
                cause=exc,
            ) from exc

    # --- pod-event dispatch (delegated to PodEventForwarder) ------------
    #
    # Thin back-compat wrappers — the heavy lifting lives on
    # :class:`PodEventForwarder` (see
    # ``training_monitor_pod_event_forwarder.py``). Tests that
    # exercise ``monitor._dispatch_event`` / ``monitor._maybe_log_status``
    # directly continue to work without modification; the wrappers
    # forward to the same logic the WS consumer loop uses.

    def _dispatch_event(
        self,
        event: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Back-compat wrapper — delegates to :meth:`PodEventForwarder.dispatch_event`."""
        return self._get_forwarder().dispatch_event(event)

    def _maybe_log_status(self, payload: dict[str, Any]) -> None:
        """Back-compat wrapper — delegates to :meth:`PodEventForwarder.maybe_log_status`."""
        self._get_forwarder().maybe_log_status(payload)

    def _handle_trainer_exited(
        self,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Translate a ``trainer_exited`` event to a terminal dict (or raise).

        Phase D-aware payload shape (``schema_version=2``):
        ``{exit_code, signal, cancellation_requested, schema_version,
        code, message, traceback_summary, wall_seconds,
        payload_source}``. Legacy producers (pre-D) omit the new fields
        — the consumer falls back to the old exit-code reasoning.

        Decoding policy (Phase D):

        * ``payload_source == "trainer_file"`` → trusted typed payload.
          ``code == TRAINING_OOM`` raises :class:`TrainingOOMError`;
          ``code == INTERNAL_ERROR`` raises :class:`InternalError`;
          any other code raises :class:`TrainingFailedError` (we never
          surface arbitrary :class:`ErrorCode` values as their own
          exception class here — the training stage's failure shape
          is one of three categories from the pipeline's perspective).
        * ``payload_source == "sigkill_heuristic"`` → SIGKILL/137; same
          mapping as ``TRAINING_OOM`` above.
        * ``payload_source == "missing"`` or no Phase D fields → fall
          back to the legacy exit-code heuristic.

        The trainer's last stdout/stderr lives on disk in
        ``trainer.stdio.log`` and is rendered by
        :meth:`_collect_death_diagnostics` from the SCP-pulled local
        copy — single source, no embedded WS tail.
        """
        duration = max(0.0, time.time() - self._training_start_time)
        exit_code = payload.get("exit_code")
        signal_name = payload.get("signal")
        cancelled = bool(payload.get("cancellation_requested"))

        # Phase D — schema_version=2 fields. ``payload_source`` is the
        # discriminator: ``"none"`` means the supervisor saw a clean
        # exit (rc==0); anything else means we have decoded data we
        # can lean on instead of the legacy heuristic.
        payload_source = payload.get("payload_source")
        typed_code = payload.get("code")
        typed_message = payload.get("message")
        typed_wall_seconds = payload.get("wall_seconds")
        if isinstance(typed_wall_seconds, (int, float)) and typed_wall_seconds > duration:
            # Trust the trainer's anchor when it's strictly greater —
            # the supervisor's ``time.time()`` may have started later
            # than the trainer's monotonic clock (e.g. delayed
            # spawn). Never go backwards.
            duration = float(typed_wall_seconds)

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
            return {
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

        if cancelled:
            self._fail(
                f"training cancelled (exit_code={exit_code}, signal={signal_name})",
                duration,
                code="TRAINING_CANCELLED",
            )

        # Phase D — typed-code dispatch when the trainer wrote a payload
        # OR the supervisor synthesised one via the SIGKILL heuristic.
        if payload_source in {"trainer_file", "sigkill_heuristic"} and typed_code:
            message = typed_message or (
                f"trainer exited non-zero (exit_code={exit_code}, signal={signal_name})"
            )
            self._raise_typed(typed_code, message, duration)

        if exit_code is None and signal_name is None:
            # Process died without a parsed exit code — surface as a
            # generic training failure. Phase 5 will fold the previous
            # ``on_process_died`` callback into a typed event.
            self._fail("trainer process died (no exit code)", duration)

        self._fail(
            f"trainer exited non-zero (exit_code={exit_code}, signal={signal_name})",
            duration,
        )
        # Unreachable — _fail always raises; keeps type checker happy.
        raise AssertionError("_fail must raise")

    def _raise_typed(
        self,
        code: str,
        message: str,
        duration: float,
    ) -> None:
        """Raise the typed exception class matching Phase D ``code``.

        Three categories from the pipeline stage's perspective:

        * ``TRAINING_OOM`` → :class:`TrainingOOMError` (5xx, retryable
          with smaller batch / more VRAM).
        * ``INTERNAL_ERROR`` → :class:`InternalError` (server bug;
          report).
        * everything else → :class:`TrainingFailedError` (catch-all
          for typed-but-non-OOM training failures).

        Always raises — never returns. Phase 4: callbacks removed;
        Phase 5 will replace the previous ``on_training_failed`` banner
        with a typed event.
        """
        # String comparison rather than ``ErrorCode(code)`` — the wire
        # payload carries the enum value as a bare string, and a
        # producer with a code we don't yet know about should NOT
        # crash the consumer (forward-compat: future error codes
        # degrade to ``TrainingFailedError``, never to a parse error).
        if code == "TRAINING_OOM":
            raise TrainingOOMError(
                detail=message,
                context={"duration_seconds": duration, "phase_d_typed": True},
            )
        if code == "INTERNAL_ERROR":
            raise InternalError(
                detail=message,
                context={"duration_seconds": duration, "phase_d_typed": True},
            )
        raise TrainingFailedError(
            detail=message,
            context={
                "duration_seconds": duration,
                "phase_d_typed": True,
                "legacy_code": code,
            },
        )

    def _terminal_from_state(
        self,
        state: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Backstop: derive terminal dict from a bare FSM state event.

        Raises ``TrainingFailedError`` for non-success terminal states.
        """
        duration = max(0.0, time.time() - self._training_start_time)
        message = payload.get("message") if isinstance(payload, dict) else None

        if state == _TERMINAL_COMPLETED:
            return {"status": "completed", "duration_seconds": duration}

        if state == _TERMINAL_CANCELLED:
            self._fail(
                f"training cancelled ({message or 'no detail'})",
                duration,
                code="TRAINING_CANCELLED",
            )

        self._fail(
            f"training failed ({message or 'no detail'})",
            duration,
        )
        raise AssertionError("_fail must raise")  # unreachable

    def _fail(
        self,
        message: str,
        duration: float,
        *,
        code: str = "TRAINING_FAILED",
    ) -> None:
        """Common path for FAILED / CANCELLED → typed ``TrainingFailedError`` raise.

        Always raises — never returns. Phase 4: legacy callback hooks
        removed; Phase 5 will fold the previous ``on_training_failed``
        banner into a typed event.
        """
        raise TrainingFailedError(
            detail=message,
            context={"legacy_code": code},
        )

    async def _fallback_to_status(
        self,
        client: JobClient,
        job_id: str,
    ) -> dict[str, Any]:
        """Buffer rolled past us — refetch :meth:`JobClient.get_status`
        and translate the snapshot to a terminal dict (or raise).

        We deliberately do NOT try to resume subscribing here —
        events between the truncation and ``get_status`` are gone,
        and re-subscribing would race the very same buffer rollover
        again. The orchestrator's restart logic re-runs the monitor
        if the job is somehow still active.
        """
        try:
            snap = await client.get_status(job_id)
        except JobClientError as exc:
            raise TrainingFailedError(
                detail=(
                    "WebSocket replay buffer rolled over and "
                    f"GET /jobs/{job_id} failed: {exc}"
                ),
                context={"legacy_code": "MONITOR_STATUS_FETCH_FAILED"},
                cause=exc,
            ) from exc

        state = (snap.get("state") or "").lower()
        if state in _TERMINAL_STATES:
            return self._terminal_from_state(state, snap)
        raise TrainingFailedError(
            detail=(
                f"WebSocket replay truncated and FSM is still "
                f"non-terminal ({state!r}); cannot resume "
                "monitoring without losing events"
            ),
            context={"legacy_code": "MONITOR_REPLAY_TRUNCATED"},
        )
