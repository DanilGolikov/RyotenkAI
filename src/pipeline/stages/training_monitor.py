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

from src.api.clients.job_client import (
    JobClientError,
    JobNotFoundError,
    ReplayTruncatedError,
)
from src.constants import CONSOLE_LINE_WIDTH, LOG_DOWNLOAD_INTERVAL_DEFAULT, SSH_PORT_DEFAULT
from src.pipeline.stages.base import PipelineStage
from src.pipeline.stages.constants import PipelineContextKeys, StageNames
from src.utils.logger import logger
from src.utils.result import AppError, Err, Ok, Result, TrainingError

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

    from src.api.clients.job_client import JobClient
    from src.api.services.tunnel_service import SSHTunnelManager
    from src.config.secrets.model import Secrets
    from src.utils.config import PipelineConfig


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

        # Phase 11.E — no try/finally for tunnel teardown anymore.
        # Resources captured on ``self`` are torn down in :meth:`cleanup`
        # at orchestrator finalize-time, AFTER ModelRetriever has run.
        # An exception escaping ``_watch`` propagates up the stack
        # cleanly — orchestrator's reverse-order cleanup() will still
        # tear down the captured handles.
        watch_result = asyncio.run(self._watch(client, job_id))

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
        self, context: dict[str, Any],
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
                    "[MONITOR] %s exists at %s but is unreadable (%s); "
                    "skipping reconciliation",
                    marker_name, marker_path, exc,
                )
                # Don't try the next marker — disk read failure
                # likely means the whole attempt dir is in a bad
                # shape; bail.
                return

            run_id = payload.get("run_id")
            if not run_id:
                logger.info(
                    "[MONITOR] %s at %s has no run_id; MLflow "
                    "reconciliation skipped (no run to address)",
                    marker_name, marker_path,
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
                getattr(run, "info", None), "status", None,
            )

            if current_status != "RUNNING":
                # Already terminal (KILLED / FAILED / FINISHED) —
                # something else closed the run. No-op.
                logger.debug(
                    "[MONITOR] MLflow run %s already terminal "
                    "(status=%s); %s reconciliation no-op",
                    run_id, current_status, reason,
                )
                return

            client.set_terminated(run_id=run_id, status=target_status)
            # Use warning level for cancellation (operator should
            # know SIGKILL fallback fired) and info for natural
            # completion (expected post-sleep recovery).
            log_fn = (
                logger.warning if marker_name == "cancelled.marker"
                else logger.info
            )
            extra_note = ""
            if marker_name == "completion.marker" and payload.get(
                "flush_timed_out",
            ):
                extra_note = (
                    " — flush_timed_out=true; some metrics may be "
                    "missing"
                )
            log_fn(
                "[MONITOR] reconciled MLflow run %s: RUNNING → %s "
                "(%s indicated %s)%s",
                run_id, target_status, marker_name, reason, extra_note,
            )
        except Exception as exc:
            logger.warning(
                "[MONITOR] %s reconciliation for run_id=%s failed: %s "
                "— operator may need to manually set the MLflow run "
                "status",
                marker_name, run_id, exc,
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
                job_id, since=self._last_offset,
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
                    message=f"runner reports unknown job {job_id!r} "
                    "(pod restart wiped state?)",
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
            return Err(
                TrainingError(
                    message=f"runner client error: {exc}",
                    code="MONITOR_CLIENT_ERROR",
                ),
            )

    def _dispatch_event(
        self, event: dict[str, Any],
    ) -> Result[dict[str, Any], AppError] | None:
        """Fire callbacks; return a terminal Result or ``None`` to keep
        listening.

        Recognised event kinds (everything else is logged at debug
        and ignored):
        - ``health_snapshot`` → ``on_resource_check``
        - ``trainer_exited`` → ``on_training_completed`` /
          ``on_training_failed`` / ``on_process_died`` based on
          payload, then return terminal Result
        - ``stop_requested``, ``pod_stop_attempt`` etc. → log only
        """
        kind = event.get("kind") or ""
        payload = event.get("payload") or {}

        if kind == "health_snapshot" and self._callbacks.on_resource_check:
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

    def _handle_trainer_exited(
        self, payload: dict[str, Any],
    ) -> Result[dict[str, Any], AppError]:
        """Translate a ``trainer_exited`` event to a terminal Result.

        Payload shape (per :class:`Supervisor._reap`):
        ``{"exit_code": int, "signal": str | None,
           "cancellation_requested": bool}``
        """
        duration = max(0.0, time.time() - self._training_start_time)
        exit_code = payload.get("exit_code")
        signal_name = payload.get("signal")
        cancelled = bool(payload.get("cancellation_requested"))

        if exit_code == 0 and not cancelled:
            if self._callbacks.on_training_completed:
                self._callbacks.on_training_completed(duration)
            return Ok({
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
            })

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
        self, state: str, payload: dict[str, Any],
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
        self, client: JobClient, job_id: str,
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
                    message=(
                        "WebSocket replay buffer rolled over and "
                        f"GET /jobs/{job_id} failed: {exc}"
                    ),
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
