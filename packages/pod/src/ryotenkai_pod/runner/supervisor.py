"""Trainer subprocess supervisor — Phase 2.

The supervisor is the **only place** in the runner that spawns,
signals, and reaps the trainer child. It owns:

- a single ``asyncio.subprocess.Process`` at a time (Phase 1
  guarantee: one active job → one trainer at a time),
- the trainer's process-group id (a fresh session via
  ``start_new_session=True``) so SIGTERM can be sent to the whole
  group via :func:`os.killpg` — the trainer plus its dataloader
  workers and any fork()ed children all receive the stop request,
- background asyncio tasks that pump the trainer's stdout / stderr
  **to a single ground-truth file on disk** (single-writer
  ``trainer.stdio.log``) — survives import-time crashes, native
  faulthandler dumps land here too because faulthandler defaults
  to stderr,
- the lifecycle bridge to :class:`JobLifecycleFSM` — every state
  transition originates here, never in the API layer.

Data plane vs control plane:
- Trainer stdout/stderr lines go ONLY to the stdio log file. They
  do NOT flow through the EventBus (no ``trainer_log`` events).
  Mac picks them up via SCP delta-pull from
  ``<workspace>/logs/trainer.stdio.log``.
- The EventBus carries ONLY control-plane events (``trainer_spawned``,
  ``trainer_exited``, ``cancellation_*``) and telemetry events
  (``mlflow_*``, ``health_snapshot``) published by the trainer's
  loopback HTTP callbacks and the runner's HealthReporter.

State machine (mirrored on top of :class:`JobLifecycleFSM`)::

    spawn()       → preparing → running         (legal if no active job)
    natural exit  → running   → completed | failed
    request_stop  → running   → stopping → completed | cancelled | failed
    shutdown()    → whatever  → failed if non-terminal

Two-phase shutdown::

    request_stop(grace_seconds)
        ↓
    os.killpg(pgid, SIGTERM)
        ↓
    wait up to ``grace_seconds``
        ↓ alive?
    os.killpg(pgid, SIGKILL) — final escalation
        ↓
    proc.wait() — reap and parse exit code

Exit-code interpretation (POSIX):
- ``0`` → ``completed`` after natural exit, or ``cancelled`` after a
  prior ``request_stop`` (the trainer's signal handler caught SIGTERM,
  saved a checkpoint, and exited 0).
- ``> 128`` → killed by signal (UNIX convention rc = 128 + signal_no);
  ``signal.Signals(rc - 128).name`` is logged in the FSM message and
  the FSM transitions to ``failed`` (or ``cancelled`` if a prior
  ``request_stop`` set ``self._cancellation_requested`` — meaning
  the SIGKILL came from us).
- ``< 0`` → Python's ``Popen.returncode`` convention for "killed by
  signal -N" (asyncio subprocesses report returncode = -SIGNAL on
  some platforms). Treated identically to ``> 128``.
- any other non-zero → ``failed``.

Stream pumping:
The supervisor reads each stream line-by-line. Each non-empty line
gets an ``[OUT]`` or ``[ERR]`` prefix and is appended to the stdio
log file. Disk-write failures are suppressed (defensive: a stuck
disk must not break trainer pump — pumps draining the pipe is the
load-bearing invariant). Native faulthandler dumps land here too
because Python's faulthandler writes to stderr by default.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING

from ryotenkai_pod.runner.state import (
    InvalidTransitionError,
    JobState,
)

if TYPE_CHECKING:
    from ryotenkai_pod.runner.event_bus import EventBus
    from ryotenkai_pod.runner.state import JobLifecycleFSM

__all__ = [
    "Supervisor",
    "SupervisorBusy",
    "SupervisorError",
    "TerminalHook",
]


# Async callback fired after every FSM transition into a terminal state
# (completed / failed / cancelled). The single argument is the terminal
# state name (``"completed"`` etc.). Used by :mod:`src.runner.main` to
# wire :func:`src.runner.pod_terminator.run_terminal_hook` — i.e.
# pick between podStop / podTerminate based on the Phase 11.B decision
# matrix once training finishes. Keeping this protocol generic (rather
# than ``PodTerminator``-typed) means the supervisor stays
# provider-agnostic.
TerminalHook = Callable[[str], Awaitable[None]]


# Default grace period before SIGKILL escalation. 30 s is enough for
# HuggingFace Trainer to finish the current step and run on_save —
# checkpoint write speed dominates this.
DEFAULT_GRACE_SECONDS = 30.0


class SupervisorError(RuntimeError):
    """Base for supervisor-specific errors."""


class SupervisorBusy(SupervisorError):
    """A trainer is already running; refuse to start a second one."""


class Supervisor:
    """Async wrapper around a single trainer subprocess.

    Construct once per :class:`fastapi.FastAPI` app instance (lives on
    ``app.state`` alongside the FSM and EventBus). Phase 2 binds a
    1:1 instance per server; Phase 5+ may grow into a registry once
    multi-job lands.
    """

    def __init__(
        self,
        fsm: "JobLifecycleFSM",
        bus: "EventBus",
        *,
        terminal_hook: TerminalHook | None = None,
        stdio_log_path: Path | None = None,
    ) -> None:
        self._fsm = fsm
        self._bus = bus
        self._proc: asyncio.subprocess.Process | None = None
        self._pgid: int | None = None
        self._stdout_task: asyncio.Task[None] | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self._wait_task: asyncio.Task[None] | None = None
        self._escalation_task: asyncio.Task[None] | None = None
        # Set when ``request_stop`` runs; consulted by the natural-exit
        # handler so it knows whether to land in ``cancelled`` or
        # ``failed`` on a non-zero exit.
        self._cancellation_requested = False
        # Phase 9.C: epoch-ms anchor stamped on ``request_stop`` so
        # downstream telemetry events (``cancellation_finalized`` from
        # the trainer-side callback, ``cancellation_completed`` from
        # this supervisor's reap path) can carry ``latency_ms``
        # measured against a single point in time. ``None`` means
        # "no cancellation requested" — used to gate the completed
        # event so natural exits don't emit cancellation telemetry.
        self._cancellation_started_at_ms: int | None = None
        # Optional post-terminal callback (see :data:`TerminalHook`).
        # Production wires this to RunPod auto-stop; tests usually
        # leave it ``None``.
        self._terminal_hook: TerminalHook | None = terminal_hook
        # Pod-side ground-truth for trainer stdout/stderr. When set,
        # ``_pump_stream`` appends every line (with kind prefix) to
        # this file — single-writer, append-only, byte-buffered.
        # ``None`` means stdio capture is disabled (used by test
        # harnesses that don't need the file artefact).
        self._stdio_log_path: Path | None = stdio_log_path
        self._stdio_log_file = None  # opened in _spawn, closed in shutdown
        # Phase D — trainer subprocess working directory captured on
        # spawn so :meth:`_reap` can read
        # ``<workdir>/trainer-exit.json`` (the structured exit payload
        # the trainer's exit_reporter writes on failure). ``None`` when
        # no workdir was set on spawn (legacy / test paths); the
        # supervisor then falls through to the exit-code heuristic.
        self._workdir: Path | None = None

    # --- read-only accessors ------------------------------------------------

    @property
    def is_running(self) -> bool:
        """``True`` if a trainer subprocess exists and hasn't exited."""
        return self._proc is not None and self._proc.returncode is None

    @property
    def pgid(self) -> int | None:
        return self._pgid

    # --- lifecycle ----------------------------------------------------------

    async def submit_and_spawn(
        self,
        job_id: str,
        command: list[str],
        *,
        env: dict[str, str] | None = None,
        workdir: Path | None = None,
    ) -> None:
        """Atomically submit + spawn the trainer.

        FSM transitions ``→ preparing → running``. If the subprocess
        spawn fails (e.g. the binary doesn't exist), the FSM is
        rolled to ``failed`` before the exception is re-raised — a
        partial half-submitted state is never observable.

        Raises:
            SupervisorBusy: a previous trainer is still running, or
                the FSM holds an active non-terminal job.
            FileNotFoundError / OSError: from ``subprocess`` if the
                command can't be launched.
        """
        if self.is_running:
            raise SupervisorBusy("a trainer subprocess is already running")

        # ``preparing`` first — visible to subscribers immediately.
        try:
            self._fsm.submit(job_id)
        except InvalidTransitionError as exc:
            raise SupervisorBusy(f"job in non-terminal state: {exc}") from exc

        self._bus.publish(
            "job_submitted",
            {"job_id": job_id, "sequence": 0},
        )

        try:
            await self._spawn(command, env=env, workdir=workdir)
        except BaseException:
            # Spawn blew up — roll the FSM to failed so future
            # restore_or_init doesn't see a stuck ``preparing`` state.
            with contextlib.suppress(InvalidTransitionError):
                self._fsm.transition(JobState.FAILED, message="spawn_failed")
            self._bus.publish("spawn_failed", {})
            raise

    async def _spawn(
        self,
        command: list[str],
        *,
        env: dict[str, str] | None = None,
        workdir: Path | None = None,
    ) -> None:
        """Internal: launch the subprocess and start the pumps.

        Separated from :meth:`submit_and_spawn` so the rollback path
        is the only place that knows about the FSM transitions.
        """

        # ``start_new_session=True`` puts the child in a fresh session
        # (and so a fresh process group) — this lets us SIGTERM the
        # whole group later via os.killpg without hitting our own
        # process. Inheriting env (None default) plus optional override.
        merged_env: dict[str, str] | None = None
        if env is not None:
            merged_env = {**os.environ, **env}

        # Open the stdio capture file BEFORE spawning the subprocess
        # so the very first byte the pumps write lands on disk. If the
        # path's parent directory doesn't exist, create it (idempotent).
        # Failures are tolerated — capture is best-effort observability,
        # never load-bearing.
        if self._stdio_log_path is not None and self._stdio_log_file is None:
            try:
                self._stdio_log_path.parent.mkdir(parents=True, exist_ok=True)
                # Append-binary, line-buffered (buffering=1 only works
                # in text mode → use buffering=0 + manual flush per write).
                self._stdio_log_file = open(  # noqa: SIM115 — closed in shutdown
                    self._stdio_log_path, "ab", buffering=0,
                )
            except OSError as exc:
                import logging
                logging.getLogger(__name__).warning(
                    "[SUPERVISOR] could not open stdio log file %s: %s",
                    self._stdio_log_path, exc,
                )
                self._stdio_log_file = None

        # Phase D — remember the subprocess cwd so :meth:`_reap` can
        # locate ``<workdir>/trainer-exit.json`` written by the
        # trainer's exit_reporter on failure paths. ``None`` workdir
        # means the supervisor inherited uvicorn's cwd; we do NOT
        # default to ``Path.cwd()`` because that would race with
        # concurrent runs in the same FastAPI process (Phase 5+).
        self._workdir = Path(workdir) if workdir is not None else None

        self._proc = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(workdir) if workdir is not None else None,
            env=merged_env,
            start_new_session=True,
        )
        # In a fresh session, the child IS the new pgid leader — i.e.
        # pgid == child.pid. We cache it because killpg races with
        # the child reaping itself: once the child dies, getpgid()
        # raises, but we still want to know the (now-defunct) group
        # for forensics.
        try:
            self._pgid = os.getpgid(self._proc.pid)
        except ProcessLookupError:
            # Trainer died before getpgid resolved — extremely rare.
            self._pgid = self._proc.pid

        self._cancellation_requested = False

        # Pump stdout / stderr in the background. Each stream gets
        # its own task so a long-line writer on stderr doesn't block
        # stdout reads.
        self._stdout_task = asyncio.create_task(
            self._pump_stream(self._proc.stdout, kind="stdout"),
            name="supervisor.stdout_pump",
        )
        self._stderr_task = asyncio.create_task(
            self._pump_stream(self._proc.stderr, kind="stderr"),
            name="supervisor.stderr_pump",
        )

        # Reap the process in another task — completing FSM transition
        # on natural exit. Holding the reference prevents the task from
        # being GC'd mid-flight.
        self._wait_task = asyncio.create_task(
            self._reap(),
            name="supervisor.reap",
        )

        self._fsm.transition(JobState.RUNNING, message="trainer_spawned")
        self._bus.publish(
            "trainer_spawned",
            {"pid": self._proc.pid, "pgid": self._pgid, "command": list(command)},
        )

    async def request_stop(
        self, *, grace_seconds: float = DEFAULT_GRACE_SECONDS,
    ) -> None:
        """Two-phase graceful stop. FSM → ``stopping`` synchronously,
        SIGKILL escalation runs in background.

        Synchronous part:
        - FSM transition to ``stopping`` so subscribers see the new
          state before the response unblocks the caller.
        - Publish ``stop_requested`` event.
        - SIGTERM the process group.

        Asynchronous tail (a background task on the same loop):
        - Wait up to ``grace_seconds`` for the subprocess to exit.
        - If it doesn't, publish ``stop_escalated`` and SIGKILL the
          group. The reap task takes care of the FSM transition to
          ``cancelled`` / ``failed``.

        This split keeps the API endpoint responsive — POST /stop
        returns 202 the moment the SIGTERM is in flight, not after
        the grace window has elapsed.

        Idempotent: a second call while ``stopping`` is a no-op
        (the FSM transition would raise; we swallow it).
        """
        if not self.is_running:
            return

        try:
            self._fsm.transition(JobState.STOPPING, message="stop_requested")
        except InvalidTransitionError:
            return

        # Phase 9.C — anchor for the cancellation chain's latency
        # bookkeeping. Stamped BEFORE bus publishes so the event
        # payload carries the same anchor downstream consumers see.
        from ryotenkai_shared.observability.cancellation_telemetry import (
            CANCELLATION_STARTED,
            now_ms,
        )
        self._cancellation_started_at_ms = now_ms()
        self._cancellation_requested = True

        # Backwards-compat: keep the existing ``stop_requested`` event
        # for any consumer that grep'd it pre-9.C. The new
        # ``cancellation_started`` carries the structured telemetry
        # payload; both fire on the same FSM transition.
        self._bus.publish("stop_requested", {"grace_seconds": grace_seconds})
        self._bus.publish(
            CANCELLATION_STARTED,
            {
                "requested_at_ms": self._cancellation_started_at_ms,
                "grace_seconds": grace_seconds,
                # ``reason`` is informational — supervisor doesn't know
                # why the user stopped; it just got the request. Other
                # callers (idle_detector, max_lifetime) populate it
                # via a future ``request_stop(reason="idle_timeout")``
                # parameter — wired separately if/when those paths
                # gain dedicated reason strings. Default is the
                # canonical "user clicked stop" reason.
                "reason": "user_stop",
            },
        )
        self._signal_group(signal.SIGTERM)

        # Background escalation. Held as a task reference so it isn't
        # GC'd before it gets to run.
        self._escalation_task = asyncio.create_task(
            self._wait_and_escalate(grace_seconds),
            name="supervisor.escalate",
        )

    async def _wait_and_escalate(self, grace_seconds: float) -> None:
        """SIGKILL the process group if SIGTERM didn't reap within
        ``grace_seconds``. Runs as a fire-and-forget background task
        spawned by :meth:`request_stop`.
        """
        if self._proc is None:  # pragma: no cover — defensive
            return
        try:
            await asyncio.wait_for(self._proc.wait(), timeout=grace_seconds)
        except TimeoutError:
            self._bus.publish(
                "stop_escalated",
                {"reason": "grace_expired", "grace_seconds": grace_seconds},
            )
            self._signal_group(signal.SIGKILL)

    async def shutdown(self) -> None:
        """Lifespan-exit hook — kill the trainer and wait for terminal state.

        Called from :func:`src.runner.main._lifespan` on FastAPI
        shutdown. Uses a 5 s SIGTERM grace because uvicorn already
        shortened its own grace; a longer wait here would be ignored
        anyway.

        We wait on ``_wait_task`` (the reap loop) so the FSM lands
        in a terminal state before the lifespan tears down — without
        this, an out-of-order ``bus.close()`` would clip the final
        ``trainer_exited`` event.
        """
        if self.is_running:
            await self.request_stop(grace_seconds=5.0)

        # Reap MUST land before we cancel pumps — it drives the
        # final FSM transition. Bound by a hard timeout so a wedged
        # subprocess doesn't keep the lifespan open forever.
        if self._wait_task is not None and not self._wait_task.done():
            with contextlib.suppress(asyncio.CancelledError, TimeoutError, Exception):
                await asyncio.wait_for(
                    asyncio.shield(self._wait_task), timeout=10.0,
                )

        # Anything still alive (pumps, escalation task) — cancel.
        for task in (
            self._stdout_task,
            self._stderr_task,
            self._escalation_task,
        ):
            if task is not None and not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

        # Close the stdio capture file last — pumps must drain into it
        # before the handle goes away. Suppressing because a closed
        # bus / fs error here cannot affect terminal state.
        if self._stdio_log_file is not None:
            with contextlib.suppress(OSError, ValueError):
                self._stdio_log_file.close()
            self._stdio_log_file = None

    # --- internals ----------------------------------------------------------

    def _signal_group(self, sig: int) -> None:
        """Send ``sig`` to the cached process group; tolerate races.

        ``ProcessLookupError`` means the group has already been reaped.
        ``PermissionError`` would only happen if we lost ownership —
        impossible inside the same UID, but caught defensively.
        """
        if self._pgid is None:
            return
        try:
            os.killpg(self._pgid, sig)
        except (ProcessLookupError, PermissionError):
            pass

    async def _pump_stream(
        self,
        stream: asyncio.StreamReader | None,
        *,
        kind: str,
    ) -> None:
        """Read ``stream`` line-by-line; append each line to the stdio
        capture file with a kind prefix.

        Writes to the file are best-effort — disk-full / stale-handle
        OSErrors are suppressed so the pump never blocks on disk
        contention. The pumps drain the OS pipe buffer; failure to
        drain would cause the subprocess to SIGPIPE its own writes,
        which is the load-bearing invariant.

        Output format per line:
            ``[OUT] <line>\\n`` for stdout
            ``[ERR] <line>\\n`` for stderr

        ``kind`` is one of ``"stdout"`` / ``"stderr"``. Empty lines
        are dropped.
        """
        if stream is None:  # pragma: no cover — defensive
            return
        prefix = b"[ERR] " if kind == "stderr" else b"[OUT] "
        while True:
            try:
                raw = await stream.readline()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # pragma: no cover — best effort
                self._bus.publish(
                    "stream_error",
                    {"kind": kind, "error": repr(exc)},
                )
                return
            if not raw:
                return  # EOF — pipe closed (subprocess exited or stream done)
            # ``raw`` includes the trailing newline (or is empty on EOF
            # which we already handled). Drop blank lines to keep the
            # file tight.
            stripped = raw.rstrip(b"\n")
            if not stripped:
                continue
            if self._stdio_log_file is not None:
                with contextlib.suppress(OSError, ValueError):
                    self._stdio_log_file.write(prefix + stripped + b"\n")

    async def _reap(self) -> None:
        """Wait on the subprocess and drive the final FSM transition."""
        if self._proc is None:  # pragma: no cover — defensive
            return

        rc = await self._proc.wait()

        # Drain any tail from the pumps before publishing the final
        # event — give them up to 2 s to finish. ALL exceptions are
        # suppressed: a stuck pump (timeout), a cancelled pump
        # (CancelledError), or a pump that raised mid-readline must
        # never prevent the FSM from reaching its terminal state.
        # Without this, a slow stderr writer could leave a job
        # forever in ``running`` because the reap task crashed
        # before its FSM transition.
        for task in (self._stdout_task, self._stderr_task):
            if task is not None and not task.done():
                with contextlib.suppress(
                    asyncio.CancelledError, TimeoutError, Exception,
                ):
                    await asyncio.wait_for(task, timeout=2.0)

        signal_name = _signal_name_from_rc(rc)
        message_parts: list[str] = [f"exit_code={rc}"]
        if signal_name is not None:
            message_parts.append(f"signal={signal_name}")
        message = ", ".join(message_parts)

        # Decide terminal state.
        if rc == 0:
            target = (
                JobState.COMPLETED
                if not self._cancellation_requested
                else JobState.CANCELLED
            )
        elif self._cancellation_requested:
            # We sent the signal, even if SIGKILL — landing as
            # ``cancelled`` matches the user's intent, and the rc /
            # signal name is preserved in the message for forensics.
            target = JobState.CANCELLED
        else:
            target = JobState.FAILED

        # Phase D — resolve the structured exit payload BEFORE publishing
        # ``trainer_exited``. Three branches:
        #   1. ``<workdir>/trainer-exit.json`` exists + validates →
        #      use its typed fields (code / message / traceback_summary).
        #   2. File absent + rc==0 → success; no failure fields.
        #   3. File absent + rc!=0 → synthesise payload. SIGKILL
        #      (137 / -9) gets ``TRAINING_OOM`` (R-SIGKILL heuristic:
        #      false-positive OOM beats noisy INTERNAL_ERROR for the
        #      common cgroup-OOM scenario where atexit doesn't run).
        #      Other non-zero rc gets ``INTERNAL_ERROR`` with the rc
        #      stamped into the message.
        exit_payload = _read_exit_payload(self._workdir)
        synthesized_payload = exit_payload is None and not self._cancellation_requested and (
            rc != 0
        )

        trainer_exited_event = _build_trainer_exited_event(
            rc=rc,
            signal_name=signal_name,
            cancellation_requested=self._cancellation_requested,
            payload=exit_payload,
            synthesize_if_missing=synthesized_payload,
        )

        # Publish first, transition second — ensures the WS subscriber
        # sees the final event slot before the FSM closes. Payload stays
        # minimal: trainer stdout/stderr ground truth lives in
        # ``trainer.stdio.log`` on disk and is pulled by Mac via SCP.
        # Pre-2026-05-02 we briefly embedded a stderr/stdout tail here
        # (PR-B) but that duplicated the postmortem dump on Mac and was
        # reverted; the diagnostic-grace window (PR-C) gives Mac enough
        # room to SCP before pod teardown.
        self._bus.publish("trainer_exited", trainer_exited_event)

        try:
            self._fsm.transition(target, message=message)
        except InvalidTransitionError:
            # The FSM is already terminal — extremely rare race where
            # ``shutdown()`` ran during reap. Tolerate.
            pass

        # Phase 9.C — emit ``cancellation_completed`` telemetry only
        # when the run was actually cancelled. Natural exits and
        # crashes still get ``trainer_exited`` (handled above) but
        # the cancellation-specific event is silent for them — keeps
        # the operator dashboards focused on real stop chains.
        if (
            self._cancellation_requested
            and self._cancellation_started_at_ms is not None
        ):
            from ryotenkai_shared.observability.cancellation_telemetry import (
                CANCELLATION_COMPLETED,
                latency_ms_since,
            )
            self._bus.publish(
                CANCELLATION_COMPLETED,
                {
                    "total_latency_ms": latency_ms_since(
                        self._cancellation_started_at_ms,
                    ),
                    "terminal_state": target.value,
                    "exit_code": rc,
                    "signal": signal_name,
                    "requested_at_ms": self._cancellation_started_at_ms,
                },
            )

        # Fire the terminal hook AFTER the FSM transition lands. Errors
        # in the hook (e.g. RunPod GraphQL outage) must not affect the
        # supervisor's terminal state — the hook implementation is
        # expected to publish its own outcome event for forensics.
        if self._terminal_hook is not None:
            with contextlib.suppress(Exception):
                await self._terminal_hook(target.value)

        # Close the stdio capture file once pumps have drained (the
        # ``await asyncio.wait_for(task, timeout=2.0)`` calls above
        # ensure the pumps are EOF'd before we land here). Leaving
        # the handle open across runs is fine on the same Supervisor
        # instance — but we close eagerly so on-disk size is exposed
        # to log_manager scp without waiting for shutdown().
        if self._stdio_log_file is not None:
            with contextlib.suppress(OSError, ValueError):
                self._stdio_log_file.close()
            self._stdio_log_file = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Phase D — TrainerExitPayload reader + event builder
# ---------------------------------------------------------------------------


# Exit codes that POSIX shells / Python report for SIGKILL. Both shapes
# can be observed depending on whether the killing signal came from the
# kernel (138, 137 on POSIX = 128 + 9) or asyncio's negative-rc shape.
_SIGKILL_EXIT_CODES: frozenset[int] = frozenset({137, -9})

# Schema version stamped on every ``trainer_exited`` event payload.
# Bumped to ``2`` in Phase D — pre-D events had no ``schema_version``
# field; consumers that need backward compatibility check ``event.get(
# "schema_version", 1)``.
TRAINER_EXITED_EVENT_SCHEMA_VERSION: int = 2


def _read_exit_payload(workdir: "Path | None"):  # type: ignore[name-defined]
    """Read + validate ``<workdir>/trainer-exit.json``; ``None`` if absent/invalid.

    The trainer's :mod:`ryotenkai_pod.trainer.exit_reporter` writes
    this file atomically on failure. Any read / parse error is logged
    at WARN and treated as if the file were missing, so the
    supervisor falls through to the exit-code heuristic.
    """
    if workdir is None:
        return None
    # Local imports keep the supervisor's cold-path import surface small —
    # the contract module pulls Pydantic which we don't need on the
    # success path.
    from pathlib import Path as _Path

    from ryotenkai_shared.contracts.trainer_exit import (
        TRAINER_EXIT_FILENAME,
        TrainerExitPayload,
    )

    target = _Path(workdir) / TRAINER_EXIT_FILENAME
    try:
        return TrainerExitPayload.read_from(target)
    except Exception as exc:  # noqa: BLE001 — defensive
        import logging
        logging.getLogger(__name__).warning(
            "[SUPERVISOR] trainer-exit.json at %s failed to parse: %s",
            target, exc,
        )
        return None


def _build_trainer_exited_event(
    *,
    rc: int,
    signal_name: str | None,
    cancellation_requested: bool,
    payload,  # TrainerExitPayload | None
    synthesize_if_missing: bool,
) -> dict:
    """Build the ``trainer_exited`` event body (Phase D, schema_version=2).

    Legacy fields (``exit_code``, ``signal``, ``cancellation_requested``)
    are preserved verbatim so older consumers keep working. New fields:

    * ``schema_version`` — pinned at :data:`TRAINER_EXITED_EVENT_SCHEMA_VERSION`.
    * ``code`` — :class:`ErrorCode` value or ``None`` on natural success.
    * ``message`` — single-line human description; ``None`` on success.
    * ``traceback_summary`` — sanitised tail; ``None`` when no
      traceback was available (incl. SIGKILL paths).
    * ``wall_seconds`` — copied from the payload if present.
    * ``payload_source`` — one of ``"trainer_file"``, ``"sigkill_heuristic"``,
      ``"missing"``, or ``"none"`` (rc==0 and clean exit).
      Lets the control side distinguish trusted typed signals from
      synthesised ones.
    """
    from ryotenkai_shared.contracts.problem_details import ErrorCode

    body: dict = {
        "exit_code": rc,
        "signal": signal_name,
        "cancellation_requested": cancellation_requested,
        "schema_version": TRAINER_EXITED_EVENT_SCHEMA_VERSION,
        "code": None,
        "message": None,
        "traceback_summary": None,
        "wall_seconds": None,
        "payload_source": "none",
    }

    if payload is not None:
        body["code"] = payload.code.value
        body["message"] = payload.message
        body["traceback_summary"] = payload.traceback_summary
        body["wall_seconds"] = payload.wall_seconds
        body["payload_source"] = "trainer_file"
        return body

    if not synthesize_if_missing:
        return body

    # File absent on a failed exit — synthesise.
    if rc in _SIGKILL_EXIT_CODES:
        # R-SIGKILL heuristic: cgroup OOM-killer / external SIGKILL.
        # Documented in
        # ``docs/plans/sharded-stargazing-wigderson.md`` Layer 6.
        body["code"] = ErrorCode.TRAINING_OOM.value
        body["message"] = (
            "Trainer killed by signal SIGKILL — likely OOM "
            f"(exit_code={rc})"
        )
        body["payload_source"] = "sigkill_heuristic"
        return body

    body["code"] = ErrorCode.INTERNAL_ERROR.value
    body["message"] = f"trainer exited without payload, exit_code={rc}"
    body["payload_source"] = "missing"
    return body


def _signal_name_from_rc(rc: int) -> str | None:
    """Translate ``rc`` to a signal name, or ``None`` if rc was a
    normal Python exit code.

    POSIX shells encode signal-N exits as 128+N (so 137 = SIGKILL on
    Linux). asyncio's ``Process.returncode`` reports negative values
    on some platforms (-N where N is the signal number).
    """
    signal_no: int | None = None
    if rc < 0:
        signal_no = -rc
    elif rc > 128:
        signal_no = rc - 128

    if signal_no is None:
        return None

    try:
        return signal.Signals(signal_no).name
    except ValueError:
        return f"signal-{signal_no}"
