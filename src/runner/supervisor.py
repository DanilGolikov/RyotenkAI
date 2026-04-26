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
  into the :class:`EventBus`,
- the lifecycle bridge to :class:`JobLifecycleFSM` — every state
  transition originates here, never in the API layer.

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
becomes a single :class:`Event` via :meth:`EventBus.publish`. Phase 2
emits raw lines untouched; Phase 3+ may add a TrainerCallback hook
on top that produces structured events while raw stdout still flows
through this path.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
from pathlib import Path
from typing import TYPE_CHECKING

from src.runner.state import (
    InvalidTransitionError,
    JobState,
)

if TYPE_CHECKING:
    from src.runner.event_bus import EventBus
    from src.runner.state import JobLifecycleFSM

__all__ = [
    "Supervisor",
    "SupervisorError",
    "SupervisorBusy",
]


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

    def __init__(self, fsm: "JobLifecycleFSM", bus: "EventBus") -> None:
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

        self._cancellation_requested = True
        self._bus.publish("stop_requested", {"grace_seconds": grace_seconds})
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
        """Read ``stream`` line-by-line; each line becomes a bus event."""
        if stream is None:  # pragma: no cover — defensive
            return
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
            line = raw.decode("utf-8", errors="replace").rstrip("\n")
            if not line:
                continue
            self._bus.publish("trainer_log", {"kind": kind, "line": line})

    async def _reap(self) -> None:
        """Wait on the subprocess and drive the final FSM transition."""
        if self._proc is None:  # pragma: no cover — defensive
            return

        rc = await self._proc.wait()

        # Drain any tail from the pumps before publishing the final
        # event — give them one tick to finish.
        for task in (self._stdout_task, self._stderr_task):
            if task is not None and not task.done():
                with contextlib.suppress(asyncio.CancelledError):
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

        # Publish first, transition second — ensures the WS subscriber
        # sees the final event slot before the FSM closes.
        self._bus.publish(
            "trainer_exited",
            {
                "exit_code": rc,
                "signal": signal_name,
                "cancellation_requested": self._cancellation_requested,
            },
        )

        try:
            self._fsm.transition(target, message=message)
        except InvalidTransitionError:
            # The FSM is already terminal — extremely rare race where
            # ``shutdown()`` ran during reap. Tolerate.
            pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
