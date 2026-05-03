"""Phase 2 — :class:`Supervisor` integration tests with real subprocesses.

These exercise actual fork + signal + reap semantics that the
``MockSupervisor`` in :mod:`conftest` deliberately skips. Each test
spawns a small Python subprocess via the supervisor and asserts on
the FSM transitions / events / exit codes the supervisor produces.

Tests use ``sys.executable`` so the trainer subprocess always
matches the test interpreter — no PATH games, no "python vs python3"
foot-guns.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import pytest_asyncio

from ryotenkai_pod.runner.event_bus import EventBus
from ryotenkai_pod.runner.state import (
    JobLifecycleFSM,
    JobState,
)
from ryotenkai_pod.runner.supervisor import Supervisor, SupervisorBusy

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fsm(tmp_path: Path) -> JobLifecycleFSM:
    f = JobLifecycleFSM(workspace_dir=tmp_path)
    f.restore_or_init()
    return f


@pytest.fixture
def bus() -> EventBus:
    return EventBus(capacity=100)


@pytest.fixture
def stdio_log_path(tmp_path: Path) -> Path:
    """Per-test stdio capture file. Single-writer; test reads it
    after the supervisor reaps."""
    return tmp_path / "logs" / "trainer.stdio.log"


@pytest_asyncio.fixture
async def supervisor(
    fsm: JobLifecycleFSM, bus: EventBus, stdio_log_path: Path,
) -> AsyncIterator[Supervisor]:
    """Real :class:`Supervisor` with cleanup on test exit.

    ``shutdown()`` is idempotent and safe even if the test never
    spawned a subprocess, so we always call it.
    """
    s = Supervisor(fsm, bus, stdio_log_path=stdio_log_path)
    try:
        yield s
    finally:
        await s.shutdown()


def _py(code: str) -> list[str]:
    """Build an argv that runs ``code`` in the test's Python."""
    return [sys.executable, "-c", code]


def _read_stdio_lines(path: Path, kind: str | None = None) -> list[str]:
    """Read captured stdio file and return lines, optionally filtered.

    Each line in the file has form ``[OUT] <text>`` or ``[ERR] <text>``.
    Pass ``kind="stdout"`` to keep only OUT lines, ``"stderr"`` for ERR,
    or ``None`` for everything (with prefix stripped).
    """
    if not path.exists():
        return []
    raw = path.read_text(encoding="utf-8", errors="replace")
    out: list[str] = []
    for line in raw.splitlines():
        if line.startswith("[OUT] "):
            if kind in (None, "stdout"):
                out.append(line[6:])
        elif line.startswith("[ERR] "):
            if kind in (None, "stderr"):
                out.append(line[6:])
    return out


async def _wait_for_state(
    fsm: JobLifecycleFSM, target: JobState, timeout: float = 5.0,
) -> None:
    """Poll the FSM until ``target`` is reached or timeout fires."""
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        snap = fsm.current()
        if snap is not None and snap.state == target:
            return
        await asyncio.sleep(0.02)
    snap = fsm.current()
    raise AssertionError(
        f"FSM did not reach {target.value} within {timeout}s; "
        f"last state = {snap.state.value if snap else 'None'}",
    )


# ---------------------------------------------------------------------------
# Happy-path lifecycle
# ---------------------------------------------------------------------------


class TestNaturalExit:
    async def test_exit_zero_lands_in_completed(
        self, supervisor: Supervisor, fsm: JobLifecycleFSM,
    ) -> None:
        await supervisor.submit_and_spawn("j-1", _py("pass"))
        # ``pass`` exits immediately. Wait for the reap task to land.
        await _wait_for_state(fsm, JobState.COMPLETED)
        snap = fsm.current()
        assert snap is not None
        assert "exit_code=0" in snap.message

    async def test_exit_nonzero_lands_in_failed(
        self, supervisor: Supervisor, fsm: JobLifecycleFSM,
    ) -> None:
        await supervisor.submit_and_spawn(
            "j-1", _py("import sys; sys.exit(7)"),
        )
        await _wait_for_state(fsm, JobState.FAILED)
        snap = fsm.current()
        assert snap is not None
        assert "exit_code=7" in snap.message

    async def test_workdir_sets_subprocess_cwd(
        self, supervisor: Supervisor, fsm: JobLifecycleFSM,
        stdio_log_path: Path, tmp_path: Path,
    ) -> None:
        """``submit_and_spawn(..., workdir=PATH)`` must set the
        subprocess's cwd so relative paths resolve from there.

        Regression guard for the trainer-cwd bug: previously the
        workdir argument was accepted but never propagated to the
        subprocess (see ``src/runner/api/jobs.py`` — old code
        dropped it). The trainer then inherited uvicorn's ``/root``
        and FileNotFoundError'd on its config arg.

        We verify by spawning a Python that prints its CWD; the
        line lands in the stdio capture file.
        """
        run_dir = tmp_path / "run-cwd-check"
        run_dir.mkdir()

        # Spawn a child that just prints its cwd — fast enough that
        # the FSM races to COMPLETED before the test wraps up.
        await supervisor.submit_and_spawn(
            "j-cwd", _py("import os; print(os.getcwd())"),
            workdir=run_dir,
        )
        await _wait_for_state(fsm, JobState.COMPLETED)

        # Read the stdio capture file and look for the cwd line.
        cwd_lines = _read_stdio_lines(stdio_log_path, kind="stdout")
        assert any(str(run_dir) in line for line in cwd_lines), (
            f"expected child cwd to be {run_dir!s} but stdio lines "
            f"didn't contain it; got: {cwd_lines!r}"
        )

    async def test_stdout_lines_captured_to_file(
        self, supervisor: Supervisor, fsm: JobLifecycleFSM,
        stdio_log_path: Path,
    ) -> None:
        await supervisor.submit_and_spawn(
            "j-1", _py("print('hello'); print('world')"),
        )
        await _wait_for_state(fsm, JobState.COMPLETED)
        lines = _read_stdio_lines(stdio_log_path, kind="stdout")
        assert "hello" in lines
        assert "world" in lines

    async def test_stderr_lines_captured_to_file(
        self, supervisor: Supervisor, fsm: JobLifecycleFSM,
        stdio_log_path: Path,
    ) -> None:
        """Trainer stderr (incl. ImportError tracebacks) must land
        in the stdio capture file independently of stdout."""
        # Print to stderr explicitly so we can verify per-stream split.
        await supervisor.submit_and_spawn(
            "j-stderr",
            _py("import sys; sys.stderr.write('boom\\n'); sys.exit(0)"),
        )
        await _wait_for_state(fsm, JobState.COMPLETED)
        err_lines = _read_stdio_lines(stdio_log_path, kind="stderr")
        assert "boom" in err_lines

    async def test_no_trainer_log_event_published(
        self, supervisor: Supervisor, bus: EventBus, fsm: JobLifecycleFSM,
    ) -> None:
        """Data plane no longer routes through the bus — log lines
        go to the file, not to ``trainer_log`` events. This test pins
        the contract."""
        await supervisor.submit_and_spawn(
            "j-no-log-events", _py("print('hello'); print('world')"),
        )
        await _wait_for_state(fsm, JobState.COMPLETED)
        log_events = [
            e for e in list(bus._buffer) if e.kind == "trainer_log"
        ]
        assert log_events == [], (
            f"expected no trainer_log events on the bus; got {log_events!r}"
        )


# ---------------------------------------------------------------------------
# Stop semantics
# ---------------------------------------------------------------------------


class TestRequestStop:
    async def test_sigterm_long_running_lands_in_cancelled(
        self, supervisor: Supervisor, fsm: JobLifecycleFSM,
    ) -> None:
        # Subprocess installs a SIGTERM handler that exits 0 — the
        # supervisor sees rc=0 + cancellation_requested → cancelled.
        code = (
            "import signal, time\n"
            "signal.signal(signal.SIGTERM, lambda *_: __import__('sys').exit(0))\n"
            "time.sleep(60)\n"
        )
        await supervisor.submit_and_spawn("j-1", _py(code))
        await _wait_for_state(fsm, JobState.RUNNING)

        await supervisor.request_stop(grace_seconds=2.0)
        await _wait_for_state(fsm, JobState.CANCELLED, timeout=3.0)
        snap = fsm.current()
        assert snap is not None
        assert snap.state == JobState.CANCELLED

    async def test_sigkill_escalation_when_term_ignored(
        self, supervisor: Supervisor, fsm: JobLifecycleFSM, bus: EventBus,
        stdio_log_path: Path,
    ) -> None:
        # Subprocess swallows SIGTERM — the supervisor must escalate
        # to SIGKILL after the grace window. The "READY" print is a
        # ready barrier: we don't send SIGTERM until we know the
        # trainer has installed its ignore-handler. Without it, a
        # SIGTERM landing during ``import signal`` kills the
        # subprocess fast and the escalation never fires.
        code = (
            "import signal, time, sys\n"
            "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
            "print('READY', flush=True)\n"
            "time.sleep(60)\n"
        )
        await supervisor.submit_and_spawn("j-1", _py(code))
        await _wait_for_state(fsm, JobState.RUNNING)

        # Wait for READY on stdout — the trainer's signal handler
        # is installed by the time this line lands in the file.
        async def _wait_ready() -> None:
            deadline = asyncio.get_event_loop().time() + 5.0
            while asyncio.get_event_loop().time() < deadline:
                if "READY" in _read_stdio_lines(stdio_log_path, kind="stdout"):
                    return
                await asyncio.sleep(0.02)
            raise AssertionError("trainer never printed READY")

        await _wait_ready()

        await supervisor.request_stop(grace_seconds=0.3)
        await _wait_for_state(fsm, JobState.CANCELLED, timeout=3.0)

        kinds = [e.kind for e in list(bus._buffer)]
        assert "stop_escalated" in kinds
        snap = fsm.current()
        assert snap is not None
        # exit_code carries SIGKILL = 9 (rendered as -9 on POSIX
        # asyncio subprocesses, or 137 if the kernel reports +128).
        assert snap.state == JobState.CANCELLED
        assert "SIGKILL" in snap.message

    async def test_stop_when_idle_is_noop(
        self, supervisor: Supervisor,
    ) -> None:
        # Calling request_stop with no active subprocess is a no-op.
        await supervisor.request_stop(grace_seconds=0.1)
        assert not supervisor.is_running


# ---------------------------------------------------------------------------
# Native crash
# ---------------------------------------------------------------------------


class TestNativeCrash:
    async def test_segv_lands_in_failed_with_signal_name(
        self, supervisor: Supervisor, fsm: JobLifecycleFSM,
    ) -> None:
        # ``os._exit(139)`` simulates 128 + SIGSEGV (11) — the
        # supervisor's exit-code parser must translate 139 to
        # SIGSEGV.
        await supervisor.submit_and_spawn(
            "j-1", _py("import os; os._exit(139)"),
        )
        await _wait_for_state(fsm, JobState.FAILED)
        snap = fsm.current()
        assert snap is not None
        assert "SIGSEGV" in snap.message
        assert "exit_code=139" in snap.message


# ---------------------------------------------------------------------------
# Single-active enforcement
# ---------------------------------------------------------------------------


class TestSupervisorBusy:
    async def test_double_spawn_raises(
        self, supervisor: Supervisor, fsm: JobLifecycleFSM,
    ) -> None:
        await supervisor.submit_and_spawn(
            "j-1", _py("import time; time.sleep(60)"),
        )
        await _wait_for_state(fsm, JobState.RUNNING)
        with pytest.raises(SupervisorBusy):
            await supervisor.submit_and_spawn("j-2", _py("pass"))

    async def test_spawn_failure_rolls_fsm_to_failed(
        self, supervisor: Supervisor, fsm: JobLifecycleFSM,
    ) -> None:
        # Bogus command → FileNotFoundError. The FSM must land in
        # ``failed`` (rolled forward by submit_and_spawn) rather than
        # leaving a stuck ``preparing``.
        with pytest.raises((FileNotFoundError, OSError)):
            await supervisor.submit_and_spawn(
                "j-1", ["/no/such/binary"],
            )
        snap = fsm.current()
        assert snap is not None
        assert snap.state == JobState.FAILED


# ---------------------------------------------------------------------------
# Process group
# ---------------------------------------------------------------------------


class TestProcessGroup:
    async def test_pgid_is_distinct_from_parent(
        self, supervisor: Supervisor, fsm: JobLifecycleFSM,
    ) -> None:
        # Sanity: the trainer is in a fresh session/group, so its
        # pgid is NOT our pgid. Without ``start_new_session=True``
        # they would coincide and SIGTERM would kill the test runner.
        await supervisor.submit_and_spawn(
            "j-1", _py("import time; time.sleep(60)"),
        )
        await _wait_for_state(fsm, JobState.RUNNING)
        assert supervisor.pgid is not None
        assert supervisor.pgid != os.getpgrp()

        await supervisor.request_stop(grace_seconds=0.5)
        await _wait_for_state(fsm, JobState.CANCELLED, timeout=2.0)


# ---------------------------------------------------------------------------
# Shutdown
# ---------------------------------------------------------------------------


class TestReapResilience:
    async def test_stuck_pump_does_not_block_reap(
        self, supervisor: Supervisor, fsm: JobLifecycleFSM,
    ) -> None:
        # Replace the stdout pump with a never-finishing task before
        # the trainer exits — the reap drain must time out cleanly
        # and still drive the FSM to its terminal state. Regression
        # for "_reap drain doesn't suppress TimeoutError" — without
        # the fix the reap task crashes and the FSM stays stuck in
        # ``running``.
        await supervisor.submit_and_spawn("j-1", _py("pass"))
        await _wait_for_state(fsm, JobState.RUNNING)

        # Override the stdout pump with a never-resolving sleep.
        # Cancel the original first so we don't leak a task.
        if supervisor._stdout_task is not None:
            supervisor._stdout_task.cancel()
        supervisor._stdout_task = asyncio.create_task(asyncio.sleep(60))

        # FSM must still reach a terminal state — the reap fixes
        # the pump-tail timeout.
        await _wait_for_state(fsm, JobState.COMPLETED, timeout=5.0)


# ---------------------------------------------------------------------------
# Terminal hook (Phase 4.4 — pod auto-stop wiring)
# ---------------------------------------------------------------------------


class TestTerminalHook:
    """Supervisor must fire its ``terminal_hook`` exactly once on the
    FSM's terminal transition. This wires the RunPod auto-stop in
    production — a regression here means we leak billing time."""

    async def test_hook_fires_with_completed_state(
        self, fsm: JobLifecycleFSM, bus: EventBus,
    ) -> None:
        seen: list[str] = []

        async def _hook(state: str) -> None:
            seen.append(state)

        s = Supervisor(fsm, bus, terminal_hook=_hook)
        try:
            await s.submit_and_spawn("j-hook-ok", _py("pass"))
            await _wait_for_state(fsm, JobState.COMPLETED)
        finally:
            await s.shutdown()
        # Single fire, exactly the terminal name (not e.g. ``running``).
        assert seen == ["completed"]

    async def test_hook_fires_with_failed_state(
        self, fsm: JobLifecycleFSM, bus: EventBus,
    ) -> None:
        seen: list[str] = []

        async def _hook(state: str) -> None:
            seen.append(state)

        s = Supervisor(fsm, bus, terminal_hook=_hook)
        try:
            await s.submit_and_spawn("j-hook-fail", _py("import sys; sys.exit(2)"))
            await _wait_for_state(fsm, JobState.FAILED)
        finally:
            await s.shutdown()
        assert seen == ["failed"]

    async def test_hook_exception_does_not_unset_terminal_state(
        self, fsm: JobLifecycleFSM, bus: EventBus,
    ) -> None:
        # A hook that raises must not roll the FSM back — the pod
        # must reach its terminal state regardless of GraphQL outage.
        async def _bad_hook(_state: str) -> None:
            raise RuntimeError("graphql unreachable")

        s = Supervisor(fsm, bus, terminal_hook=_bad_hook)
        try:
            await s.submit_and_spawn("j-bad-hook", _py("pass"))
            await _wait_for_state(fsm, JobState.COMPLETED)
        finally:
            await s.shutdown()
        snap = fsm.current()
        assert snap is not None
        assert snap.state == JobState.COMPLETED


class TestShutdown:
    async def test_shutdown_kills_running_trainer(
        self, fsm: JobLifecycleFSM, bus: EventBus,
    ) -> None:
        # Build the supervisor manually so we can call shutdown()
        # explicitly inside the test (the fixture would do it on
        # teardown, but we want to assert behaviour here).
        s = Supervisor(fsm, bus)
        await s.submit_and_spawn(
            "j-1", _py("import time; time.sleep(60)"),
        )
        await _wait_for_state(fsm, JobState.RUNNING)

        await s.shutdown()
        # After shutdown the process is dead and the FSM is terminal.
        snap = fsm.current()
        assert snap is not None
        assert snap.state.is_terminal
        assert not s.is_running


# ---------------------------------------------------------------------------
# Phase 9.C — cancellation telemetry chain
# ---------------------------------------------------------------------------


class TestCancellationTelemetry:
    """Phase 9.C wires the supervisor into the cancellation event
    chain. The two events the supervisor owns are
    ``cancellation_started`` (FSM running → stopping) and
    ``cancellation_completed`` (FSM stopping → terminal).

    The contract these tests pin:
    * ``cancellation_started`` carries ``requested_at_ms`` (epoch
      anchor), ``grace_seconds``, and ``reason``.
    * Legacy ``stop_requested`` event still fires alongside (no
      backwards-compat break for old subscribers).
    * ``cancellation_completed`` fires ONLY when stop was requested
      — natural exits / native crashes do NOT emit it.
    * ``cancellation_completed.total_latency_ms`` is anchored to
      ``cancellation_started.requested_at_ms`` and never negative.
    """

    @staticmethod
    def _events_of(bus: EventBus, kind: str) -> list[dict]:
        return [
            e.payload for e in list(bus._buffer) if e.kind == kind
        ]

    async def test_request_stop_emits_cancellation_started(
        self, supervisor: Supervisor, fsm: JobLifecycleFSM, bus: EventBus,
    ) -> None:
        from ryotenkai_shared.observability.cancellation_telemetry import (
            CANCELLATION_STARTED,
        )

        code = (
            "import signal, time\n"
            "signal.signal(signal.SIGTERM, lambda *_: __import__('sys').exit(0))\n"
            "time.sleep(60)\n"
        )
        await supervisor.submit_and_spawn("j-cs", _py(code))
        await _wait_for_state(fsm, JobState.RUNNING)
        await supervisor.request_stop(grace_seconds=2.0)
        await _wait_for_state(fsm, JobState.CANCELLED, timeout=3.0)

        started = self._events_of(bus, CANCELLATION_STARTED)
        assert len(started) == 1
        payload = started[0]
        assert "requested_at_ms" in payload
        assert isinstance(payload["requested_at_ms"], int)
        assert payload["requested_at_ms"] > 0
        assert payload["grace_seconds"] == 2.0
        assert payload["reason"] == "user_stop"

    async def test_legacy_stop_requested_still_emitted(
        self, supervisor: Supervisor, fsm: JobLifecycleFSM, bus: EventBus,
    ) -> None:
        # Backwards compat — pre-9.C consumers grep'd ``stop_requested``.
        # The 9.C addition is additive; the legacy event still fires.
        code = (
            "import signal, time\n"
            "signal.signal(signal.SIGTERM, lambda *_: __import__('sys').exit(0))\n"
            "time.sleep(60)\n"
        )
        await supervisor.submit_and_spawn("j-legacy", _py(code))
        await _wait_for_state(fsm, JobState.RUNNING)
        await supervisor.request_stop(grace_seconds=2.0)
        await _wait_for_state(fsm, JobState.CANCELLED, timeout=3.0)

        legacy = self._events_of(bus, "stop_requested")
        assert len(legacy) == 1
        assert legacy[0] == {"grace_seconds": 2.0}

    async def test_request_stop_emits_cancellation_completed(
        self, supervisor: Supervisor, fsm: JobLifecycleFSM, bus: EventBus,
    ) -> None:
        from ryotenkai_shared.observability.cancellation_telemetry import (
            CANCELLATION_COMPLETED,
            CANCELLATION_STARTED,
        )

        code = (
            "import signal, time\n"
            "signal.signal(signal.SIGTERM, lambda *_: __import__('sys').exit(0))\n"
            "time.sleep(60)\n"
        )
        await supervisor.submit_and_spawn("j-cc", _py(code))
        await _wait_for_state(fsm, JobState.RUNNING)
        await supervisor.request_stop(grace_seconds=2.0)
        await _wait_for_state(fsm, JobState.CANCELLED, timeout=3.0)

        started = self._events_of(bus, CANCELLATION_STARTED)
        completed = self._events_of(bus, CANCELLATION_COMPLETED)
        assert len(started) == 1
        assert len(completed) == 1

        completed_payload = completed[0]
        assert "total_latency_ms" in completed_payload
        assert isinstance(completed_payload["total_latency_ms"], int)
        assert completed_payload["total_latency_ms"] >= 0
        # ``requested_at_ms`` from started must equal the one in
        # completed — same anchor for downstream joins.
        assert (
            completed_payload["requested_at_ms"]
            == started[0]["requested_at_ms"]
        )
        assert completed_payload["terminal_state"] == "cancelled"
        # exit_code can be 0 (clean signal handler exit) or -15
        # (asyncio's "killed by SIGTERM" convention). Both are valid
        # cancellation outcomes — what matters is that the FSM
        # respected the stop request.
        assert completed_payload["exit_code"] in (0, -15, 143)

    async def test_natural_exit_does_not_emit_cancellation_completed(
        self, supervisor: Supervisor, fsm: JobLifecycleFSM, bus: EventBus,
    ) -> None:
        from ryotenkai_shared.observability.cancellation_telemetry import (
            CANCELLATION_COMPLETED,
            CANCELLATION_STARTED,
        )

        # No request_stop — pure natural exit. Neither cancellation
        # event should fire; operator dashboards must stay focused on
        # actual stop chains.
        await supervisor.submit_and_spawn("j-nat", _py("pass"))
        await _wait_for_state(fsm, JobState.COMPLETED)

        assert self._events_of(bus, CANCELLATION_STARTED) == []
        assert self._events_of(bus, CANCELLATION_COMPLETED) == []

    async def test_native_crash_does_not_emit_cancellation_completed(
        self, supervisor: Supervisor, fsm: JobLifecycleFSM, bus: EventBus,
    ) -> None:
        from ryotenkai_shared.observability.cancellation_telemetry import (
            CANCELLATION_COMPLETED,
            CANCELLATION_STARTED,
        )

        # rc=1 (failed, but no stop request). Cancellation events are
        # silent — this is FSM=failed without a user stop.
        await supervisor.submit_and_spawn(
            "j-crash", _py("import sys; sys.exit(1)"),
        )
        await _wait_for_state(fsm, JobState.FAILED)

        assert self._events_of(bus, CANCELLATION_STARTED) == []
        assert self._events_of(bus, CANCELLATION_COMPLETED) == []

    async def test_sigkill_escalation_still_emits_completed(
        self, supervisor: Supervisor, fsm: JobLifecycleFSM, bus: EventBus,
        stdio_log_path: Path,
    ) -> None:
        # Trainer ignores SIGTERM → SIGKILL escalates → FSM=cancelled.
        # ``cancellation_completed`` MUST still fire (we requested
        # stop, even if escalated). ``terminal_state`` is "cancelled"
        # because the supervisor honors the stop request even for
        # SIGKILL exits.
        from ryotenkai_shared.observability.cancellation_telemetry import (
            CANCELLATION_COMPLETED,
        )

        code = (
            "import signal, time, sys\n"
            "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
            "print('READY', flush=True)\n"
            "time.sleep(60)\n"
        )
        await supervisor.submit_and_spawn("j-kill", _py(code))
        await _wait_for_state(fsm, JobState.RUNNING)

        # Wait for READY then request stop with tight grace.
        async def _wait_ready() -> None:
            deadline = asyncio.get_event_loop().time() + 5.0
            while asyncio.get_event_loop().time() < deadline:
                if "READY" in _read_stdio_lines(stdio_log_path, kind="stdout"):
                    return
                await asyncio.sleep(0.02)
            raise AssertionError("trainer never printed READY")

        await _wait_ready()
        await supervisor.request_stop(grace_seconds=0.3)
        await _wait_for_state(fsm, JobState.CANCELLED, timeout=3.0)

        completed = self._events_of(bus, CANCELLATION_COMPLETED)
        assert len(completed) == 1
        assert completed[0]["terminal_state"] == "cancelled"
        assert completed[0]["signal"] == "SIGKILL"

    async def test_completed_latency_never_negative(
        self, supervisor: Supervisor, fsm: JobLifecycleFSM, bus: EventBus,
    ) -> None:
        # Invariant: even if the test runs on a clock-skewed CI box,
        # ``total_latency_ms`` is clamped at 0 by ``latency_ms_since``.
        from ryotenkai_shared.observability.cancellation_telemetry import (
            CANCELLATION_COMPLETED,
        )

        code = (
            "import signal, time\n"
            "signal.signal(signal.SIGTERM, lambda *_: __import__('sys').exit(0))\n"
            "time.sleep(60)\n"
        )
        await supervisor.submit_and_spawn("j-inv", _py(code))
        await _wait_for_state(fsm, JobState.RUNNING)
        await supervisor.request_stop(grace_seconds=2.0)
        await _wait_for_state(fsm, JobState.CANCELLED, timeout=3.0)

        completed = self._events_of(bus, CANCELLATION_COMPLETED)
        assert len(completed) == 1
        assert completed[0]["total_latency_ms"] >= 0


