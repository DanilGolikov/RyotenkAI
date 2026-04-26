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

from src.runner.event_bus import EventBus
from src.runner.state import (
    JobLifecycleFSM,
    JobState,
)
from src.runner.supervisor import Supervisor, SupervisorBusy

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


@pytest_asyncio.fixture
async def supervisor(
    fsm: JobLifecycleFSM, bus: EventBus,
) -> "AsyncIterator[Supervisor]":
    """Real :class:`Supervisor` with cleanup on test exit.

    ``shutdown()`` is idempotent and safe even if the test never
    spawned a subprocess, so we always call it.
    """
    s = Supervisor(fsm, bus)
    try:
        yield s
    finally:
        await s.shutdown()


def _py(code: str) -> list[str]:
    """Build an argv that runs ``code`` in the test's Python."""
    return [sys.executable, "-c", code]


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

    async def test_stdout_lines_emitted_as_events(
        self, supervisor: Supervisor, bus: EventBus, fsm: JobLifecycleFSM,
    ) -> None:
        await supervisor.submit_and_spawn(
            "j-1", _py("print('hello'); print('world')"),
        )
        await _wait_for_state(fsm, JobState.COMPLETED)
        log_events = [
            e for e in list(bus._buffer)
            if e.kind == "trainer_log" and e.payload.get("kind") == "stdout"
        ]
        lines = [e.payload["line"] for e in log_events]
        assert "hello" in lines
        assert "world" in lines


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
        # is installed by the time this event arrives.
        async def _wait_ready() -> None:
            deadline = asyncio.get_event_loop().time() + 5.0
            while asyncio.get_event_loop().time() < deadline:
                for ev in list(bus._buffer):
                    if (
                        ev.kind == "trainer_log"
                        and ev.payload.get("line") == "READY"
                    ):
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


