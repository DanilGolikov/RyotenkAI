"""Shared fixtures for runner unit tests.

Two flavours:

- :data:`runner_client` — TestClient backed by a *mock* supervisor
  (no real subprocess fork). Use for API contract tests where we
  exercise wire shape and FSM transitions without paying for a
  Python interpreter spawn per test.
- :data:`runner_client_real` — TestClient backed by the real
  :class:`Supervisor`. Use for the supervisor's own integration
  tests in :mod:`test_supervisor` where we need actual fork +
  signal handling.

Each fixture rebuilds the FastAPI app via ``create_app()`` so test
isolation is automatic.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from fastapi.testclient import TestClient

from src.runner.main import create_app
from src.runner.state import (
    InvalidTransitionError,
    JobState,
)
from src.runner.supervisor import SupervisorBusy

if TYPE_CHECKING:
    from collections.abc import Iterator

    from src.runner.event_bus import EventBus
    from src.runner.state import JobLifecycleFSM


# ---------------------------------------------------------------------------
# MockSupervisor
# ---------------------------------------------------------------------------


class MockSupervisor:
    """Drop-in replacement for :class:`src.runner.supervisor.Supervisor`.

    Implements the same public surface (``submit_and_spawn``,
    ``request_stop``, ``shutdown``, ``is_running``, ``pgid``) but
    drives the FSM without spawning a subprocess. Tests can:

    - assert on FSM transitions (the "real" supervisor does the
      same ones — the FSM is never aware it's mocked);
    - call :meth:`finish` / :meth:`fail` to deterministically
      drive the trainer-exited transition without waiting on
      a real reap.

    Use the ``mock_supervisor`` fixture below (returns the instance
    bound to ``app.state.supervisor``) to interact with it from
    inside an API test.
    """

    def __init__(self, fsm: "JobLifecycleFSM", bus: "EventBus") -> None:
        self._fsm = fsm
        self._bus = bus
        self._running = False
        self.last_command: list[str] | None = None
        self.last_env: dict[str, str] | None = None
        self.stop_requests: int = 0

    # --- supervisor protocol ----------------------------------------------

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def pgid(self) -> int | None:
        return None

    async def submit_and_spawn(
        self,
        job_id: str,
        command: list[str],
        *,
        env: dict[str, str] | None = None,
        workdir: Path | None = None,
    ) -> None:
        if self._running:
            raise SupervisorBusy("a trainer subprocess is already running")
        try:
            self._fsm.submit(job_id)
        except InvalidTransitionError as exc:
            raise SupervisorBusy(f"job in non-terminal state: {exc}") from exc

        self._bus.publish("job_submitted", {"job_id": job_id, "sequence": 0})

        # No real subprocess; just transition to running.
        self._fsm.transition(JobState.RUNNING, message="trainer_spawned")
        self._bus.publish(
            "trainer_spawned",
            {"pid": -1, "pgid": -1, "command": list(command), "mock": True},
        )

        self.last_command = list(command)
        self.last_env = dict(env) if env is not None else None
        self._running = True

    async def request_stop(self, *, grace_seconds: float = 30.0) -> None:
        self.stop_requests += 1
        if not self._running:
            return
        try:
            self._fsm.transition(JobState.STOPPING, message="stop_requested")
        except InvalidTransitionError:
            return
        self._bus.publish("stop_requested", {"grace_seconds": grace_seconds})

    async def shutdown(self) -> None:
        if self._running:
            await self.request_stop(grace_seconds=0.0)
            self.finish(exit_code=130, cancelled=True)

    # --- test driver ------------------------------------------------------

    def finish(self, *, exit_code: int = 0, cancelled: bool = False) -> None:
        """Deterministically drive the trainer-exited transition.

        Mirrors the production logic in :meth:`Supervisor._reap`:
        - exit_code == 0 → ``completed`` (or ``cancelled`` if
          ``cancelled=True``)
        - exit_code != 0 + cancelled → ``cancelled``
        - exit_code != 0 + not cancelled → ``failed``
        """
        if not self._running:
            raise RuntimeError("MockSupervisor.finish() called while not running")

        if exit_code == 0:
            target = JobState.CANCELLED if cancelled else JobState.COMPLETED
        elif cancelled:
            target = JobState.CANCELLED
        else:
            target = JobState.FAILED

        self._bus.publish(
            "trainer_exited",
            {"exit_code": exit_code, "cancellation_requested": cancelled},
        )
        try:
            self._fsm.transition(target, message=f"exit_code={exit_code}")
        except InvalidTransitionError:
            pass
        self._running = False

    def fail(self, *, exit_code: int = 1) -> None:
        """Convenience: drive directly to FAILED."""
        self.finish(exit_code=exit_code, cancelled=False)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def runner_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> "Iterator[TestClient]":
    """TestClient with a :class:`MockSupervisor` bound to ``app.state``.

    Construction injects ``MockSupervisor`` as the supervisor factory
    on :func:`create_app`, so the real :class:`Supervisor` never even
    instantiates. Tests that need real subprocess semantics use
    :func:`runner_client_real`.
    """
    monkeypatch.setenv("RYOTENKAI_WORKSPACE", str(tmp_path))
    app = create_app(supervisor_factory=MockSupervisor)
    with TestClient(app) as client:
        yield client


@pytest.fixture
def runner_client_real(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> "Iterator[TestClient]":
    """TestClient with the real :class:`Supervisor` bound to ``app.state``.

    Use only for tests in :mod:`test_supervisor` that need actual
    process semantics (signals, exit codes, pump timing).
    """
    monkeypatch.setenv("RYOTENKAI_WORKSPACE", str(tmp_path))
    with TestClient(create_app()) as client:
        yield client
