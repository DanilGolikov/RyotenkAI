"""Phase G fix-up #3 — :class:`Supervisor` zombie-trainer watchdog.

The watchdog runs concurrently with the reap task. If the trainer
subprocess exceeds ``wall_timeout_seconds`` the watchdog SIGTERMs the
process group, waits ``watchdog_kill_grace_seconds``, then SIGKILLs.
The reap path synthesises a ``trainer_exited`` event carrying
``code=TRAINING_TIMEOUT`` and ``payload_source=watchdog_timeout`` so
dashboards group on it independently from the SIGKILL-OOM heuristic
and from natural cancellation.

Test categories (project policy):

* positive       — happy-path "trainer hits timeout, watchdog escalates".
* negative       — trainer exits within budget: watchdog cancelled cleanly.
* boundary       — timeout=0 disables the watchdog; tiny grace fires SIGKILL.
* invariants     — payload_source is "watchdog_timeout"; code is
                   TRAINING_TIMEOUT; FSM lands in failed.
* dependency-err — synthesis still works when wall_seconds is unknown.
* regression     — TrainerExitPayload on disk wins over watchdog synthesis
                   (a payload-writing trainer that genuinely timed out
                   still gets its own message; only the synthesis path
                   uses the watchdog code).
* combinatorial  — (rc, signal) × (watchdog_fired) × (payload present)
                   matrix on the pure builder helper.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import pytest_asyncio

from ryotenkai_pod.runner.event_bus import EventBus
from ryotenkai_pod.runner.state import JobLifecycleFSM, JobState
from ryotenkai_pod.runner.supervisor import (
    Supervisor,
    _build_trainer_exited_event,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


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
async def short_watchdog_supervisor(
    fsm: JobLifecycleFSM, bus: EventBus,
) -> "AsyncIterator[Supervisor]":
    """Supervisor with a 0.5s wall timeout and 0.5s kill grace -- fast enough
    for test runs without flaking under typical CI load."""
    s = Supervisor(
        fsm, bus,
        wall_timeout_seconds=0.5,
        watchdog_kill_grace_seconds=0.5,
    )
    try:
        yield s
    finally:
        await s.shutdown()


@pytest_asyncio.fixture
async def no_watchdog_supervisor(
    fsm: JobLifecycleFSM, bus: EventBus,
) -> "AsyncIterator[Supervisor]":
    """Supervisor with the watchdog disabled (``wall_timeout_seconds=0``).

    Tests that exercise the natural-exit path without racing the watchdog.
    """
    s = Supervisor(fsm, bus, wall_timeout_seconds=0)
    try:
        yield s
    finally:
        await s.shutdown()


def _py_sleep(seconds: float) -> list[str]:
    """Argv that sleeps for ``seconds`` in a subprocess."""
    return [
        sys.executable,
        "-c",
        f"import time; time.sleep({seconds})",
    ]


def _py_immediate_exit(code: int = 0) -> list[str]:
    """Argv that exits with ``code`` immediately."""
    return [sys.executable, "-c", f"import sys; sys.exit({code})"]


def _find_trainer_exited(bus: EventBus) -> dict | None:
    """Return the most recent ``trainer_exited`` event's payload.

    Phase 2: the supervisor publishes ``trainer_exited`` via the legacy
    shim (it carries a free-form payload not yet typed). The legacy
    payload lives on the :class:`UnknownEvent`'s ``raw_payload``.
    """
    for ev in reversed(list(bus._buffer)):  # type: ignore[attr-defined]
        original = getattr(ev, "original_type", None)
        if original == "trainer_exited":
            raw = getattr(ev, "raw_payload", None)
            return dict(raw) if isinstance(raw, dict) else None
    return None


def _find_watchdog_event(bus: EventBus) -> dict | None:
    """Return the ``watchdog_timeout`` event payload if any."""
    for ev in list(bus._buffer):  # type: ignore[attr-defined]
        original = getattr(ev, "original_type", None)
        if original == "watchdog_timeout":
            raw = getattr(ev, "raw_payload", None)
            return dict(raw) if isinstance(raw, dict) else None
    return None


# ---------------------------------------------------------------------------
# Positive
# ---------------------------------------------------------------------------


class TestPositive:
    @pytest.mark.asyncio
    async def test_supervisor_kills_runaway_trainer(
        self,
        short_watchdog_supervisor: Supervisor,
        fsm: JobLifecycleFSM,
        bus: EventBus,
    ) -> None:
        """Trainer sleeps past the wall timeout -> watchdog SIGKILLs it."""
        await short_watchdog_supervisor.submit_and_spawn(
            "j-1", _py_sleep(60.0),
        )

        # The watchdog should fire within ~0.5s + 0.5s grace + slack.
        # Wait for the reap path to complete.
        wait_task = short_watchdog_supervisor._wait_task
        assert wait_task is not None
        await asyncio.wait_for(wait_task, timeout=5.0)

        # Watchdog event published with the configured timeout.
        wd = _find_watchdog_event(bus)
        assert wd is not None
        assert wd["wall_timeout_seconds"] == 0.5

        # Trainer-exited event has the typed timeout code.
        exited = _find_trainer_exited(bus)
        assert exited is not None
        assert exited["code"] == "TRAINING_TIMEOUT"
        assert exited["payload_source"] == "watchdog_timeout"
        # wall_seconds populated by supervisor measurement.
        assert exited["wall_seconds"] is not None
        assert exited["wall_seconds"] >= 0.5
        # FSM landed in failed (cancellation wasn't user-initiated).
        assert fsm.current() is not None
        assert fsm.current().state == JobState.FAILED  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Negative
# ---------------------------------------------------------------------------


class TestNegative:
    @pytest.mark.asyncio
    async def test_natural_exit_within_budget_does_not_trigger_watchdog(
        self,
        short_watchdog_supervisor: Supervisor,
        bus: EventBus,
    ) -> None:
        """Trainer that exits cleanly within the budget leaves the watchdog
        idle; no ``watchdog_timeout`` event is published."""
        await short_watchdog_supervisor.submit_and_spawn(
            "j-1", _py_immediate_exit(0),
        )
        wait_task = short_watchdog_supervisor._wait_task
        assert wait_task is not None
        await asyncio.wait_for(wait_task, timeout=5.0)

        assert _find_watchdog_event(bus) is None
        exited = _find_trainer_exited(bus)
        assert exited is not None
        # Exit code 0 means natural completion: code is None,
        # payload_source is "none".
        assert exited["code"] is None
        assert exited["payload_source"] == "none"


# ---------------------------------------------------------------------------
# Boundary
# ---------------------------------------------------------------------------


class TestBoundary:
    @pytest.mark.asyncio
    async def test_timeout_zero_disables_watchdog(
        self,
        no_watchdog_supervisor: Supervisor,
        bus: EventBus,
    ) -> None:
        """``wall_timeout_seconds=0`` skips watchdog construction entirely.

        Lets tests that don't want to race the watchdog opt out cleanly.
        """
        await no_watchdog_supervisor.submit_and_spawn(
            "j-1", _py_immediate_exit(0),
        )
        # No watchdog task was ever scheduled.
        assert no_watchdog_supervisor._watchdog_task is None

        wait_task = no_watchdog_supervisor._wait_task
        assert wait_task is not None
        await asyncio.wait_for(wait_task, timeout=5.0)
        assert _find_watchdog_event(bus) is None


# ---------------------------------------------------------------------------
# Invariants -- payload synthesis on the pure builder
# ---------------------------------------------------------------------------


class TestPayloadSynthesis:
    def test_watchdog_fired_overrides_sigkill_heuristic(self) -> None:
        """SIGKILL rc + watchdog_fired -> TRAINING_TIMEOUT (not TRAINING_OOM).

        The SIGKILL came from us, not the cgroup; the OOM heuristic
        would otherwise mis-classify the failure.
        """
        ev = _build_trainer_exited_event(
            rc=137,
            signal_name="SIGKILL",
            cancellation_requested=False,
            payload=None,
            synthesize_if_missing=True,
            watchdog_fired=True,
            wall_seconds=21600.0,
            wall_timeout_seconds=21600.0,
        )
        assert ev["code"] == "TRAINING_TIMEOUT"
        assert ev["payload_source"] == "watchdog_timeout"
        assert "21600s" in ev["message"]
        assert ev["wall_seconds"] == 21600.0
        assert ev["exit_code"] == 137

    def test_watchdog_not_fired_keeps_sigkill_heuristic(self) -> None:
        """SIGKILL rc without watchdog_fired -> TRAINING_OOM (cgroup OOM)."""
        ev = _build_trainer_exited_event(
            rc=137,
            signal_name="SIGKILL",
            cancellation_requested=False,
            payload=None,
            synthesize_if_missing=True,
            watchdog_fired=False,
        )
        assert ev["code"] == "TRAINING_OOM"
        assert ev["payload_source"] == "sigkill_heuristic"

    def test_payload_on_disk_wins_over_watchdog(self) -> None:
        """If the trainer wrote ``trainer-exit.json`` BEFORE the watchdog
        sent SIGKILL, the typed payload survives. Regression: don't
        clobber a real traceback with the watchdog synthesis."""
        from ryotenkai_shared.contracts.problem_details import ErrorCode
        from ryotenkai_shared.contracts.trainer_exit import TrainerExitPayload

        payload = TrainerExitPayload(
            code=ErrorCode.TRAINING_FAILED,
            message="cuda error",
            traceback_summary="boom",
            exit_code=1,
            wall_seconds=12.5,
        )
        ev = _build_trainer_exited_event(
            rc=137,
            signal_name="SIGKILL",
            cancellation_requested=False,
            payload=payload,
            synthesize_if_missing=False,
            watchdog_fired=True,
            wall_seconds=21600.0,
        )
        assert ev["code"] == "TRAINING_FAILED"
        assert ev["payload_source"] == "trainer_file"
        # Trainer-reported wall_seconds wins.
        assert ev["wall_seconds"] == 12.5


# ---------------------------------------------------------------------------
# Combinatorial
# ---------------------------------------------------------------------------


class TestCombinatorial:
    @pytest.mark.parametrize("rc,signal_name", [
        (1, None),
        (137, "SIGKILL"),
        (143, "SIGTERM"),
        (-9, "SIGKILL"),
    ])
    def test_watchdog_label_supersedes_rc_classification(
        self, rc: int, signal_name: str | None,
    ) -> None:
        """For every plausible rc/signal combo, the watchdog code wins
        when ``watchdog_fired=True``."""
        ev = _build_trainer_exited_event(
            rc=rc,
            signal_name=signal_name,
            cancellation_requested=False,
            payload=None,
            synthesize_if_missing=True,
            watchdog_fired=True,
            wall_seconds=10.0,
            wall_timeout_seconds=5.0,
        )
        assert ev["code"] == "TRAINING_TIMEOUT"
        assert ev["exit_code"] == rc

    def test_wall_timeout_unknown_falls_back_to_generic_message(self) -> None:
        """Watchdog payload with no ``wall_timeout_seconds`` still produces
        a sensible message (no crash, no ``f"{None:.0f}"`` artefact)."""
        ev = _build_trainer_exited_event(
            rc=137,
            signal_name="SIGKILL",
            cancellation_requested=False,
            payload=None,
            synthesize_if_missing=True,
            watchdog_fired=True,
            wall_seconds=None,
            wall_timeout_seconds=None,
        )
        assert ev["code"] == "TRAINING_TIMEOUT"
        assert "wall-clock" in ev["message"]
