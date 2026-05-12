"""``FakeTrainerSpawner`` — canonical fake for :class:`ITrainerSpawner`.

In-memory state machine modelling the trainer subprocess lifecycle the
:class:`ryotenkai_pod.runner.supervisor.Supervisor` exposes. No real
subprocess — events are emitted on a deterministic schedule driven by
the injected :class:`Clock`.

State machine::

    PREPARING -> RUNNING -> {COMPLETED | FAILED | CANCELLED}
                     \
                      -> STOPPING -> {COMPLETED | CANCELLED | FAILED}

Chaos surface:

* :meth:`inject_oom_next_spawn` — the next spawn lands in FAILED with
  ``exit_code=137``
* :meth:`inject_callback_failure` — the next event emit raises
* :meth:`inject_slow_start_ms` — adds a fake sleep before RUNNING
* :meth:`inject_signal_ignored` — next ``send_signal`` is a no-op
* :meth:`reset_chaos`
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any

from ryotenkai_shared.infrastructure.trainer_spawner import (
    ITrainerSpawner,
    TrainerEvent,
    TrainerHandle,
    TrainerSpawnError,
    TrainerSpec,
    TrainerStatus,
)
from tests._harness.clock import Clock, RealClock


@dataclass
class _ChaosState:
    oom_next_spawn: bool = False
    callback_failures_remaining: int = 0
    slow_start_seconds: float = 0.0
    signal_ignored: bool = False


@dataclass
class _TrainerEntry:
    trainer_id: str
    spec: TrainerSpec
    status: TrainerStatus
    pid: int
    pgid: int
    exit_code: int | None = None
    events: list[TrainerEvent] = field(default_factory=list)
    event_counter: int = 0


class FakeTrainerSpawner:
    """Deterministic in-memory fake for :class:`ITrainerSpawner`."""

    def __init__(self, *, clock: Clock | None = None) -> None:
        self._clock: Clock = clock if clock is not None else RealClock()
        self._trainers: dict[str, _TrainerEntry] = {}
        self._chaos = _ChaosState()
        self._id_counter = itertools.count(start=1)
        self._pid_counter = itertools.count(start=10000)
        # ``Supervisor`` enforces single-active — mirror that so tests
        # observe the same shape.
        self._active_id: str | None = None

    # ------------------------------------------------------------------
    # Chaos surface
    # ------------------------------------------------------------------

    def inject_oom_next_spawn(self) -> None:
        self._chaos.oom_next_spawn = True

    def inject_callback_failure(self, count: int = 1) -> None:
        if count < 0:
            raise ValueError("count must be non-negative")
        self._chaos.callback_failures_remaining = count

    def inject_slow_start_ms(self, ms: int) -> None:
        if ms < 0:
            raise ValueError("ms must be non-negative")
        self._chaos.slow_start_seconds = ms / 1000.0

    def inject_signal_ignored(self) -> None:
        self._chaos.signal_ignored = True

    def reset_chaos(self) -> None:
        self._chaos = _ChaosState()

    # ------------------------------------------------------------------
    # Inspection helpers (test-only, not part of ITrainerSpawner)
    # ------------------------------------------------------------------

    def get_entry(self, trainer_id: str) -> _TrainerEntry:
        return self._trainers[trainer_id]

    def list_trainers(self) -> list[str]:
        return list(self._trainers.keys())

    def emit_event(
        self,
        trainer_id: str,
        kind: str,
        payload: dict[str, Any] | None = None,
    ) -> TrainerEvent:
        entry = self._trainers[trainer_id]
        if self._chaos.callback_failures_remaining > 0:
            self._chaos.callback_failures_remaining -= 1
            raise TrainerSpawnError("fake injected callback failure")
        event = TrainerEvent(
            offset=entry.event_counter,
            kind=kind,
            payload=dict(payload or {}),
            timestamp=self._clock.now(),
        )
        entry.events.append(event)
        entry.event_counter += 1
        return event

    def force_terminal(
        self,
        trainer_id: str,
        target: TrainerStatus,
        *,
        exit_code: int = 0,
    ) -> None:
        """Drive the trainer to a terminal state directly (test helper)."""
        if not target.is_terminal:
            raise ValueError(f"target {target!r} is not terminal")
        entry = self._trainers[trainer_id]
        entry.status = target
        entry.exit_code = exit_code
        # Emit a synthetic exit event; mirrors Supervisor's
        # ``trainer_exited`` so consumers can spot terminal transitions.
        self.emit_event(
            trainer_id,
            "trainer_exited",
            {"exit_code": exit_code, "terminal_state": target.value},
        )
        if self._active_id == trainer_id:
            self._active_id = None

    # ------------------------------------------------------------------
    # Snapshot
    # ------------------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        return {
            "active_id": self._active_id,
            "trainers": {
                tid: {
                    "spec": {
                        "job_id": entry.spec.job_id,
                        "command": list(entry.spec.command),
                        "env": [list(kv) for kv in entry.spec.env],
                        "workdir": entry.spec.workdir,
                    },
                    "status": entry.status.value,
                    "pid": entry.pid,
                    "pgid": entry.pgid,
                    "exit_code": entry.exit_code,
                    "events": [
                        {
                            "offset": e.offset,
                            "kind": e.kind,
                            "payload": dict(e.payload),
                            "timestamp": e.timestamp,
                        }
                        for e in entry.events
                    ],
                }
                for tid, entry in self._trainers.items()
            },
            "chaos": {
                "oom_next_spawn": self._chaos.oom_next_spawn,
                "callback_failures_remaining": self._chaos.callback_failures_remaining,
                "slow_start_seconds": self._chaos.slow_start_seconds,
                "signal_ignored": self._chaos.signal_ignored,
            },
        }

    # ------------------------------------------------------------------
    # ITrainerSpawner surface
    # ------------------------------------------------------------------

    async def spawn(self, spec: TrainerSpec) -> TrainerHandle:
        if self._active_id is not None:
            raise TrainerSpawnError(
                f"a trainer is already active: {self._active_id}",
            )

        trainer_id = f"t-{next(self._id_counter):04d}"
        pid = next(self._pid_counter)
        pgid = pid
        entry = _TrainerEntry(
            trainer_id=trainer_id,
            spec=spec,
            status=TrainerStatus.PREPARING,
            pid=pid,
            pgid=pgid,
        )
        self._trainers[trainer_id] = entry
        self._active_id = trainer_id

        # Emit the prepare/spawn events that real Supervisor publishes.
        self.emit_event(trainer_id, "job_submitted", {"job_id": spec.job_id})

        # Optional slow-start latency.
        if self._chaos.slow_start_seconds > 0:
            await self._clock.sleep(self._chaos.slow_start_seconds)

        if self._chaos.oom_next_spawn:
            self._chaos.oom_next_spawn = False
            entry.status = TrainerStatus.FAILED
            entry.exit_code = 137
            self.emit_event(
                trainer_id,
                "trainer_exited",
                {"exit_code": 137, "signal": "SIGKILL", "terminal_state": "failed"},
            )
            self._active_id = None
            return TrainerHandle(trainer_id=trainer_id, pid=pid, pgid=pgid)

        entry.status = TrainerStatus.RUNNING
        self.emit_event(
            trainer_id,
            "trainer_spawned",
            {"pid": pid, "pgid": pgid, "command": list(spec.command)},
        )
        return TrainerHandle(trainer_id=trainer_id, pid=pid, pgid=pgid)

    async def send_signal(self, trainer_id: str, sig: int) -> None:
        if trainer_id not in self._trainers:
            raise TrainerSpawnError(f"unknown trainer_id: {trainer_id!r}")
        if self._chaos.signal_ignored:
            self._chaos.signal_ignored = False
            return
        entry = self._trainers[trainer_id]
        if entry.status.is_terminal:
            return  # SIGCHLD on a reaped process — best-effort no-op
        # SIGTERM (15) → STOPPING; SIGKILL (9) → CANCELLED immediately.
        if sig == 9:
            entry.status = TrainerStatus.CANCELLED
            entry.exit_code = 137
            self.emit_event(
                trainer_id,
                "trainer_exited",
                {"exit_code": 137, "signal": "SIGKILL", "terminal_state": "cancelled"},
            )
            if self._active_id == trainer_id:
                self._active_id = None
            return
        # SIGTERM / other — transition to STOPPING; tests drive the
        # eventual reap via ``force_terminal``.
        if entry.status == TrainerStatus.RUNNING:
            entry.status = TrainerStatus.STOPPING
            self.emit_event(trainer_id, "stop_requested", {"signal": sig})

    async def wait(
        self, trainer_id: str, *, timeout: float | None = None,
    ) -> TrainerStatus:
        if trainer_id not in self._trainers:
            raise TrainerSpawnError(f"unknown trainer_id: {trainer_id!r}")
        # In a deterministic fake we don't actually block — terminal
        # transitions are explicit via ``force_terminal``. If the caller
        # passed a timeout and we're not terminal yet, raise TimeoutError
        # to mirror real semantics.
        entry = self._trainers[trainer_id]
        if entry.status.is_terminal:
            return entry.status
        if timeout is None:
            # No deadline + non-terminal → simulate "still running".
            # The fake's contract is "tests drive transitions"; calling
            # wait() without a timeout from a still-running trainer is
            # a test bug — surface it loudly.
            raise TrainerSpawnError(
                f"trainer {trainer_id!r} is not terminal and no timeout given",
            )
        await self._clock.sleep(timeout)
        # After sleeping the entry might have transitioned (other task
        # called ``force_terminal``); re-check.
        if entry.status.is_terminal:
            return entry.status
        raise TimeoutError(f"trainer {trainer_id!r} did not terminate in {timeout}s")

    async def read_events(
        self, trainer_id: str, *, since: int = 0,
    ) -> list[TrainerEvent]:
        if trainer_id not in self._trainers:
            raise TrainerSpawnError(f"unknown trainer_id: {trainer_id!r}")
        entry = self._trainers[trainer_id]
        return [e for e in entry.events if e.offset >= since]

    async def status(self, trainer_id: str) -> TrainerStatus:
        if trainer_id not in self._trainers:
            raise TrainerSpawnError(f"unknown trainer_id: {trainer_id!r}")
        return self._trainers[trainer_id].status


# Static guarantee — fail fast at module import if the fake drifts from
# the Protocol shape.
_runtime_check: ITrainerSpawner = FakeTrainerSpawner()


__all__ = [
    "FakeTrainerSpawner",
]
