"""Phase 4 — Provider-agnostic ``ITrainerSpawner`` Protocol.

Extracted additively in 2026-05-11 as part of the Phase 4 testing-architecture
overhaul. Rationale: :class:`ryotenkai_pod.runner.supervisor.Supervisor`
is the sole production class that spawns / signals / reaps the trainer
subprocess. The Protocol abstracts that surface so component tests can
exercise lifecycle wiring (runner → trainer events → FSM transitions)
through a deterministic in-memory fake without forking real subprocesses.

This module is **definition-only** — no production class implements
``ITrainerSpawner`` yet. The compliance test parametrizes over
``[fake, real]`` but ``real`` is currently ``pytest.skip``-ed.

Scope of the Protocol:

The concrete :class:`Supervisor` exposes a sprawling surface tightly
coupled to ``asyncio.subprocess`` / signal handling / FSM transitions.
The Protocol intentionally narrows that to the in-process control plane
needed by Mac-side orchestrators and tests:

* ``spawn(spec) -> TrainerHandle`` — start a trainer; return a handle
  whose ``trainer_id`` is opaque to the caller.
* ``send_signal(trainer_id, sig)`` — request termination / stop /
  pause via SIGTERM / SIGINT / SIGSTOP semantics.
* ``wait(trainer_id, timeout=None)`` — await terminal state; returns
  the :class:`TrainerStatus` snapshot at terminal time.
* ``read_events(trainer_id, since=0)`` — drain pending control-plane
  events since the given offset.
* ``status(trainer_id) -> TrainerStatus`` — current snapshot.

The Protocol is **async** even when the production implementation is
sync; ``Supervisor`` is already async and the fake naturally is too.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable


class TrainerSpawnError(RuntimeError):
    """Base error from :class:`ITrainerSpawner` operations."""


@dataclass(frozen=True)
class TrainerSpec:
    """Frozen description of one trainer subprocess invocation.

    Mirrors the arguments :meth:`Supervisor.submit_and_spawn` takes,
    narrowed to fields the Mac side genuinely controls. Resource
    requirements (GPU count, memory etc.) are out-of-scope — the
    Protocol only cares about subprocess lifecycle.
    """

    job_id: str
    command: tuple[str, ...]
    env: tuple[tuple[str, str], ...] = ()
    workdir: str | None = None


class TrainerStatus(StrEnum):
    """Coarse lifecycle states observable through the Protocol."""

    PREPARING = "preparing"
    RUNNING = "running"
    STOPPING = "stopping"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def is_terminal(self) -> bool:
        return self in {TrainerStatus.COMPLETED, TrainerStatus.FAILED, TrainerStatus.CANCELLED}


@dataclass(frozen=True)
class TrainerHandle:
    """Handle returned by :meth:`ITrainerSpawner.spawn`."""

    trainer_id: str
    pid: int | None = None
    pgid: int | None = None


@dataclass(frozen=True)
class TrainerEvent:
    """Control-plane event emitted by a trainer (fake or real).

    Mirrors :class:`ryotenkai_pod.runner.event_bus.Event`'s public shape
    but lives in shared/ so callers don't need pod-side imports.

    ``offset`` is a monotonic per-trainer sequence (0 is the first
    event), so consumers can drive resumption with
    ``read_events(since=last_offset + 1)``.
    """

    offset: int
    kind: str
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0


@runtime_checkable
class ITrainerSpawner(Protocol):
    """Async lifecycle surface for trainer subprocesses (definition-only).

    Implementations:

    * (Phase 4+ TODO) wrapper around
      :class:`ryotenkai_pod.runner.supervisor.Supervisor` that adapts
      its FSM + EventBus surface into the Protocol shape.
    * :class:`tests._fakes.trainer.FakeTrainerSpawner` — in-memory
      deterministic stand-in for component / property / chaos tests.
    """

    async def spawn(self, spec: TrainerSpec) -> TrainerHandle:
        """Spawn the trainer subprocess described by ``spec``.

        Raises:
            TrainerSpawnError: when a trainer is already running, or
                the launch itself failed.
        """
        ...

    async def send_signal(self, trainer_id: str, sig: int) -> None:
        """Forward ``sig`` to the trainer process group.

        Tolerates "process is already gone" via best-effort semantics.
        Raises :class:`TrainerSpawnError` for unknown ``trainer_id``.
        """
        ...

    async def wait(self, trainer_id: str, *, timeout: float | None = None) -> TrainerStatus:
        """Await terminal state; return the final :class:`TrainerStatus`.

        Raises:
            TrainerSpawnError: when ``trainer_id`` is unknown.
            TimeoutError: when ``timeout`` elapses before terminal.
        """
        ...

    async def read_events(
        self, trainer_id: str, *, since: int = 0,
    ) -> list[TrainerEvent]:
        """Return every event with ``offset >= since`` in monotonic order."""
        ...

    async def status(self, trainer_id: str) -> TrainerStatus:
        """Return the current :class:`TrainerStatus`."""
        ...


__all__ = [
    "ITrainerSpawner",
    "TrainerEvent",
    "TrainerHandle",
    "TrainerSpawnError",
    "TrainerSpec",
    "TrainerStatus",
]
