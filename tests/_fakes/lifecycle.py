"""``FakePodLifecycleClient`` — canonical fake for :class:`IPodLifecycleClient`.

State machine (matches the operator-facing dashboard vocabulary in
:class:`PodTerminalOutcome`):

::

    PROVISIONING -> RUNNING <-> STOPPING -> STOPPED -> TERMINATED
                                              ^---------'
                                       (resume returns RUNNING)

* ``terminate`` is idempotent: STOPPED, RUNNING → TERMINATED;
  TERMINATED → ``already_terminated``.
* ``pause`` is idempotent: RUNNING → STOPPED;
  STOPPED → ``already_stopped``;
  TERMINATED → ``skipped`` (terminal state can't be stopped — matches
  the spec invariant from the user prompt).
* ``resume`` is idempotent: STOPPED → RUNNING (``resumed``);
  RUNNING → ``already_running``;
  TERMINATED → ``failed`` with the failure reason.

Chaos surface:

* :meth:`set_pod_state` — direct state override
* :meth:`inject_failure_on` — next N calls of an action return ``failed``
* :meth:`make_eventually_consistent` — schedules transition rather than
  applying it synchronously, exposing eventual-consistency bugs
* :meth:`reset_chaos` — back to clean state
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from ryotenkai_shared.infrastructure.lifecycle import (
    LifecycleActionResult,
    PodTerminalOutcome,
)
from tests._harness.clock import Clock, RealClock


class PodState(StrEnum):
    PROVISIONING = "PROVISIONING"
    RUNNING = "RUNNING"
    STOPPING = "STOPPING"
    STOPPED = "STOPPED"
    TERMINATED = "TERMINATED"


_RESUME_OK = "resumed"
_RESUME_ALREADY_RUNNING = "already_running"


@dataclass
class _PendingTransition:
    target: PodState
    fire_at: float
    action: str


@dataclass
class _PodEntry:
    pod_id: str
    state: PodState
    pending: _PendingTransition | None = None
    failure_overrides: dict[str, int] = field(default_factory=dict)


class FakePodLifecycleClient:
    """In-memory deterministic fake for :class:`IPodLifecycleClient`."""

    def __init__(
        self,
        *,
        provider_name: str = "fake",
        clock: Clock | None = None,
    ) -> None:
        self._provider_name = provider_name
        self._clock: Clock = clock if clock is not None else RealClock()
        self._pods: dict[str, _PodEntry] = {}
        self._eventually_consistent: dict[str, float] = {}
        self._call_history: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Pod registry
    # ------------------------------------------------------------------

    def register_pod(self, pod_id: str, *, state: PodState = PodState.RUNNING) -> None:
        self._pods[pod_id] = _PodEntry(pod_id=pod_id, state=state)

    def set_pod_state(self, pod_id: str, state: PodState | str) -> None:
        if isinstance(state, str):
            state = PodState(state)
        if pod_id not in self._pods:
            self._pods[pod_id] = _PodEntry(pod_id=pod_id, state=state)
        else:
            self._pods[pod_id].state = state
            self._pods[pod_id].pending = None

    def get_pod_state(self, pod_id: str) -> PodState | None:
        entry = self._pods.get(pod_id)
        if entry is None:
            return None
        # WHY reify on read: tests want to observe eventual consistency by
        # advancing the clock and checking state. Without reification the
        # state stays stale until the next ``_dispatch`` call.
        self._reify_pending_if_due(entry)
        return entry.state

    # ------------------------------------------------------------------
    # Chaos surface
    # ------------------------------------------------------------------

    def inject_failure_on(self, pod_id: str, action: str, count: int = 1) -> None:
        if action not in {"terminate", "stop", "resume"}:
            raise ValueError(f"action must be terminate|stop|resume, got {action!r}")
        if count < 0:
            raise ValueError("count must be non-negative")
        entry = self._pods.setdefault(pod_id, _PodEntry(pod_id=pod_id, state=PodState.RUNNING))
        entry.failure_overrides[action] = count

    def make_eventually_consistent(self, pod_id: str, transition_delay_seconds: float = 1.0) -> None:
        if transition_delay_seconds < 0:
            raise ValueError("transition_delay_seconds must be non-negative")
        self._eventually_consistent[pod_id] = transition_delay_seconds

    def reset_chaos(self) -> None:
        for entry in self._pods.values():
            entry.failure_overrides.clear()
            entry.pending = None
        self._eventually_consistent.clear()

    # ------------------------------------------------------------------
    # Snapshot
    # ------------------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        return {
            "provider_name": self._provider_name,
            "pods": {
                pod_id: {
                    "state": entry.state.value,
                    "pending": (
                        {
                            "target": entry.pending.target.value,
                            "fire_at": entry.pending.fire_at,
                            "action": entry.pending.action,
                        }
                        if entry.pending is not None
                        else None
                    ),
                    "failure_overrides": dict(entry.failure_overrides),
                }
                for pod_id, entry in self._pods.items()
            },
            "eventually_consistent": dict(self._eventually_consistent),
            "call_history": list(self._call_history),
        }

    # ------------------------------------------------------------------
    # IPodLifecycleClient surface
    # ------------------------------------------------------------------

    @property
    def provider_name(self) -> str:
        return self._provider_name

    async def terminate(self, *, resource_id: str) -> LifecycleActionResult:
        return await self._dispatch(
            resource_id=resource_id,
            action="terminate",
            target_state=PodState.TERMINATED,
            success_outcome=PodTerminalOutcome.TERMINATED,
            already_outcome=PodTerminalOutcome.ALREADY_TERMINATED,
            already_states={PodState.TERMINATED},
        )

    async def pause(self, *, resource_id: str) -> LifecycleActionResult:
        return await self._dispatch(
            resource_id=resource_id,
            action="stop",
            target_state=PodState.STOPPED,
            success_outcome=PodTerminalOutcome.STOPPED,
            already_outcome=PodTerminalOutcome.ALREADY_STOPPED,
            already_states={PodState.STOPPED},
        )

    async def resume(self, *, resource_id: str) -> LifecycleActionResult:
        return await self._dispatch(
            resource_id=resource_id,
            action="resume",
            target_state=PodState.RUNNING,
            success_outcome=_RESUME_OK,
            already_outcome=_RESUME_ALREADY_RUNNING,
            already_states={PodState.RUNNING},
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _dispatch(
        self,
        *,
        resource_id: str,
        action: str,
        target_state: PodState,
        success_outcome: str,
        already_outcome: str,
        already_states: set[PodState],
    ) -> LifecycleActionResult:
        entry = self._pods.setdefault(resource_id, _PodEntry(pod_id=resource_id, state=PodState.RUNNING))
        self._reify_pending_if_due(entry)

        # Failure injection always wins over state checks so chaos tests
        # observe a failed call regardless of the pod's actual state.
        if entry.failure_overrides.get(action, 0) > 0:
            entry.failure_overrides[action] -= 1
            self._call_history.append({"action": action, "pod_id": resource_id, "outcome": "failed"})
            return LifecycleActionResult(
                outcome=PodTerminalOutcome.FAILED,
                attempts_made=1,
                last_error=f"fake_injected_failure for {action}",
            )

        # ``stop`` on a TERMINATED pod is the spec's "skipped" — terminal
        # state can't be stopped. Resume on TERMINATED is failed because
        # the resource no longer exists.
        if action == "stop" and entry.state == PodState.TERMINATED:
            self._call_history.append({"action": action, "pod_id": resource_id, "outcome": "skipped"})
            return LifecycleActionResult(outcome=PodTerminalOutcome.SKIPPED, attempts_made=1)

        if action == "resume" and entry.state == PodState.TERMINATED:
            self._call_history.append({"action": action, "pod_id": resource_id, "outcome": "failed"})
            return LifecycleActionResult(
                outcome=PodTerminalOutcome.FAILED,
                attempts_made=1,
                last_error="cannot resume a terminated pod",
            )

        if entry.state in already_states:
            self._call_history.append({"action": action, "pod_id": resource_id, "outcome": already_outcome})
            return LifecycleActionResult(outcome=already_outcome, attempts_made=1)

        delay = self._eventually_consistent.get(resource_id)
        if delay is not None and delay > 0:
            entry.pending = _PendingTransition(
                target=target_state,
                fire_at=self._clock.now() + delay,
                action=action,
            )
            self._call_history.append({"action": action, "pod_id": resource_id, "outcome": success_outcome})
            return LifecycleActionResult(outcome=success_outcome, attempts_made=1)

        entry.state = target_state
        self._call_history.append({"action": action, "pod_id": resource_id, "outcome": success_outcome})
        return LifecycleActionResult(outcome=success_outcome, attempts_made=1)

    def _reify_pending_if_due(self, entry: _PodEntry) -> None:
        if entry.pending is None:
            return
        if self._clock.now() >= entry.pending.fire_at:
            entry.state = entry.pending.target
            entry.pending = None


__all__ = [
    "FakePodLifecycleClient",
    "PodState",
]
