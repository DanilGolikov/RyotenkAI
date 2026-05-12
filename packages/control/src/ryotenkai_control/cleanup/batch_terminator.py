"""Batch terminator — terminate a list of pods through ``IPodLifecycleClient``.

Phase 1 greenfield helper. Used by Mac-side cleanup workflows that need
to drive multiple pods into terminal state (workspace cleanup, fleet
sweeps after a release, etc.). Tolerates per-pod failures: a single bad
pod doesn't abort the batch.

Design notes:

* The collaborator is the abstract :class:`IPodLifecycleClient`
  Protocol — same interface the runner uses internally. Tests inject
  :class:`tests._fakes.lifecycle.FakePodLifecycleClient` to drive the
  state machine deterministically.
* No mocks anywhere. The helper is pure logic over a Protocol.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ryotenkai_shared.infrastructure.lifecycle import (
    LifecycleActionResult,
    PodTerminalOutcome,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from ryotenkai_shared.infrastructure.lifecycle import IPodLifecycleClient


@dataclass(frozen=True)
class BatchTerminationReport:
    """Per-batch summary of a :class:`BatchPodTerminator.terminate_all` call."""

    total: int
    terminated: int
    already_terminated: int
    failed: int
    failures: tuple[tuple[str, str], ...] = field(default_factory=tuple)
    outcomes: dict[str, str] = field(default_factory=dict)


class BatchPodTerminator:
    """Drive a list of pods to a terminal state."""

    def __init__(self, *, client: IPodLifecycleClient) -> None:
        self._client = client

    async def terminate_all(self, pod_ids: Iterable[str]) -> BatchTerminationReport:
        outcomes: dict[str, str] = {}
        failures: list[tuple[str, str]] = []
        terminated = 0
        already_terminated = 0
        failed = 0
        total = 0

        for pod_id in pod_ids:
            total += 1
            result: LifecycleActionResult = await self._client.terminate(resource_id=pod_id)
            outcomes[pod_id] = result.outcome
            if result.outcome == PodTerminalOutcome.TERMINATED:
                terminated += 1
            elif result.outcome == PodTerminalOutcome.ALREADY_TERMINATED:
                already_terminated += 1
            else:
                failed += 1
                failures.append((pod_id, result.last_error or result.outcome))

        return BatchTerminationReport(
            total=total,
            terminated=terminated,
            already_terminated=already_terminated,
            failed=failed,
            failures=tuple(failures),
            outcomes=outcomes,
        )


__all__ = ["BatchPodTerminator", "BatchTerminationReport"]
