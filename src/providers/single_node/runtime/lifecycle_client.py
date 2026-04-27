"""Phase 14.B — single-node NoOp impl of :class:`IPodLifecycleClient`.

Single-node has no cloud lifecycle to act on (host is always-on,
operator-managed). All three Protocol methods return
``LifecycleActionResult(outcome="skipped", attempts_made=0)`` —
preserves the pre-14.B
:data:`~src.runner.pod_terminator.PodTerminalOutcome.SKIPPED`
vocabulary that operator dashboards already grep for.

The client carries NO transport dependencies — pure stdlib. A
linting test in :mod:`src.tests.unit.providers.single_node.runtime`
pins this so a future refactor can't accidentally pull in httpx
through this path.
"""

from __future__ import annotations

from src.constants import PROVIDER_SINGLE_NODE
from src.runner.pod_terminator import PodTerminalOutcome
from src.runner.runtime.lifecycle_client import (
    IPodLifecycleClient,
    LifecycleActionResult,
)

__all__ = ["NoOpPodLifecycleClient"]


# Single shared sentinel — every NoOp call returns the same instance.
# Frozen dataclass so sharing is safe; saves us a dataclass
# construction per terminal hook.
_SKIPPED_RESULT = LifecycleActionResult(
    outcome=PodTerminalOutcome.SKIPPED,
    attempts_made=0,
)


class NoOpPodLifecycleClient:
    """Single-node :class:`IPodLifecycleClient` — every action is a no-op.

    Conforms to
    :class:`~src.runner.runtime.lifecycle_client.IPodLifecycleClient`.

    The ``resource_id`` argument is accepted for Protocol parity but
    ignored — single-node has no resource concept (the host is
    operator-managed and not tracked by the runner).
    """

    @property
    def provider_name(self) -> str:
        return PROVIDER_SINGLE_NODE

    async def terminate(self, *, resource_id: str) -> LifecycleActionResult:
        return _SKIPPED_RESULT

    async def pause(self, *, resource_id: str) -> LifecycleActionResult:
        return _SKIPPED_RESULT

    async def resume(self, *, resource_id: str) -> LifecycleActionResult:
        return _SKIPPED_RESULT


# Static guarantee — fail fast at module import if the Protocol
# shape drifts from this impl.
_runtime_check: IPodLifecycleClient = NoOpPodLifecycleClient()  # noqa: F841
