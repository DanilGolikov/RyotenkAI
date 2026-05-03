"""Pod-lifecycle protocol + outcome vocabulary (provider-agnostic).

Both the runner-side ``PodTerminator`` and the provider-side
lifecycle adapters (RunPod, single_node) speak this contract.
Hosting it in shared (ADR row 7) keeps providers and pod from
importing each other across the workspace boundary.
"""

from __future__ import annotations

from ryotenkai_shared.infrastructure.lifecycle.availability import PodAvailability
from ryotenkai_shared.infrastructure.lifecycle.outcomes import PodTerminalOutcome
from ryotenkai_shared.infrastructure.lifecycle.protocol import (
    IPodLifecycleClient,
    LifecycleActionResult,
)

__all__ = [
    "IPodLifecycleClient",
    "LifecycleActionResult",
    "PodAvailability",
    "PodTerminalOutcome",
]
