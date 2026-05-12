"""Phase 5 — real :class:`IPodLifecycleClient` adapter.

Additive only — the existing
:class:`ryotenkai_providers.runpod.runtime.lifecycle_client.RunPodPodLifecycleClient`
already implements the Protocol shape (Phase 14.B made the runtime
client conform to ``IPodLifecycleClient``). This module re-exports it
as :class:`RunPodLifecycleAdapter` so the live-protocol-compliance
test has a stable import path even if the underlying class moves.

The compliance test wires this adapter pointed at any reachable URL
(default: the fake-runpod sidecar). A real-RunPod sandbox is wired
via the standard ``RUNPOD_API_KEY`` env var.
"""

from __future__ import annotations

from ryotenkai_providers.runpod.runtime.lifecycle_client import RunPodPodLifecycleClient
from ryotenkai_shared.infrastructure.lifecycle import IPodLifecycleClient

#: Alias of the existing runner-side GraphQL impl; satisfies the Protocol.
RunPodLifecycleAdapter = RunPodPodLifecycleClient

# Static guarantee.
_runtime_check: IPodLifecycleClient = RunPodLifecycleAdapter(api_key="_compile_check")  # noqa: F841


__all__ = ["RunPodLifecycleAdapter"]
