"""Phase 1 — Provider-agnostic ``IRunPodAPI`` Protocol.

Extracted additively in 2026-05-10 as part of the greenfield testing
architecture rollout (see
``docs/plans/structured-hopping-starfish.md`` Decision 1, table row 3).

Why a new Protocol rather than reusing :class:`IPodLifecycleClient`?
   * :class:`IPodLifecycleClient` only covers the three terminal-state
     mutations (``terminate``/``pause``/``resume``). RunPod's GraphQL
     surface that the control-plane talks to also includes pod
     enumeration, query, and richer error shapes (rate-limit, partial
     responses) that are useful to surface in tests.
   * The existing concrete classes
     (:class:`ryotenkai_providers.runpod.training.api_client.RunPodAPIClient`
     and :class:`ryotenkai_providers.runpod.runtime.lifecycle_client.RunPodPodLifecycleClient`)
     already speak similar shapes; this Protocol just makes the surface
     DI-able for greenfield component tests without forcing a real
     refactor of the call sites today.

This module is **definition-only** — nothing in production yet
implements ``IRunPodAPI``. The compliance test (in
``tests/contract/protocol_compliance/``) parametrizes over
``[fake, real]`` but ``real`` is currently a ``pytest.skip(...)`` until
Phase 1+ wires :class:`RunPodAPIClient` into the Protocol.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class RunPodInfo:
    """Normalized snapshot of a RunPod pod's observable state.

    Phase 1 deliberately keeps the surface minimal — control-plane
    callers ask "what's its desired status, is it ready, what's its
    SSH endpoint" — so the fake stays honest. Mirrors the ``runtime``
    payload shape consumed by
    :class:`ryotenkai_providers.runpod.models.PodSnapshot` but uses
    typed fields rather than raw GraphQL dicts.
    """

    pod_id: str
    desired_status: str
    runtime_status: str | None = None
    ssh_host: str | None = None
    ssh_port: int | None = None
    cost_per_hr: float | None = None
    machine_id: str | None = None
    extras: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class RunPodLifecycleResponse:
    """Outcome of a single ``stop_pod``/``terminate_pod``/``resume_pod`` call.

    Distinct from
    :class:`ryotenkai_shared.infrastructure.lifecycle.LifecycleActionResult`
    because the Mac-side control-plane historically thinks in
    ``Result``/``ProviderError`` shapes — ``RunPodLifecycleResponse``
    bridges the gap by carrying both a structured outcome and an
    optional error payload.

    Attributes:
        outcome:
            One of ``"ok"``, ``"already_done"``, ``"not_found"``,
            ``"failed"``. ``already_done`` covers RunPod's idempotency
            markers (already-stopped/already-terminated/etc.).
        message:
            Human-readable reason. Empty string on success.
    """

    outcome: str
    message: str = ""


class RunPodAPIError(Exception):
    """Base error for the IRunPodAPI surface."""


class RunPodRateLimitedError(RunPodAPIError):
    """Upstream returned 429 (or RunPod's documented "rate limit" marker)."""


class RunPodTransientError(RunPodAPIError):
    """5xx or other transient upstream failure — caller should retry."""


class RunPodPartialResponseError(RunPodAPIError):
    """Response was 200 but the GraphQL payload was missing required fields.

    Surfaces parser-fragility scenarios; the chaos-mode equivalent in
    :class:`tests._fakes.runpod.FakeRunPodAPI` injects this so
    the parser-tolerance code in
    :mod:`ryotenkai_providers.runpod.models` can be exercised
    deterministically.
    """


@runtime_checkable
class IRunPodAPI(Protocol):
    """Provider-agnostic surface for RunPod-flavoured pod control.

    Implemented by:

    * (Phase 1+ TODO) :class:`ryotenkai_providers.runpod.training.api_client.RunPodAPIClient`
      after a refactor that swaps its internal SDK adapter for the
      Protocol.
    * :class:`tests._fakes.runpod.FakeRunPodAPI` — in-memory
      deterministic stand-in for component tests.

    All methods are ``async`` even when the production SDK is sync so
    the Protocol is uniform (the fake is naturally async). Where the
    Mac-side ``RunPodAPIClient`` exposes sync methods, an additive
    refactor wrapping each in ``asyncio.to_thread`` is the planned
    bridge.
    """

    async def find_pod(self, pod_id: str) -> RunPodInfo | None:
        """Return ``None`` if the pod does not exist; otherwise its snapshot.

        Mirrors :meth:`RunPodAPIClient.query_pod` semantics: missing
        pod is *not* an error — it's a successful "the pod is gone".
        """
        ...

    async def list_pods(self) -> list[RunPodInfo]:
        """Enumerate visible pods. Empty list when none exist.

        Mirrors the SDK's ``runpod.get_pods()`` shape consumed by
        :class:`ryotenkai_providers.runpod.inference.pods.api_client.RunPodInferenceAPIClient`.
        """
        ...

    async def stop_pod(self, pod_id: str) -> RunPodLifecycleResponse:
        """Pause a pod (preserving the volume). Idempotent."""
        ...

    async def terminate_pod(self, pod_id: str) -> RunPodLifecycleResponse:
        """Permanently destroy a pod. Idempotent."""
        ...

    async def resume_pod(self, pod_id: str) -> RunPodLifecycleResponse:
        """Wake a previously paused pod. Idempotent."""
        ...


__all__ = [
    "IRunPodAPI",
    "RunPodAPIError",
    "RunPodInfo",
    "RunPodLifecycleResponse",
    "RunPodPartialResponseError",
    "RunPodRateLimitedError",
    "RunPodTransientError",
]
