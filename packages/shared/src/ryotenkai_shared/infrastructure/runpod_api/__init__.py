"""Provider-agnostic RunPod API Protocol (definition-only in Phase 1).

See :mod:`ryotenkai_shared.infrastructure.runpod_api.protocol` for the
``IRunPodAPI`` Protocol, ``RunPodInfo`` / ``RunPodLifecycleResponse``
DTOs, and the typed error hierarchy
(``RunPodAPIError`` → ``RunPodRateLimitedError`` /
``RunPodTransientError`` / ``RunPodPartialResponseError``).
"""

from __future__ import annotations

from ryotenkai_shared.infrastructure.runpod_api.protocol import (
    IRunPodAPI,
    RunPodAPIError,
    RunPodInfo,
    RunPodLifecycleResponse,
    RunPodPartialResponseError,
    RunPodRateLimitedError,
    RunPodTransientError,
)

__all__ = [
    "IRunPodAPI",
    "RunPodAPIError",
    "RunPodInfo",
    "RunPodLifecycleResponse",
    "RunPodPartialResponseError",
    "RunPodRateLimitedError",
    "RunPodTransientError",
]
