"""Provider-agnostic ``IHFHubClient`` Protocol (definition-only in Phase 4)."""

from __future__ import annotations

from ryotenkai_shared.infrastructure.hf_hub.protocol import (
    HFAuthError,
    HFHubError,
    HFNotFoundError,
    HFRateLimitedError,
    HFRepoInfo,
    HFTransientError,
    IHFHubClient,
)

__all__ = [
    "HFAuthError",
    "HFHubError",
    "HFNotFoundError",
    "HFRateLimitedError",
    "HFRepoInfo",
    "HFTransientError",
    "IHFHubClient",
]
