"""Provider-agnostic ``IJobClient`` Protocol (definition-only in Phase 4)."""

from __future__ import annotations

from ryotenkai_shared.infrastructure.job_client.protocol import (
    IJobClient,
    JobClientNetworkError,
    JobClientNotFoundError,
    JobClientProtocolError,
    JobClientRateLimitedError,
    JobSubmissionResult,
)

__all__ = [
    "IJobClient",
    "JobClientNetworkError",
    "JobClientNotFoundError",
    "JobClientProtocolError",
    "JobClientRateLimitedError",
    "JobSubmissionResult",
]
