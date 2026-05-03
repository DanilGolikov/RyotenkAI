"""Backward-compat shim — canonical home moved to ``src.utils.clients.job_client``.

Removed at the start of Phase B (monorepo packagization, see
``docs/plans/2026-05-03-monorepo-uv-workspace-packagization.md`` §B.4).
"""

from __future__ import annotations

from ryotenkai_shared.utils.clients.job_client import (
    DEFAULT_RECONNECT_MAX_DELAY,
    DEFAULT_REQUEST_TIMEOUT,
    JobClient,
    JobClientError,
    JobNotFoundError,
    ReplayTruncatedError,
)

__all__ = [
    "DEFAULT_RECONNECT_MAX_DELAY",
    "DEFAULT_REQUEST_TIMEOUT",
    "JobClient",
    "JobClientError",
    "JobNotFoundError",
    "ReplayTruncatedError",
]
