"""Backward-compat shim — canonical home moved to ``src.utils.clients``.

Removed at the start of Phase B (monorepo packagization, see
``docs/plans/2026-05-03-monorepo-uv-workspace-packagization.md`` §B.4).
"""

from __future__ import annotations

from src.utils.clients.job_client import (
    JobClient,
    JobClientError,
    JobNotFoundError,
    ReplayTruncatedError,
)

__all__ = [
    "JobClient",
    "JobClientError",
    "JobNotFoundError",
    "ReplayTruncatedError",
]
