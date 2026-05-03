"""HTTP / SSH-tunnel clients used by Mac control-plane and pipeline.

Originally housed under ``src.api.clients`` and ``src.api.services``;
moved here in Phase A.2 of monorepo packagization (see
``docs/plans/2026-05-03-monorepo-uv-workspace-packagization.md``) so
pipeline-side and CLI-side callers no longer have to import from
``src.api`` (root cause of the ``api ↔ pipeline`` cycle, plan §2.4-A).
"""

from __future__ import annotations

from ryotenkai_shared.utils.clients.job_client import (
    JobClient,
    JobClientError,
    JobNotFoundError,
    ReplayTruncatedError,
)
from ryotenkai_shared.utils.clients.ssh_tunnel import (
    SSHTunnelEndpoint,
    SSHTunnelError,
    SSHTunnelManager,
)

__all__ = [
    "JobClient",
    "JobClientError",
    "JobNotFoundError",
    "ReplayTruncatedError",
    "SSHTunnelEndpoint",
    "SSHTunnelError",
    "SSHTunnelManager",
]
