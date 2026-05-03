"""Backward-compat shim — canonical home moved to ``src.pipeline.heartbeat.heartbeat``.

Removed at the start of Phase B (monorepo packagization, see
``docs/plans/2026-05-03-monorepo-uv-workspace-packagization.md`` §B.4).
"""

from __future__ import annotations

from src.pipeline.heartbeat.heartbeat import (
    DEFAULT_PING_INTERVAL_SECONDS,
    DEFAULT_TTL_SECONDS,
    ControlPlaneHeartbeat,
)

__all__ = [
    "ControlPlaneHeartbeat",
    "DEFAULT_PING_INTERVAL_SECONDS",
    "DEFAULT_TTL_SECONDS",
]
