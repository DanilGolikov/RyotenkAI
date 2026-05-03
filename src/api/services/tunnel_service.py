"""Backward-compat shim — canonical home moved to ``src.utils.clients.ssh_tunnel``.

Removed at the start of Phase B (monorepo packagization, see
``docs/plans/2026-05-03-monorepo-uv-workspace-packagization.md`` §B.4).
"""

from __future__ import annotations

from src.utils.clients.ssh_tunnel import (
    DEFAULT_LOCAL_PORT_RANGE,
    DEFAULT_REMOTE_PORT,
    SSHTunnelEndpoint,
    SSHTunnelError,
    SSHTunnelManager,
)

__all__ = [
    "DEFAULT_LOCAL_PORT_RANGE",
    "DEFAULT_REMOTE_PORT",
    "SSHTunnelEndpoint",
    "SSHTunnelError",
    "SSHTunnelManager",
]
