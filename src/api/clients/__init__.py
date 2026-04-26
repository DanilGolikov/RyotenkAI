"""Mac-side clients that talk to the in-pod runner over an SSH tunnel.

The runner exposes loopback-only HTTP (``127.0.0.1:8080``) inside the
pod. Mac control-plane code reaches it through ``ssh -L`` (managed by
:class:`src.api.services.tunnel_service.SSHTunnelManager`) and a
:class:`JobClient` instance pointed at the local end of the tunnel.

Why a dedicated package: existing :mod:`src.api.services` modules host
business logic that runs *inside* the API process; the runner clients
talk to a remote service over the wire and need different test seams
(transport mocks, WebSocket reconnect drills) — splitting them out
keeps the public service surface clean.
"""

from __future__ import annotations

from src.api.clients.job_client import (
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
