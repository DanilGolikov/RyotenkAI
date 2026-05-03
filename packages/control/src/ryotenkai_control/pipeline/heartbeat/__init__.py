"""Pipeline-side control-plane heartbeat.

Mac orchestrator pings the runner's ``/api/v1/control/heartbeat`` every
30 s while the pipeline is alive — a defense against the implicit-WS
heartbeat going stale during long pure-SCP work (e.g. adapter download).

Lived in ``src.api.services.control_plane_heartbeat`` until Phase A.2
of the monorepo packagization (plan §A.2). Moved here because it is a
*pipeline-driven* client of the runner, not a service that runs inside
the control-plane API process — hosting it under ``src.api.services``
caused the ``pipeline → api`` half of the cycle (plan §2.4-A).
"""

from __future__ import annotations

from ryotenkai_control.pipeline.heartbeat.heartbeat import (
    DEFAULT_PING_INTERVAL_SECONDS,
    DEFAULT_TTL_SECONDS,
    ControlPlaneHeartbeat,
)

__all__ = [
    "ControlPlaneHeartbeat",
    "DEFAULT_PING_INTERVAL_SECONDS",
    "DEFAULT_TTL_SECONDS",
]
