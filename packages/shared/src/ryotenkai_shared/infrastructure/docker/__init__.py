"""Docker infrastructure — Protocol + production impl.

The legacy module-level functions in :mod:`ryotenkai_shared.utils.docker`
remain available for callers that haven't migrated to the Protocol
form yet (Phase 4A is additive).
"""

from __future__ import annotations

from ryotenkai_shared.infrastructure.docker.local import LocalDockerClient
from ryotenkai_shared.infrastructure.docker.protocol import IDockerClient

__all__ = ["IDockerClient", "LocalDockerClient"]
