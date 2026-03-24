from __future__ import annotations

from pydantic import Field

from ...base import StrictBaseModel

# NOTE: Runtime imports are required for Pydantic field types.
from ..ssh import SSHConfig  # noqa: TC001


class SingleNodeConnectConfig(StrictBaseModel):
    """
    Connection config for single_node provider.

    SSH connection modes:
    1. Alias mode (recommended):
        connect:
          ssh:
            alias: pc  # From ~/.ssh/config

    2. Explicit mode:
        connect:
          ssh:
            host: <your-gpu-host>
            port: 22
            user: <your-user>
            key_path: ~/.ssh/id_ed25519
    """

    ssh: SSHConfig = Field(..., description="Unified SSH config (alias or host+user)")


__all__ = [
    "SingleNodeConnectConfig",
]
