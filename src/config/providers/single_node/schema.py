from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from pydantic import Field

from ...base import StrictBaseModel

# NOTE: Runtime imports are required for Pydantic field types.
from ..ssh import SSHConfig  # noqa: TC001
from .cleanup import SingleNodeCleanupConfig
from .connect import SingleNodeConnectConfig  # noqa: TC001
from .inference import SingleNodeInferenceConfig
from .training import SingleNodeTrainingConfig  # noqa: TC001


class SingleNodeConfig(StrictBaseModel):
    """
    Unified single_node provider config (training + inference).

    New structure (v3):
        providers:
          single_node:
            connect:
              ssh:
                alias: pc
            cleanup:
              cleanup_workspace: true
              keep_on_error: true
              on_interrupt: true
            training:
              workspace_path: /home/user/workspace
              training_start_timeout: 30
            inference:
              serve:
                host: "127.0.0.1"
                port: 8000
                workspace: "/home/user/inference"

    Benefits:
    - Single SSH config (no duplication)
    - Clear separation: connect (transport) / training / inference
    - Logical grouping of all single_node settings
    """

    connect: SingleNodeConnectConfig = Field(..., description="Connection config (SSH)")
    cleanup: SingleNodeCleanupConfig = Field(
        default_factory=SingleNodeCleanupConfig,  # type: ignore[arg-type]
        description="Cleanup policy (workspace + interrupt handling)",
    )
    training: SingleNodeTrainingConfig = Field(..., description="Training-specific settings")
    inference: SingleNodeInferenceConfig = Field(
        default_factory=SingleNodeInferenceConfig,  # type: ignore[arg-type]
        description="Inference-specific settings",
    )

    # =========================================================================
    # Convenience properties (shortcuts)
    # =========================================================================

    @property
    def ssh(self) -> SSHConfig:
        """Shortcut to SSH config."""
        return self.connect.ssh

    @property
    def workspace_path(self) -> str:
        """Shortcut to training workspace path."""
        return self.training.workspace_path

    @property
    def cleanup_workspace(self) -> bool:
        """Shortcut to cleanup_workspace setting."""
        return self.cleanup.cleanup_workspace

    @property
    def keep_on_error(self) -> bool:
        """Shortcut to keep_on_error setting."""
        return self.cleanup.keep_on_error

    @property
    def training_start_timeout(self) -> int:
        """Shortcut to training_start_timeout setting."""
        return self.training.training_start_timeout

    @property
    def gpu_type(self) -> str | None:
        """Shortcut to gpu_type setting."""
        return self.training.gpu_type

    @property
    def mock_mode(self) -> bool:
        """Shortcut to mock_mode setting."""
        return self.training.mock_mode

    # =========================================================================
    # SSH helpers (for backward compatibility with existing code)
    # =========================================================================

    @property
    def is_alias_mode(self) -> bool:
        """Check if using SSH alias mode."""
        return self.connect.ssh.alias is not None

    def get_ssh_host_for_client(self) -> str:
        """Get host for SSHClient (alias or actual host)."""
        ssh = self.connect.ssh
        return ssh.alias if ssh.alias else str(ssh.host)

    def get_ssh_user_for_client(self) -> str | None:
        """Get user for SSHClient (None if using alias)."""
        ssh = self.connect.ssh
        return None if ssh.alias else ssh.user

    def get_ssh_port_for_client(self) -> int:
        """Get SSH port (ignored in alias mode; configured in ~/.ssh/config)."""
        from ..constants import SSH_PORT_DEFAULT

        return int(self.connect.ssh.port or SSH_PORT_DEFAULT)

    def resolve_ssh_key_path_for_client(self) -> str | None:
        """
        Resolve SSH key path for SSHClient explicit mode.

        Resolution:
            1. ssh.key_path (direct path)
            2. ssh.key_env (environment variable)
            3. None (SSHClient will auto-detect keys)
        """
        ssh = self.connect.ssh
        if ssh.alias:
            return None

        if ssh.key_path:
            path = Path(ssh.key_path).expanduser()
            if path.exists():
                return str(path)
            raise ValueError(f"SSH key not found at: {ssh.key_path}")

        if ssh.key_env:
            env_value = os.environ.get(ssh.key_env)
            if env_value:
                path = Path(env_value).expanduser()
                if path.exists():
                    return str(path)
                raise ValueError(f"SSH key from ${ssh.key_env}='{env_value}' not found")
            raise ValueError(f"Environment variable {ssh.key_env} not set")

        return None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SingleNodeConfig:
        """Create config from dictionary."""
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return self.model_dump(exclude_none=True)


__all__ = [
    "SingleNodeConfig",
]
