from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..providers.ssh import SSHConfig


def validate_ssh(cfg: SSHConfig) -> None:
    """Validate SSHConfig connection mode (alias vs explicit host+user)."""

    # If ssh.alias is provided → it's enough to consider config valid.
    if cfg.alias:
        return

    # Otherwise require explicit host+user.
    if not cfg.host or not cfg.user:
        raise ValueError("SSHConfig requires either ssh.alias or (ssh.host + ssh.user)")


__all__ = [
    "validate_ssh",
]
