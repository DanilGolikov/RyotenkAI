"""Provider-agnostic ``ISSHClient`` Protocol (definition-only in Phase 4)."""

from __future__ import annotations

from ryotenkai_shared.infrastructure.ssh.protocol import (
    ISSHClient,
    SSHCommandResult,
    SSHConnectionError,
    SSHError,
    SSHExecError,
    SSHTransferError,
)

__all__ = [
    "ISSHClient",
    "SSHCommandResult",
    "SSHConnectionError",
    "SSHError",
    "SSHExecError",
    "SSHTransferError",
]
