"""Phase 4 — Provider-agnostic ``ISSHClient`` Protocol.

Extracted additively in 2026-05-11. The concrete production class
:class:`ryotenkai_shared.utils.ssh_client.SSHClient` already exposes
the upload/download/exec surface; the Protocol narrows it to async
methods that the Mac control-plane and tests use, intentionally
omitting helpers like ``test_connection`` (which is a higher-level
retry loop) and the legacy ``_StallDetected`` exception (an internal).

**Definition-only**: nothing in production implements ``ISSHClient``
yet. The compliance test parametrizes over ``[fake, real]`` but
``real`` ``pytest.skip``-s.

Connection model: the Protocol assumes a connected session — there is
no explicit ``connect()`` call. Implementations either connect lazily
on first use (production) or accept a pre-built "connected" state
(fake). Disconnect is via :meth:`close`.

Filesystem model: the Protocol mirrors the four real operations
(upload file, download file, upload directory, download directory)
plus arbitrary command exec. Each method returns a small typed result
shape; transport failures raise :class:`SSHError` subclasses so
callers can react to "host unreachable" vs "command failed" without
parsing stderr strings.

Vendor-exception isolation
--------------------------
:class:`SSHError` and its subclasses (:class:`SSHConnectionError`,
:class:`SSHExecError`, :class:`SSHTransferError`) are transport-layer
types. They MUST be caught and translated into typed
:class:`ryotenkai_shared.errors.RyotenkAIError` subclasses
(``SSHConnectionFailedError`` / ``SSHExecFailedError`` / ``SSHTransferFailedError``)
inside the adapter modules; downstream code (control, pod,
providers.training, providers.inference, providers.single_node) is
forbidden from importing this module directly. The boundary is enforced
by the importlinter contract *"Vendor SDK exception types stay inside
infrastructure adapters"* in ``pyproject.toml``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


class SSHError(Exception):
    """Base error for the ``ISSHClient`` surface."""


class SSHConnectionError(SSHError):
    """Transport-level failure: host unreachable, timeout, auth failure."""


class SSHExecError(SSHError):
    """Remote command returned non-zero or timed out mid-stream."""


class SSHTransferError(SSHError):
    """File or directory transfer (scp/tar pipe) failed."""


@dataclass(frozen=True)
class SSHCommandResult:
    """Structured result of one remote command.

    Mirrors the (success, stdout, stderr) tuple returned by
    :meth:`SSHClient.exec_command` but as a frozen typed shape with an
    explicit exit code. ``exit_code == 0`` is the success contract.
    """

    exit_code: int
    stdout: str
    stderr: str

    @property
    def success(self) -> bool:
        return self.exit_code == 0


@runtime_checkable
class ISSHClient(Protocol):
    """Async surface for an SSH connection to a remote host."""

    @property
    def host(self) -> str:
        """The remote host this client is bound to (for logging)."""
        ...

    async def exec(
        self,
        command: str,
        *,
        timeout: float | None = None,
    ) -> SSHCommandResult:
        """Run ``command`` synchronously on the remote host.

        Returns the typed :class:`SSHCommandResult` regardless of
        ``exit_code``. Only transport / connection failures raise
        :class:`SSHConnectionError`.
        """
        ...

    async def upload_file(
        self,
        local_path: str,
        remote_path: str,
        *,
        timeout: float | None = None,
    ) -> None:
        """SCP a single file to the remote host.

        Raises :class:`SSHTransferError` on transfer failure (including
        a missing local source).
        """
        ...

    async def download_file(
        self,
        remote_path: str,
        local_path: str,
        *,
        timeout: float | None = None,
    ) -> None:
        """SCP a single file from the remote host.

        Raises :class:`SSHTransferError` on transfer failure.
        """
        ...

    async def file_exists(self, remote_path: str) -> bool:
        """Return whether ``remote_path`` is a regular file on the remote host."""
        ...

    async def close(self) -> None:
        """Close the connection (best-effort, idempotent)."""
        ...


__all__ = [
    "ISSHClient",
    "SSHCommandResult",
    "SSHConnectionError",
    "SSHError",
    "SSHExecError",
    "SSHTransferError",
]
