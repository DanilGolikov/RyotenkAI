"""Shared SSH helpers used by all deployment components.

Pure module-level functions, no class state — each consumer (CodeSyncer,
FileUploader, DependencyInstaller, TrainingLauncher) imports and calls
directly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.utils.ssh_client import SSHClient


def build_ssh_opts(ssh_client: SSHClient) -> str:
    """Build SSH options string reusing the client's ControlMaster socket.

    When ``ssh_client`` exposes ``ssh_base_opts`` (always the case for a
    real :class:`~src.utils.ssh_client.SSHClient`), reuse them so that
    rsync/tar-over-ssh share the persistent TCP connection opened by
    earlier SSH operations.

    Falls back to a legacy/mock-friendly path that constructs ``-i / -p``
    flags from ``key_path`` / ``port`` attributes when ``ssh_base_opts``
    is missing — this branch is exercised by unit tests with ``MagicMock``
    SSH clients.
    """
    base_opts: list[str] | None = getattr(ssh_client, "ssh_base_opts", None)
    if base_opts:
        key_path = getattr(ssh_client, "key_path", None)
        port = getattr(ssh_client, "port", None)
        parts: list[str] = []
        if isinstance(key_path, str) and key_path:
            parts.extend(["-i", key_path])
        if isinstance(port, int) and port:
            parts.extend(["-p", str(port)])
        parts.extend(base_opts)
        return " ".join(parts)

    # Legacy / mock fallback
    alias_mode_attr = getattr(ssh_client, "is_alias_mode", None)
    if not isinstance(alias_mode_attr, bool):
        alias_mode_attr = getattr(ssh_client, "_is_alias_mode", False)
    is_alias_mode = bool(alias_mode_attr) if isinstance(alias_mode_attr, bool) else False

    if is_alias_mode:
        return "-o StrictHostKeyChecking=no"

    opts_parts: list[str] = []
    key_path = getattr(ssh_client, "key_path", None)
    if isinstance(key_path, str) and key_path:
        opts_parts.append(f"-i {key_path}")
    port = getattr(ssh_client, "port", None)
    if isinstance(port, int) and port:
        opts_parts.append(f"-p {port}")
    opts_parts.append("-o StrictHostKeyChecking=no")
    return " ".join(opts_parts)


__all__ = ["build_ssh_opts"]
