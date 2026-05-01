"""TCP-probe helper for :class:`PodSshWaiter`.

Closes the cold-SSH gap: RunPod marks a pod RUNNING with an SSH
endpoint allocated, but ``sshd`` itself may take a few seconds longer
to start listening. Without this probe, downstream stages get
``Connection refused`` and the launch fails with a misleading error.

Kept as a free function (injectable) so tests can swap in a deterministic
fake without monkeypatching ``socket``.
"""

from __future__ import annotations

import socket


def default_tcp_probe(host: str, port: int, timeout_s: float) -> bool:
    """Return ``True`` if a TCP connection to ``(host, port)`` succeeds within
    ``timeout_s`` seconds.

    Catches every error from ``socket.create_connection`` — the caller
    only needs the boolean verdict, not the underlying OS error code.
    """
    try:
        with socket.create_connection((host, port), timeout=timeout_s):
            return True
    except OSError:
        return False


__all__ = ["default_tcp_probe"]
