"""Ephemeral TCP port allocator.

Bind to port 0, read the kernel-assigned port, close the socket, and return.
Race-prone in principle (the port could be reused before our subprocess
binds), but the reuse window in tests is short and we tolerate the
occasional retry inside :class:`Stack.boot`.
"""

from __future__ import annotations

import socket


def allocate_port_block(n: int) -> list[int]:
    if n < 1:
        raise ValueError("n must be >= 1")
    sockets: list[socket.socket] = []
    try:
        for _ in range(n):
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.bind(("127.0.0.1", 0))
            sockets.append(s)
        ports = [s.getsockname()[1] for s in sockets]
    finally:
        for s in sockets:
            s.close()
    return ports


def allocate_port() -> int:
    return allocate_port_block(1)[0]


__all__ = ["allocate_port", "allocate_port_block"]
