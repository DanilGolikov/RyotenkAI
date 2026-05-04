"""Log streaming DTOs (Phase 2 PR-2.3).

``GET /api/v1/logs/{name}`` lets the Mac client tail pod-side log
files via simple range HTTP polling, replacing the SSH ``stat`` /
``tail -c`` protocol the legacy :class:`LogManager` used.

Anti-path-traversal: ``LogName`` is a closed StrEnum (whitelist). The
endpoint NEVER accepts arbitrary paths from the Mac client — only
the enum values map to known files on the pod.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field

from ._strict import _StrictModel


class LogName(StrEnum):
    """Whitelist of log files the runner is willing to tail.

    Pattern modelled on ``kubectl logs --container=<name>``: a small
    closed enum is enough to cover every operational use case and
    forbids the entire class of path-traversal bugs that come with
    accepting raw paths.

    * ``TRAINER_STDIO`` — trainer subprocess stdout/stderr ground
      truth, written by the runner's :class:`Supervisor` pump
      (``[OUT]``/``[ERR]`` per-line prefixed for atomic ordering).
    * ``RUNNER`` — uvicorn / runner stdout, captures pre-import
      failures and lifecycle events.
    """

    TRAINER_STDIO = "trainer_stdio"
    RUNNER = "runner"


class LogChunkResponse(_StrictModel):
    """One range read of a remote log file.

    The Mac client polls ``GET /logs/{name}?offset=N&limit_bytes=M``
    every 5 s; each call returns the bytes from ``offset`` up to
    ``min(offset + limit_bytes, total_size)``. Then the client
    advances ``offset`` to ``next_offset`` for the next poll.
    """

    content: str = Field(
        description=(
            "UTF-8 chunk read from the remote file starting at "
            "``offset``. Decoded with ``errors='replace'`` so a "
            "torn multi-byte sequence at the chunk boundary doesn't "
            "fail the response."
        ),
    )
    total_size: int = Field(
        ge=0,
        description="Total file size in bytes (``stat -c %s`` equivalent).",
    )
    next_offset: int = Field(
        ge=0,
        description=(
            "Byte position the caller should pass as ``offset`` on "
            "the next poll. Equals ``offset + len(content_bytes)``."
        ),
    )
    truncated: bool = Field(
        default=False,
        description=(
            "True when the chunk hit ``limit_bytes`` before reaching "
            "EOF — caller should keep polling without sleep."
        ),
    )


class LogSizeResponse(_StrictModel):
    """Lightweight ``GET /logs/{name}/size`` — file size only.

    Used by the Mac client to decide whether a poll is even worth
    issuing (``size > offset``); cheaper than fetching a zero-byte
    chunk just to read ``total_size``.
    """

    size_bytes: int = Field(ge=0)


__all__ = [
    "LogChunkResponse",
    "LogName",
    "LogSizeResponse",
]
