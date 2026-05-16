"""``GET /api/v1/logs/{name}`` — range-read tail of pod-side log files.

Replaces the SSH ``stat -c %s`` / ``tail -c <delta>`` protocol the
legacy :class:`LogManager` (Mac-side) used to issue. The endpoint
returns the bytes from ``offset`` up to ``min(offset + limit_bytes,
total_size)`` plus the metadata the caller needs to advance:

    {
        "content": "...UTF-8 chunk...",
        "total_size": 12345,
        "next_offset": 8192,
        "truncated": false
    }

Anti-path-traversal: the path component is a closed
:class:`LogName` enum — the Mac client cannot ask for an arbitrary
file. The endpoint maps the enum value through
:class:`PodLayout` so the file path source-of-truth stays in one
place.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, Query, Request

from ryotenkai_shared.errors import (
    LogNameInvalidError,
    LogNotAvailableError,
    LogOffsetOutOfRangeError,
    RunnerNotReadyError,
)
from ryotenkai_shared.contracts.runner_api.logs import (
    LogChunkResponse,
    LogName,
    LogSizeResponse,
)
from ryotenkai_shared.utils.pod_layout import PodLayout

router = APIRouter(prefix="/logs", tags=["logs"])


# 10 MB hard cap — protects the runner from a misbehaving client
# requesting the entire pod-side log file in one go. Mac clients
# tail in 8 KB chunks by design (see Mac LogFetcher).
LIMIT_BYTES_DEFAULT = 8 * 1024
LIMIT_BYTES_MAX = 10 * 1024 * 1024


def _get_pod_layout(request: Request) -> PodLayout:
    """Resolve the per-run :class:`PodLayout` stored on
    :attr:`app.state.pod_layout` by the lifespan."""
    layout = getattr(request.app.state, "pod_layout", None)
    if layout is None:
        raise RunnerNotReadyError(
            detail="pod layout not initialised on app.state",
        )
    return layout


def _resolve_log_path(layout: PodLayout, name: LogName) -> Path:
    """Map :class:`LogName` to the actual pod-side file path.

    No string manipulation of the name parameter — the closed enum
    is the only entry point, so directory traversal is structurally
    impossible.
    """
    if name is LogName.TRAINER_STDIO:
        return Path(str(layout.trainer_stdio_log))
    if name is LogName.RUNNER:
        return Path(str(layout.runner_log))
    raise LogNameInvalidError(
        detail=f"unhandled LogName={name!r} (programming bug)",
        context={"log_name": str(name)},
    )


@router.get("/{name}", response_model=LogChunkResponse)
def read_log(
    name: LogName,
    *,
    layout: PodLayout = Depends(_get_pod_layout),  # noqa: B008
    offset: int = Query(default=0, ge=0),
    limit_bytes: int = Query(
        default=LIMIT_BYTES_DEFAULT,
        ge=1,
        le=LIMIT_BYTES_MAX,
    ),
) -> LogChunkResponse:
    """Range read from a pod-side log file."""
    path = _resolve_log_path(layout, name)

    try:
        total_size = path.stat().st_size
    except FileNotFoundError:
        raise LogNotAvailableError(
            detail=f"log {name.value!r} does not exist on disk yet",
        ) from None
    except OSError as exc:
        raise LogNotAvailableError(
            detail=f"log {name.value!r} stat failed: {exc}",
            cause=exc,
        ) from exc

    if offset > total_size:
        raise LogOffsetOutOfRangeError(
            detail=(
                f"offset={offset} > total_size={total_size}; "
                "client must reset its cursor to 0 (likely the file "
                "was rotated/truncated mid-poll)."
            ),
            context={"offset": offset, "total_size": total_size},
        )

    if offset == total_size:
        # Tail caught up — return empty chunk so the client has a
        # consistent contract (always a 200 with the new total).
        return LogChunkResponse(
            content="", total_size=total_size,
            next_offset=offset, truncated=False,
        )

    available = total_size - offset
    capped = min(limit_bytes, available)
    truncated = capped < available

    try:
        with path.open("rb") as fh:
            fh.seek(offset)
            raw = fh.read(capped)
    except OSError as exc:
        raise LogNotAvailableError(
            detail=f"log {name.value!r} read failed: {exc}",
            cause=exc,
        ) from exc

    # ``errors='replace'`` so a torn multi-byte UTF-8 sequence at the
    # chunk boundary doesn't blow up the response. The Mac client
    # may still see a U+FFFD on the seam — acceptable for log tail.
    content = raw.decode("utf-8", errors="replace")

    return LogChunkResponse(
        content=content,
        total_size=total_size,
        next_offset=offset + len(raw),
        truncated=truncated,
    )


@router.get("/{name}/size", response_model=LogSizeResponse)
def get_log_size(
    name: LogName,
    *,
    layout: PodLayout = Depends(_get_pod_layout),  # noqa: B008
) -> LogSizeResponse:
    """Lightweight size-only query — cheaper than a zero-byte read."""
    path = _resolve_log_path(layout, name)
    try:
        size = path.stat().st_size
    except FileNotFoundError:
        # Distinguish "no log yet" from "wrong name" — name is enum-
        # validated, so a missing file always means "not yet written".
        # We return 0 instead of 404 to keep the polling contract
        # simple: client polls size, sees 0, sleeps and tries again.
        return LogSizeResponse(size_bytes=0)
    except OSError as exc:
        raise LogNotAvailableError(
            detail=f"log {name.value!r} stat failed: {exc}",
            cause=exc,
        ) from exc
    return LogSizeResponse(size_bytes=size)


__all__ = ["router"]
