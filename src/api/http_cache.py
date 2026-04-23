"""HTTP conditional-GET helpers for sibling-client state reads.

The web UI polls state endpoints every few seconds per open tab. Most of
those requests hit ``pipeline_state.json`` files that haven't changed since
the previous poll — we answer ``304 Not Modified`` instead of re-sending
the full body.

We key cache validators on the ``mtime_ns`` of the underlying JSON file:

* ``ETag: W/"<mtime_ns>"`` — weak validator (we don't byte-compare bodies,
  only freshness).
* ``Last-Modified: <RFC 7231 HTTP-date>`` — coarser, seconds-resolution
  fallback that browsers/fetch honour automatically.

Why both: ``If-None-Match`` is what modern clients (React Query, SWR,
browsers) send first and is more precise; ``If-Modified-Since`` is the
safety net for proxies / tools that only understand HTTP-date.

Precedence (per RFC 7232 §6): if ``If-None-Match`` is present we use only
that. ``If-Modified-Since`` is consulted only when ``If-None-Match`` is
absent.
"""

from __future__ import annotations

from datetime import UTC, datetime
from email.utils import format_datetime, parsedate_to_datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastapi import Request, Response


def etag_for(mtime_ns: int) -> str:
    """Build a weak ETag from a filesystem ``mtime_ns``.

    Weak (``W/``) because we're signalling semantic equivalence, not byte
    equality — the same state can serialize to slightly different bytes
    (key ordering, whitespace) under edge cases, and clients must not rely
    on strong equality here.
    """
    return f'W/"{mtime_ns}"'


def last_modified_for(mtime_ns: int) -> str:
    """Build an RFC 7231 HTTP-date from ``mtime_ns``.

    HTTP-date is seconds-resolution; sub-second state changes that happen
    to land in the same second will still be caught by the ETag path.
    """
    seconds = mtime_ns / 1_000_000_000
    dt = datetime.fromtimestamp(seconds, tz=UTC)
    return format_datetime(dt, usegmt=True)


def is_fresh(request: Request, mtime_ns: int) -> bool:
    """Return True iff the client already has a fresh copy.

    Implements the RFC 7232 precedence:
      1. ``If-None-Match`` wins when present.
      2. ``If-Modified-Since`` consulted only as fallback.

    The returned value tells the caller to emit ``304 Not Modified``.
    """
    inm = request.headers.get("if-none-match")
    if inm:
        expected = etag_for(mtime_ns)
        # Clients may send a list ("W/\"1\", W/\"2\"") — trim and match any.
        return any(token.strip() == expected for token in inm.split(","))

    ims = request.headers.get("if-modified-since")
    if ims:
        try:
            client_dt = parsedate_to_datetime(ims)
        except (TypeError, ValueError):
            return False
        if client_dt is None:
            return False
        if client_dt.tzinfo is None:
            client_dt = client_dt.replace(tzinfo=UTC)
        # HTTP-date is seconds-resolution — floor our mtime before comparing.
        server_seconds = mtime_ns // 1_000_000_000
        client_seconds = int(client_dt.timestamp())
        return client_seconds >= server_seconds

    return False


def apply_cache_headers(response: Response, mtime_ns: int) -> None:
    """Attach ``ETag`` + ``Last-Modified`` to an outgoing response.

    ``Cache-Control: no-cache`` forces the client to revalidate on every
    request (which is what we want — polling) rather than silently serving
    from local cache without asking us. The ETag / Last-Modified then make
    that revalidation cheap (304 with empty body).
    """
    response.headers["ETag"] = etag_for(mtime_ns)
    response.headers["Last-Modified"] = last_modified_for(mtime_ns)
    response.headers["Cache-Control"] = "no-cache"


__all__ = [
    "apply_cache_headers",
    "etag_for",
    "is_fresh",
    "last_modified_for",
]
