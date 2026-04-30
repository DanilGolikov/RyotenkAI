"""REST endpoint for the Mac-side runner event mirror.

PR2 of the trainer-log-file plan. The pipeline's
:class:`TrainingMonitor` writes every runner event into
``runs/<id>/attempts/<n>/events/events_mirror.jsonl`` via
:class:`EventMirrorWriter`. This router exposes that file as a
``since=N&limit=K&kind=...`` paginated REST endpoint so:

* Frontend can do a cold-replay of past events without an SSH tunnel
  (useful when the run has finished and the pod is gone).
* CLI tools can ``curl`` the events stream for ad-hoc analysis.
* Reconnect / catch-up paths in the WebSocket relay
  (:mod:`src.api.ws.run_events`) reuse this same reader.

The mirror file format matches the pod-side
:class:`src.runner.event_journal.JournalRecord`:
``{"v": int, "offset": int, "ts": str, "kind": str, "payload": dict}``.
We don't validate the schema strictly — corrupted lines are skipped
with a debug log so a single bad write doesn't take down the
endpoint.
"""

from __future__ import annotations

import json
from pathlib import Path  # noqa: TC003 — runtime-needed for FastAPI Depends signature
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query

from src.api.dependencies import resolve_run_dir
from src.pipeline.stages.managers.event_mirror import EventMirrorWriter
from src.utils.logger import logger

router = APIRouter(prefix="/runs/{run_id:path}", tags=["events"])


def _mirror_path_for_attempt(run_dir: Path, attempt_no: int) -> Path:
    """Resolve ``<run_dir>/attempts/attempt_<n>/events/events_mirror.jsonl``.

    Mirrors the layout :class:`EventMirrorWriter` writes to. Fails
    with HTTP 404 if the attempt directory doesn't exist; an empty
    or missing mirror file is treated as "no events yet" (200 with
    empty list) so the frontend can poll a fresh run that hasn't
    produced any events yet.
    """
    attempt_dir = run_dir / "attempts" / f"attempt_{attempt_no}"
    if not attempt_dir.is_dir():
        raise HTTPException(
            status_code=404,
            detail={
                "code": "attempt_not_found",
                "message": f"attempt_{attempt_no} not found under {run_dir.name}",
            },
        )
    return (
        attempt_dir
        / EventMirrorWriter.EVENTS_DIR_NAME
        / EventMirrorWriter.MIRROR_FILE_NAME
    )


def _read_events(
    mirror_path: Path,
    *,
    since: int,
    limit: int,
    kind: str | None,
) -> tuple[list[dict[str, Any]], int]:
    """Stream-read the mirror file and return events with offset >= since.

    Returns ``(events, total_scanned)`` — ``total_scanned`` is the
    number of journal records the file contained at read time, useful
    for a UI progress meter / sanity check ("server saw N events
    total"). We slice ``limit`` from the head of the filtered list,
    NOT from raw line count, so ``kind=trainer_log&limit=200`` returns
    200 trainer-log events even if there are 5000 unrelated events
    in between.

    Corrupt lines (invalid JSON, missing offset) are skipped with a
    debug log. The mirror is append-only; partial writes are
    extremely rare and a single bad line should not 500 the endpoint.
    """
    if not mirror_path.exists():
        return [], 0

    matched: list[dict[str, Any]] = []
    scanned = 0
    try:
        with mirror_path.open(encoding="utf-8") as fp:
            for raw_line in fp:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    logger.debug(
                        "[run_events] skipping malformed line in %s",
                        mirror_path,
                    )
                    continue
                if not isinstance(event, dict):
                    continue
                offset = event.get("offset")
                if not isinstance(offset, int):
                    continue
                scanned += 1
                if offset < since:
                    continue
                if kind is not None and event.get("kind") != kind:
                    continue
                if len(matched) >= limit:
                    # Hard cap. ``scanned`` counts every offset-bearing
                    # record seen so the response's ``total_scanned``
                    # is still meaningful, but appending more would
                    # break the strict ``len(events) <= limit``
                    # contract clients rely on for pagination.
                    break
                matched.append(event)
    except OSError as exc:
        # Mirror is being rotated or transient FS error; surface 503
        # so the caller retries rather than getting a partial empty
        # response.
        raise HTTPException(
            status_code=503,
            detail={
                "code": "mirror_read_error",
                "message": str(exc),
            },
        ) from exc

    return matched, scanned


@router.get("/attempts/{attempt_no}/events", summary="Cold-replay event mirror")
def get_run_events(
    attempt_no: int,
    since: int = Query(0, ge=0, description="Return events with offset >= since."),
    limit: int = Query(200, ge=1, le=2000, description="Max events to return."),
    kind: str | None = Query(
        None,
        description="Optional event-kind filter (e.g. 'trainer_log', 'health_snapshot').",
    ),
    run_dir: Path = Depends(resolve_run_dir),
) -> dict[str, Any]:
    """Return a slice of the ``events_mirror.jsonl`` file.

    Response shape::

        {
            "events": [{...EventResponse-like...}, ...],
            "next_since": int,   # cursor for the next request
            "total_scanned": int # records seen so far in the mirror
        }

    Pagination contract: pass ``next_since`` from the previous
    response back as ``since=`` to fetch the next page. When the
    response contains fewer events than ``limit`` and the events
    list ends at the current tail, ``next_since`` equals the
    largest ``offset + 1`` we returned (or the original ``since``
    if no events matched, so polling is monotonic).
    """
    mirror_path = _mirror_path_for_attempt(run_dir, attempt_no)
    events, scanned = _read_events(
        mirror_path, since=since, limit=limit, kind=kind,
    )
    next_since = (
        max(int(e["offset"]) for e in events) + 1
        if events
        else since
    )
    return {
        "events": events,
        "next_since": next_since,
        "total_scanned": scanned,
    }
