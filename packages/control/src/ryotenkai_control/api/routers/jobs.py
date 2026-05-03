"""Phase 7.2 — control-plane proxy to the in-pod runner.

The Web UI cannot open SSH tunnels itself (sandbox), so for live
training views the browser hits these endpoints, the FastAPI
control plane opens a short-lived SSH tunnel + JobClient, fetches
state from the runner, and returns plain JSON.

Endpoints, all read-only except ``stop``:

- ``GET  /api/v1/runs/{run_id}/job/status``       FSM snapshot.
- ``GET  /api/v1/runs/{run_id}/job/events?since=N&limit=N``
                                                  Buffered event slice.
- ``GET  /api/v1/runs/{run_id}/job/logs?since=N&limit=N&stream=...``
                                                  Trainer stdout/stderr —
                                                  filtered slice of
                                                  ``trainer_log`` events.
- ``POST /api/v1/runs/{run_id}/job/stop``         Graceful stop request.

Each call opens and closes a fresh tunnel. That's an extra ~1 s of
latency per poll vs. holding a long-lived connection — acceptable
for an MVP polling UI (browser polls every 2 s; tunnel reuse can
land in a follow-up if it becomes a bottleneck).

Source of truth: ``attempts/<n>/job_submission.json`` written by
:class:`TrainingLauncher` (Phase 6.3a). If the file is missing we
return 404 — the run never reached the launcher (or pre-dates the
new flow).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query

from ryotenkai_control.api.dependencies import resolve_run_dir

router = APIRouter(prefix="/runs/{run_id:path}/job", tags=["job"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _latest_attempt_dir(run_dir: Path, attempt: int | None) -> Path:
    """Pick which attempt's submission file to read.

    Mirrors the CLI's :func:`src.cli.commands.job._resolve_attempt_dir`
    but raises HTTP errors instead of calling ``die``."""
    runs = sorted(
        (run_dir / "attempts").glob("attempt_*"),
        key=lambda p: int(p.name.split("_")[-1])
        if p.name.split("_")[-1].isdigit()
        else 0,
    )
    if not runs:
        raise HTTPException(
            status_code=404,
            detail={
                "code": "no_attempts",
                "message": f"no attempts/ subdirectories under {run_dir}",
            },
        )
    if attempt is None:
        return runs[-1]
    target = run_dir / "attempts" / f"attempt_{attempt}"
    if not target.is_dir():
        raise HTTPException(
            status_code=404,
            detail={
                "code": "attempt_not_found",
                "message": f"attempt_{attempt} not found",
                "available": [p.name for p in runs],
            },
        )
    return target


def _load_submission(attempt_dir: Path) -> Any:
    from ryotenkai_control.pipeline.state.job_submission import (
        JobSubmissionLoadError,
        load_job_submission,
    )

    try:
        return load_job_submission(attempt_dir)
    except JobSubmissionLoadError as exc:
        raise HTTPException(
            status_code=404,
            detail={
                "code": "job_submission_missing",
                "message": str(exc),
            },
        ) from exc


async def _with_runner(submission, fn):  # type: ignore[no-untyped-def]
    """Open SSH tunnel + JobClient, run ``fn(client, job_id)``,
    tear everything down. Mirror of the CLI's helper — duplicated
    here rather than imported to keep the API router free of CLI
    deps."""
    from ryotenkai_shared.utils.clients.job_client import JobClient
    from ryotenkai_shared.utils.clients.ssh_tunnel import (
        SSHTunnelEndpoint,
        SSHTunnelManager,
    )

    endpoint = SSHTunnelEndpoint(
        host=submission.ssh_host,
        port=submission.ssh_port,
        username=submission.ssh_username,
        key_path=submission.ssh_key_path,
    )
    tunnel = SSHTunnelManager(endpoint)
    await tunnel.open()
    try:
        client = JobClient(tunnel.base_url)
        try:
            return await fn(client, submission.job_id)
        finally:
            await client.aclose()
    finally:
        await tunnel.close()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/status")
async def get_status(
    attempt: int | None = Query(default=None, ge=1),
    run_dir: Path = Depends(resolve_run_dir),
) -> dict[str, Any]:
    """Return submission metadata + current FSM snapshot."""
    submission = _load_submission(_latest_attempt_dir(run_dir, attempt))

    async def _go(client, job_id):  # type: ignore[no-untyped-def]
        return await client.get_status(job_id)

    try:
        snapshot = await _with_runner(submission, _go)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=502,
            detail={"code": "runner_unreachable", "message": str(exc)},
        ) from exc
    return {"submission": submission.to_dict(), "snapshot": snapshot}


@router.get("/events")
async def get_events(
    attempt: int | None = Query(default=None, ge=1),
    since: int = Query(default=0, ge=0),
    limit: int = Query(default=200, ge=1, le=2000),
    run_dir: Path = Depends(resolve_run_dir),
) -> dict[str, Any]:
    """Return up to ``limit`` events with ``offset >= since``.

    Bounded — the UI polls every couple of seconds and only needs
    the slice since its last cursor; capping at 2000 prevents a
    pathological ``since=0`` request from saturating the WS replay
    when the buffer is full.
    """
    submission = _load_submission(_latest_attempt_dir(run_dir, attempt))

    async def _go(client, job_id):  # type: ignore[no-untyped-def]
        events: list[dict[str, Any]] = []
        async for event in client.subscribe_events(
            job_id, since=since, max_reconnect_attempts=0,
        ):
            events.append(event)
            if len(events) >= limit:
                break
        return events

    try:
        events = await _with_runner(submission, _go)
    except Exception as exc:  # noqa: BLE001
        # ``subscribe_events`` is best-effort here — return whatever
        # we managed to capture before the WS closed. UI will retry
        # on next poll.
        return {
            "events": [],
            "next_since": since,
            "error": {"code": "stream_failed", "message": str(exc)},
        }

    next_since = (
        max(int(e.get("offset", since)) for e in events) + 1
        if events
        else since
    )
    return {"events": events, "next_since": next_since}


_DEFAULT_LOG_STREAMS: tuple[str, ...] = ("stdout", "stderr")


@router.get("/logs")
async def get_logs(
    attempt: int | None = Query(default=None, ge=1),
    since: int = Query(default=0, ge=0),
    limit: int = Query(default=200, ge=1, le=2000),
    stream: list[str] = Query(
        default=list(_DEFAULT_LOG_STREAMS),
        description="Filter to ``stdout`` and/or ``stderr``.",
    ),
    run_dir: Path = Depends(resolve_run_dir),
) -> dict[str, Any]:
    """Return up to ``limit`` ``trainer_log`` events with ``offset >= since``.

    The trainer subprocess pipes its stdout / stderr through the
    runner's :class:`Supervisor`, which emits each line as a
    ``trainer_log`` event with ``payload={"kind": "stdout"|"stderr",
    "line": "..."}``. This endpoint surfaces those events as the
    file-tail fallback for cases when the structured event callback
    (``RunnerEventCallback``) self-disabled — the supervisor pump
    has no opt-out, so every line of trainer output is observable.

    Same partial-failure shape as ``/events``: a transient WebSocket
    failure mid-stream returns whatever was captured plus a
    structured ``error`` so the polling UI keeps the cursor and
    retries on the next tick.
    """
    submission = _load_submission(_latest_attempt_dir(run_dir, attempt))
    streams = {s for s in stream if s in _DEFAULT_LOG_STREAMS}
    if not streams:
        # Empty stream filter is a misconfiguration — refuse rather
        # than silently return no logs forever.
        raise HTTPException(
            status_code=422,
            detail={
                "code": "invalid_stream_filter",
                "message": (
                    "stream must contain one or more of "
                    f"{_DEFAULT_LOG_STREAMS}"
                ),
            },
        )

    async def _go(client, job_id):  # type: ignore[no-untyped-def]
        events: list[dict[str, Any]] = []
        async for event in client.subscribe_events(
            job_id, since=since, max_reconnect_attempts=0,
        ):
            if event.get("kind") != "trainer_log":
                continue
            payload = event.get("payload") or {}
            if payload.get("kind") not in streams:
                continue
            events.append(event)
            if len(events) >= limit:
                break
        return events

    try:
        events = await _with_runner(submission, _go)
    except Exception as exc:  # noqa: BLE001
        return {
            "events": [],
            "next_since": since,
            "error": {"code": "stream_failed", "message": str(exc)},
        }

    next_since = (
        max(int(e.get("offset", since)) for e in events) + 1
        if events
        else since
    )
    return {"events": events, "next_since": next_since}


@router.post("/stop", status_code=202)
async def stop_job(
    attempt: int | None = Query(default=None, ge=1),
    grace: float | None = Query(default=None, ge=0),
    run_dir: Path = Depends(resolve_run_dir),
) -> dict[str, Any]:
    """Forward a graceful-stop request to the runner."""
    submission = _load_submission(_latest_attempt_dir(run_dir, attempt))

    async def _go(client, job_id):  # type: ignore[no-untyped-def]
        return await client.request_stop(job_id, grace_seconds=grace)

    try:
        return await _with_runner(submission, _go)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=502,
            detail={"code": "runner_unreachable", "message": str(exc)},
        ) from exc
