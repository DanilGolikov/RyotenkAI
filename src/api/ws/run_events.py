"""WebSocket relay: pod-runner event stream → frontend.

PR2 of the trainer-log-file plan. Frontend cannot open SSH tunnels
itself, so the Mac control plane proxies the runner's WebSocket
event stream (``/api/v1/jobs/<job>/events``) into a Mac-side
WebSocket the browser CAN connect to.

Frame protocol (server → client):

* ``{"type": "init", "since": int, "phase": "catchup"|"live"}`` —
  sent right after WS accept; tells the client the cursor we're
  starting from and whether the file mirror has any backlog.
* ``{"type": "event", "event": {...}}`` — one runner event. The
  ``event`` object is identical to what the runner publishes
  (``offset, ts, kind, payload``).
* ``{"type": "eof", "reason": str}`` — server-initiated end of
  stream (run terminal, runner unreachable, mirror done with no
  live source).

Close codes:

* ``1000`` — clean close (run reached terminal state).
* ``4404`` — run / attempt dir not found on disk.
* ``4410`` — runner reported ``ReplayTruncatedError`` (offset gone).
* ``4503`` — SSH tunnel could not be opened.
* ``1011`` — internal error.

Two phases:

1. **Catchup** — read ``events_mirror.jsonl`` from ``since=`` to EOF,
   send each line as a frame. No SSH involvement. This is what makes
   "open ``/runs/<id>/live`` an hour after the run finished"
   work — the pod is gone but the mirror lives on the Mac.

2. **Live** — if the run looks like it's still active (job_submission
   exists + last mirror event is not terminal), open an SSH tunnel
   + JobClient, subscribe with ``since=last_seen+1`` and relay each
   event through the WS. The mirror keeps being written by the
   pipeline-side TrainingMonitor; we don't write it from here to
   avoid double writes.

This endpoint is read-only — frontend never sends frames; we ignore
client messages.
"""

from __future__ import annotations

import json
from pathlib import Path  # noqa: TC003 — runtime-needed for FastAPI Depends signature
from typing import Any

from fastapi import APIRouter, Depends, Query, WebSocket, WebSocketDisconnect

from src.api.clients.job_client import (
    JobClientError,
    JobNotFoundError,
    ReplayTruncatedError,
)
from src.api.dependencies import resolve_run_dir
from src.pipeline.stages.managers.event_mirror import EventMirrorWriter
from src.pipeline.state.job_submission import (
    JobSubmissionLoadError,
    load_job_submission,
)
from src.utils.logger import logger

router = APIRouter()


# Close codes (4xxx are application-defined, mirror runner conventions
# from src/runner/api/jobs.py and src/api/clients/job_client.py).
_CODE_CLEAN = 1000
_CODE_INTERNAL = 1011
_CODE_NOT_FOUND = 4404
_CODE_TRUNCATED = 4410
_CODE_TUNNEL_FAILED = 4503

# Catchup chunking — yield to the event loop every N events so a
# big mirror replay doesn't starve other coroutines / WS pongs.
_CATCHUP_YIELD_EVERY = 200

# Terminal kinds — once we see one, the run is done. We close the
# WS cleanly without trying to subscribe to a (now non-existent)
# live runner.
_TERMINAL_KINDS = frozenset({"trainer_exited"})


def _attempt_dir_for(run_dir: Path, attempt_no: int) -> Path:
    return run_dir / "attempts" / f"attempt_{attempt_no}"


def _mirror_path(attempt_dir: Path) -> Path:
    return (
        attempt_dir
        / EventMirrorWriter.EVENTS_DIR_NAME
        / EventMirrorWriter.MIRROR_FILE_NAME
    )


@router.websocket("/runs/{run_id:path}/attempts/{attempt_no}/events/stream")
async def stream_run_events(
    websocket: WebSocket,
    attempt_no: int,
    since: int = Query(0, ge=0),
    run_dir: Path = Depends(resolve_run_dir),
) -> None:
    """Relay endpoint: catch up from mirror, then live-tail via runner.

    See module docstring for protocol. ``since`` is the offset cursor
    the client passes; on reconnect it should be ``last_seen+1``.
    """
    attempt_dir = _attempt_dir_for(run_dir, attempt_no)
    if not attempt_dir.is_dir():
        await websocket.close(code=_CODE_NOT_FOUND, reason="attempt_not_found")
        return

    await websocket.accept()
    await websocket.send_json({"type": "init", "since": since, "phase": "catchup"})

    # Phase 1: catch up from mirror.
    try:
        last_seen, terminal_seen = await _replay_mirror(
            websocket, _mirror_path(attempt_dir), since=since,
        )
    except WebSocketDisconnect:
        return
    except Exception as exc:
        logger.debug("[run_events.ws] mirror replay failed: %s", exc)
        await _safe_close(websocket, code=_CODE_INTERNAL, reason="mirror_read_error")
        return

    if terminal_seen:
        # Mirror already contains a terminal — nothing to do live.
        await _safe_send(websocket, {"type": "eof", "reason": "terminal_in_mirror"})
        await _safe_close(websocket, code=_CODE_CLEAN, reason="run_terminal")
        return

    # Phase 2: try to attach to a live runner. If submission file is
    # absent the run never reached the launcher (or ran to completion
    # before the file became readable) — close cleanly. EOF is the
    # natural signal.
    try:
        submission = load_job_submission(attempt_dir)
    except JobSubmissionLoadError:
        await _safe_send(websocket, {"type": "eof", "reason": "no_live_source"})
        await _safe_close(websocket, code=_CODE_CLEAN, reason="no_submission")
        return

    next_since = last_seen + 1 if last_seen >= since else since
    try:
        await _live_relay(websocket, submission, next_since=next_since)
    except WebSocketDisconnect:
        return


# ---------------------------------------------------------------------------
# Phase 1 — mirror replay
# ---------------------------------------------------------------------------


async def _replay_mirror(
    websocket: WebSocket, mirror_path: Path, *, since: int,
) -> tuple[int, bool]:
    """Stream the mirror file to the client.

    Returns ``(last_seen_offset, terminal_seen)``. ``last_seen_offset``
    is ``since - 1`` if no events matched, so the caller's
    ``last_seen + 1`` arithmetic still produces the correct
    "subscribe from this offset" value.
    """
    last_seen = since - 1
    terminal_seen = False
    if not mirror_path.exists():
        return last_seen, terminal_seen

    sent_count = 0
    with mirror_path.open(encoding="utf-8") as fp:
        for raw_line in fp:
            line = raw_line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict):
                continue
            offset = event.get("offset")
            if not isinstance(offset, int) or offset < since:
                continue
            await websocket.send_json({"type": "event", "event": event})
            last_seen = offset
            sent_count += 1
            if event.get("kind") in _TERMINAL_KINDS:
                terminal_seen = True
            if sent_count % _CATCHUP_YIELD_EVERY == 0:
                # Cooperative yield: let WS pongs / cancel signals run.
                import asyncio

                await asyncio.sleep(0)

    return last_seen, terminal_seen


# ---------------------------------------------------------------------------
# Phase 2 — live relay via SSH tunnel + JobClient
# ---------------------------------------------------------------------------


async def _live_relay(
    websocket: WebSocket,
    submission: Any,
    *,
    next_since: int,
) -> None:
    """Open SSH tunnel + JobClient, subscribe, relay until terminal/disconnect.

    On any runner-side error we close the WebSocket with the matching
    code and return. The tunnel and client are torn down regardless.
    """
    # Lazy imports — avoid forcing the whole runner-client stack on
    # every API process if no one ever opens this endpoint.
    from src.api.clients.job_client import JobClient
    from src.api.services.tunnel_service import (
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
    try:
        await tunnel.open()
    except Exception as exc:
        logger.debug("[run_events.ws] tunnel.open failed: %s", exc)
        await _safe_close(
            websocket, code=_CODE_TUNNEL_FAILED, reason="tunnel_open_failed",
        )
        return

    await _safe_send(
        websocket, {"type": "init", "since": next_since, "phase": "live"},
    )

    client = JobClient(tunnel.base_url)
    try:
        try:
            async for event in client.subscribe_events(
                submission.job_id, since=next_since,
            ):
                await websocket.send_json({"type": "event", "event": event})
                if event.get("kind") in _TERMINAL_KINDS:
                    await _safe_send(
                        websocket, {"type": "eof", "reason": "trainer_exited"},
                    )
                    await _safe_close(websocket, code=_CODE_CLEAN, reason="terminal")
                    return
            # Stream ended without a terminal kind — runner closed for
            # some other reason. Tell the client and close cleanly.
            await _safe_send(websocket, {"type": "eof", "reason": "runner_eof"})
            await _safe_close(websocket, code=_CODE_CLEAN, reason="runner_eof")
        except JobNotFoundError:
            await _safe_close(websocket, code=_CODE_NOT_FOUND, reason="job_not_found")
        except ReplayTruncatedError:
            await _safe_close(
                websocket, code=_CODE_TRUNCATED, reason="replay_truncated",
            )
        except JobClientError as exc:
            logger.debug("[run_events.ws] JobClient error: %s", exc)
            await _safe_close(
                websocket, code=_CODE_INTERNAL, reason="runner_client_error",
            )
    finally:
        try:
            await client.aclose()
        finally:
            await tunnel.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _safe_send(websocket: WebSocket, payload: dict[str, Any]) -> None:
    """``send_json`` that swallows disconnect-during-send. The caller
    is expected to also handle :class:`WebSocketDisconnect` — this is
    just defence-in-depth so a write race during teardown doesn't
    bubble up as a 500."""
    try:
        await websocket.send_json(payload)
    except (RuntimeError, WebSocketDisconnect):
        return


async def _safe_close(websocket: WebSocket, *, code: int, reason: str) -> None:
    try:
        await websocket.close(code=code, reason=reason)
    except RuntimeError:
        # Already closed.
        return
