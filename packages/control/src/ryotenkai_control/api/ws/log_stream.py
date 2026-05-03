from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi import APIRouter, Depends, Query, WebSocket, WebSocketDisconnect

from src.api.config import ApiSettings
from src.api.dependencies import get_settings, resolve_run_dir
from src.api.services.log_service import resolve_log_path
from src.api.ws.live_tail import LiveLogTail
from src.pipeline.run_queries import effective_pipeline_status
from src.pipeline.state import PipelineStateLoadError, PipelineStateStore

router = APIRouter()


_TERMINAL_STATUSES = {"completed", "failed", "interrupted", "stale", "skipped"}


@router.websocket("/runs/{run_id:path}/attempts/{attempt_no}/logs/stream")
async def stream_logs(
    websocket: WebSocket,
    attempt_no: int,
    file: str = Query(default="pipeline.log"),
    from_offset: int | None = Query(default=None),
    run_dir: Path = Depends(resolve_run_dir),
    settings: ApiSettings = Depends(get_settings),
) -> None:
    try:
        log_path = resolve_log_path(run_dir, attempt_no, file)
    except ValueError:
        await websocket.close(code=4400, reason="unsupported_file")
        return

    await websocket.accept()
    tail = LiveLogTail()

    # Resolve starting offset. Default: seek to EOF so client only gets new lines.
    if from_offset is not None:
        tail.path = log_path
        tail.offset = max(0, int(from_offset))
    else:
        tail.path = log_path
        if log_path.exists():
            tail.offset = log_path.stat().st_size
        else:
            tail.offset = 0

    await websocket.send_json({"type": "init", "file": file, "offset": tail.offset})

    state_store = PipelineStateStore(run_dir)
    last_state_mtime = 0.0
    last_status: str | None = None
    poll_seconds = max(0.05, settings.log_stream_poll_interval_ms / 1000.0)

    try:
        while True:
            # 1. Emit any new log lines.
            try:
                new_lines = await asyncio.to_thread(tail.read_new_lines)
            except OSError:
                new_lines = []
            if new_lines:
                await websocket.send_json({"type": "chunk", "lines": new_lines, "offset": tail.offset})

            # 2. Watch pipeline_state.json mtime for stage-status transitions.
            status, mtime = await asyncio.to_thread(_read_state_status, state_store)
            if mtime != last_state_mtime or status != last_status:
                last_state_mtime = mtime
                last_status = status
                if status is not None:
                    await websocket.send_json({"type": "state", "status": status})

            if status in _TERMINAL_STATUSES and not new_lines:
                # Drain a tiny bit more then close cleanly.
                await asyncio.sleep(poll_seconds)
                try:
                    trailing = await asyncio.to_thread(tail.read_new_lines)
                except OSError:
                    trailing = []
                if trailing:
                    await websocket.send_json({"type": "chunk", "lines": trailing, "offset": tail.offset})
                await websocket.send_json({"type": "eof"})
                return

            await asyncio.sleep(poll_seconds)
    except WebSocketDisconnect:
        return


def _read_state_status(store: PipelineStateStore) -> tuple[str | None, float]:
    try:
        mtime = store.state_path.stat().st_mtime
    except OSError:
        return None, 0.0
    try:
        state = store.load()
    except PipelineStateLoadError:
        return None, mtime
    return effective_pipeline_status(state), mtime
