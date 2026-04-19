from __future__ import annotations

import time
from pathlib import Path

import pytest


def _log_path(runs_dir: Path, run_id: str) -> Path:
    return runs_dir / run_id / "attempts" / "attempt_1" / "pipeline.log"


def test_ws_log_stream_delivers_new_lines(client, seed_completed_run) -> None:
    run_dir = seed_completed_run("run_ws_1")
    log = _log_path(run_dir.parent, "run_ws_1")

    with client.websocket_connect("/api/v1/runs/run_ws_1/attempts/1/logs/stream?file=pipeline.log&from_offset=0") as ws:
        init = ws.receive_json()
        assert init["type"] == "init"
        assert init["offset"] == 0

        # First poll returns existing content.
        deadline = time.time() + 2.0
        received_chunk = False
        while time.time() < deadline:
            msg = ws.receive_json()
            if msg["type"] == "chunk":
                received_chunk = True
                assert any("pipeline start" in line for line in msg["lines"])
                break
            if msg["type"] == "eof":
                break
        assert received_chunk


def test_ws_rejects_unknown_file(client, seed_completed_run) -> None:
    seed_completed_run("run_ws_2")
    from starlette.websockets import WebSocketDisconnect

    with pytest.raises(WebSocketDisconnect) as exc_info:
        with client.websocket_connect(
            "/api/v1/runs/run_ws_2/attempts/1/logs/stream?file=evil.sh"
        ):
            pass
    assert exc_info.value.code == 4400
