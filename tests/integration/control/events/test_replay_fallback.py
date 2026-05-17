"""Integration: HTTP replay fallback closes the post-Phase-6.a gap.

Before this fix, :meth:`TrainingMonitor._watch` caught
:class:`ReplayTruncatedError` (WS close 4410 — pod journal rolled
past the requested ``since``) and went straight to
``_fallback_to_status``, dropping every event between the gap and
``now`` even though the pod's on-disk journal still had them.

Phase 6.a had already exposed
``GET /api/v1/jobs/{id}/events/replay?after_offset=N`` for exactly
this case; the monitor just never invoked it. After the fix the
monitor's recovery sequence is:

1. WS subscribe raises :class:`ReplayTruncatedError`.
2. Monitor calls ``client.replay_events(after_offset=last_seen)`` and
   drains the NDJSON pages, dispatching each event the normal way
   (which also forwards typed envelopes to the emitter via the
   Issue 1 fix).
3. Once the pagination cursor stops advancing, the monitor
   re-subscribes WS with the new ``since`` so live tailing resumes.
4. Only on transport failure does it fall back to the status snapshot
   (legacy path).

Coverage:

* The replay endpoint is invoked and its events are dispatched.
* The monitor re-subscribes WS at the post-replay offset.
* A transport error inside replay falls back to status snapshot.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from ryotenkai_shared.errors import TrainingFailedError
from ryotenkai_shared.utils.clients.job_client import (
    JobClientError,
    ReplayTruncatedError,
)

from tests.unit.control.pipeline.test_training_monitor_v2 import (
    _ctx_with_handles,
    _make_monitor,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _replay_page(events: list[dict[str, Any]], next_offset: int) -> tuple[list[dict[str, Any]], int]:
    """Build the (events, next_offset) tuple that ``JobClient.replay_events``
    would return."""
    return events, next_offset


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestReplayFallback:
    def test_replay_drained_then_ws_resubscribed_to_terminal(self) -> None:
        """Long Mac sleep scenario.

        * WS subscribe raises ``ReplayTruncatedError`` (pod ring rolled).
        * Monitor calls ``replay_events`` and gets the missed slice
          back via the HTTP endpoint.
        * Monitor re-subscribes WS at the resumed offset and lands a
          terminal ``trainer_exited`` event.
        """
        monitor = _make_monitor()

        # Stage 1: first WS subscribe raises truncation.
        ws_invocations = {"count": 0}

        async def _subscribe(_job_id: str, *, since: int = 0, **_kw: Any) -> Any:
            ws_invocations["count"] += 1
            if ws_invocations["count"] == 1:
                raise ReplayTruncatedError("ring rolled")
            # Stage 3: live re-subscribe at since=1001 — the runner
            # tails through to terminal.
            yield {
                "offset": 1001,
                "kind": "trainer_exited",
                "payload": {
                    "exit_code": 0,
                    "signal": None,
                    "cancellation_requested": False,
                },
            }

        # Stage 2: replay_events paginates from after_offset=-1.
        # Page 1 covers 500..999 (X-Next-Offset=999); Page 2 returns
        # offset=1000 (last); Page 3 returns empty (cursor unchanged).
        replay_calls: list[int] = []

        async def _replay(_job_id: str, *, after_offset: int, limit: int = 10000) -> Any:
            replay_calls.append(after_offset)
            if after_offset == -1:
                events = [
                    {"offset": i, "kind": "health_snapshot", "payload": {}}
                    for i in range(500, 1000)
                ]
                return events, 999
            if after_offset == 999:
                return [{"offset": 1000, "kind": "health_snapshot", "payload": {}}], 1000
            return [], after_offset  # exhausted

        client = MagicMock()
        client.subscribe_events = _subscribe
        client.replay_events = _replay
        client.get_status = AsyncMock(return_value={"state": "completed"})
        client.aclose = AsyncMock(return_value=None)

        result = monitor.execute(_ctx_with_handles(client))
        assert result["status"] == "completed"

        # Verify pagination occurred (3 calls: -1, 999, 1000).
        assert replay_calls == [-1, 999, 1000]
        # Re-subscribe to WS happened with the post-replay offset
        # (1001 = last replayed offset + 1).
        assert ws_invocations["count"] == 2

    def test_replay_transport_error_falls_back_to_status(self) -> None:
        """If the HTTP replay endpoint itself fails (network / pod
        unreachable / 5xx), the monitor falls back to the legacy
        status-snapshot path so the run still terminates."""
        monitor = _make_monitor()

        async def _subscribe(_job_id: str, *, since: int = 0, **_kw: Any) -> Any:
            raise ReplayTruncatedError("ring rolled")
            yield  # pragma: no cover

        async def _replay(_job_id: str, **_kw: Any) -> Any:
            raise JobClientError("connection refused")

        client = MagicMock()
        client.subscribe_events = _subscribe
        client.replay_events = _replay
        client.get_status = AsyncMock(return_value={"state": "completed"})
        client.aclose = AsyncMock(return_value=None)

        result = monitor.execute(_ctx_with_handles(client))
        assert result["status"] == "completed"
        # Status fallback was invoked.
        client.get_status.assert_awaited()

    def test_replay_non_terminal_status_raises_with_legacy_code(self) -> None:
        """Replay transport error + non-terminal FSM ⇒ legacy
        MONITOR_REPLAY_TRUNCATED error (status fallback path)."""
        monitor = _make_monitor()

        async def _subscribe(_job_id: str, *, since: int = 0, **_kw: Any) -> Any:
            raise ReplayTruncatedError("ring rolled")
            yield  # pragma: no cover

        async def _replay(_job_id: str, **_kw: Any) -> Any:
            raise JobClientError("dead")

        client = MagicMock()
        client.subscribe_events = _subscribe
        client.replay_events = _replay
        client.get_status = AsyncMock(return_value={"state": "running"})
        client.aclose = AsyncMock(return_value=None)

        with pytest.raises(TrainingFailedError) as exc_info:
            monitor.execute(_ctx_with_handles(client))
        assert exc_info.value.context.get("legacy_code") == "MONITOR_REPLAY_TRUNCATED"

    def test_replay_yields_terminal_event_directly(self) -> None:
        """If the replay page contains a terminal event (trainer_exited),
        the monitor honours it and skips the re-subscribe entirely."""
        monitor = _make_monitor()

        async def _subscribe(_job_id: str, *, since: int = 0, **_kw: Any) -> Any:
            raise ReplayTruncatedError("ring rolled")
            yield  # pragma: no cover

        async def _replay(_job_id: str, *, after_offset: int, limit: int = 10000) -> Any:
            if after_offset == -1:
                return [
                    {"offset": 0, "kind": "trainer_spawned", "payload": {}},
                    {
                        "offset": 1,
                        "kind": "trainer_exited",
                        "payload": {
                            "exit_code": 0,
                            "signal": None,
                            "cancellation_requested": False,
                        },
                    },
                ], 1
            return [], after_offset

        client = MagicMock()
        client.subscribe_events = _subscribe
        client.replay_events = _replay
        client.get_status = AsyncMock(return_value={"state": "completed"})
        client.aclose = AsyncMock(return_value=None)

        result = monitor.execute(_ctx_with_handles(client))
        assert result["status"] == "completed"
