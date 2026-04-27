"""Phase 11.E — :meth:`PodTerminator._check_heartbeat_with_retries`.

Pin the retry contract:
* First read True → no retries (fast path).
* First read False AND terminal=completed → up to N retries with
  T-second sleep between each.
* Recovery on any retry → switch to alive=True.
* All retries exhausted → alive=False, full attempts_used.
* Non-completed terminal states → no retries (preserves Phase 9
  semantics for cancelled / failed).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.runner.heartbeat import MacHeartbeat
from src.runner.pod_terminator import PodTerminator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_terminator(
    *,
    sleep: AsyncMock | None = None,
    heartbeat_retry_attempts: int = 3,
    heartbeat_retry_tick: float = 10.0,
) -> PodTerminator:
    return PodTerminator(
        sleep=sleep or AsyncMock(),
        heartbeat_retry_attempts=heartbeat_retry_attempts,
        heartbeat_retry_tick_seconds=heartbeat_retry_tick,
    )


def _heartbeat_with_alive_sequence(values: list[bool]) -> MacHeartbeat:
    """Build a fake heartbeat whose ``is_alive()`` cycles through
    the provided values."""
    hb = MacHeartbeat()
    iter_values = iter(values)
    hb.is_alive = lambda: next(iter_values, False)  # type: ignore[method-assign]
    hb.age_seconds = lambda: None  # type: ignore[method-assign]
    return hb


# ---------------------------------------------------------------------------
# 1. Positive — fast path no retries
# ---------------------------------------------------------------------------


class TestPositive:
    @pytest.mark.asyncio
    async def test_first_read_alive_skips_retry_loop(self) -> None:
        sleep = AsyncMock()
        terminator = _make_terminator(sleep=sleep)
        heartbeat = _heartbeat_with_alive_sequence([True])

        events: list[tuple[str, dict[str, Any]]] = []
        bus_publish = lambda kind, payload: events.append((kind, payload))  # noqa: E731

        alive, attempts = await terminator._check_heartbeat_with_retries(
            heartbeat=heartbeat,
            terminal_state="completed",
            bus_publish=bus_publish,
        )
        assert alive is True
        assert attempts == 0
        # No retry events emitted.
        assert all("retry" not in kind for kind, _ in events)
        # No sleep — fast path returns immediately.
        sleep.assert_not_awaited()


# ---------------------------------------------------------------------------
# 2. Recovery — heartbeat returns on Nth retry
# ---------------------------------------------------------------------------


class TestRecovery:
    @pytest.mark.asyncio
    async def test_heartbeat_recovers_on_second_attempt(self) -> None:
        sleep = AsyncMock()
        terminator = _make_terminator(sleep=sleep)
        # Sequence: first read False (triggers retry), retry 1 also
        # False, retry 2 True (recovery).
        heartbeat = _heartbeat_with_alive_sequence([False, False, True])

        events: list[tuple[str, dict[str, Any]]] = []
        alive, attempts = await terminator._check_heartbeat_with_retries(
            heartbeat=heartbeat,
            terminal_state="completed",
            bus_publish=lambda k, p: events.append((k, p)),
        )

        assert alive is True
        assert attempts == 2
        # Two sleeps before recovery.
        assert sleep.await_count == 2
        # Telemetry: started + recovered.
        kinds = [k for k, _ in events]
        assert "pod_terminal_heartbeat_retry_started" in kinds
        assert "pod_terminal_heartbeat_retry_recovered" in kinds
        # No exhausted event when recovery succeeds.
        assert "pod_terminal_heartbeat_retry_exhausted" not in kinds

    @pytest.mark.asyncio
    async def test_heartbeat_recovers_on_first_retry(self) -> None:
        sleep = AsyncMock()
        terminator = _make_terminator(sleep=sleep)
        heartbeat = _heartbeat_with_alive_sequence([False, True])

        alive, attempts = await terminator._check_heartbeat_with_retries(
            heartbeat=heartbeat,
            terminal_state="completed",
            bus_publish=lambda k, p: None,
        )
        assert alive is True
        assert attempts == 1


# ---------------------------------------------------------------------------
# 3. Exhausted — all retries fail
# ---------------------------------------------------------------------------


class TestExhausted:
    @pytest.mark.asyncio
    async def test_all_retries_fail(self) -> None:
        sleep = AsyncMock()
        terminator = _make_terminator(sleep=sleep, heartbeat_retry_attempts=3)
        heartbeat = _heartbeat_with_alive_sequence([False, False, False, False])

        events: list[tuple[str, dict[str, Any]]] = []
        alive, attempts = await terminator._check_heartbeat_with_retries(
            heartbeat=heartbeat,
            terminal_state="completed",
            bus_publish=lambda k, p: events.append((k, p)),
        )
        assert alive is False
        assert attempts == 3
        # All 3 retries waited.
        assert sleep.await_count == 3
        kinds = [k for k, _ in events]
        assert "pod_terminal_heartbeat_retry_started" in kinds
        assert "pod_terminal_heartbeat_retry_exhausted" in kinds


# ---------------------------------------------------------------------------
# 4. Non-completed — no retries
# ---------------------------------------------------------------------------


class TestNoRetryOnOtherStates:
    @pytest.mark.asyncio
    async def test_cancelled_skips_retries(self) -> None:
        # User-stop has well-defined outcome regardless of heartbeat;
        # retries pure waste of time.
        sleep = AsyncMock()
        terminator = _make_terminator(sleep=sleep)
        heartbeat = _heartbeat_with_alive_sequence([False])

        alive, attempts = await terminator._check_heartbeat_with_retries(
            heartbeat=heartbeat,
            terminal_state="cancelled",
            bus_publish=lambda k, p: None,
        )
        assert alive is False
        assert attempts == 0
        sleep.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_failed_skips_retries(self) -> None:
        # Same logic for failed runs.
        sleep = AsyncMock()
        terminator = _make_terminator(sleep=sleep)
        heartbeat = _heartbeat_with_alive_sequence([False])

        alive, attempts = await terminator._check_heartbeat_with_retries(
            heartbeat=heartbeat,
            terminal_state="failed",
            bus_publish=lambda k, p: None,
        )
        assert alive is False
        assert attempts == 0
        sleep.assert_not_awaited()


# ---------------------------------------------------------------------------
# 5. Custom retry knobs
# ---------------------------------------------------------------------------


class TestCustomKnobs:
    @pytest.mark.asyncio
    async def test_custom_retry_count_respected(self) -> None:
        sleep = AsyncMock()
        terminator = _make_terminator(
            sleep=sleep, heartbeat_retry_attempts=5,
        )
        heartbeat = _heartbeat_with_alive_sequence([False] * 10)

        alive, attempts = await terminator._check_heartbeat_with_retries(
            heartbeat=heartbeat,
            terminal_state="completed",
            bus_publish=lambda k, p: None,
        )
        assert alive is False
        assert attempts == 5
        assert sleep.await_count == 5

    @pytest.mark.asyncio
    async def test_custom_tick_passed_to_sleep(self) -> None:
        sleep = AsyncMock()
        terminator = _make_terminator(
            sleep=sleep, heartbeat_retry_tick=2.5,
        )
        heartbeat = _heartbeat_with_alive_sequence([False, False])

        await terminator._check_heartbeat_with_retries(
            heartbeat=heartbeat,
            terminal_state="completed",
            bus_publish=lambda k, p: None,
        )
        sleep.assert_awaited_with(2.5)
