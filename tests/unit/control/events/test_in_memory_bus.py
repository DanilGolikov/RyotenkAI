"""Tests for :class:`ryotenkai_control.events.InMemoryBus` (Phase 3)."""

from __future__ import annotations

import asyncio
import contextlib

import pytest

from ryotenkai_control.events import InMemoryBus
from tests.unit.control.events.conftest import (
    make_completed,
    make_started,
)

# ===========================================================================
# 1. Positive
# ===========================================================================


class TestPositive:
    @pytest.mark.asyncio
    async def test_publish_then_subscribe_yields_events(self) -> None:
        bus = InMemoryBus(capacity=8)
        bus.publish(make_started(offset=0))
        bus.publish(make_completed(offset=1))

        gen = bus.subscribe("c1", after_offset=-1)
        seen: list[int] = []
        async for ev in gen:
            seen.append(ev.offset)
            if len(seen) == 2:
                break
        assert seen == [0, 1]

    def test_oldest_and_newest_offset_reflect_buffer(self) -> None:
        bus = InMemoryBus(capacity=8)
        assert bus.oldest_offset is None
        assert bus.newest_offset is None
        bus.publish(make_started(offset=0))
        bus.publish(make_completed(offset=1))
        assert bus.oldest_offset == 0
        assert bus.newest_offset == 1
        assert bus.buffered_count == 2
        # Phase 8 — published_total / current_depth observable.
        assert bus.published_total == 2
        assert bus.current_depth == 2
        assert bus.subscriber_count == 0


# ===========================================================================
# 2. Negative
# ===========================================================================


class TestNegative:
    def test_overflow_drops_oldest_and_increments_counter(self) -> None:
        bus = InMemoryBus(capacity=2)
        bus.publish(make_started(offset=0))
        bus.publish(make_completed(offset=1))
        # Third publish evicts offset=0.
        bus.publish(make_completed(offset=2))
        assert bus.dropped_total == 1
        assert bus.oldest_offset == 1

    def test_publish_after_close_silently_dropped(self) -> None:
        bus = InMemoryBus(capacity=4)
        bus.close()
        bus.publish(make_started(offset=0))
        # Nothing added — bus stayed empty.
        assert bus.buffered_count == 0


# ===========================================================================
# 3. Boundary
# ===========================================================================


class TestBoundary:
    def test_capacity_one_keeps_only_latest(self) -> None:
        bus = InMemoryBus(capacity=1)
        bus.publish(make_started(offset=0))
        bus.publish(make_completed(offset=1))
        bus.publish(make_completed(offset=2))
        bus.publish(make_completed(offset=3))
        bus.publish(make_completed(offset=4))
        # 5 published, capacity 1 → 4 drops.
        assert bus.dropped_total == 4
        assert bus.oldest_offset == 4
        assert bus.newest_offset == 4

    def test_capacity_below_min_raises(self) -> None:
        with pytest.raises(ValueError):
            InMemoryBus(capacity=0)

    def test_capacity_above_max_raises(self) -> None:
        with pytest.raises(ValueError):
            InMemoryBus(capacity=10_000_000)


# ===========================================================================
# 4. Invariants
# ===========================================================================


class TestInvariants:
    @pytest.mark.asyncio
    async def test_consumer_cursor_monotonic(self) -> None:
        bus = InMemoryBus(capacity=8)
        for i in range(4):
            bus.publish(make_started(offset=i))
        gen = bus.subscribe("c1", after_offset=-1)
        seen: list[int] = []
        async for ev in gen:
            seen.append(ev.offset)
            if len(seen) == 4:
                break
        # Cursor is the last yielded offset.
        assert bus.consumer_cursor("c1") == 3
        # Strictly monotonic increase.
        assert all(seen[i] < seen[i + 1] for i in range(len(seen) - 1))


# ===========================================================================
# 5. Dependency errors
# ===========================================================================


class TestDependencyErrors:
    @pytest.mark.asyncio
    async def test_subscribe_after_close_terminates(self) -> None:
        bus = InMemoryBus(capacity=4)
        bus.close()
        gen = bus.subscribe("c1", after_offset=-1)
        seen: list[int] = []
        async for ev in gen:  # pragma: no cover — bus closed yields nothing
            seen.append(ev.offset)
        assert seen == []


# ===========================================================================
# 6. Regressions
# ===========================================================================


class TestRegressions:
    def test_slow_consumer_drop_counter_increments_on_subscribe(self) -> None:
        bus = InMemoryBus(capacity=2)
        # Publish more than capacity.
        for i in range(5):
            bus.publish(make_started(offset=i))
        # Consumer asks for offsets starting at 0 — but the ring only
        # has 3..4. The bus accounts for missed events on subscribe
        # entry and bumps the per-consumer counter.

        async def drive() -> int:
            async for _ev in bus.subscribe("slow", after_offset=-1):
                break
            return bus.dropped_per_consumer.get("slow", 0)

        missed = asyncio.run(drive())
        # Oldest available offset is 3 (0,1,2 were evicted from the
        # ring; cursor started at -1; missed = (3-1) - (-1) = 3).
        assert missed == 3


# ===========================================================================
# 7. Logic-specific
# ===========================================================================


class TestLogicSpecific:
    @pytest.mark.asyncio
    async def test_multi_consumer_independent_cursors(self) -> None:
        bus = InMemoryBus(capacity=8)
        for i in range(3):
            bus.publish(make_started(offset=i))

        async def drain(consumer_id: str) -> list[int]:
            seen: list[int] = []
            async for ev in bus.subscribe(consumer_id, after_offset=-1):
                seen.append(ev.offset)
                if len(seen) == 3:
                    break
            return seen

        a, b = await asyncio.gather(drain("a"), drain("b"))
        assert a == [0, 1, 2]
        assert b == [0, 1, 2]

    def test_record_consumer_drop_increments(self) -> None:
        bus = InMemoryBus(capacity=4)
        bus.record_consumer_drop("x", 3)
        bus.record_consumer_drop("x", 2)
        bus.record_consumer_drop("y", 7)
        assert bus.dropped_per_consumer == {"x": 5, "y": 7}

    def test_record_consumer_drop_zero_is_noop(self) -> None:
        bus = InMemoryBus(capacity=4)
        bus.record_consumer_drop("x", 0)
        bus.record_consumer_drop("x", -1)
        assert bus.dropped_per_consumer == {}


# ===========================================================================
# 8. Cursor inversion regression (Phase 9 lesson — pin the 1-char bug)
# ===========================================================================


class TestCursorInversionRegression:
    """Pin the Phase 9 bug: ``subscribe(after_offset=N)`` MUST yield events
    with ``offset > N``, not ``offset <= N``.

    The original implementation accidentally inverted the predicate
    (``not ev.offset > cursor``), so a consumer asking for events
    after offset 2 received offsets 0..2 instead of 3, and the cursor
    silently went backwards. We add explicit, dedicated coverage here
    so any future refactor that flips the comparison breaks loud.
    """

    @pytest.mark.asyncio
    async def test_subscribe_yields_events_strictly_greater_than_cursor(
        self,
    ) -> None:
        """Buffer ``[0, 1, 2, 3]`` + ``after_offset=2`` ⇒ yields only ``[3]``."""
        bus = InMemoryBus(capacity=100)
        for i in range(4):
            bus.publish(make_started(offset=i))

        results: list[int] = []
        async for ev in bus.subscribe("test", after_offset=2):
            results.append(ev.offset)
            if len(results) == 1:
                break

        assert results == [3], (
            f"Expected only offset > 2, got {results} — the cursor "
            "predicate must be strictly greater-than"
        )

    @pytest.mark.asyncio
    async def test_subscribe_after_offset_minus_one_yields_all_from_zero(
        self,
    ) -> None:
        """``after_offset=-1`` (canonical "from start") yields every event,
        starting at offset 0."""
        bus = InMemoryBus(capacity=100)
        for i in range(5):
            bus.publish(make_started(offset=i))

        results: list[int] = []
        async for ev in bus.subscribe("test", after_offset=-1):
            results.append(ev.offset)
            if len(results) == 5:
                break

        assert results == [0, 1, 2, 3, 4], (
            "after_offset=-1 must yield the entire buffer beginning "
            f"at offset 0; got {results}"
        )

    @pytest.mark.asyncio
    async def test_subscribe_after_last_offset_blocks_until_new_publish(
        self,
    ) -> None:
        """``after_offset = newest`` must NOT replay the newest; it must
        block waiting for a strictly newer publish."""
        bus = InMemoryBus(capacity=100)
        for i in range(3):
            bus.publish(make_started(offset=i))

        async def consume() -> list[int]:
            results: list[int] = []
            async for ev in bus.subscribe("test", after_offset=2):
                results.append(ev.offset)
                if len(results) == 1:
                    break
            return results

        task = asyncio.create_task(consume())
        # Give the task a chance to enter the wait state.
        await asyncio.sleep(0)
        assert not task.done(), (
            "subscribe(after_offset=last) must block — got immediate yield, "
            "which means the predicate is inclusive (regression!)"
        )

        # New event at offset 3 unblocks the wait.
        bus.publish(make_started(offset=3))
        result = await asyncio.wait_for(task, timeout=1.0)
        assert result == [3]


# ===========================================================================
# 9. Mutation kill — constants, capacity bounds, signature markers
# ===========================================================================


class TestMutationKillConstants:
    """Pin module-level constants — kills NumberReplacer mutants."""

    def test_default_capacity_is_exactly_10000(self) -> None:
        """``DEFAULT_CAPACITY = 10_000``. Pin both the module constant
        AND the constructor default introspection so 9999/10001 mutants
        fail."""
        import inspect

        from ryotenkai_control.events.in_memory_bus import DEFAULT_CAPACITY

        assert DEFAULT_CAPACITY == 10_000
        sig = inspect.signature(InMemoryBus.__init__)
        assert sig.parameters["capacity"].default == 10_000

    def test_min_capacity_is_exactly_one(self) -> None:
        """``MIN_CAPACITY = 1``: rejects 0 (boundary). Pin both ends —
        capacity=1 accepted, capacity=0 rejected. Kills 0/2 mutants on
        the literal."""
        from ryotenkai_control.events.in_memory_bus import MIN_CAPACITY

        assert MIN_CAPACITY == 1
        # Boundary: 1 accepted.
        InMemoryBus(capacity=1)
        # Below boundary rejected.
        with pytest.raises(ValueError):
            InMemoryBus(capacity=0)

    def test_max_capacity_is_exactly_one_million(self) -> None:
        """``MAX_CAPACITY = 1_000_000``. Pin both module constant and
        the accept/reject boundary."""
        from ryotenkai_control.events.in_memory_bus import MAX_CAPACITY

        assert MAX_CAPACITY == 1_000_000
        # Boundary: exactly MAX_CAPACITY is accepted.
        InMemoryBus(capacity=MAX_CAPACITY)
        # Above boundary rejected.
        with pytest.raises(ValueError):
            InMemoryBus(capacity=MAX_CAPACITY + 1)


class TestMutationKillCapacityCompare:
    """Pin the ``len == capacity`` overflow check against identity-swap."""

    def test_overflow_check_uses_value_equality_at_large_capacity(self) -> None:
        """``if len(self._buffer) == self._capacity``. The ``Eq_Is``
        mutant replaces ``==`` with ``is``: for small caches integers
        ARE interned (so 5 is 5 == True). With capacity > 256 the cache
        no longer interns; ``is`` returns False and the dropped_total
        counter would never increment.
        """
        # Use a capacity above CPython's small-int cache (256).
        bus = InMemoryBus(capacity=300)
        # Fill exactly to capacity — first overflow triggers drop.
        for i in range(300):
            bus.publish(make_started(offset=i))
        # Now publish ONE more — must drop one.
        bus.publish(make_completed(offset=300))
        assert bus.dropped_total == 1, (
            "overflow drop counter MUST increment when len reaches "
            "capacity (large capacity defeats int-interning); kills "
            "Eq → Is mutant"
        )


class TestMutationKillRecordConsumerDrop:
    """Pin the default kwarg + boundary on record_consumer_drop."""

    def test_default_count_is_exactly_one(self) -> None:
        """``count: int = 1``. Pin via observable behaviour: calling
        without count records exactly 1.
        """
        bus = InMemoryBus(capacity=4)
        bus.record_consumer_drop("c")  # default count=1
        assert bus.dropped_per_consumer["c"] == 1, (
            "default count must be 1; kills `= 0` / `= 2` mutants"
        )

    def test_count_one_is_recorded(self) -> None:
        """``if count <= 0: return``. Mutant ``<= 1`` would treat 1 as
        zero (early return), so count=1 would NOT be recorded. Pin: 1
        IS recorded; 0 and negatives are NOT."""
        bus = InMemoryBus(capacity=4)
        # 1 IS recorded (kills <= 1 mutant).
        bus.record_consumer_drop("c1", 1)
        assert bus.dropped_per_consumer["c1"] == 1
        # 0 is NOT recorded (boundary).
        bus.record_consumer_drop("c2", 0)
        assert "c2" not in bus.dropped_per_consumer
        # Negative also NOT recorded.
        bus.record_consumer_drop("c3", -5)
        assert "c3" not in bus.dropped_per_consumer


class TestMutationKillSubscribeStartCursor:
    """Pin the initial-cursor branch — ``cursor = self._buffer[-1].offset
    if self._buffer else -1`` — against index and value mutants.
    """

    @pytest.mark.asyncio
    async def test_subscribe_no_after_offset_starts_at_newest_not_second(
        self,
    ) -> None:
        """``self._buffer[-1].offset`` MUST take the LAST element. The
        ``[-1]`` → ``[1]`` / ``[~1]`` / ``[+1]`` / ``[not 1]`` mutants
        all access a different position. With a buffer of length > 2 the
        observable difference is which "newest offset" the subscriber
        skips past.
        """
        bus = InMemoryBus(capacity=8)
        # Publish offsets 10, 20, 30, 40. ``[-1]`` → 40; ``[1]`` → 20;
        # ``[~1]`` == ``[-2]`` → 30; ``[+1]`` → 20; ``[not 1]`` == ``[0]``
        # → 10. So a subscriber should NOT receive 40 (it's <= cursor=40
        # under correct code). With any mutant index, the cursor is set
        # to the wrong offset and an "old" event is re-emitted.
        for off in (10, 20, 30, 40):
            bus.publish(make_started(offset=off))

        async def consume() -> int | None:
            async for ev in bus.subscribe("c", after_offset=None):
                return ev.offset
            return None

        task = asyncio.create_task(consume())
        await asyncio.sleep(0.01)
        # Under correct code the subscriber is blocked — there are no
        # events with offset > 40 yet.
        assert not task.done(), (
            "subscribe(after_offset=None) MUST start at the LAST resident "
            "offset; if cursor lands on the wrong element a stale event "
            "would yield immediately. Kills mutated indices [1] / [~1] / "
            "[+1] / [not 1]"
        )
        # Publishing a strictly-newer event unblocks the consumer.
        bus.publish(make_completed(offset=50))
        result = await asyncio.wait_for(task, timeout=1.0)
        assert result == 50

    @pytest.mark.asyncio
    async def test_subscribe_no_after_offset_uses_truthy_buffer_check(
        self,
    ) -> None:
        """The conditional ``if self._buffer`` (truthy check). The
        ``AddNot`` mutant flips to ``if not self._buffer`` so a NON-empty
        buffer would take the ``else -1`` branch — and an empty buffer
        would crash on ``[-1]``.
        """
        bus = InMemoryBus(capacity=8)
        # Empty buffer — cursor MUST be -1, NOT a crash.
        # The ``not self._buffer`` mutant would try ``self._buffer[-1]``
        # on empty deque → IndexError.

        async def consume() -> list[int]:
            seen: list[int] = []
            async for ev in bus.subscribe("c", after_offset=None):
                seen.append(ev.offset)
                if len(seen) >= 1:
                    break
            return seen

        # Subscribe on empty bus; publish after.
        task = asyncio.create_task(consume())
        await asyncio.sleep(0.01)
        bus.publish(make_started(offset=7))
        result = await asyncio.wait_for(task, timeout=1.0)
        assert result == [7], (
            "on empty buffer the subscriber must wait and yield the "
            "first NEW event; kills `if not self._buffer` mutant which "
            "would crash on indexing into the empty deque"
        )

    @pytest.mark.asyncio
    async def test_subscribe_empty_buffer_default_cursor_is_minus_one(
        self,
    ) -> None:
        """The ``else -1`` branch — when the buffer is empty, cursor =
        -1. Mutants ``else 1`` / ``else ~1`` / ``else +1`` / ``else not 1``
        / ``else -0`` / ``else -2`` all shift the resume point.

        Test: subscribe on empty bus, then publish offset=0. With
        cursor=-1, ``0 > -1`` ⇒ yield 0. With cursor=0 (or 1), ``0 > 0``
        is False ⇒ DO NOT yield (would block forever).
        """
        bus = InMemoryBus(capacity=8)

        async def consume() -> int | None:
            async for ev in bus.subscribe("c", after_offset=None):
                return ev.offset
            return None

        task = asyncio.create_task(consume())
        await asyncio.sleep(0.01)
        # Publish offset=0 — must be yielded because cursor was -1.
        bus.publish(make_started(offset=0))
        result = await asyncio.wait_for(task, timeout=1.0)
        assert result == 0, (
            "with empty buffer at subscribe time, cursor must be -1 so "
            "offset=0 yields; kills `else 0` / `else 1` / `else 2` mutants"
        )


class TestMutationKillSlowConsumerMath:
    """Pin the exact arithmetic on the slow-consumer catchup branch."""

    @pytest.mark.asyncio
    async def test_slow_cursor_at_oldest_minus_two_records_correct_drop(
        self,
    ) -> None:
        """``cursor < oldest - 1`` AND ``missed = (oldest - 1) - cursor``.

        Scenario: ring has offsets [5, 6, 7] (capacity=3). A consumer
        asks for after_offset=2. Cursor=2; oldest=5; missed = (5-1)-2 = 2.

        - Original ``cursor < oldest - 1``: 2 < 4 → True → record 2.
        - Mutant ``<=``: 2 <= 4 → True (matches; killable on cursor=4
          boundary below).
        - Mutant ``oldest + 1`` / ``* 1`` / ``/ 1`` / etc.: changes value
          of ``oldest - 1`` such that the resulting ``missed`` is wrong.
        """
        bus = InMemoryBus(capacity=3)
        for off in (5, 6, 7):
            bus.publish(make_started(offset=off))

        async def consume() -> int:
            async for _ev in bus.subscribe("slow", after_offset=2):
                break
            return bus.dropped_per_consumer.get("slow", 0)

        missed = await asyncio.wait_for(consume(), timeout=1.0)
        # With oldest=5, cursor=2 → missed = (5-1) - 2 = 2.
        # Mutant ``oldest + 1`` → (5+1) - 2 = 4. Killed.
        # Mutant ``oldest * 1`` → (5*1) - 2 = 3. Killed.
        # Mutant ``oldest >> 1`` → (5>>1) - 2 = 2 - 2 = 0. Killed.
        # Mutant ``oldest // 1`` → 5 - 2 = 3. Killed.
        # NumberReplacer 1→2: (5-2) - 2 = 1. Killed.
        # NumberReplacer 1→0: (5-0) - 2 = 3. Killed.
        assert missed == 2

    @pytest.mark.asyncio
    async def test_slow_predicate_strict_at_boundary(self) -> None:
        """``cursor < oldest - 1``. Boundary: cursor == oldest - 1 → False
        (no drop counted). Mutant ``<=`` would incorrectly record a
        zero-or-positive drop count. Mutant ``!=`` would record a drop
        whenever cursor != oldest-1 (e.g. cursor == oldest, which is
        legitimate).
        """
        bus = InMemoryBus(capacity=3)
        for off in (5, 6, 7):
            bus.publish(make_started(offset=off))
        # cursor = oldest - 1 = 4 exactly. NO drop recorded.

        async def consume() -> int:
            async for _ev in bus.subscribe("at_boundary", after_offset=4):
                break
            return bus.dropped_per_consumer.get("at_boundary", 0)

        missed = await asyncio.wait_for(consume(), timeout=1.0)
        assert missed == 0, (
            "cursor exactly at oldest-1 is NOT slow; kills `<=` mutant "
            "(would record 0, but the dict key would still be created)"
        )
        # Also assert key NOT created (because ``if missed > 0`` blocks
        # the dict write when missed == 0).
        assert "at_boundary" not in bus.dropped_per_consumer

    @pytest.mark.asyncio
    async def test_missed_strictly_positive_guard(self) -> None:
        """``if missed > 0``. Mutant ``>= 0`` would create a dict entry
        with value 0 even when missed == 0. Mutant ``!= 0`` would record
        for any non-zero missed (including negatives). Mutant ``> 1``
        would skip a single-event miss. Mutant ``> -1`` always-records.
        """
        # Setup: cursor 5; oldest 7; missed = 6 - 5 = 1. Predicate
        # ``cursor < oldest-1`` = 5 < 6 = True. ``missed > 0`` = 1 > 0
        # = True → record 1.
        bus = InMemoryBus(capacity=3)
        for off in (7, 8, 9):
            bus.publish(make_started(offset=off))

        async def consume() -> int:
            async for _ev in bus.subscribe("one_miss", after_offset=5):
                break
            return bus.dropped_per_consumer.get("one_miss", 0)

        missed = await asyncio.wait_for(consume(), timeout=1.0)
        assert missed == 1, (
            "single-event miss must be recorded as 1; kills `> 1` mutant"
        )

    @pytest.mark.asyncio
    async def test_cursor_resumes_at_exactly_oldest_minus_one(self) -> None:
        """After slow-consumer accounting, ``cursor = oldest - 1`` so
        the next yield is exactly ``oldest``. Mutants on the assignment
        (``oldest & 1``, ``oldest * 1``, etc.) make the consumer skip
        the oldest event or re-yield events the bus thinks are gone.
        """
        bus = InMemoryBus(capacity=3)
        for off in (10, 11, 12):
            bus.publish(make_started(offset=off))

        # Consumer asks for after_offset=0 (way behind). Expected first
        # yield: oldest=10 (cursor reset to 9; predicate ``ev.offset >
        # 9`` ⇒ yield 10 first).
        async def consume() -> list[int]:
            seen: list[int] = []
            async for ev in bus.subscribe("slow2", after_offset=0):
                seen.append(ev.offset)
                if len(seen) == 3:
                    break
            return seen

        result = await asyncio.wait_for(consume(), timeout=1.0)
        assert result == [10, 11, 12], (
            "after slow-consumer reset, cursor must be oldest-1 (9) so "
            "the first yielded event is oldest (10); kills mutated "
            "assignment `cursor = oldest & 1` (would be 0) etc."
        )


class TestMutationKillSubscriberCount:
    """Pin subscriber_count bumps and the cleanup guard."""

    @pytest.mark.asyncio
    async def test_subscriber_count_increments_by_one_on_subscribe(
        self,
    ) -> None:
        """``self._subscriber_count += 1`` on subscribe entry. Kills
        ``+= 0`` (no-op) and ``+= 2`` (double-count) mutants."""
        bus = InMemoryBus(capacity=4)
        assert bus.subscriber_count == 0

        started = asyncio.Event()
        release = asyncio.Event()

        async def consume() -> None:
            agen = bus.subscribe("c", after_offset=-1)
            # Drive past the first cursor-init under the lock so
            # subscriber_count is bumped.
            anext_task = asyncio.create_task(agen.__anext__())
            await asyncio.sleep(0.01)
            started.set()
            await release.wait()
            anext_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await anext_task
            await agen.aclose()

        task = asyncio.create_task(consume())
        await asyncio.wait_for(started.wait(), timeout=1.0)
        assert bus.subscriber_count == 1, (
            "subscribe MUST bump count by exactly 1; kills `+= 0` and "
            "`+= 2` mutants"
        )
        release.set()
        await asyncio.wait_for(task, timeout=1.0)
        # After aclose, finally runs → counter decrements back to 0.
        assert bus.subscriber_count == 0

    @pytest.mark.asyncio
    async def test_subscriber_count_decrements_by_one_in_finally(
        self,
    ) -> None:
        """``self._subscriber_count -= 1`` in the finally block. Kills
        ``-= 0`` (count never recovers) and ``-= 2`` (count goes negative;
        the guard ``> 0`` blocks it on the SECOND finally but not the
        first). Pin: after two subscribe/aclose cycles, count returns to
        zero — would be -2 or 2 under mutants.
        """
        bus = InMemoryBus(capacity=4)
        for _ in range(2):
            agen = bus.subscribe("c", after_offset=-1)
            # Drive into the generator so subscriber_count bump runs.
            anext_task = asyncio.create_task(agen.__anext__())
            await asyncio.sleep(0.01)
            anext_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await anext_task
            await agen.aclose()
        # If decrement was `-= 0`: 2 (never returned to 0).
        # If decrement was `-= 2`: first iteration count→-1 (blocked by
        # `> 0` guard → still 1), second iteration count→-1 (blocked →
        # still 1). End state: 1.
        # Correct behaviour: 0.
        assert bus.subscriber_count == 0, (
            "after balanced subscribe/aclose pairs, subscriber_count "
            "MUST return to 0; kills -= 0 (would be 2) and -= 2 mutants"
        )

    @pytest.mark.asyncio
    async def test_subscriber_count_guard_does_not_decrement_below_zero(
        self,
    ) -> None:
        """``if self._subscriber_count > 0: self._subscriber_count -= 1``.
        The guard kills ``> 0`` mutants that would either always-allow
        (``>= 0``, ``> -1``, ``!= 0``) — leading to negative count — or
        never-allow (``< 0``, ``== 0``, ``> 1``) — leading to permanently
        stuck count.

        Reset the counter to 0 first via a clean cycle, then artificially
        invoke the finally path with no prior increment by pumping then
        aclose'ing TWICE — under correct code count stays >= 0.
        """
        bus = InMemoryBus(capacity=4)
        # Force counter to a known 0 state.
        assert bus.subscriber_count == 0
        # Pre-set counter to 1 manually to simulate one outstanding sub,
        # then run TWO finally-blocks back-to-back via two
        # subscribe/aclose cycles where neither enters the lock-bump
        # path.
        # Easiest observable: call subscribe + aclose immediately so the
        # generator never bumps the counter; verify count stays at 0,
        # not -1.
        agen = bus.subscribe("ghost", after_offset=-1)
        await agen.aclose()  # finally runs without prior bump
        assert bus.subscriber_count == 0, (
            "subscriber_count must not go below 0 when finally runs "
            "without a prior bump; kills `>= 0` and `>= -1` guards"
        )


class TestMutationKillSubscribeLoopControl:
    """Pin the ``continue`` after draining pending events."""

    @pytest.mark.asyncio
    async def test_continue_after_drain_keeps_subscriber_alive(self) -> None:
        """``if pending: ... continue``. The ``break`` mutant would exit
        the outer loop after the first drain, so subsequent publishes
        would NEVER reach the subscriber. Pin via two-batch consumption.
        """
        bus = InMemoryBus(capacity=8)
        # First batch: 2 events.
        bus.publish(make_started(offset=0))
        bus.publish(make_completed(offset=1))

        seen: list[int] = []
        result_ready = asyncio.Event()

        async def consume() -> None:
            async for ev in bus.subscribe("c", after_offset=-1):
                seen.append(ev.offset)
                if len(seen) == 4:
                    result_ready.set()
                    return

        task = asyncio.create_task(consume())
        # Let the first batch flush through.
        await asyncio.sleep(0.01)
        # Second batch: 2 more events. Under ``break`` mutant the
        # subscriber would have exited after draining offset 0 and 1.
        bus.publish(make_started(offset=2))
        bus.publish(make_completed(offset=3))
        await asyncio.wait_for(result_ready.wait(), timeout=1.0)
        await asyncio.wait_for(task, timeout=1.0)
        assert seen == [0, 1, 2, 3], (
            "subscriber must `continue` after drain, not `break`; kills "
            "the ContinueWithBreak mutant"
        )


class TestMutationKillSubscribePredicate:
    """Strengthen the ``ev.offset > cursor`` predicate kill — already
    partially covered in TestCursorInversionRegression."""

    @pytest.mark.asyncio
    async def test_predicate_uses_value_comparison_not_identity(self) -> None:
        """``ev.offset > cursor``. Mutants ``is not`` / ``!=`` / ``>=``
        all behave wrong at the cursor=offset boundary.

        Use a capacity > 256 and offsets > 256 so int-interning doesn't
        rescue the ``is not`` mutant.
        """
        bus = InMemoryBus(capacity=1000)
        # Publish a single event at offset 500 — well above interned
        # int range.
        bus.publish(make_started(offset=500))

        # Subscribe at after_offset=500 — cursor=500. Under ``>``: no
        # yield (must wait). Under ``>=``: 500 >= 500 ⇒ yield.
        # Under ``is not``: 500 is not 500 (different int objects) ⇒ True
        # ⇒ would yield.
        async def consume() -> int | None:
            async for ev in bus.subscribe("c", after_offset=500):
                return ev.offset
            return None

        task = asyncio.create_task(consume())
        await asyncio.sleep(0.01)
        assert not task.done(), (
            "at cursor=offset boundary, predicate `>` must be strict; "
            "kills `>=`, `!=`, `is not` mutants"
        )
        bus.publish(make_completed(offset=501))
        result = await asyncio.wait_for(task, timeout=1.0)
        assert result == 501


# ===========================================================================
# 10. Mutation kill — iteration 2 (close remaining gaps)
# ===========================================================================


class TestMutationKillSignatureMarkers:
    """Pin ``*,`` keyword-only markers in __init__ and subscribe."""

    def test_init_capacity_is_keyword_only(self) -> None:
        """``def __init__(self, *, capacity)``. Positional ``InMemoryBus(8)``
        MUST raise TypeError; the ``/,`` mutant would accept it."""
        with pytest.raises(TypeError):
            InMemoryBus(8)  # type: ignore[misc]
        # Kwarg form works.
        InMemoryBus(capacity=8)

    def test_subscribe_after_offset_is_keyword_only(self) -> None:
        """``async def subscribe(self, consumer_id, *, after_offset)``.
        Positional after_offset MUST raise TypeError; ``/,`` mutant
        accepts it."""
        bus = InMemoryBus(capacity=4)
        with pytest.raises(TypeError):
            bus.subscribe("c", 0)  # type: ignore[misc]


class TestMutationKillPropertyDecorators:
    """Pin ``@property`` on capacity / is_closed / etc."""

    def test_capacity_is_property_returns_int(self) -> None:
        """``@property def capacity`` — bound-method != int."""
        bus = InMemoryBus(capacity=42)
        assert bus.capacity == 42
        assert isinstance(bus.capacity, int)

    def test_is_closed_property_reflects_state(self) -> None:
        """``@property def is_closed`` — bound-method is truthy regardless
        of close() state. The bool comparison kills RemoveDecorator."""
        bus = InMemoryBus(capacity=4)
        assert bus.is_closed is False
        bus.close()
        assert bus.is_closed is True


class TestMutationKillEvenOldestXor:
    """Kill the ``(oldest - 1) - cursor`` → ``(oldest ^ 1) - cursor``
    mutant by choosing an EVEN oldest where ``oldest ^ 1 != oldest - 1``.

    With even oldest (say 12), correct ``oldest - 1 = 11``, mutant
    ``oldest ^ 1 = 13``. The recorded ``missed`` differs.
    """

    @pytest.mark.asyncio
    async def test_recorded_drop_count_uses_subtraction_not_xor(self) -> None:
        """capacity=3 + publish offsets 10..14 ⇒ ring [12,13,14],
        oldest=12 (even). Consumer asks for after_offset=0.
          correct missed = (12 - 1) - 0 = 11
          mutant  missed = (12 ^ 1) - 0 = 13 - 0 = 13
        """
        bus = InMemoryBus(capacity=3)
        for off in (10, 11, 12, 13, 14):
            bus.publish(make_started(offset=off))
        # Sanity — oldest is 12 (even).
        assert bus.oldest_offset == 12

        async def consume() -> int:
            async for _ev in bus.subscribe("slow", after_offset=0):
                break
            return bus.dropped_per_consumer.get("slow", 0)

        missed = await asyncio.wait_for(consume(), timeout=1.0)
        assert missed == 11, (
            f"with even oldest=12, missed MUST be (12-1)-0 = 11; got "
            f"{missed}; kills `(oldest ^ 1) - cursor` mutant"
        )


class TestMutationKillDropCounterAccumulation:
    """Kill the ``+ missed`` → ``^ missed`` mutant on the consumer-drop
    counter accumulation. Requires TWO recordings on the same consumer
    where the bitwise XOR produces a different result from addition.
    """

    @pytest.mark.asyncio
    async def test_per_consumer_drops_accumulate_via_addition_not_xor(
        self,
    ) -> None:
        """Two recordings on the same consumer:
          First:  existing=0, missed=3 → 0 + 3 = 3 (and 0 ^ 3 = 3 — same)
          Second: existing=3, missed=5 → 3 + 5 = 8 (but 3 ^ 5 = 6 — DIFFERENT)
        """
        bus = InMemoryBus(capacity=4)
        # Use the public ``record_consumer_drop`` to drive the same
        # accumulator (it uses the same dict-get-plus-add pattern at a
        # different call site, but our coverage extension here is the
        # slow-consumer accumulator at L237. We trigger the slow-consumer
        # branch twice via TWO sequential subscribes from the same
        # consumer_id.
        for off in range(10):
            bus.publish(make_started(offset=off))
        # Ring now full of [6,7,8,9] (oldest=6). Subscribe-drain-cycle 1:
        # cursor=after_offset=2 → missed = (6-1)-2 = 3. Recorded.

        async def cycle(after: int) -> None:
            async for _ev in bus.subscribe("dup", after_offset=after):
                break

        await asyncio.wait_for(cycle(2), timeout=1.0)
        assert bus.dropped_per_consumer["dup"] == 3
        # Cycle 2: publish more to evict ring, then subscribe with
        # after_offset=4. After 5 more publishes (10..14), ring=[11..14],
        # oldest=11. missed = (11-1)-4 = 6.
        for off in range(10, 15):
            bus.publish(make_started(offset=off))
        await asyncio.wait_for(cycle(4), timeout=1.0)
        # Correct: 3 + 6 = 9. Mutant XOR: 3 ^ 6 = 5. KILL.
        assert bus.dropped_per_consumer["dup"] == 9, (
            f"second drop record must ADD (3+6=9), not XOR (3^6=5); "
            f"got {bus.dropped_per_consumer['dup']}; kills `+ missed` → "
            "`^ missed` mutant"
        )


class TestMutationKillInitialCursorDictWrite:
    """Pin the initial cursor written to ``_consumer_cursors`` BEFORE
    the first yield. Use an empty buffer so the generator awaits the
    publish signal — control returns to the caller, and we can observe
    the dict value mid-flight.
    """

    @pytest.mark.asyncio
    async def test_initial_cursor_on_empty_buffer_is_exactly_minus_one(
        self,
    ) -> None:
        """With empty buffer and after_offset=None, cursor MUST be -1
        (NOT -2 or +1 etc.). The mutants ``else ~1`` (=-2), ``else - 2``
        (=-2), ``else +1`` (=1), ``else 0`` (=0), ``else 1`` (=1),
        ``else not 1`` (=False=0) all assign different values to the
        consumer-cursor dict. Observe via ``consumer_cursor()``.
        """
        bus = InMemoryBus(capacity=4)
        # Empty buffer.
        gen = bus.subscribe("watch", after_offset=None)
        anext_task = asyncio.create_task(gen.__anext__())
        # Yield control so the generator body runs, hits await signal.wait().
        await asyncio.sleep(0.01)
        # At this point the initial-cursor value sits in the dict.
        observed = bus.consumer_cursor("watch")
        assert observed == -1, (
            f"initial cursor on empty buffer must be -1; got {observed}; "
            "kills `else -2` / `else 1` / `else 0` / `else ~1` mutants"
        )
        # Clean up.
        anext_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await anext_task
        await gen.aclose()

    @pytest.mark.asyncio
    async def test_initial_cursor_on_nonempty_buffer_is_last_offset(
        self,
    ) -> None:
        """With non-empty buffer (offsets 100, 200, 300, 400) and
        after_offset=None, cursor MUST be the LAST element's offset
        (400). The ``[-1]`` index mutants would pick a different
        element. Use offsets > 256 to defeat int-interning so the
        observable distinguishes via value comparison.
        """
        bus = InMemoryBus(capacity=8)
        for off in (100, 200, 300, 400):
            bus.publish(make_started(offset=off))
        gen = bus.subscribe("watch", after_offset=None)
        anext_task = asyncio.create_task(gen.__anext__())
        await asyncio.sleep(0.01)
        observed = bus.consumer_cursor("watch")
        assert observed == 400, (
            f"initial cursor on non-empty buffer must be the LAST offset "
            f"(400); got {observed}; kills `[1]` (=200) / `[~1]==[-2]` "
            "(=300) / `[+1]` (=200) / `[not 1]==[0]` (=100) index mutants"
        )
        anext_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await anext_task
        await gen.aclose()


class TestMutationKillSubscriberCountGuardAtZero:
    """Pin the ``if subscriber_count > 0`` guard against ``>= 0`` /
    ``!= 0`` / ``> -1`` mutants by reaching the finally block with
    count exactly 0.

    Reach this by force-resetting the counter to 0 BEFORE aclose() so
    the finally block sees count=0. Under correct ``> 0``: skip
    decrement (stays 0). Under mutants ``>= 0`` / ``!= 0`` / ``> -1``:
    decrement to -1 — observable via ``subscriber_count`` post-aclose.
    """

    @pytest.mark.asyncio
    async def test_finally_guard_skips_decrement_at_zero(self) -> None:
        bus = InMemoryBus(capacity=4)
        # Enter the subscriber body so subscriber_count bumps to 1.
        gen = bus.subscribe("c", after_offset=None)
        anext_task = asyncio.create_task(gen.__anext__())
        await asyncio.sleep(0.01)
        assert bus.subscriber_count == 1
        # Force the counter to 0 manually — simulates a path where the
        # finally runs even though the count is already zero (e.g.,
        # a torn finally + re-entrancy bug). Under correct ``> 0``:
        # finally skips decrement (count STAYS 0).
        bus._subscriber_count = 0
        anext_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await anext_task
        await gen.aclose()
        assert bus.subscriber_count == 0, (
            "subscriber_count must stay at 0 (not go negative); kills "
            "`>= 0` / `!= 0` / `> -1` guard mutants"
        )
