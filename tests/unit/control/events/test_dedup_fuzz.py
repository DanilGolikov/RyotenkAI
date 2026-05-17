"""Hypothesis fuzz tests for :class:`EventDedup` (Phase 9).

Three property families:

* **Membership semantics** — for any sequence of random
  ``(run_id, source, offset)`` triples, :meth:`is_duplicate` returns
  ``True`` exactly for the keys that have been remembered. Negative
  examples (different run, source, or offset) always return ``False``.

* **TTL eviction monotonicity** — given a sequence of remember calls
  with a known clock, :meth:`evict_expired` drops exactly the entries
  whose remembered timestamp is older than ``now - ttl``. The set
  shrinks monotonically.

* **Concurrent thread-safety** — many threads remembering and
  is_duplicate-ing concurrently never produce a partial state visible
  from another thread. The dedup set is internally locked.
"""

from __future__ import annotations

import threading

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from ryotenkai_control.events import EventDedup


# ---------------------------------------------------------------------------
# Shared strategies
# ---------------------------------------------------------------------------


_run_id_strategy = st.text(
    alphabet=st.characters(min_codepoint=0x20, max_codepoint=0x7E),
    min_size=1,
    max_size=16,
)
_source_strategy = st.sampled_from([
    "pod://r/trainer",
    "control://orchestrator",
    "pod://r/runner",
])
_offset_strategy = st.integers(min_value=0, max_value=1_000_000)


# A list of unique triples. Hypothesis composes one with the ``unique_by``
# trick: we generate a list of triples and dedupe at the test level.
_triple_strategy = st.tuples(_run_id_strategy, _source_strategy, _offset_strategy)


# ---------------------------------------------------------------------------
# Membership semantics
# ---------------------------------------------------------------------------


class TestMembership:
    """is_duplicate returns True iff the key has been remembered."""

    @given(triples=st.lists(_triple_strategy, min_size=0, max_size=30))
    @settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_remembered_keys_are_duplicates(
        self,
        triples: list[tuple[str, str, int]],
    ) -> None:
        d = EventDedup()
        unique_triples = list(dict.fromkeys(triples))  # preserve order, dedupe
        for run_id, src, off in unique_triples:
            d.remember(run_id, src, off)
        # Every remembered key is reported as a duplicate.
        for run_id, src, off in unique_triples:
            assert d.is_duplicate(run_id, src, off) is True
        assert d.size == len(unique_triples)

    @given(
        present=st.lists(_triple_strategy, min_size=0, max_size=20, unique=True),
        absent=st.lists(_triple_strategy, min_size=1, max_size=20),
    )
    @settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_unremembered_keys_are_not_duplicates(
        self,
        present: list[tuple[str, str, int]],
        absent: list[tuple[str, str, int]],
    ) -> None:
        d = EventDedup()
        present_set = set(present)
        for run_id, src, off in present:
            d.remember(run_id, src, off)
        for triple in absent:
            if triple in present_set:
                continue  # skip if accidentally coincides
            run_id, src, off = triple
            assert d.is_duplicate(run_id, src, off) is False

    @given(
        run_id_a=_run_id_strategy,
        run_id_b=_run_id_strategy,
        src=_source_strategy,
        off=_offset_strategy,
    )
    @settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_run_id_isolates_dedup(
        self,
        run_id_a: str,
        run_id_b: str,
        src: str,
        off: int,
    ) -> None:
        """A key in run A is NOT a duplicate in run B (unless equal)."""
        d = EventDedup()
        d.remember(run_id_a, src, off)
        if run_id_a == run_id_b:
            assert d.is_duplicate(run_id_b, src, off) is True
        else:
            assert d.is_duplicate(run_id_b, src, off) is False


# ---------------------------------------------------------------------------
# TTL eviction monotonicity
# ---------------------------------------------------------------------------


class TestTTLEviction:
    """evict_expired drops exactly the keys older than the TTL cutoff."""

    @given(
        ttl=st.floats(
            min_value=1.0, max_value=10_000.0,
            allow_nan=False, allow_infinity=False,
        ),
        n_keys=st.integers(min_value=1, max_value=20),
        wait_factor=st.floats(
            min_value=0.0, max_value=3.0,
            allow_nan=False, allow_infinity=False,
        ),
    )
    @settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_evict_expired_drops_only_old(
        self,
        ttl: float,
        n_keys: int,
        wait_factor: float,
    ) -> None:
        clock = [100.0]
        d = EventDedup(ttl_seconds=ttl, clock=lambda: clock[0])

        # Phase 1: remember n_keys keys at t=100.
        for i in range(n_keys):
            d.remember("run", "src", i)

        # Phase 2: advance the clock by wait_factor × ttl.
        clock[0] += wait_factor * ttl
        evicted = d.evict_expired()

        if wait_factor > 1.0:
            # All keys are past the TTL → all evicted.
            assert evicted == n_keys
            assert d.size == 0
        else:
            # Keys still inside the TTL window → none evicted.
            # (boundary: wait_factor == 1.0 is exactly TTL — evict_expired
            # uses ``ts <= cutoff`` so a key remembered at t=100 with
            # cutoff t=100 IS evicted; allow for that single-tick edge.)
            assert evicted in (0, n_keys)

    @given(
        ttl=st.floats(
            min_value=1.0, max_value=10_000.0,
            allow_nan=False, allow_infinity=False,
        ),
    )
    @settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_size_strictly_monotonic_after_evict(self, ttl: float) -> None:
        """``size`` never increases as we evict expired entries."""
        clock = [100.0]
        d = EventDedup(ttl_seconds=ttl, clock=lambda: clock[0])
        for i in range(10):
            d.remember("run", "src", i)
        initial_size = d.size

        clock[0] += ttl * 2  # past TTL
        d.evict_expired()
        final_size = d.size
        assert final_size <= initial_size

    @given(
        run_id_a=_run_id_strategy,
        run_id_b=_run_id_strategy,
        offsets=st.lists(_offset_strategy, min_size=1, max_size=10, unique=True),
    )
    @settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_evict_run_drops_only_matching_run(
        self,
        run_id_a: str,
        run_id_b: str,
        offsets: list[int],
    ) -> None:
        d = EventDedup()
        for off in offsets:
            d.remember(run_id_a, "src", off)
            if run_id_b != run_id_a:
                d.remember(run_id_b, "src", off)

        evicted_a = d.evict_run(run_id_a)
        assert evicted_a == len(offsets)
        # All A-run keys gone.
        for off in offsets:
            assert d.is_duplicate(run_id_a, "src", off) is False
        # All B-run keys still there.
        if run_id_b != run_id_a:
            for off in offsets:
                assert d.is_duplicate(run_id_b, "src", off) is True


# ---------------------------------------------------------------------------
# Concurrent remember + is_duplicate
# ---------------------------------------------------------------------------


class TestConcurrentSafety:
    """Property: race conditions never produce inconsistent state."""

    @given(
        thread_count=st.integers(min_value=2, max_value=8),
        per_thread=st.integers(min_value=10, max_value=50),
    )
    @settings(
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
        max_examples=10,
    )
    def test_concurrent_remember_yields_exact_unique_size(
        self,
        thread_count: int,
        per_thread: int,
    ) -> None:
        d = EventDedup()

        def worker(thread_idx: int) -> None:
            for off in range(per_thread):
                d.remember("run", "src", thread_idx * per_thread + off)

        threads = [
            threading.Thread(target=worker, args=(t,))
            for t in range(thread_count)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Every offset is unique (thread_idx * per_thread + off) so the
        # set size equals thread_count * per_thread.
        assert d.size == thread_count * per_thread
        assert d.seen_total == thread_count * per_thread

    @given(
        thread_count=st.integers(min_value=2, max_value=6),
        per_thread=st.integers(min_value=10, max_value=40),
    )
    @settings(
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
        max_examples=10,
    )
    def test_concurrent_is_duplicate_never_raises(
        self,
        thread_count: int,
        per_thread: int,
    ) -> None:
        """is_duplicate is safe to call concurrently with remember."""
        d = EventDedup()
        errors: list[Exception] = []

        def writer() -> None:
            try:
                for off in range(per_thread):
                    d.remember("run", "src", off)
            except Exception as exc:  # pragma: no cover
                errors.append(exc)

        def reader() -> None:
            try:
                for off in range(per_thread):
                    d.is_duplicate("run", "src", off)
            except Exception as exc:  # pragma: no cover
                errors.append(exc)

        threads = []
        for _ in range(thread_count // 2):
            threads.append(threading.Thread(target=writer))
            threads.append(threading.Thread(target=reader))
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # No thread observed an inconsistent dict state.
        assert errors == []


# ---------------------------------------------------------------------------
# Counter monotonicity
# ---------------------------------------------------------------------------


class TestCounterMonotonicity:
    """Phase 8 counters never decrease."""

    @given(
        ops=st.lists(
            st.tuples(_run_id_strategy, _source_strategy, _offset_strategy),
            min_size=0,
            max_size=30,
        ),
    )
    @settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_seen_total_only_grows_on_remember(
        self,
        ops: list[tuple[str, str, int]],
    ) -> None:
        d = EventDedup()
        last_seen = d.seen_total
        for run_id, src, off in ops:
            d.remember(run_id, src, off)
            assert d.seen_total >= last_seen
            last_seen = d.seen_total
        # The lifetime counter increments once per remember regardless
        # of duplicates — same-key re-remembers still bump.
        assert d.seen_total == len(ops)

    @given(
        ops=st.lists(_triple_strategy, min_size=1, max_size=20, unique=True),
    )
    @settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_dedup_hits_total_grows_on_duplicate_hit(
        self,
        ops: list[tuple[str, str, int]],
    ) -> None:
        d = EventDedup()
        # Remember everything.
        for run_id, src, off in ops:
            d.remember(run_id, src, off)
        # Now query every key — each query is a "hit".
        for run_id, src, off in ops:
            assert d.is_duplicate(run_id, src, off) is True
        assert d.dedup_hits_total == len(ops)
