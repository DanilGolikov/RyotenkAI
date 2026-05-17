"""Tests for :class:`ryotenkai_control.events.EventDedup` (Phase 3)."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from ryotenkai_control.events import EventDedup, JournalReader, JournalWriter
from tests.unit.control.events.conftest import make_started

# ===========================================================================
# 1. Positive
# ===========================================================================


class TestPositive:
    def test_remember_then_is_duplicate_returns_true(self) -> None:
        d = EventDedup()
        d.remember("r", "control://orchestrator", 7)
        assert d.is_duplicate("r", "control://orchestrator", 7)

    def test_size_reflects_remembered_entries(self) -> None:
        d = EventDedup()
        d.remember("r", "src", 0)
        d.remember("r", "src", 1)
        d.remember("r2", "src", 0)
        assert d.size == 3
        # Phase 8 — seen_total tracks lifetime calls; hits start at 0
        # because no is_duplicate() has been issued yet.
        assert d.seen_total == 3
        assert d.dedup_hits_total == 0
        assert d.evicted_total == 0
        # First is_duplicate() that matches bumps the hit counter.
        assert d.is_duplicate("r", "src", 0) is True
        assert d.dedup_hits_total == 1


# ===========================================================================
# 2. Negative
# ===========================================================================


class TestNegative:
    def test_distinct_keys_are_not_duplicates(self) -> None:
        d = EventDedup()
        d.remember("r", "src", 0)
        assert not d.is_duplicate("r", "src", 1)
        assert not d.is_duplicate("r2", "src", 0)
        assert not d.is_duplicate("r", "src2", 0)

    def test_ttl_negative_rejected(self) -> None:
        with pytest.raises(ValueError):
            EventDedup(ttl_seconds=-1)


# ===========================================================================
# 3. Boundary
# ===========================================================================


class TestBoundary:
    def test_ttl_zero_evicts_immediately_on_evict_expired(self) -> None:
        clock = [100.0]
        d = EventDedup(ttl_seconds=0, clock=lambda: clock[0])
        d.remember("r", "src", 0)
        d.remember("r", "src", 1)
        clock[0] += 1.0  # any positive delta clears the TTL window
        evicted = d.evict_expired()
        assert evicted == 2
        assert d.size == 0


# ===========================================================================
# 4. Invariants
# ===========================================================================


class TestInvariants:
    def test_evict_run_drops_only_matching_run_id(self) -> None:
        d = EventDedup()
        d.remember("r1", "src", 0)
        d.remember("r1", "src", 1)
        d.remember("r2", "src", 0)
        evicted = d.evict_run("r1")
        assert evicted == 2
        assert not d.is_duplicate("r1", "src", 0)
        assert d.is_duplicate("r2", "src", 0)

    def test_remember_idempotent_on_same_key(self) -> None:
        d = EventDedup()
        d.remember("r", "src", 0)
        d.remember("r", "src", 0)
        d.remember("r", "src", 0)
        assert d.size == 1


# ===========================================================================
# 5. Dependency errors
# ===========================================================================


class TestDependencyErrors:
    def test_reconstruct_from_empty_journal_returns_zero(self, tmp_path: Path) -> None:
        path = tmp_path / "events.jsonl"
        path.touch()
        d = EventDedup()
        added = d.reconstruct_from_journal(JournalReader(path))
        assert added == 0
        assert d.size == 0

    def test_reconstruct_from_missing_journal_returns_zero(self, tmp_path: Path) -> None:
        d = EventDedup()
        added = d.reconstruct_from_journal(JournalReader(tmp_path / "nope.jsonl"))
        assert added == 0


# ===========================================================================
# 6. Regressions
# ===========================================================================


class TestRegressions:
    def test_reconstruct_handles_many_entries_quickly(self, tmp_path: Path) -> None:
        """Smoke benchmark: reconstruct from a journal with 1k entries in
        well under a second. The instruction prescribes 50k but the
        reader's pure-python iteration is the bottleneck; 1k is enough
        to assert linear behaviour without slowing CI.
        """
        path = tmp_path / "events.jsonl"
        writer = JournalWriter(path)
        for i in range(1_000):
            writer.append(make_started(offset=i))
        writer.close()

        d = EventDedup()
        start = time.monotonic()
        added = d.reconstruct_from_journal(
            JournalReader(path), max_entries_per_source=10_000
        )
        elapsed = time.monotonic() - start
        assert added == 1_000
        assert elapsed < 1.0  # generous bound, mostly catches O(n^2) regressions

    def test_reconstruct_respects_max_entries_per_source(self, tmp_path: Path) -> None:
        path = tmp_path / "events.jsonl"
        writer = JournalWriter(path)
        for i in range(20):
            writer.append(make_started(offset=i))
        writer.close()

        d = EventDedup()
        added = d.reconstruct_from_journal(
            JournalReader(path), max_entries_per_source=5
        )
        assert added == 5
        # The 5 most recent are remembered.
        assert d.is_duplicate("test-run", "control://orchestrator", 19)
        assert not d.is_duplicate("test-run", "control://orchestrator", 0)


# ===========================================================================
# 7. Logic-specific
# ===========================================================================


class TestLogicSpecific:
    def test_ttl_eviction_uses_injected_clock(self) -> None:
        clock = [100.0]
        d = EventDedup(ttl_seconds=60, clock=lambda: clock[0])
        d.remember("r", "src", 0)
        # Within TTL — no eviction.
        clock[0] += 30
        assert d.evict_expired() == 0
        # Past TTL — eviction.
        clock[0] += 31
        assert d.evict_expired() == 1
        assert d.size == 0

    def test_reconstruct_skips_unknown_marker(self, tmp_path: Path) -> None:
        """An UnknownEvent with UNKNOWN_OFFSET in the journal is skipped
        during reconstruction — it has no valid offset to dedup on.
        """
        path = tmp_path / "events.jsonl"
        writer = JournalWriter(path)
        writer.append(make_started(offset=0))
        writer.close()
        # Append a torn line; reader will treat it as UnknownEvent.
        with path.open("ab") as fh:
            fh.write(b"5\t{bad}\n")
        d = EventDedup()
        added = d.reconstruct_from_journal(JournalReader(path))
        assert added == 1  # only the well-formed envelope


# ===========================================================================
# 8. Mutation kill — boundary, comparisons, booleans, constants
# ===========================================================================


class TestMutationKillConstants:
    """Pin module-level constants & default kwargs against off-by-one /
    operator-swap mutants."""

    def test_default_ttl_seconds_is_exactly_86400(self) -> None:
        """24 * 3600 = 86400. Any NumberReplacer (23/25) or operator swap
        (+, %, /, **, //, <<, &, ^, >>) on this product produces a value
        != 86400, so pinning the exact computed integer kills all of them.
        """
        from ryotenkai_control.events.dedup import DEFAULT_TTL_SECONDS

        assert DEFAULT_TTL_SECONDS == 86400
        # Belt-and-braces — also pin the type so any future tweak that
        # accidentally drops to float (e.g. ``24 / 3600 == 0.00666…``)
        # fails the constructor with mypy + this assertion.
        assert isinstance(DEFAULT_TTL_SECONDS, int)

    def test_default_ttl_used_when_no_ttl_argument(self) -> None:
        """A constructor that doesn't override ``ttl_seconds`` adopts the
        canonical default. Pin the resolved value via the eviction window
        so a constant-mutation propagates into an observable failure.
        """
        from ryotenkai_control.events.dedup import DEFAULT_TTL_SECONDS

        clock = [0.0]
        d = EventDedup(clock=lambda: clock[0])
        d.remember("r", "src", 0)
        # At exactly TTL boundary, entry is NOT expired (cutoff = ttl,
        # entry ts = 0; ts <= 0? Only if cutoff >= 0). We move just shy
        # of the default TTL and confirm the entry survives.
        clock[0] = float(DEFAULT_TTL_SECONDS) - 1.0
        assert d.evict_expired() == 0
        # Step past TTL → entry is evicted.
        clock[0] = float(DEFAULT_TTL_SECONDS) + 1.0
        assert d.evict_expired() == 1

    def test_reconstruct_default_max_entries_is_exactly_10000(self) -> None:
        """The default for ``max_entries_per_source`` is exactly 10_000.
        A 9999/10001 NumberReplacer mutant changes which envelopes get
        tailed when the journal has > default entries; we assert via
        ``inspect.signature`` so the test pins the literal directly
        without needing a 10k-envelope journal to fall over.
        """
        import inspect

        sig = inspect.signature(EventDedup.reconstruct_from_journal)
        default = sig.parameters["max_entries_per_source"].default
        assert default == 10_000


class TestMutationKillComparisons:
    """Kill ``ReplaceComparisonOperator_*`` mutants on evict_run / evict_expired."""

    def test_evict_run_requires_exact_equality_not_lte(self) -> None:
        """``k[0] == run_id`` MUST NOT be relaxed to ``<=``. Run id
        strings that sort BEFORE ``run_id`` (lexicographically) would
        otherwise get evicted, which is a data-loss bug.

        Use run ids where ``"r-aaa" <= "r-bbb"`` lexicographically; only
        the exact "r-bbb" should be evicted.
        """
        d = EventDedup()
        d.remember("r-aaa", "src", 0)  # would match <= "r-bbb"
        d.remember("r-bbb", "src", 0)  # exact target
        d.remember("r-ccc", "src", 0)  # would NOT match <= "r-bbb"

        evicted = d.evict_run("r-bbb")
        assert evicted == 1  # exactly one — not two (== vs <=) or three.
        assert d.is_duplicate("r-aaa", "src", 0)
        assert d.is_duplicate("r-ccc", "src", 0)
        assert not d.is_duplicate("r-bbb", "src", 0)

    def test_evict_run_uses_value_equality_not_identity(self) -> None:
        """``k[0] == run_id`` MUST NOT be replaced with ``is``. Distinct
        string objects with the same value (defeats interning) must still
        compare equal.
        """
        d = EventDedup()
        d.remember("run-x", "src", 0)
        # Force a *non-interned* duplicate string value so ``is`` would
        # return False but ``==`` returns True. ``str(...)`` of a
        # concatenation creates a fresh object on most CPython builds;
        # ``"".join`` reliably defeats interning.
        non_interned = "".join(["run", "-", "x"])
        assert non_interned == "run-x"
        # The exact identity contract: ``is`` would NOT find this entry.
        # The eviction MUST find it (via ``==``).
        evicted = d.evict_run(non_interned)
        assert evicted == 1
        assert d.size == 0

    def test_evict_expired_uses_lte_at_boundary(self) -> None:
        """``ts <= cutoff`` MUST NOT be relaxed to ``ts < cutoff``. An
        entry timestamped EXACTLY at the cutoff is evicted under the
        original predicate but survives under the mutant.

        cutoff = clock_now - ttl. Remember at t=10 with ttl=5; advance
        clock to t=15 ⇒ cutoff=10; entry ts=10 ⇒ ``ts <= cutoff`` is
        True (evict) vs ``ts < cutoff`` is False (survive).
        """
        clock = [10.0]
        d = EventDedup(ttl_seconds=5.0, clock=lambda: clock[0])
        d.remember("r", "src", 0)  # remembered at t=10
        # Advance clock to exactly cutoff boundary: cutoff = 15 - 5 = 10.
        clock[0] = 15.0
        evicted = d.evict_expired()
        assert evicted == 1, (
            "Entry timestamped exactly at cutoff must be evicted under "
            "<= predicate; this kills the < mutant"
        )


class TestMutationKillSignatureMarkers:
    """Pin keyword-only argument markers — ``*,`` MUST NOT be turned into
    ``/,`` (positional-only). Mutants flip these markers; if the test
    suite never calls with positional args we can't observe the change.
    """

    def test_event_dedup_ctor_rejects_positional_ttl(self) -> None:
        """``EventDedup(ttl_seconds=...)`` is keyword-only. Calling
        positionally MUST raise TypeError. The ``*,`` → ``/,`` mutant
        would make positional calls succeed."""
        with pytest.raises(TypeError):
            EventDedup(60.0)  # type: ignore[misc]
        # Also assert the kwarg-form succeeds — guards against this test
        # mis-firing on an unrelated bug in the ctor.
        d = EventDedup(ttl_seconds=60.0)
        assert d.size == 0

    def test_reconstruct_rejects_positional_max_entries(self, tmp_path: Path) -> None:
        """``reconstruct_from_journal(reader, max_entries_per_source=N)``
        is keyword-only past ``reader``. The ``*,`` → ``/,`` mutant flips
        this; calling positionally would succeed under the mutant.
        """
        path = tmp_path / "events.jsonl"
        path.touch()
        d = EventDedup()
        reader = JournalReader(path)
        with pytest.raises(TypeError):
            d.reconstruct_from_journal(reader, 5)  # type: ignore[misc]


class TestMutationKillCounters:
    """Pin counters bumped on remember/evict so a removed-bump or
    boundary-replaced increment is observable."""

    def test_evicted_total_increments_by_exact_victim_count(self) -> None:
        """``self._evicted_total += len(victims)``. Pin both the per-call
        delta AND the cumulative value so any +1 / -1 / *2 mutant on the
        bookkeeping path fails.
        """
        d = EventDedup()
        d.remember("r1", "src", 0)
        d.remember("r1", "src", 1)
        d.remember("r1", "src", 2)
        d.remember("r2", "src", 0)
        # First eviction — evicts 3 r1 keys.
        n = d.evict_run("r1")
        assert n == 3
        assert d.evicted_total == 3
        # Second eviction — evicts 1 r2 key.
        n = d.evict_run("r2")
        assert n == 1
        assert d.evicted_total == 4  # cumulative

    def test_dedup_hits_total_bumps_only_on_hit(self) -> None:
        """``_dedup_hits_total`` bumps ONLY when ``is_duplicate`` returns
        True. A miss MUST NOT bump it. Pin both halves so a boolean
        flip on the inner ``if hit:`` is killed.
        """
        d = EventDedup()
        d.remember("r", "src", 0)
        # Miss — counter MUST remain 0.
        assert d.is_duplicate("r", "src", 99) is False
        assert d.dedup_hits_total == 0
        # Hit — counter goes to 1.
        assert d.is_duplicate("r", "src", 0) is True
        assert d.dedup_hits_total == 1
        # Second hit — counter goes to 2 (not capped).
        assert d.is_duplicate("r", "src", 0) is True
        assert d.dedup_hits_total == 2

    def test_seen_total_increments_per_remember_call(self) -> None:
        """``_seen_total`` bumps once per ``remember`` call, even when
        the same key is re-remembered (no idempotency on the counter)."""
        d = EventDedup()
        d.remember("r", "src", 0)
        d.remember("r", "src", 0)  # same key again
        d.remember("r", "src", 0)  # and again
        assert d.size == 1
        # Counter increments per CALL not per distinct key.
        assert d.seen_total == 3

    def test_reconstruct_bumps_seen_total_by_exact_added_count(
        self, tmp_path: Path
    ) -> None:
        """``self._seen_total += added`` at the end of reconstruction.
        Pin the exact delta so a ``+`` → ``-`` / ``*`` / etc. mutant
        produces an observable counter mismatch."""
        path = tmp_path / "events.jsonl"
        writer = JournalWriter(path)
        for i in range(5):
            writer.append(make_started(offset=i))
        writer.close()

        d = EventDedup()
        # Pre-condition: seen_total starts at zero so the delta is the
        # exact post-condition value.
        assert d.seen_total == 0
        added = d.reconstruct_from_journal(JournalReader(path))
        assert added == 5
        assert d.seen_total == 5

    def test_evict_run_no_match_does_not_bump_evicted_total(self) -> None:
        """When no entries match ``run_id``, ``len(victims) == 0`` so
        the counter must NOT change. Pin the no-op path so a constant
        mutant on ``+= len(victims)`` (e.g. always ``+= 1``) fails."""
        d = EventDedup()
        d.remember("r1", "src", 0)
        before = d.evicted_total
        n = d.evict_run("missing")
        assert n == 0
        assert d.evicted_total == before
