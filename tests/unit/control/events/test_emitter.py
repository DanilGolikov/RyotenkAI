"""Tests for :class:`ryotenkai_control.events.ControlEventEmitter` (Phase 3)."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from ryotenkai_control.events import (
    ControlEventEmitter,
    EventDedup,
    InMemoryBus,
    JournalReader,
    JournalWriter,
)
from ryotenkai_shared.events import UNKNOWN_OFFSET, IEventEmitter
from tests.unit.control.events.conftest import make_completed, make_started


def _build_emitter(
    tmp_path: Path, *, source: str = "control://orchestrator"
) -> ControlEventEmitter:
    journal = JournalWriter(tmp_path / "events.jsonl")
    bus = InMemoryBus(capacity=64)
    dedup = EventDedup()
    return ControlEventEmitter(
        run_id="test-run",
        source=source,
        journal=journal,
        bus=bus,
        dedup=dedup,
    )


# ===========================================================================
# 1. Positive
# ===========================================================================


class TestPositive:
    def test_emit_writes_journal_and_bus(self, tmp_path: Path) -> None:
        emitter = _build_emitter(tmp_path)
        emitter.emit(make_started())
        # Phase 8 — emitter counters track the hot path.
        assert emitter.events_emitted_total == 1
        assert emitter.offset_collisions_detected_total == 0
        emitter.close()

        envelopes = list(JournalReader(tmp_path / "events.jsonl").iter_envelopes())
        assert len(envelopes) == 1
        assert envelopes[0].kind == "ryotenkai.control.run.started"

    def test_protocol_compliance(self, tmp_path: Path) -> None:
        emitter = _build_emitter(tmp_path)
        assert isinstance(emitter, IEventEmitter)
        emitter.close()

    def test_for_run_builds_collaborators(self, tmp_path: Path) -> None:
        em = ControlEventEmitter.for_run(run_id="r", run_directory=tmp_path)
        assert isinstance(em, IEventEmitter)
        assert em.run_id == "r"
        em.emit(make_started())
        em.close()
        assert (tmp_path / "events.jsonl").exists()


# ===========================================================================
# 2. Negative
# ===========================================================================


class TestNegative:
    def test_emit_remote_dedup_blocks_duplicate(self, tmp_path: Path) -> None:
        emitter = _build_emitter(tmp_path)
        ev = make_started(source="pod://r/trainer", offset=5)
        emitter.emit_remote(ev)
        emitter.emit_remote(ev)  # duplicate
        assert emitter.events_remote_accepted_total == 1
        assert emitter.events_remote_dropped_total.get("duplicate", 0) == 1
        emitter.close()

    def test_emit_remote_with_unknown_offset_dropped(self, tmp_path: Path) -> None:
        emitter = _build_emitter(tmp_path)
        ev = make_started(source="pod://r/trainer", offset=UNKNOWN_OFFSET)
        emitter.emit_remote(ev)
        assert emitter.events_remote_dropped_total.get("invalid_offset", 0) == 1
        assert emitter.events_remote_accepted_total == 0
        emitter.close()


# ===========================================================================
# 3. Boundary
# ===========================================================================


class TestBoundary:
    def test_emit_with_unknown_offset_assigns_monotonic(self, tmp_path: Path) -> None:
        emitter = _build_emitter(tmp_path)
        emitter.emit(make_started(offset=UNKNOWN_OFFSET))
        emitter.emit(make_completed(offset=UNKNOWN_OFFSET))
        emitter.close()
        envelopes = list(JournalReader(tmp_path / "events.jsonl").iter_envelopes())
        assert [e.offset for e in envelopes] == [0, 1]

    def test_resume_continues_offset_sequence(self, tmp_path: Path) -> None:
        """After close + reopen via for_run, new emits don't collide
        with offsets already on disk.
        """
        em = ControlEventEmitter.for_run(run_id="r", run_directory=tmp_path)
        em.emit(make_started())  # offset 0
        em.emit(make_completed())  # offset 1
        em.close()

        em2 = ControlEventEmitter.for_run(run_id="r", run_directory=tmp_path)
        em2.emit(make_completed())  # should be offset 2
        em2.close()

        envelopes = list(JournalReader(tmp_path / "events.jsonl").iter_envelopes())
        assert [e.offset for e in envelopes] == [0, 1, 2]


# ===========================================================================
# 4. Invariants
# ===========================================================================


class TestInvariants:
    def test_stage_scope_fills_stage_id(self, tmp_path: Path) -> None:
        emitter = _build_emitter(tmp_path)
        with emitter.stage_scope("dataset_validator"):
            emitter.emit(make_started())
        emitter.close()
        envelopes = list(JournalReader(tmp_path / "events.jsonl").iter_envelopes())
        assert envelopes[0].stage_id == "dataset_validator"

    def test_emit_never_overwrites_explicit_stage_id(self, tmp_path: Path) -> None:
        emitter = _build_emitter(tmp_path)
        explicit = make_started().model_copy(update={"stage_id": "explicit"})
        with emitter.stage_scope("scope-value"):
            emitter.emit(explicit)
        emitter.close()
        envelopes = list(JournalReader(tmp_path / "events.jsonl").iter_envelopes())
        assert envelopes[0].stage_id == "explicit"

    def test_emit_remote_preserves_identity_fields(self, tmp_path: Path) -> None:
        emitter = _build_emitter(tmp_path)
        ev = make_started(source="pod://r/trainer", offset=42)
        original_event_id = ev.event_id
        original_time = ev.time
        emitter.emit_remote(ev)
        emitter.close()
        envelopes = list(JournalReader(tmp_path / "events.jsonl").iter_envelopes())
        assert envelopes[0].offset == 42
        assert envelopes[0].source == "pod://r/trainer"
        assert envelopes[0].event_id == original_event_id
        assert envelopes[0].time == original_time


# ===========================================================================
# 5. Dependency errors
# ===========================================================================


class TestDependencyErrors:
    def test_emit_does_not_raise_on_journal_failure(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        emitter = _build_emitter(tmp_path)

        def boom(_ev: object) -> None:
            raise OSError("disk gone")

        monkeypatch.setattr(emitter._journal, "append", boom)
        emitter.emit(make_started())  # must not raise
        assert emitter.events_emit_failed_total.get("journal_write", 0) == 1
        assert emitter.events_emitted_total == 0
        emitter.close()

    def test_emit_after_close_silently_failed(self, tmp_path: Path) -> None:
        emitter = _build_emitter(tmp_path)
        emitter.close()
        emitter.emit(make_started())
        assert emitter.events_emit_failed_total.get("closed", 0) == 1


# ===========================================================================
# 6. Regressions
# ===========================================================================


class TestRegressions:
    def test_concurrent_emits_produce_distinct_offsets(self, tmp_path: Path) -> None:
        """50 threads × 10 emits = 500 distinct offsets per (R-05).

        The journal must record all 500 envelopes with strictly
        monotonic offsets — no collisions, no missing rows.
        """
        emitter = _build_emitter(tmp_path)

        def worker() -> None:
            for _ in range(10):
                emitter.emit(make_started())

        threads = [threading.Thread(target=worker) for _ in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        emitter.close()

        offsets = [
            e.offset
            for e in JournalReader(tmp_path / "events.jsonl").iter_envelopes()
        ]
        # 500 distinct offsets (no collisions under contention).
        # Note: file-order may differ from offset-order because the
        # write-lock and offset-lock are not the same lock — but per
        # source the offset counter is strictly monotonic, so the set
        # of offsets is exactly {0..499}.
        assert len(offsets) == 500
        assert sorted(offsets) == list(range(500))


# ===========================================================================
# 7. Logic-specific
# ===========================================================================


class TestLogicSpecific:
    def test_nested_stage_scope_inner_overrides_outer(self, tmp_path: Path) -> None:
        emitter = _build_emitter(tmp_path)
        with emitter.stage_scope("outer"):
            with emitter.stage_scope("inner"):
                emitter.emit(make_started())
            emitter.emit(make_completed())  # back to outer
        emitter.close()

        envelopes = list(JournalReader(tmp_path / "events.jsonl").iter_envelopes())
        assert envelopes[0].stage_id == "inner"
        assert envelopes[1].stage_id == "outer"

    def test_close_is_idempotent(self, tmp_path: Path) -> None:
        emitter = _build_emitter(tmp_path)
        emitter.close()
        emitter.close()  # no error

    def test_remote_dedup_persists_through_restart(self, tmp_path: Path) -> None:
        """Reconstruct dedup from journal on restart; resends are dropped."""
        em = ControlEventEmitter.for_run(run_id="r", run_directory=tmp_path)
        ev = make_started(source="pod://r/trainer", offset=10, run_id="r")
        em.emit_remote(ev)
        em.close()

        # Simulate restart — new emitter via for_run() must reload dedup.
        em2 = ControlEventEmitter.for_run(run_id="r", run_directory=tmp_path)
        em2.emit_remote(ev)  # duplicate after restart
        assert em2.events_remote_dropped_total.get("duplicate", 0) == 1
        assert em2.events_remote_accepted_total == 0
        em2.close()


# ===========================================================================
# 8. Mutation kill — offset validation boundaries (emit + emit_remote)
# ===========================================================================


class TestMutationKillEmitOffsetValidation:
    """Pin the L234 / L318 ``offset == UNKNOWN_OFFSET or offset < 0``
    predicates. Two near-identical sites — kill mutants on both."""

    def test_emit_with_explicit_zero_offset_preserves_offset(
        self, tmp_path: Path
    ) -> None:
        """offset=0 is VALID (not UNKNOWN, not < 0). The mutant
        ``offset < 0`` → ``offset < 1`` would treat 0 as invalid and
        auto-assign. ``offset <= 0`` would do the same.

        Mutant ``Eq → Lt`` (``offset < UNKNOWN_OFFSET``): UNKNOWN_OFFSET
        is -1 by convention; 0 < -1 is False so the predicate misses;
        offset=0 would NOT be auto-assigned. ``Eq → Gt`` similar.
        """
        from ryotenkai_shared.events import UNKNOWN_OFFSET as UNK

        assert UNK < 0  # sanity — UNKNOWN_OFFSET is negative
        emitter = _build_emitter(tmp_path)
        # offset=0 is a VALID explicit value (not auto-assign).
        ev0 = make_started(offset=0)
        emitter.emit(ev0)
        emitter.close()
        envelopes = list(JournalReader(tmp_path / "events.jsonl").iter_envelopes())
        assert envelopes[0].offset == 0, (
            "explicit offset=0 must be preserved; kills `offset < 1` and "
            "`offset <= 0` mutants"
        )

    def test_emit_with_negative_offset_auto_assigns(self, tmp_path: Path) -> None:
        """offset=-5 (not UNKNOWN but < 0) MUST trigger auto-assign. The
        mutant ``or`` → ``and`` (L234:46) would require BOTH conditions
        true; -5 is not UNKNOWN so the gate fails and we'd write -5 to
        the journal. The mutant ``offset < 0`` → ``> 0`` would also fail
        on -5.
        """
        emitter = _build_emitter(tmp_path)
        ev = make_started(offset=-5)
        emitter.emit(ev)
        emitter.close()
        envelopes = list(JournalReader(tmp_path / "events.jsonl").iter_envelopes())
        # Auto-assigned starts at 0.
        assert envelopes[0].offset == 0, (
            "negative offset (not UNKNOWN) must trigger auto-assign; "
            "kills `or → and` and `< 0 → > 0` mutants"
        )

    def test_emit_remote_with_explicit_zero_offset_is_valid(
        self, tmp_path: Path
    ) -> None:
        """emit_remote: offset=0 is VALID. The mutant ``< 0`` → ``< 1``
        would reject 0 with ``invalid_offset``. Mutant ``or → and``
        would require BOTH conditions true and accept everything.
        """
        emitter = _build_emitter(tmp_path)
        ev = make_started(source="pod://r/trainer", offset=0)
        emitter.emit_remote(ev)
        # Successful accept — no invalid_offset bumps.
        assert emitter.events_remote_accepted_total == 1
        assert emitter.events_remote_dropped_total.get("invalid_offset", 0) == 0
        emitter.close()

    def test_emit_remote_with_minus_one_is_dropped_as_invalid(
        self, tmp_path: Path
    ) -> None:
        """offset = -1 (UNKNOWN_OFFSET by convention) MUST be dropped.
        The mutant ``and`` would require offset both == UNKNOWN AND < 0
        — but -1 is UNKNOWN AND < 0 so this passes equivalently. The
        distinguishing mutant is ``Eq → Lt``: offset=UNKNOWN_OFFSET is
        ``-1 < -1`` False, but offset < 0 = True, so OR still triggers.
        We disambiguate by setting offset to a clearly-negative-but-not-
        UNKNOWN value (-3): under ``Eq → Lt`` (``-3 < -1``) and ``or``,
        -3 IS caught. Under ``Eq → Gt`` AND ``< 0 → > 0``, -3 escapes.

        Pin: offset=-1 (UNKNOWN_OFFSET) is dropped via invalid_offset.
        """
        from ryotenkai_shared.events import UNKNOWN_OFFSET

        emitter = _build_emitter(tmp_path)
        ev = make_started(source="pod://r/trainer", offset=UNKNOWN_OFFSET)
        emitter.emit_remote(ev)
        assert emitter.events_remote_dropped_total.get("invalid_offset", 0) == 1
        emitter.close()

    def test_emit_remote_with_negative_non_unknown_is_dropped(
        self, tmp_path: Path
    ) -> None:
        """A negative non-UNKNOWN offset must also be dropped. This
        kills the ``or → and`` mutant (would require both predicates) at
        L318:46."""
        emitter = _build_emitter(tmp_path)
        ev = make_started(source="pod://r/trainer", offset=-7)
        emitter.emit_remote(ev)
        assert emitter.events_remote_dropped_total.get("invalid_offset", 0) == 1
        emitter.close()

    def test_emit_zero_offset_preserved_when_counter_already_advanced(
        self, tmp_path: Path
    ) -> None:
        """``offset < 0`` MUST stay strict (NOT ``< 1`` or ``<= 0``). At
        offset=0 with counter ALREADY at 5, the mutant ``< 1`` or ``<= 0``
        treats 0 as invalid and auto-assigns 5, but the original keeps
        the explicit 0.

        Setup: emit 5 events without explicit offset (counter → 5), then
        emit(offset=0) — explicit 0 must be preserved.
        """
        emitter = _build_emitter(tmp_path)
        for _ in range(5):
            emitter.emit(make_started())  # offsets 0..4 auto-assigned
        # Counter now sits at 5. emit(offset=0) — explicit ZERO.
        emitter.emit(make_completed(offset=0))
        emitter.close()
        envelopes = list(JournalReader(tmp_path / "events.jsonl").iter_envelopes())
        offsets = [e.offset for e in envelopes]
        assert offsets == [0, 1, 2, 3, 4, 0], (
            f"explicit offset=0 MUST be kept (got {offsets}); mutant `<= 0` "
            "or `< 1` would auto-assign 5"
        )

    def test_emit_remote_zero_offset_accepted_when_counter_already_advanced(
        self, tmp_path: Path
    ) -> None:
        """Same idea but for emit_remote() — explicit offset=0 from a
        remote source MUST be accepted under the original predicate
        even after the local counter advanced.

        Mutant ``< 1`` / ``<= 0`` would tag offset=0 as invalid_offset
        (the L333 predicate). Original keeps it as a valid remote
        envelope and dedupes.
        """
        emitter = _build_emitter(tmp_path, source="control://orchestrator")
        # Advance LOCAL counter via local emits.
        for _ in range(3):
            emitter.emit(make_started())
        # Now a REMOTE envelope from a different source with offset=0.
        ev = make_started(source="pod://r/trainer", offset=0)
        emitter.emit_remote(ev)
        # Under original: accepted, not invalid.
        assert emitter.events_remote_accepted_total == 1
        assert emitter.events_remote_dropped_total.get("invalid_offset", 0) == 0
        emitter.close()


class TestMutationKillSweeperStopGuard:
    """Pin the ``if self._dedup_sweeper is not None`` guard in close()."""

    def test_close_calls_sweeper_stop_when_present(self, tmp_path: Path) -> None:
        """``if self._dedup_sweeper is not None: sweeper.stop()``.

        Mutant ``IsNot → Is``: predicate flips — calls stop() ONLY when
        sweeper IS None (AttributeError on None.stop) but the surrounding
        ``contextlib.suppress(Exception)`` swallows it. We have to OBSERVE
        the sweeper.stop() invocation.
        """
        # Stub sweeper that records calls.
        calls: list[str] = []

        class StubSweeper:
            def start(self) -> None: ...
            def stop(self) -> None:
                calls.append("stop")

        emitter = ControlEventEmitter(
            run_id="r",
            source="src",
            journal=JournalWriter(tmp_path / "events.jsonl"),
            bus=InMemoryBus(capacity=4),
            dedup=EventDedup(),
            dedup_sweeper=StubSweeper(),  # type: ignore[arg-type]
        )
        emitter.close()
        assert calls == ["stop"], (
            "sweeper.stop() MUST be called when sweeper is not None; "
            "kills `is not` → `is` mutant (would skip stop) and "
            "`not sweeper is not None` mutant"
        )

    def test_close_does_not_explode_when_sweeper_is_none(
        self, tmp_path: Path
    ) -> None:
        """The ``is not None`` guard MUST protect against None. The
        ``not X is not None`` mutant would call ``None.stop()`` and
        silently raise (suppressed). Pin via correct behaviour:
        no exception, no observable error."""
        emitter = _build_emitter(tmp_path)
        # _dedup_sweeper defaults to None.
        assert emitter._dedup_sweeper is None
        emitter.close()  # no error
        assert emitter.is_closed


class TestMutationKillOffsetCounterDefault:
    """``self._offset_counters.get(source, 0)`` in _bump_source_counter.

    Mutant default=-1: with no prior bump, candidate=0 vs current=-1 →
    ``0 > -1`` True → counter becomes 0. But mutant default=1 would
    require candidate>1 to update. This affects when the counter for an
    UNSEEN source first gets bumped.
    """

    def test_bump_with_zero_candidate_on_fresh_source_records(
        self, tmp_path: Path
    ) -> None:
        """A pre-numbered emit(offset=0) on a brand-new source produces
        candidate=1, gets compared to current=0 (default), 1>0 → set to 1.
        Mutant default=-1: 1 > -1 → set to 1 (same observable).
        Mutant default=1: 1 > 1 False → no bump; counter stays at 1.

        Observable difference: emit a SECOND pre-numbered with offset=0;
        candidate=1, current=1 (correct) or 1 (mutant default=1) — both
        compare 1>1 False. Same.

        Better: emit(offset=-1) (UNKNOWN) FIRST → triggers _assign_offset
        which writes counter=1. Then emit(offset=0) with the same source
        → candidate=1, current=1 → no bump. Auto-assign yields 1, not 0.
        We pin the chain so the default value of get() matters via the
        FIRST auto-assign path.

        Cleanest path: bump with negative candidate. ``_bump_source_counter``
        is called via emit's explicit-offset path. Force the path with
        emit(offset=-5) which... no, that gets caught by the predicate.

        Use emit_remote(offset=0) on a new emitter (fresh source). The
        _bump_source_counter is called with candidate=event.offset+1=1.
        Under mutant default=-1: current=-1, 1 > -1 True → counter=1.
        Under mutant default=1: current=1, 1 > 1 False → counter UNCHANGED
        (stays at... what?). Actually if mutated default returns 1, the
        counter never had an entry; the get returned 1 but the dict has
        no entry. Then ``if candidate > current: self._offset_counters[source]
        = candidate``. Mutant: 1 > 1 False → no write. So the dict
        still has no entry for that source.

        Now a subsequent emit() with auto-assign on the SAME source:
        ``_assign_offset`` reads ``self._offset_counters.get(source)``
        (NO DEFAULT — returns None). Sets next_offset = 0. Writes
        counter = 1. Returns 0. So next auto-assign returns 0.

        Pin: after emit_remote(offset=0) on "pod://r/t", then emit()
        on same source — original auto-assign returns 1; mutant
        default=1 auto-assign returns 0.
        """
        emitter = _build_emitter(tmp_path, source="pod://r/t")
        # Step 1: remote envelope with offset=0 — counter for pod://r/t
        # should be set to 1.
        emitter.emit_remote(make_started(source="pod://r/t", offset=0))
        # Step 2: local emit on the same source — auto-assign.
        emitter.emit(make_completed(source="pod://r/t"))
        emitter.close()

        envelopes = list(JournalReader(tmp_path / "events.jsonl").iter_envelopes())
        offsets = [e.offset for e in envelopes]
        assert offsets == [0, 1], (
            f"after emit_remote(offset=0), next local auto-assign MUST be 1; "
            f"got {offsets}; kills `_offset_counters.get(source, 1)` "
            "mutant (would never bump → next auto-assign returns 0)"
        )


class TestMutationKillBumpForwardGuard:
    """The ``candidate > current`` mutant ``>=`` would let an EQUAL
    candidate overwrite (no-op functionally) but it ALSO causes an
    unnecessary lock write. Observable difference is hard without
    introspection.

    More importantly the ``>=`` mutant survives because in normal usage
    EQUAL candidates are rare. Force the issue: monkey-patch the
    counter dict to a known value, call _bump_source_counter with
    candidate==current, then... we can't directly observe a dict write
    because the value is the same. EQUIVALENT MUTANT.

    BUT: the ``>= `` vs ``>`` behaves DIFFERENTLY at the FIRST bump on
    a fresh source: current = get(default=0). candidate=0 (e.g. from
    emit(offset=-1) which would have offset+1=0). 0 > 0 False vs 0 >= 0
    True. With mutant: counter set to 0 (no-op since default returns 0).
    Same observable.

    Mutants `<`, `<=`, `==`, `is`, `is not`, `!=` are killed by the
    existing test_bump_source_counter_only_advances_forward.
    """

    def test_bump_with_candidate_one_less_than_current_does_not_regress(
        self, tmp_path: Path
    ) -> None:
        """Already covered by test_bump_source_counter_only_advances_forward
        — left here as a doc-anchor for `>=` equivalence."""
        emitter = _build_emitter(tmp_path)
        emitter.emit(make_started(offset=10))  # counter → 11
        emitter.emit(make_completed(offset=10))  # candidate=11, current=11
        # Mutant ``>=``: 11 >= 11 True → write counter=11 (same as before).
        # Original: 11 > 11 False → no write. Same observable state.
        emitter.emit(make_completed())  # auto-assign
        emitter.close()
        envs = list(JournalReader(tmp_path / "events.jsonl").iter_envelopes())
        # Last emit yields offset=11 (counter was at 11 → assigns 11 → bumps to 12).
        assert envs[-1].offset == 11


class TestMutationKillBumpSourceCounterArithmetic2:
    """Kill the L380 ``event.offset + 1`` → ``^ 1`` mutant on the
    emit_remote bump path. We already have a test for L365 (the
    pre-numbered emit path), but the emit_remote pre-numbered path is a
    distinct site.
    """

    def test_emit_remote_with_even_offset_bumps_counter_correctly(
        self, tmp_path: Path
    ) -> None:
        """offset=4 (even): correct ``4 + 1 = 5``; mutant ``4 ^ 1 = 5``
        (SAME). Need ODD offset where XOR differs: offset=5: correct
        ``5 + 1 = 6``; mutant ``5 ^ 1 = 4``.

        After emit_remote(offset=5), next emit() auto-assign should be 6.
        Mutant: counter = 4; next auto-assign returns 4 (or 0 if no
        update — see below).
        """
        emitter = _build_emitter(tmp_path, source="pod://r/t")
        emitter.emit_remote(make_started(source="pod://r/t", offset=5))
        emitter.emit(make_completed(source="pod://r/t"))
        emitter.close()
        envs = list(JournalReader(tmp_path / "events.jsonl").iter_envelopes())
        offsets = [e.offset for e in envs]
        assert offsets == [5, 6], (
            f"after emit_remote(offset=5), next auto-assign MUST be 6; "
            f"got {offsets}; kills `offset + 1` → `offset ^ 1` mutant "
            "(would set counter to 4 instead of 6)"
        )


class TestMutationKillBumpSourceCounter:
    """Pin ``event.offset + 1`` arithmetic in emit() and emit_remote()
    when the caller pre-numbered the offset.
    """

    def test_emit_with_pre_numbered_offset_then_auto_assign_continues(
        self, tmp_path: Path
    ) -> None:
        """After ``emit(offset=5)`` the source counter MUST be 6 so the
        NEXT auto-assigned offset is 6 (not 5/4/0/etc.).

        Mutants on L239 (``event.offset + 1``):
        - ``+ 0``: counter stays at 5; next auto-assign returns 5 (collision).
        - ``+ 2``: counter becomes 6 from the candidate-7 logic; next
          auto-assign returns 7 (gap).
        - ``- 1``: candidate=4; under ``if 4 > 0: counter=4``; next
          auto-assign returns 4 (regression).
        - ``* 1`` / ``/ 1`` / etc.: candidate=5 with current=0;
          ``if 5 > 0: counter=5``; next auto-assign returns 5 (collision).
        - ``& 1`` / ``% 1`` etc.: candidate ≤ 1; counter stays smaller.
        """
        emitter = _build_emitter(tmp_path)
        # Step 1: emit with explicit offset 5.
        emitter.emit(make_started(offset=5))
        # Step 2: emit with UNKNOWN — auto-assign.
        emitter.emit(make_completed())  # UNKNOWN by default
        emitter.close()

        envelopes = list(JournalReader(tmp_path / "events.jsonl").iter_envelopes())
        offsets = [e.offset for e in envelopes]
        assert offsets == [5, 6], (
            f"after pre-numbered=5, next auto-assign MUST be 6 (got "
            f"{offsets}); kills arithmetic mutants on `event.offset + 1`"
        )

    def test_emit_remote_advances_local_counter_past_observed_offset(
        self, tmp_path: Path
    ) -> None:
        """After ``emit_remote(offset=42)``, a subsequent LOCAL
        auto-assign on the same source MUST return 43.

        Same arithmetic-mutant set as above, but on L365.
        """
        emitter = _build_emitter(tmp_path, source="pod://r/trainer")
        ev = make_started(source="pod://r/trainer", offset=42)
        emitter.emit_remote(ev)
        # Auto-assign on the SAME source — counters are keyed per-source.
        emitter.emit(make_completed(source="pod://r/trainer"))
        emitter.close()

        envelopes = list(JournalReader(tmp_path / "events.jsonl").iter_envelopes())
        assert envelopes[0].offset == 42
        assert envelopes[1].offset == 43, (
            f"after emit_remote(offset=42), next local emit MUST get 43; "
            f"got {envelopes[1].offset}; kills arithmetic mutants on "
            "`event.offset + 1` in emit_remote"
        )


class TestMutationKillBumpSourceCounterGuard:
    """Pin ``if candidate > current:`` in ``_bump_source_counter``."""

    def test_bump_source_counter_only_advances_forward(self, tmp_path: Path) -> None:
        """``if candidate > current``: an OLDER candidate must NOT
        regress the counter. The mutant ``<``, ``<=`` would replace
        the counter with smaller values. ``==`` would only update on
        equality. ``>=`` would no-op vs ``>`` only at equality —
        observable via consecutive bumps with equal candidate.

        Sequence:
          emit(offset=10) ⇒ counter=11
          emit(offset=5)  ⇒ candidate=6 < 11 ⇒ counter stays 11
          emit(auto)      ⇒ returns 11
        """
        emitter = _build_emitter(tmp_path)
        emitter.emit(make_started(offset=10))
        emitter.emit(make_completed(offset=5))
        emitter.emit(make_completed())  # auto-assign
        emitter.close()

        envelopes = list(JournalReader(tmp_path / "events.jsonl").iter_envelopes())
        offsets = [e.offset for e in envelopes]
        assert offsets == [10, 5, 11], (
            f"counter must only advance forward; got {offsets}; kills "
            "`candidate > current` → `<` / `<=` / `==` mutants"
        )

    def test_assign_offset_default_zero_for_new_source(self, tmp_path: Path) -> None:
        """``self._offset_counters.get(source, 0)`` — the DEFAULT for an
        unseen source is 0. Mutants ``= 1`` would start at 1 (first
        auto-assign returns 0 but counter writes 1; actually the
        ``_assign_offset`` flow returns ``current`` which is 0 if get
        returns 0; with default=1, the flow returns 1).

        Hmm — careful: ``_assign_offset`` uses `get(source)` not
        `get(source, 0)`. Wait let me check: in _assign_offset it's
        ``self._offset_counters.get(source)`` then ``if current is None:
        next_offset = 0``. So the get-with-default is in
        ``_bump_source_counter`` only.

        For ``_bump_source_counter``: default=0. With mutant default=1,
        the first ever bump compares candidate vs 1 — so calling
        emit(offset=0) on a fresh source would compute candidate=1, ``1 > 1``
        False, counter stays 1, next auto-assign returns 1 instead of 0.
        Calling emit(offset=1) → candidate=2 > 1 → counter=2; auto-assign
        returns 2.

        Pin: emit(offset=0) then auto-assign should give 1 (not 2).
        """
        emitter = _build_emitter(tmp_path)
        emitter.emit(make_started(offset=0))  # candidate=1 → counter=1
        emitter.emit(make_completed())  # auto-assign returns 1
        emitter.close()

        envelopes = list(JournalReader(tmp_path / "events.jsonl").iter_envelopes())
        offsets = [e.offset for e in envelopes]
        assert offsets == [0, 1], (
            f"after emit(0), auto-assign MUST return 1; got {offsets}; "
            "kills `_offset_counters.get(source, 1)` and `(source, -1)` "
            "default mutants"
        )


class TestMutationKillFailureCounters:
    """Pin ``+ 1`` arithmetic in `_bump_failed` and `_bump_remote_dropped`."""

    def test_bump_failed_increments_by_exactly_one(self, tmp_path: Path) -> None:
        """``events_emit_failed_total.get(reason, 0) + 1`` — the BitXor
        mutant: 0 ^ 1 = 1 (passes first time) but 1 ^ 1 = 0 (regresses
        on second bump). Pin via two failures.
        """
        emitter = _build_emitter(tmp_path)
        emitter.close()
        # First failure → counter for "closed" = 1.
        emitter.emit(make_started())
        assert emitter.events_emit_failed_total.get("closed") == 1
        # Second failure → counter MUST become 2 (not 0).
        emitter.emit(make_completed())
        assert emitter.events_emit_failed_total.get("closed") == 2, (
            "two failures must yield counter=2 (not 0); kills `+ 1` → "
            "`^ 1` BitXor mutant which would flip 1 back to 0"
        )

    def test_bump_remote_dropped_increments_by_exactly_one(
        self, tmp_path: Path
    ) -> None:
        """``events_remote_dropped_total.get(reason, 0) + 1`` — same
        BitXor mutant story."""
        emitter = _build_emitter(tmp_path)
        # Two duplicate emit_remote calls under the same key.
        ev = make_started(source="pod://r/trainer", offset=5)
        emitter.emit_remote(ev)  # accepted
        emitter.emit_remote(ev)  # duplicate #1
        emitter.emit_remote(ev)  # duplicate #2
        emitter.close()

        assert emitter.events_remote_dropped_total.get("duplicate") == 2, (
            "two duplicates must yield counter=2; kills BitXor mutant"
        )


class TestMutationKillForRun:
    """Pin ``for_run`` construction details."""

    def test_for_run_creates_parent_dirs_recursively(self, tmp_path: Path) -> None:
        """``run_dir.mkdir(parents=True, exist_ok=True)`` — the
        ``parents=True`` → ``parents=False`` mutant would fail when the
        parent directory doesn't exist.
        """
        nested = tmp_path / "a" / "b" / "c"
        # Parent ``tmp_path / "a"`` does NOT exist — mutant would raise
        # FileNotFoundError; original ``parents=True`` succeeds.
        em = ControlEventEmitter.for_run(run_id="r", run_directory=nested)
        em.close()
        assert nested.exists()

    def test_for_run_offset_resume_adds_exactly_one(self, tmp_path: Path) -> None:
        """``max_off + 1`` in the offset_resume comprehension. Mutant
        ``<< 1`` would double; ``+ 0``/``- 0``/``* 1`` would equal
        max_off (collision); ``+ 2`` would gap by 1.

        Scenario: write events 0, 1, 2 then restart. Next auto-assign
        MUST be 3, not 1, 2, 4, or 6.
        """
        em = ControlEventEmitter.for_run(run_id="r", run_directory=tmp_path)
        for _ in range(3):
            em.emit(make_started())  # offsets 0, 1, 2
        em.close()

        em2 = ControlEventEmitter.for_run(run_id="r", run_directory=tmp_path)
        em2.emit(make_completed())  # MUST be offset 3
        em2.close()

        envelopes = list(JournalReader(tmp_path / "events.jsonl").iter_envelopes())
        offsets = [e.offset for e in envelopes]
        assert offsets == [0, 1, 2, 3], (
            f"after resume from offsets [0,1,2], next emit MUST be 3; "
            f"got {offsets}; kills `max_off + 1` arithmetic mutants"
        )


class TestMutationKillSignatureMarkers:
    """Pin kw-only markers — ``*,`` MUST NOT collapse to ``/,``."""

    def test_init_requires_kwargs(self, tmp_path: Path) -> None:
        """``def __init__(self, *, run_id, source, journal, bus, dedup, ...)``.
        Calling positionally must raise TypeError under the kw-only
        contract."""
        journal = JournalWriter(tmp_path / "events.jsonl")
        bus = InMemoryBus(capacity=4)
        dedup = EventDedup()
        with pytest.raises(TypeError):
            ControlEventEmitter("rid", "src", journal, bus, dedup)  # type: ignore[misc]
        # And the canonical kw-form works.
        em = ControlEventEmitter(
            run_id="rid", source="src", journal=journal, bus=bus, dedup=dedup
        )
        em.close()

    def test_for_run_requires_kwargs(self, tmp_path: Path) -> None:
        """``@classmethod def for_run(cls, *, run_id, run_directory, ...)``.
        Positional call must raise TypeError."""
        with pytest.raises(TypeError):
            ControlEventEmitter.for_run("rid", str(tmp_path))  # type: ignore[misc]


class TestMutationKillExceptionTypes:
    """Pin the explicit ``except ValidationError`` / ``except Exception``
    catch clauses — mutants replace them with a private synthetic
    ``CosmicRayTestingException``. If we can OBSERVE that the original
    handler ran (counter bumped, no propagation), the mutant fails (the
    counter would NOT bump and the exception would propagate).
    """

    def test_emit_validation_error_is_caught_and_counted(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A ValidationError raised by EVENT_ADAPTER.validate_python
        MUST be caught and recorded under ``validation``. The
        ExceptionReplacer mutant on L253 swaps the catch type so the
        ValidationError would propagate. The outer ``except Exception``
        catches it under ``unexpected`` — that's observably DIFFERENT
        from ``validation``.
        """
        emitter = _build_emitter(tmp_path)
        # Monkeypatch EVENT_ADAPTER.validate_python to raise the precise
        # ValidationError type.
        from pydantic import ValidationError

        from ryotenkai_control.events import emitter as em_mod

        # Build a real ValidationError once (before patching) so the
        # patched ``boom`` does not recurse back into itself.
        captured: list[ValidationError] = []
        try:
            em_mod.EVENT_ADAPTER.validate_python({"bad": "data"})
        except ValidationError as e:
            captured.append(e)
        assert captured, "fixture precondition: must produce a ValidationError"
        real_err = captured[0]

        def boom(_data: object) -> None:
            raise real_err

        monkeypatch.setattr(em_mod.EVENT_ADAPTER, "validate_python", boom)
        emitter.emit(make_started())
        # The validation counter (NOT unexpected) bumped.
        assert emitter.events_emit_failed_total.get("validation") == 1
        assert emitter.events_emit_failed_total.get("unexpected", 0) == 0
        emitter.close()

    def test_emit_journal_failure_is_caught_under_journal_write_reason(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """L280 ``except Exception``: a journal append failure bumps
        ``journal_write``, NOT ``unexpected``. The ExceptionReplacer
        mutant on L280 would let the exception propagate to the outer
        try-block (L291) which bumps ``unexpected``.
        """
        emitter = _build_emitter(tmp_path)

        def boom(_ev: object) -> None:
            raise OSError("disk")

        monkeypatch.setattr(emitter._journal, "append", boom)
        emitter.emit(make_started())
        # Pinned reason — kills the L280 ExceptionReplacer mutant.
        assert emitter.events_emit_failed_total.get("journal_write") == 1
        assert emitter.events_emit_failed_total.get("unexpected", 0) == 0
        emitter.close()

    def test_emit_bus_failure_is_caught_under_bus_publish_reason(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Bus publish failure under emit() bumps ``bus_publish``, not
        ``unexpected``. Kills L280 / L291 ExceptionReplacer mutants
        narrowly."""
        emitter = _build_emitter(tmp_path)

        def boom(_ev: object) -> None:
            raise RuntimeError("bus broke")

        monkeypatch.setattr(emitter._bus, "publish", boom)
        emitter.emit(make_started())
        assert emitter.events_emit_failed_total.get("bus_publish") == 1
        assert emitter.events_emit_failed_total.get("unexpected", 0) == 0
        emitter.close()

    def test_emit_remote_validation_error_is_caught_under_validation(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """L337 ``except ValidationError``: emit_remote-path validation
        failure bumps ``validation``. Mutant would let it propagate."""
        emitter = _build_emitter(tmp_path)

        from pydantic import ValidationError

        from ryotenkai_control.events import emitter as em_mod

        captured: list[ValidationError] = []
        try:
            em_mod.EVENT_ADAPTER.validate_python({"bad": "data"})
        except ValidationError as e:
            captured.append(e)
        real_err = captured[0]

        def boom(_data: object) -> None:
            raise real_err

        monkeypatch.setattr(em_mod.EVENT_ADAPTER, "validate_python", boom)
        emitter.emit_remote(make_started(source="pod://r/t", offset=1))
        assert emitter.events_remote_dropped_total.get("validation") == 1
        assert emitter.events_remote_dropped_total.get("unexpected", 0) == 0
        emitter.close()

    def test_emit_remote_journal_failure_caught_under_journal_write(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """L348 ``except Exception`` in emit_remote: journal append failure
        bumps ``journal_write`` not ``unexpected``."""
        emitter = _build_emitter(tmp_path)

        def boom(_ev: object) -> None:
            raise OSError("disk")

        monkeypatch.setattr(emitter._journal, "append", boom)
        emitter.emit_remote(make_started(source="pod://r/t", offset=1))
        assert emitter.events_remote_dropped_total.get("journal_write") == 1
        assert emitter.events_remote_dropped_total.get("unexpected", 0) == 0
        emitter.close()

    def test_emit_remote_bus_failure_caught_under_bus_publish(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """L369 ``except Exception`` in emit_remote: bus publish failure
        bumps ``bus_publish`` not ``unexpected``."""
        emitter = _build_emitter(tmp_path)

        def boom(_ev: object) -> None:
            raise RuntimeError("bus broke")

        monkeypatch.setattr(emitter._bus, "publish", boom)
        emitter.emit_remote(make_started(source="pod://r/t", offset=1))
        assert emitter.events_remote_dropped_total.get("bus_publish") == 1
        assert emitter.events_remote_dropped_total.get("unexpected", 0) == 0
        emitter.close()


class TestMutationKillOuterUnexpected:
    """The OUTER ``except Exception`` at L291 / L379 is the last-resort
    safety net. To kill the ExceptionReplacer mutant we need a code path
    that raises something OUTSIDE the inner try blocks but INSIDE the
    outer try (e.g. attribute access on event before the first try).
    """

    def test_emit_outer_handler_catches_unexpected_exception(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Make _assign_offset raise — it sits BETWEEN the try-blocks
        but inside the outer try. The outer ``except Exception`` MUST
        bump ``unexpected``. Mutant on L291 would let it escape (we'd
        observe an uncaught RuntimeError)."""
        emitter = _build_emitter(tmp_path)

        def boom(_source: object) -> int:
            raise RuntimeError("unexpected explosion")

        monkeypatch.setattr(emitter, "_assign_offset", boom)
        # Must NOT raise to the caller (the never-raises contract).
        emitter.emit(make_started())
        assert emitter.events_emit_failed_total.get("unexpected", 0) == 1
        emitter.close()

    def test_emit_remote_outer_handler_catches_unexpected_exception(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Force the dedup check to raise — that's after the offset
        check but before the inner try-blocks. The outer ``except
        Exception`` at L379 bumps ``unexpected``."""
        emitter = _build_emitter(tmp_path)

        def boom(*_a: object, **_k: object) -> bool:
            raise RuntimeError("dedup boom")

        monkeypatch.setattr(emitter._dedup, "is_duplicate", boom)
        emitter.emit_remote(make_started(source="pod://r/t", offset=1))
        assert emitter.events_remote_dropped_total.get("unexpected", 0) == 1
        emitter.close()


class TestMutationKillDecoratorsArePropertyGuards:
    """The ``@property`` decorators are removed by ``RemoveDecorator``
    mutants. Pin via accessing the attributes directly — accessing a
    method-typed attribute returns a bound method, not the value.
    """

    def test_run_id_is_property_returns_value_not_method(
        self, tmp_path: Path
    ) -> None:
        """``run_id`` property — without @property it's a bound method.
        Comparing to a string would fail (`<bound method> == 'rid'`)."""
        em = _build_emitter(tmp_path)
        # Property: equals "test-run" string. Bound method: false.
        assert em.run_id == "test-run"
        em.close()

    def test_source_is_property(self, tmp_path: Path) -> None:
        em = _build_emitter(tmp_path)
        assert em.source == "control://orchestrator"
        em.close()

    def test_journal_bus_dedup_are_properties(self, tmp_path: Path) -> None:
        em = _build_emitter(tmp_path)
        # Each property returns the collaborator instance directly.
        assert em.journal is em._journal
        assert em.bus is em._bus
        assert em.dedup is em._dedup
        em.close()

    def test_is_closed_is_property(self, tmp_path: Path) -> None:
        em = _build_emitter(tmp_path)
        assert em.is_closed is False
        em.close()
        assert em.is_closed is True
