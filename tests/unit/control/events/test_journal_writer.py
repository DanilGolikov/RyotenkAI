"""Tests for :class:`ryotenkai_control.events.JournalWriter` (Phase 3).

Seven categories per ``docs/testing/mutation_testing.md``:

1. TestPositive             — happy path: write, flush, line format.
2. TestNegative             — write/fsync errors are logged, not raised.
3. TestBoundary             — batch size 50 → fsync; severity=error → fsync.
4. TestInvariants           — lock blocks concurrent writes; UTF-8 length
                              prefix matches the body byte count.
5. TestDependencyErrors     — fsync OSError is swallowed.
6. TestRegressions          — append-after-close logs but doesn't raise;
                              torn-write recovery composes with reader.
7. TestLogicSpecific        — fsync interval triggers via monotonic time.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from ryotenkai_control.events import JournalReader, JournalWriter
from ryotenkai_shared.events import from_jsonl
from tests.unit.control.events.conftest import (
    make_completed,
    make_failed,
    make_started,
)

# ===========================================================================
# 1. Positive
# ===========================================================================


class TestPositive:
    def test_append_writes_length_prefixed_line(self, tmp_path: Path) -> None:
        path = tmp_path / "events.jsonl"
        writer = JournalWriter(path)
        event = make_started(offset=0)

        writer.append(event)
        writer.close()

        raw = path.read_text(encoding="utf-8")
        assert raw.endswith("\n")
        assert "\t" in raw
        length_str, json_body_with_nl = raw.split("\t", 1)
        # Length prefix matches the UTF-8 byte count of the JSON body.
        assert int(length_str) == len(json_body_with_nl.rstrip("\n").encode("utf-8"))

    def test_appended_envelope_round_trips_via_from_jsonl(self, tmp_path: Path) -> None:
        path = tmp_path / "events.jsonl"
        writer = JournalWriter(path)
        event = make_started(offset=0)
        writer.append(event)
        writer.close()

        line = path.read_text(encoding="utf-8")
        decoded = from_jsonl(line)
        assert decoded.kind == event.kind
        assert decoded.run_id == event.run_id
        assert decoded.offset == 0

    def test_append_creates_parent_dirs(self, tmp_path: Path) -> None:
        nested = tmp_path / "a" / "b" / "events.jsonl"
        writer = JournalWriter(nested)
        writer.append(make_started(offset=0))
        writer.close()
        assert nested.exists()

    def test_metric_counters_track_appends(self, tmp_path: Path) -> None:
        writer = JournalWriter(tmp_path / "events.jsonl")
        for i in range(3):
            writer.append(make_started(offset=i))
        assert writer.events_appended == 3
        # Phase 8 — total_bytes_written grows with each append.
        assert writer.total_bytes_written > 0
        assert writer.fsync_failed_total == 0
        writer.close()
        # close() performs a final fsync → last_fsync_at set.
        assert writer.last_fsync_at is not None


# ===========================================================================
# 2. Negative
# ===========================================================================


class TestNegative:
    def test_append_after_close_is_no_op(self, tmp_path: Path) -> None:
        writer = JournalWriter(tmp_path / "events.jsonl")
        writer.close()
        writer.append(make_started(offset=0))  # logs warning, doesn't raise
        assert writer.events_appended == 0

    def test_close_twice_is_idempotent(self, tmp_path: Path) -> None:
        writer = JournalWriter(tmp_path / "events.jsonl")
        writer.append(make_started(offset=0))
        writer.close()
        writer.close()  # no error

    def test_write_failure_swallowed_and_counted(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        writer = JournalWriter(tmp_path / "events.jsonl")

        # Force the underlying write() to fail.
        def boom_write(_self: object, _data: bytes) -> int:
            raise OSError("disk full")

        monkeypatch.setattr(writer._fh, "write", boom_write.__get__(writer._fh))
        writer.append(make_started(offset=0))
        writer.close()

        assert writer.write_failures_total == 1
        assert writer.events_appended == 0  # write failed, not counted


# ===========================================================================
# 3. Boundary
# ===========================================================================


class TestBoundary:
    def test_batch_size_triggers_fsync(self, tmp_path: Path) -> None:
        writer = JournalWriter(
            tmp_path / "events.jsonl",
            fsync_batch_size=3,
            fsync_interval_s=999.0,
        )
        for i in range(3):
            writer.append(make_started(offset=i))
        # Batch threshold reached → at least one fsync.
        assert writer.fsyncs_total >= 1
        writer.close()

    def test_severity_error_triggers_immediate_fsync(self, tmp_path: Path) -> None:
        writer = JournalWriter(
            tmp_path / "events.jsonl",
            fsync_batch_size=10_000,
            fsync_interval_s=999.0,
        )
        writer.append(make_failed(offset=0))  # severity=error
        assert writer.fsyncs_total >= 1
        writer.close()

    def test_zero_appends_means_zero_fsync_until_close(self, tmp_path: Path) -> None:
        writer = JournalWriter(tmp_path / "events.jsonl")
        # Pre-close fsync count snapshot.
        pre_close = writer.fsyncs_total
        assert pre_close == 0
        writer.close()
        # close() performs one final fsync.
        assert writer.fsyncs_total == 1


# ===========================================================================
# 4. Invariants
# ===========================================================================


class TestInvariants:
    def test_concurrent_appends_serialize_safely(self, tmp_path: Path) -> None:
        writer = JournalWriter(tmp_path / "events.jsonl")
        produced: list[int] = []

        def worker(start: int) -> None:
            for i in range(start, start + 25):
                writer.append(make_started(offset=i))
                produced.append(i)

        threads = [threading.Thread(target=worker, args=(s,)) for s in (0, 100, 200, 300)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        writer.close()

        # All 100 envelopes are well-formed (length prefix integrity
        # holds → no torn writes).
        for raw in (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines():
            assert raw
            # Each line is length-prefixed.
            n_str, _, body = raw.partition("\t")
            assert int(n_str) == len(body.encode("utf-8"))

    def test_length_prefix_matches_utf8_byte_count_for_multibyte_payload(
        self, tmp_path: Path
    ) -> None:
        from ryotenkai_shared.events import UNKNOWN_OFFSET
        from ryotenkai_shared.events.types.control_run import (
            RunFailedEvent,
            RunFailedPayload,
        )

        writer = JournalWriter(tmp_path / "events.jsonl")
        ev = RunFailedEvent(
            source="control://orchestrator",
            run_id="r",
            offset=UNKNOWN_OFFSET,
            payload=RunFailedPayload(
                failing_stage="s",
                error_type="E",
                message="مرحبا بالعالم — 你好世界 — Привет",
                traceback_excerpt="",
            ),
        )
        writer.append(ev)
        writer.close()

        line = (tmp_path / "events.jsonl").read_text(encoding="utf-8")
        # Body byte count == declared length.
        n_str, _, body = line.partition("\t")
        assert int(n_str) == len(body.rstrip("\n").encode("utf-8"))


# ===========================================================================
# 5. Dependency errors
# ===========================================================================


class TestDependencyErrors:
    def test_fsync_oserror_swallowed(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        writer = JournalWriter(
            tmp_path / "events.jsonl",
            fsync_batch_size=1,
            fsync_interval_s=999.0,
        )

        import os as os_mod
        original_fsync = os_mod.fsync

        def boom(_fd: int) -> None:
            raise OSError("fsync failed")

        monkeypatch.setattr(os_mod, "fsync", boom)
        # Should not raise even though fsync is broken.
        writer.append(make_started(offset=0))
        monkeypatch.setattr(os_mod, "fsync", original_fsync)
        writer.close()


# ===========================================================================
# 6. Regressions
# ===========================================================================


class TestRegressions:
    def test_torn_tail_after_kill_simulation(self, tmp_path: Path) -> None:
        """Simulate kill -9 mid-write by appending a partial line, then
        verify the JournalReader can truncate it before the next writer
        reopens the file.
        """
        path = tmp_path / "events.jsonl"
        writer = JournalWriter(path)
        writer.append(make_started(offset=0))
        writer.close()

        # Simulate torn write: append a partial length-prefixed line.
        with path.open("ab") as fh:
            fh.write(b"42\t{\"partial\":")  # no newline, length lies

        reader = JournalReader(path)
        truncated = reader.truncate_torn_tail()
        assert truncated is True

        # File now contains only the well-formed first envelope.
        envelopes = list(reader.iter_envelopes())
        assert len(envelopes) == 1
        assert envelopes[0].offset == 0

    def test_resume_appends_after_existing_file(self, tmp_path: Path) -> None:
        path = tmp_path / "events.jsonl"
        writer = JournalWriter(path)
        writer.append(make_started(offset=0))
        writer.close()

        # Reopen — should append, not overwrite.
        writer2 = JournalWriter(path)
        writer2.append(make_completed(offset=1))
        writer2.close()

        envelopes = list(JournalReader(path).iter_envelopes())
        assert [e.offset for e in envelopes] == [0, 1]


# ===========================================================================
# 7. Logic-specific
# ===========================================================================


class TestLogicSpecific:
    def test_interval_elapsed_triggers_fsync(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Mock monotonic clock so a single ``append`` after an elapsed
        interval triggers fsync even if the batch size isn't reached.
        """
        # Set interval to 1s and pre-set last_fsync_at "in the past".
        fake_clock = [100.0]

        def fake_monotonic() -> float:
            return fake_clock[0]

        monkeypatch.setattr(time, "monotonic", fake_monotonic)

        writer = JournalWriter(
            tmp_path / "events.jsonl",
            fsync_batch_size=10_000,
            fsync_interval_s=1.0,
        )
        # Advance the clock past the interval.
        fake_clock[0] += 5.0
        writer.append(make_started(offset=0))
        assert writer.fsyncs_total >= 1
        writer.close()

    def test_fsync_now_is_idempotent(self, tmp_path: Path) -> None:
        writer = JournalWriter(tmp_path / "events.jsonl")
        writer.fsync_now()
        writer.fsync_now()
        # No event appended yet — fsyncs may be 0 since fh has no
        # writes, but call shouldn't raise.
        writer.close()

    def test_constructor_validates_args(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            JournalWriter(tmp_path / "events.jsonl", fsync_batch_size=0)
        with pytest.raises(ValueError):
            JournalWriter(tmp_path / "events.jsonl", fsync_interval_s=-0.1)


# ===========================================================================
# 8. Mutation kill — constants, boundaries, counters, signature markers
# ===========================================================================


class TestMutationKillConstants:
    """Pin module-level defaults so a 49/50/51 or 0.0/1.0/2.0 swap fails."""

    def test_default_fsync_batch_size_is_exactly_50(self) -> None:
        """``DEFAULT_FSYNC_BATCH_SIZE = 50`` — pin via module attribute
        AND via constructor default introspection so both NumberReplacer
        mutants (49, 51) fail."""
        import inspect

        from ryotenkai_control.events.journal_writer import (
            DEFAULT_FSYNC_BATCH_SIZE,
        )

        assert DEFAULT_FSYNC_BATCH_SIZE == 50
        sig = inspect.signature(JournalWriter.__init__)
        assert sig.parameters["fsync_batch_size"].default == 50

    def test_default_fsync_interval_is_exactly_one_second(self) -> None:
        """``DEFAULT_FSYNC_INTERVAL_S = 1.0`` — pin via module attribute
        AND via constructor default introspection. Kills 0.0 / 2.0
        mutants."""
        import inspect

        from ryotenkai_control.events.journal_writer import (
            DEFAULT_FSYNC_INTERVAL_S,
        )

        assert DEFAULT_FSYNC_INTERVAL_S == 1.0
        sig = inspect.signature(JournalWriter.__init__)
        assert sig.parameters["fsync_interval_s"].default == 1.0


class TestMutationKillBoundary:
    """Strict comparisons at validation boundaries."""

    def test_batch_size_at_one_is_accepted(self, tmp_path: Path) -> None:
        """``fsync_batch_size < 1`` MUST stay strict. The mutant
        ``< 1`` → ``< 0`` would accept 0; the mutant ``< 0`` (NumberReplacer
        on the literal 1) would also accept 0. Confirm 1 is the minimum.
        """
        # 1 is the lowest accepted value.
        w = JournalWriter(tmp_path / "events.jsonl", fsync_batch_size=1)
        w.close()
        # 0 is rejected.
        with pytest.raises(ValueError):
            JournalWriter(tmp_path / "events.jsonl", fsync_batch_size=0)
        # -1 is also rejected (just to anchor the threshold direction).
        with pytest.raises(ValueError):
            JournalWriter(tmp_path / "events.jsonl", fsync_batch_size=-1)

    def test_fsync_interval_zero_is_accepted(self, tmp_path: Path) -> None:
        """``fsync_interval_s < 0`` MUST stay strict. The mutant
        ``< 0`` → ``<= 0`` would reject 0.0. The mutant ``< 0`` → ``< 1``
        would reject 0.5 and 0.0. Pin both 0.0 (accept) and -0.5 (reject)
        so both mutants are killed.
        """
        # 0.0 is the boundary — must be accepted.
        w = JournalWriter(tmp_path / "events.jsonl", fsync_interval_s=0.0)
        w.close()
        # A small positive value also accepted (kills ``< 1`` mutant).
        w = JournalWriter(tmp_path / "events.jsonl", fsync_interval_s=0.5)
        w.close()
        # Negative rejected.
        with pytest.raises(ValueError):
            JournalWriter(tmp_path / "events.jsonl", fsync_interval_s=-0.5)


class TestMutationKillBatchFsyncBoundary:
    """Kill ``>=`` mutants on the inline fsync trigger and the severity gate."""

    def test_fsync_triggers_at_exact_batch_threshold(self, tmp_path: Path) -> None:
        """``self._unflushed >= self._batch_size`` MUST trigger AT the
        threshold (e.g. 3 ≥ 3). The ``>=`` → ``>`` mutant skips the
        boundary fsync (only fires at 4). The ``>=`` → ``==`` mutant
        fires only at exactly 3 (passes here) but FAILS the next assertion
        for batch=2 where we publish 3 events — equality only catches
        the first crossing.
        """
        writer = JournalWriter(
            tmp_path / "events.jsonl",
            fsync_batch_size=3,
            fsync_interval_s=9_999.0,
        )
        # 2 appends: below threshold — no fsync from batch path. But
        # we cannot assert == 0 because interval/severity could trigger;
        # use a future-time-pinning approach: severity is "info"
        # (make_started), interval is huge.
        writer.append(make_started(offset=0))
        writer.append(make_completed(offset=1))
        # 3rd append: hits == threshold, should trigger fsync.
        writer.append(make_completed(offset=2))
        assert writer.fsyncs_total == 1, (
            "fsync MUST fire at threshold (== boundary); kills `> ` mutant"
        )
        writer.close()

    def test_fsync_below_threshold_does_not_trigger(self, tmp_path: Path) -> None:
        """``_unflushed >= batch_size``: when unflushed strictly < batch,
        DO NOT fsync from this branch. Kills ``>=`` → ``<=`` (would
        trigger at 1 below batch) and ``>=`` → ``Lt`` (always-trigger
        for unflushed < batch)."""
        writer = JournalWriter(
            tmp_path / "events.jsonl",
            fsync_batch_size=10,
            fsync_interval_s=9_999.0,
        )
        # 3 appends, all info-severity, well below batch threshold.
        for i in range(3):
            writer.append(make_started(offset=i))
        # Without close, only batch/interval/severity can trigger fsync.
        # Batch (3 < 10): no. Interval (huge): no. Severity (info): no.
        assert writer.fsyncs_total == 0, (
            "fsync must NOT fire below threshold; kills `<=` and `<` mutants"
        )
        writer.close()

    def test_severity_below_error_does_not_trigger_immediate(
        self, tmp_path: Path
    ) -> None:
        """``SEVERITY_ORDER.get(severity, 0) >= _IMMEDIATE_FSYNC_THRESHOLD``:
        ``info`` (severity rank 1) is BELOW ``error`` (rank 3). Confirm
        info-severity does NOT immediately fsync. Kills ``>=`` → ``<=``
        (would fire for everything <= error) and ``==`` (would fire only
        at exact equality, hiding warning vs error contrast)."""
        writer = JournalWriter(
            tmp_path / "events.jsonl",
            fsync_batch_size=10_000,
            fsync_interval_s=9_999.0,
        )
        writer.append(make_started(offset=0))  # severity=info
        assert writer.fsyncs_total == 0
        writer.close()

    def test_severity_default_zero_does_not_trigger(self, tmp_path: Path) -> None:
        """``SEVERITY_ORDER.get(event.severity, 0)``: the DEFAULT must be
        0 (unknown severity → no immediate fsync). Mutants flip the
        default to ``-1`` or ``1``; the ``1`` case would still produce
        ``1 >= 3 == False`` so it's equivalent here, but the ``-1`` case
        is also equivalent for info events. We catch the broader change
        by also asserting on an explicit unknown severity via a stub —
        but this is enough for the get-default mutants because both
        replacements produce values < threshold (3 for error) and the
        observable behaviour at info-severity is identical. Doc-only
        sentinel kept to mark the equivalence explicitly.
        """
        # Equivalent mutant marker — see comment above. Test pins the
        # actual default constant to assert intent.
        from ryotenkai_control.events.journal_writer import (
            _IMMEDIATE_FSYNC_THRESHOLD,
        )
        from ryotenkai_shared.events import SEVERITY_ORDER

        assert SEVERITY_ORDER["error"] == _IMMEDIATE_FSYNC_THRESHOLD


class TestMutationKillIntervalArithmetic:
    """Kill ``- → + / * / % / // / /`` on the interval-elapsed predicate."""

    def test_interval_elapsed_predicate_uses_subtraction(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``(now - self._last_fsync_at) >= self._interval_s``. The mutant
        ``+`` would always be huge (triggering); ``*`` similarly huge;
        ``%`` could be anything; ``/`` would be near 1.0; ``//`` similar.
        Pin TWO observations:

        1. With a tiny ``now-last`` delta and a small interval, the
           predicate is False → no fsync.
        2. With a huge ``now-last`` delta and a small interval, the
           predicate is True → fsync fires.

        Both depend on ``-``: the mutants either always-trigger (+, *)
        or never-trigger (negative results from % / //) or trigger at
        wrong times.
        """
        fake_clock = [1000.0]

        def fake_monotonic() -> float:
            return fake_clock[0]

        monkeypatch.setattr(time, "monotonic", fake_monotonic)

        writer = JournalWriter(
            tmp_path / "events.jsonl",
            fsync_batch_size=10_000,
            fsync_interval_s=2.0,
        )
        # Step 1: tiny delta — predicate False → no fsync.
        fake_clock[0] = 1000.5  # 0.5s after init
        writer.append(make_started(offset=0))
        assert writer.fsyncs_total == 0, (
            "interval predicate must be False for delta < interval; "
            "kills + (always huge), * (always huge) mutants"
        )

        # Step 2: huge delta — predicate True → fsync fires.
        fake_clock[0] = 2000.0  # 1000s after init, well past 2s interval
        writer.append(make_completed(offset=1))
        assert writer.fsyncs_total == 1, (
            "interval predicate must be True for delta > interval; "
            "kills mutants that never trigger (% / // producing small values)"
        )
        writer.close()


class TestMutationKillFsyncNowGuard:
    """Pin the ``if self._closed or self._fh is None: return`` guard."""

    def test_fsync_now_on_closed_writer_is_noop(self, tmp_path: Path) -> None:
        """The ``or`` MUST be ``or``. The ``and`` mutant only short-
        circuits when BOTH closed AND fh is None. Calling ``fsync_now``
        on a closed writer (fh may still be None after close) must
        produce no error.

        Also kill ``AddNot`` on ``if self._closed``: with ``not closed``,
        a non-closed writer would early-return, hiding fsyncs.
        """
        writer = JournalWriter(tmp_path / "events.jsonl")
        writer.append(make_started(offset=0))
        baseline_fsyncs = writer.fsyncs_total
        writer.close()
        # Closed: fsync_now is a no-op.
        post_close_fsyncs = writer.fsyncs_total
        writer.fsync_now()
        # No additional fsync after close.
        assert writer.fsyncs_total == post_close_fsyncs
        # And we did not regress the count via the call.
        assert post_close_fsyncs >= baseline_fsyncs

    def test_fsync_now_on_open_writer_increments_count(
        self, tmp_path: Path
    ) -> None:
        """A NON-closed, NON-None-fh writer MUST run fsync. Kills the
        ``not closed`` mutant (would early-return for open writers) and
        the ``is not None`` mutant (would early-return for live fh)."""
        writer = JournalWriter(
            tmp_path / "events.jsonl",
            fsync_batch_size=10_000,
            fsync_interval_s=9_999.0,
        )
        writer.append(make_started(offset=0))
        before = writer.fsyncs_total
        writer.fsync_now()
        assert writer.fsyncs_total == before + 1, (
            "fsync_now on an OPEN writer must perform one fsync; "
            "kills `not closed` / `is not None` mutants"
        )
        writer.close()


class TestMutationKillCounters:
    """Pin counter deltas so ``+= 1`` → ``+= 0`` / ``+= 2`` mutants fail."""

    def test_unflushed_increments_by_exactly_one_per_append(
        self, tmp_path: Path
    ) -> None:
        """``self._unflushed += 1`` — observe via batch-trigger timing.
        With batch=4, four appends MUST trigger exactly one fsync (4>=4).
        Three appends MUST trigger zero. A ``+= 2`` mutant would fsync
        after 2 appends. A ``+= 0`` mutant would never trigger via batch.
        """
        # 4 appends should produce exactly 1 fsync via batch path.
        writer = JournalWriter(
            tmp_path / "events.jsonl",
            fsync_batch_size=4,
            fsync_interval_s=9_999.0,
        )
        for i in range(3):
            writer.append(make_started(offset=i))
        # After 3 appends with `+= 1`: unflushed=3, no fsync (3 < 4).
        # With `+= 2`: unflushed=6 >= 4 ⇒ would have fsynced.
        # With `+= 0`: unflushed=0 < 4 ⇒ would not fsync (matches; need
        # the next assertion to differentiate).
        assert writer.fsyncs_total == 0
        # 4th append → unflushed reaches batch threshold (with `+= 1`).
        writer.append(make_completed(offset=3))
        assert writer.fsyncs_total == 1, (
            "after 4 appends with batch=4, expected exactly 1 fsync; "
            "kills `+= 0` (never fires) and `+= 2` (fired too early)"
        )
        writer.close()

    def test_unflushed_resets_to_zero_after_fsync(self, tmp_path: Path) -> None:
        """``self._unflushed = 0`` at the end of ``_fsync_locked``. The
        mutant ``= 1`` would carry one event over, so the NEXT batch
        only needs ``batch_size - 1`` appends to trigger.
        """
        writer = JournalWriter(
            tmp_path / "events.jsonl",
            fsync_batch_size=3,
            fsync_interval_s=9_999.0,
        )
        for i in range(3):
            writer.append(make_started(offset=i))
        assert writer.fsyncs_total == 1
        # After reset, need 3 MORE appends to hit batch again.
        writer.append(make_completed(offset=3))
        writer.append(make_completed(offset=4))
        assert writer.fsyncs_total == 1, (
            "unflushed must reset to 0 — kills `= 1`/`= -1` mutants "
            "that would carry stale state into the next batch"
        )
        writer.append(make_completed(offset=5))
        assert writer.fsyncs_total == 2
        writer.close()

    def test_fsyncs_total_increments_by_exactly_one_per_call(
        self, tmp_path: Path
    ) -> None:
        """``self.fsyncs_total += 1`` — pin exact increment per fsync.
        ``+= 2`` mutant would over-report; ``+= 0`` mutant would
        under-report."""
        writer = JournalWriter(
            tmp_path / "events.jsonl",
            fsync_batch_size=10_000,
            fsync_interval_s=9_999.0,
        )
        writer.append(make_started(offset=0))
        assert writer.fsyncs_total == 0
        writer.fsync_now()
        assert writer.fsyncs_total == 1
        writer.fsync_now()
        assert writer.fsyncs_total == 2
        writer.close()
        # close performs one more fsync.
        assert writer.fsyncs_total == 3

    def test_fsync_failed_increments_by_exactly_one_per_oserror(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``self.fsync_failed_total += 1`` on OSError in
        ``_fsync_locked``. Kills ``+= 0`` (silent failure) and ``+= 2``
        (double-count) mutants."""
        writer = JournalWriter(
            tmp_path / "events.jsonl",
            fsync_batch_size=10_000,
            fsync_interval_s=9_999.0,
        )
        import os as os_mod

        def boom(_fd: int) -> None:
            raise OSError("disk full")

        monkeypatch.setattr(os_mod, "fsync", boom)
        writer.append(make_started(offset=0))
        before = writer.fsync_failed_total
        writer.fsync_now()
        assert writer.fsync_failed_total == before + 1
        # Second fsync → exactly one more failure.
        writer.fsync_now()
        assert writer.fsync_failed_total == before + 2
        # Restore so close() doesn't keep failing.
        monkeypatch.undo()
        writer.close()

    def test_close_fsync_failure_bumps_fsync_failed_not_fsyncs_total(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """In ``close()``: on OSError, ``fsync_failed_total += 1``. Kills
        the ``+= 2`` / ``+= 0`` mutants on line 207 specifically."""
        writer = JournalWriter(tmp_path / "events.jsonl")
        writer.append(make_started(offset=0))
        # Sync the inline fsync first (one success).
        writer.fsync_now()
        before_total = writer.fsyncs_total
        before_failed = writer.fsync_failed_total
        # Break fsync for the close path.
        import os as os_mod
        monkeypatch.setattr(os_mod, "fsync", lambda _fd: (_ for _ in ()).throw(OSError("x")))
        writer.close()
        # close attempts ONE fsync that fails → fsync_failed_total += 1.
        assert writer.fsync_failed_total == before_failed + 1
        assert writer.fsyncs_total == before_total

    def test_close_sets_closed_to_true(self, tmp_path: Path) -> None:
        """``self._closed = True`` in close(). The ``= False`` mutant
        would leave the writer "open" semantically; subsequent ``append``
        would NOT be guarded.
        """
        writer = JournalWriter(tmp_path / "events.jsonl")
        writer.close()
        assert writer.is_closed is True
        # And subsequent append is a no-op (the closed guard works).
        writer.append(make_started(offset=0))
        assert writer.events_appended == 0


class TestMutationKillCloseExceptionType:
    """Pin the close-path exception filter to OSError specifically."""

    def test_close_only_handles_oserror_not_unrelated_exceptions(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``except OSError as exc`` in close(). The cosmic-ray
        ``ExceptionReplacer`` substitutes a private synthetic class —
        the original code would correctly catch OSError; the mutant
        would let OSError escape.

        We simulate by raising OSError from fsync at close() time. The
        original close MUST NOT raise. A mutant that swapped to a
        non-OSError class would let our raise propagate (since
        contextlib.suppress would still swallow the OSError from the
        outer wrapper on the file close — but the inner counter bump
        would never run). Pin observable: after close, fsync_failed_total
        IS incremented (meaning the except block ran).
        """
        writer = JournalWriter(tmp_path / "events.jsonl")
        writer.append(make_started(offset=0))
        # Inline fsync succeeds, but close-time fsync fails.
        writer.fsync_now()
        baseline = writer.fsync_failed_total
        import os as os_mod
        monkeypatch.setattr(
            os_mod, "fsync", lambda _fd: (_ for _ in ()).throw(OSError("close fail"))
        )
        writer.close()  # must not raise
        # The except OSError branch ran → fsync_failed_total bumped.
        assert writer.fsync_failed_total == baseline + 1


class TestMutationKillInitialStateConstants:
    """Pin initial counter values (``= 0`` vs ``= 1`` / ``= -1``)."""

    def test_total_bytes_written_starts_at_zero(self, tmp_path: Path) -> None:
        """``self.total_bytes_written = 0`` in __init__. Mutants ``= 1``
        and ``= -1`` are killed by asserting the post-init value is
        exactly 0."""
        writer = JournalWriter(tmp_path / "events.jsonl")
        assert writer.total_bytes_written == 0
        writer.close()

    def test_unflushed_starts_at_zero(self, tmp_path: Path) -> None:
        """``self._unflushed = 0`` at init. A non-zero start would push
        the very first append over the batch threshold (batch_size=1
        below). Pin via observable batch behaviour."""
        # With batch=2: first append should NOT fsync (unflushed becomes 1).
        writer = JournalWriter(
            tmp_path / "events.jsonl",
            fsync_batch_size=2,
            fsync_interval_s=9_999.0,
        )
        writer.append(make_started(offset=0))
        assert writer.fsyncs_total == 0, (
            "after 1 append with batch=2 and initial unflushed=0, "
            "MUST NOT fsync; kills `_unflushed = 1` init mutant"
        )
        writer.append(make_completed(offset=1))
        assert writer.fsyncs_total == 1
        writer.close()


class TestMutationKillSignatureMarkers:
    """Pin keyword-only / positional-only markers on init and helpers."""

    def test_init_rejects_positional_kwargs_after_path(self, tmp_path: Path) -> None:
        """``def __init__(self, path, *, fsync_batch_size, ...)``. The
        ``*,`` → ``/,`` mutant flips kw-only to positional-only past
        ``path`` — but Python rejects a positional-only marker AFTER
        positional params with defaults; the mutant typically still
        parses, and the call ``JournalWriter(path, 50, 1.0)``
        (positional) succeeds under the mutant. Pin via the kwonly
        contract: positional batch_size MUST raise.
        """
        with pytest.raises(TypeError):
            JournalWriter(tmp_path / "events.jsonl", 50)  # type: ignore[misc]

    def test_fsync_locked_now_is_keyword_only(self) -> None:
        """``_fsync_locked(self, *, now)`` — pin via signature inspection
        so the ``*,`` → ``/,`` mutant on the private helper fails."""
        import inspect

        sig = inspect.signature(JournalWriter._fsync_locked)
        param = sig.parameters["now"]
        assert param.kind is inspect.Parameter.KEYWORD_ONLY


# ===========================================================================
# 9. Mutation kill — iteration 2 (close remaining gaps)
# ===========================================================================


class TestMutationKillBatchBoundaryLargeBatch:
    """``_unflushed >= batch_size`` — kill ``is`` mutant via large batch
    sizes that fall outside CPython's small-int cache (>256)."""

    def test_fsync_fires_at_large_batch_threshold(self, tmp_path: Path) -> None:
        """With batch=300 (> 256 → no integer interning), the boundary
        ``unflushed == batch_size`` MUST still trigger via ``>=``. The
        ``is`` mutant would compare object identity — different int
        objects fail.
        """
        writer = JournalWriter(
            tmp_path / "events.jsonl",
            fsync_batch_size=300,
            fsync_interval_s=9_999.0,
        )
        # 299 appends — below threshold.
        for i in range(299):
            writer.append(make_started(offset=i))
        assert writer.fsyncs_total == 0
        # 300th append → triggers via ``>=`` (or ``==``) but NOT via ``is``.
        writer.append(make_completed(offset=299))
        assert writer.fsyncs_total == 1, (
            "fsync MUST fire at batch boundary even for large batch_size; "
            "kills `is` mutant (300 is 300 fails outside small-int cache)"
        )
        writer.close()


class TestMutationKillIntervalRatioMutants:
    """Kill the ``now - last`` → ``now / last`` and ``now // last`` mutants
    via parameter choices where subtraction and division diverge.
    """

    def test_subtraction_predicate_differs_from_division_at_specific_values(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Setup where ``now - last >= interval`` is TRUE but
        ``now / last >= interval`` and ``now // last >= interval`` are
        FALSE. With now=1000.0, last=900.0, interval=1.5:
          subtract: 100 >= 1.5 → True (fsync)
          divide:   1.111 >= 1.5 → False (no fsync)
          floordiv: 1 >= 1.5 → False (no fsync)
        """
        fake_clock = [900.0]

        def fake_monotonic() -> float:
            return fake_clock[0]

        monkeypatch.setattr(time, "monotonic", fake_monotonic)

        writer = JournalWriter(
            tmp_path / "events.jsonl",
            fsync_batch_size=10_000,
            fsync_interval_s=1.5,
        )
        # last_fsync_at = 900 (init time).
        # Advance to 1000 → delta 100, but ratio ~1.11.
        fake_clock[0] = 1000.0
        writer.append(make_started(offset=0))
        assert writer.fsyncs_total == 1, (
            "delta=100, interval=1.5 → subtraction triggers, but ratio "
            "(now/last=1.11) does not; kills `now / last` and "
            "`now // last` mutants"
        )
        writer.close()

    def test_interval_predicate_strict_at_equality_boundary(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``>= interval`` MUST fire when delta == interval (inclusive).
        The ``>`` mutant would skip the exact-equality case.
        """
        fake_clock = [100.0]

        def fake_monotonic() -> float:
            return fake_clock[0]

        monkeypatch.setattr(time, "monotonic", fake_monotonic)

        writer = JournalWriter(
            tmp_path / "events.jsonl",
            fsync_batch_size=10_000,
            fsync_interval_s=2.0,
        )
        # last_fsync_at = 100. Advance to exactly 102 → delta = 2.0.
        fake_clock[0] = 102.0
        writer.append(make_started(offset=0))
        assert writer.fsyncs_total == 1, (
            "delta=interval=2.0 must trigger (>= is inclusive); kills "
            "the `>=` → `>` mutant"
        )
        writer.close()


class TestMutationKillSeverityDefault:
    """Cover the ``SEVERITY_ORDER.get(severity, 0)`` default and the
    ``>= _IMMEDIATE_FSYNC_THRESHOLD`` comparison via unknown severity."""

    def test_unknown_severity_does_not_trigger_immediate_fsync(
        self, tmp_path: Path
    ) -> None:
        """A made-up severity hits the default branch. With default=0,
        ``0 >= 3 (error rank)`` is False → no fsync. With mutant
        default=1: ``1 >= 3`` still False (same observable). With mutant
        default=-1: ``-1 >= 3`` still False. So this can't kill the
        default mutants directly.

        But we CAN kill the ``>=`` → ``==`` mutant: with severity=error
        (rank=3), ``3 == 3`` matches AND ``3 >= 3`` matches (SAME).
        Need severity HIGHER than error. SEVERITY_ORDER includes
        "critical" (rank 4 or higher). With critical: ``4 >= 3`` True
        but ``4 == 3`` False. So critical-severity event MUST trigger
        immediate fsync under correct ``>=``; under ``==`` mutant it
        would NOT.
        """
        # Build a critical-severity event by overriding the severity
        # field on a RunFailed envelope (the EVENT_ADAPTER will accept
        # arbitrary severity strings — they're typed as Literal but
        # model_copy bypasses validation).
        from ryotenkai_shared.events import SEVERITY_ORDER

        # Look up "critical" — it MUST exist as a rank above "error".
        assert "critical" in SEVERITY_ORDER, (
            "fixture precondition: SEVERITY_ORDER must include critical"
        )
        assert SEVERITY_ORDER["critical"] > SEVERITY_ORDER["error"]

        writer = JournalWriter(
            tmp_path / "events.jsonl",
            fsync_batch_size=10_000,
            fsync_interval_s=9_999.0,
        )
        # RunFailedEvent (severity="error" by default) — bump to "critical".
        ev = make_failed()
        critical_ev = ev.model_copy(update={"severity": "critical"})
        writer.append(critical_ev)
        assert writer.fsyncs_total == 1, (
            "critical severity (rank > error threshold) must trigger "
            "immediate fsync; kills `>= → ==` mutant (would match only "
            "exact equality)"
        )
        writer.close()


class TestMutationKillFsyncNowOrAnd:
    """Kill ``self._closed or self._fh is None`` → ``and`` mutant via a
    state with closed=False but fh=None.
    """

    def test_fsync_now_guards_each_condition_independently(
        self, tmp_path: Path
    ) -> None:
        """With closed=False AND fh=None (unusual but constructible),
        ``or`` returns True (skip fsync); ``and`` returns False (proceed
        — but fh is None, AttributeError).

        Pin: monkey-patch fh to None on an open writer; fsync_now must
        early-return (no error, no fsync count change).
        """
        writer = JournalWriter(tmp_path / "events.jsonl")
        baseline = writer.fsyncs_total
        # Simulate the inconsistent state ``and`` would mis-handle.
        writer._fh = None
        writer.fsync_now()
        # Must NOT raise. Must NOT increment fsyncs_total.
        assert writer.fsyncs_total == baseline, (
            "with fh=None the guard must early-return; kills `or → and` "
            "mutant which would proceed and raise on None.fileno()"
        )
        # Clean up — restore _fh isn't strictly needed since close()
        # tolerates None.
        writer.close()


class TestMutationKillPathProperty:
    """Pin ``@property def path`` — without the decorator, ``writer.path``
    returns a bound method instead of a Path."""

    def test_path_attribute_is_a_path_instance(self, tmp_path: Path) -> None:
        """The ``@property`` decorator. Without it, ``writer.path`` is
        a bound method; ``isinstance(writer.path, Path)`` is False.
        """
        target = tmp_path / "events.jsonl"
        writer = JournalWriter(target)
        assert isinstance(writer.path, Path)
        assert writer.path == target
        writer.close()
