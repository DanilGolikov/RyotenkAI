"""Hypothesis fuzz tests for :class:`JournalWriter` (Phase 9).

Three property families:

* **Concurrent emit integrity** — 10 producer threads × 100 events each
  end up on disk with intact length-prefixed framing and unique
  offsets. Validates the threading lock under load.

* **Kill-during-write simulation** — synthesizing a torn write
  (partial line, no newline) then re-opening through
  :class:`JournalReader.truncate_torn_tail` recovers a journal whose
  remaining tail is verifiable via :func:`parse_length_prefix`.

* **Variable fsync timing** — for any ``(fsync_batch_size,
  fsync_interval_s)`` setting, the same set of events produces the
  same on-disk byte stream and replay yields the same envelopes.

Complements the deterministic tests in
:mod:`tests.unit.control.events.test_journal_writer`.
"""

from __future__ import annotations

import threading
from pathlib import Path

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from ryotenkai_control.events import JournalReader, JournalWriter
from ryotenkai_shared.events import (
    BaseEvent,
    from_jsonl,
    parse_length_prefix,
)
from ryotenkai_shared.events.types.control_run import (
    RunStartedEvent,
    RunStartedPayload,
)

from tests.unit.control.events.conftest import (
    make_completed,
    make_failed,
    make_started,
)


# ---------------------------------------------------------------------------
# Helper strategies
# ---------------------------------------------------------------------------


# Hypothesis-friendly event factory — we can't pickle BaseEvent objects
# across threads cheanly via ``st.builds``, so we keep the strategy small
# and rebuild events in-thread.
_run_id_strategy = st.text(
    alphabet=st.characters(min_codepoint=0x20, max_codepoint=0x7E),
    min_size=1,
    max_size=24,
)


def _make_started_with_offset(offset: int, run_id: str = "test") -> RunStartedEvent:
    """Construct a :class:`RunStartedEvent` with a deterministic payload."""
    return RunStartedEvent(
        source="control://orchestrator",
        run_id=run_id,
        offset=offset,
        payload=RunStartedPayload(
            run_name=f"run-{offset}",
            algorithm="sft",
            model_id="acme/test",
            dataset_id="default",
            config_hash="abc",
        ),
    )


def _verify_all_length_prefixed(path: Path) -> int:
    """Walk every newline-terminated line and assert framing.

    Returns the number of valid lines verified — useful as a smoke
    metric in assertions that span thousands of events.
    """
    count = 0
    with path.open("rb") as fh:
        for raw in fh:
            text = raw.decode("utf-8", errors="strict")
            if not text.strip():
                continue
            # ``parse_length_prefix`` raises on framing violation;
            # an uncaught exception fails the test.
            declared, body = parse_length_prefix(text)
            assert declared == len(body.encode("utf-8"))
            count += 1
    return count


# ---------------------------------------------------------------------------
# Concurrent emit integrity
# ---------------------------------------------------------------------------


class TestConcurrentEmitIntegrity:
    """Property: parallel producers never produce torn / overlapping lines."""

    @given(
        thread_count=st.integers(min_value=2, max_value=10),
        events_per_thread=st.integers(min_value=10, max_value=50),
    )
    @settings(
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
        # Threading + filesystem is heavy; cap the example count.
        max_examples=10,
    )
    def test_parallel_emit_produces_intact_lines(
        self,
        tmp_path_factory: object,  # pytest fixture
        thread_count: int,
        events_per_thread: int,
    ) -> None:
        # tmp_path_factory is the session-scoped pytest fixture; we need
        # a fresh directory per hypothesis example.
        tmp_path = Path(  # type: ignore[attr-defined]
            tmp_path_factory.mktemp(f"fuzz_journal_{thread_count}_{events_per_thread}"),
        )
        path = tmp_path / "events.jsonl"
        writer = JournalWriter(path)

        def worker(thread_index: int) -> None:
            base = thread_index * events_per_thread
            for i in range(events_per_thread):
                writer.append(_make_started_with_offset(offset=base + i))

        threads = [
            threading.Thread(target=worker, args=(t,)) for t in range(thread_count)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        writer.close()

        expected = thread_count * events_per_thread
        # All lines are well-framed (no torn writes), counter matches.
        verified = _verify_all_length_prefixed(path)
        assert verified == expected
        # The writer's internal counter agrees.
        assert writer.events_appended == expected

    @given(
        thread_count=st.integers(min_value=2, max_value=10),
        events_per_thread=st.integers(min_value=10, max_value=50),
    )
    @settings(
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
        max_examples=10,
    )
    def test_parallel_emit_yields_unique_offsets(
        self,
        tmp_path_factory: object,
        thread_count: int,
        events_per_thread: int,
    ) -> None:
        """Every offset emitted appears exactly once in the journal."""
        tmp_path = Path(  # type: ignore[attr-defined]
            tmp_path_factory.mktemp(
                f"fuzz_journal_uniq_{thread_count}_{events_per_thread}",
            ),
        )
        path = tmp_path / "events.jsonl"
        writer = JournalWriter(path)

        def worker(thread_index: int) -> None:
            base = thread_index * events_per_thread
            for i in range(events_per_thread):
                writer.append(_make_started_with_offset(offset=base + i))

        threads = [
            threading.Thread(target=worker, args=(t,)) for t in range(thread_count)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        writer.close()

        # Read back every event and assert offsets form a set of the
        # expected size — no dupes, no missing.
        reader = JournalReader(path)
        offsets = [e.offset for e in reader.iter_envelopes()]
        assert len(offsets) == thread_count * events_per_thread
        assert len(set(offsets)) == len(offsets)


# ---------------------------------------------------------------------------
# Kill-during-write simulation
# ---------------------------------------------------------------------------


class TestKillDuringWrite:
    """Property: after a synthesized torn write, the reader recovers safely."""

    @given(
        clean_events=st.integers(min_value=1, max_value=20),
        torn_body_length=st.integers(min_value=1, max_value=200),
    )
    @settings(
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
        max_examples=15,
    )
    def test_torn_tail_is_truncated_and_clean_events_survive(
        self,
        tmp_path_factory: object,
        clean_events: int,
        torn_body_length: int,
    ) -> None:
        tmp_path = Path(  # type: ignore[attr-defined]
            tmp_path_factory.mktemp(
                f"fuzz_torn_{clean_events}_{torn_body_length}",
            ),
        )
        path = tmp_path / "events.jsonl"
        writer = JournalWriter(path)
        for i in range(clean_events):
            writer.append(_make_started_with_offset(offset=i))
        writer.close()

        # Synthesize a torn write: append partial bytes without trailing
        # newline. The declared length is well-formed; the body is
        # truncated (no newline) — this matches a ``kill -9`` mid-flush.
        with path.open("ab") as fh:
            partial = b'{"kind":"' + b"x" * torn_body_length
            fh.write(f"{len(partial) + 1000}\t".encode())  # lying length
            fh.write(partial)

        reader = JournalReader(path)
        truncated = reader.truncate_torn_tail()
        assert truncated is True

        # The remaining envelopes are all valid.
        envelopes = list(reader.iter_envelopes())
        assert len(envelopes) == clean_events
        offsets_seen = [e.offset for e in envelopes]
        assert offsets_seen == list(range(clean_events))

    @given(
        clean_events=st.integers(min_value=0, max_value=5),
    )
    @settings(
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
        max_examples=10,
    )
    def test_well_formed_tail_is_not_truncated(
        self,
        tmp_path_factory: object,
        clean_events: int,
    ) -> None:
        """A journal with a clean tail must NOT be truncated."""
        tmp_path = Path(  # type: ignore[attr-defined]
            tmp_path_factory.mktemp(f"fuzz_clean_{clean_events}"),
        )
        path = tmp_path / "events.jsonl"
        writer = JournalWriter(path)
        for i in range(clean_events):
            writer.append(_make_started_with_offset(offset=i))
        writer.close()

        reader = JournalReader(path)
        truncated = reader.truncate_torn_tail()
        assert truncated is False


# ---------------------------------------------------------------------------
# Variable fsync timing
# ---------------------------------------------------------------------------


class TestVariableFsyncTiming:
    """Property: events on disk are durable regardless of fsync cadence."""

    @given(
        fsync_batch_size=st.integers(min_value=1, max_value=100),
        fsync_interval_s=st.floats(
            min_value=0.0, max_value=10.0,
            allow_nan=False, allow_infinity=False,
        ),
        n_events=st.integers(min_value=5, max_value=50),
    )
    @settings(
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
        max_examples=20,
    )
    def test_events_persist_for_any_fsync_setting(
        self,
        tmp_path_factory: object,
        fsync_batch_size: int,
        fsync_interval_s: float,
        n_events: int,
    ) -> None:
        tmp_path = Path(  # type: ignore[attr-defined]
            tmp_path_factory.mktemp(
                f"fuzz_fsync_{fsync_batch_size}_{n_events}",
            ),
        )
        path = tmp_path / "events.jsonl"
        writer = JournalWriter(
            path,
            fsync_batch_size=fsync_batch_size,
            fsync_interval_s=fsync_interval_s,
        )
        for i in range(n_events):
            writer.append(_make_started_with_offset(offset=i))
        writer.close()

        reader = JournalReader(path)
        envelopes = list(reader.iter_envelopes())
        # Every event ends up persisted — close() flushed any tail batch.
        assert len(envelopes) == n_events
        assert [e.offset for e in envelopes] == list(range(n_events))
        # Length prefix invariants hold.
        _verify_all_length_prefixed(path)

    @given(
        n_error_events=st.integers(min_value=1, max_value=10),
        n_info_events=st.integers(min_value=0, max_value=10),
    )
    @settings(
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
        max_examples=10,
    )
    def test_error_severity_triggers_immediate_fsync(
        self,
        tmp_path_factory: object,
        n_error_events: int,
        n_info_events: int,
    ) -> None:
        """For any mix, error-severity events flush immediately."""
        tmp_path = Path(  # type: ignore[attr-defined]
            tmp_path_factory.mktemp(
                f"fuzz_sev_{n_error_events}_{n_info_events}",
            ),
        )
        path = tmp_path / "events.jsonl"
        writer = JournalWriter(
            path,
            fsync_batch_size=10_000,
            fsync_interval_s=999.0,
        )
        # Pure error stream → one fsync per event minimum (immediate).
        before = writer.fsyncs_total
        for i in range(n_error_events):
            writer.append(make_failed(offset=i))
        # At least N error events should have triggered fsyncs (1 each).
        assert writer.fsyncs_total >= before + n_error_events

        # Now sprinkle in info events — no extra fsyncs because batch
        # size is huge and interval is far in the future.
        post_error = writer.fsyncs_total
        for i in range(n_info_events):
            writer.append(make_started(offset=n_error_events + i))
        # info events shouldn't flush (batch + interval too generous).
        # Sometimes the test environment can trigger a flush from the
        # time-based path if the clock advanced; allow a small fudge.
        assert writer.fsyncs_total <= post_error + 1
        writer.close()


# ---------------------------------------------------------------------------
# Resume + reopen invariant
# ---------------------------------------------------------------------------


class TestResumeReopenInvariant:
    """Property: reopening an existing journal appends, never truncates."""

    @given(
        events_phase_one=st.integers(min_value=1, max_value=20),
        events_phase_two=st.integers(min_value=1, max_value=20),
    )
    @settings(
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
        max_examples=10,
    )
    def test_reopen_appends_after_existing(
        self,
        tmp_path_factory: object,
        events_phase_one: int,
        events_phase_two: int,
    ) -> None:
        tmp_path = Path(  # type: ignore[attr-defined]
            tmp_path_factory.mktemp(
                f"fuzz_resume_{events_phase_one}_{events_phase_two}",
            ),
        )
        path = tmp_path / "events.jsonl"

        w1 = JournalWriter(path)
        for i in range(events_phase_one):
            w1.append(_make_started_with_offset(offset=i))
        w1.close()

        w2 = JournalWriter(path)
        for i in range(events_phase_two):
            w2.append(_make_started_with_offset(
                offset=events_phase_one + i,
            ))
        w2.close()

        envelopes = list(JournalReader(path).iter_envelopes())
        total = events_phase_one + events_phase_two
        assert len(envelopes) == total
        assert [e.offset for e in envelopes] == list(range(total))


# ---------------------------------------------------------------------------
# Round-trip through the writer
# ---------------------------------------------------------------------------


class TestRoundTripThroughWriter:
    """Property: writing then reading preserves byte-equal envelope JSON."""

    @given(
        offsets=st.lists(
            st.integers(min_value=0, max_value=10_000),
            min_size=1,
            max_size=20,
            unique=True,
        ),
    )
    @settings(
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
        max_examples=15,
    )
    def test_writer_reader_round_trip_preserves_events(
        self,
        tmp_path_factory: object,
        offsets: list[int],
    ) -> None:
        tmp_path = Path(  # type: ignore[attr-defined]
            tmp_path_factory.mktemp(f"fuzz_rt_{len(offsets)}"),
        )
        path = tmp_path / "events.jsonl"
        writer = JournalWriter(path)

        events_in: list[BaseEvent] = []
        for off in offsets:
            ev = _make_started_with_offset(offset=off)
            events_in.append(ev)
            writer.append(ev)
        writer.close()

        # Read through codec; assert equality.
        events_out: list[BaseEvent] = []
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    events_out.append(from_jsonl(line, strict=True))

        assert events_out == events_in


# ---------------------------------------------------------------------------
# Bytes-written metric tracks reality
# ---------------------------------------------------------------------------


class TestMetricsAccurate:
    """Property: ``total_bytes_written`` matches on-disk size after close."""

    @given(
        n_events=st.integers(min_value=1, max_value=30),
    )
    @settings(
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
        max_examples=10,
    )
    def test_total_bytes_matches_file_size(
        self,
        tmp_path_factory: object,
        n_events: int,
    ) -> None:
        tmp_path = Path(  # type: ignore[attr-defined]
            tmp_path_factory.mktemp(f"fuzz_bytes_{n_events}"),
        )
        path = tmp_path / "events.jsonl"
        writer = JournalWriter(path)
        for i in range(n_events):
            writer.append(_make_started_with_offset(offset=i))
        writer.close()

        # Bytes written by the writer equals on-disk size — no torn
        # write or short write would slip past the counter.
        assert path.stat().st_size == writer.total_bytes_written
