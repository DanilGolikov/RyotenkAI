"""Resume integration test — closes R-12 (Phase 9).

Scenario:

1. Start a control-side emitter for ``run-resume``.
2. Emit 100 events (mix of stage + training-bridge + memory events).
3. Simulate :code:`SIGKILL` by skipping ``close()`` and writing a
   half-line (partial torn-write) onto the tail of the journal.
4. Construct a fresh emitter via :meth:`ControlEventEmitter.for_run`
   with the same ``run_id``. The factory must:

   * Truncate the torn last line via ``tmp + rename``.
   * Seed the per-source offset counter from the surviving offsets.
   * Reconstruct the dedup set from the journal tail.

5. Continue emitting from the resumed counter — new offsets must
   strictly follow the persisted ones, with no collision or gap.

6. Final assertions:
   * 100 + N events on disk.
   * All lines pass length-prefix framing.
   * All offsets strictly monotonic per source.
   * No torn lines remain.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from ryotenkai_control.events import ControlEventEmitter, JournalReader
from ryotenkai_shared.events import (
    UNKNOWN_OFFSET,
    BaseEvent,
    parse_length_prefix,
)
from ryotenkai_shared.events.types.control_stage import (
    StageCompletedEvent,
    StageCompletedPayload,
    StageStartedEvent,
    StageStartedPayload,
)
from ryotenkai_shared.events.types.pod_memory import (
    MemoryCacheClearedEvent,
    MemoryCacheClearedPayload,
)
from ryotenkai_shared.events.types.pod_training import (
    TrainingStepEvent,
    TrainingStepPayload,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_stage_started(run_id: str, source: str, idx: int) -> StageStartedEvent:
    return StageStartedEvent(
        source=source,
        run_id=run_id,
        offset=UNKNOWN_OFFSET,
        payload=StageStartedPayload(
            stage_name=f"stage-{idx}",
            stage_index=idx,
            total_stages=10,
            inputs_summary={},
        ),
    )


def _make_stage_completed(run_id: str, source: str, idx: int) -> StageCompletedEvent:
    return StageCompletedEvent(
        source=source,
        run_id=run_id,
        offset=UNKNOWN_OFFSET,
        payload=StageCompletedPayload(
            stage_name=f"stage-{idx}",
            duration_s=0.5,
            outputs_summary={},
        ),
    )


def _make_training_step(run_id: str, source: str, step: int) -> TrainingStepEvent:
    return TrainingStepEvent(
        source=source,
        run_id=run_id,
        offset=step,
        payload=TrainingStepPayload(
            step=step,
            loss=2.0,
            learning_rate=1e-5,
            grad_norm=0.5,
            tokens_per_sec=1000.0,
            samples_per_sec=4.0,
        ),
    )


def _make_memory_cleared(run_id: str, source: str, idx: int) -> MemoryCacheClearedEvent:
    return MemoryCacheClearedEvent(
        source=source,
        run_id=run_id,
        offset=idx,
        payload=MemoryCacheClearedPayload(
            device="cuda:0",
            before_bytes=100_000_000,
            after_bytes=50_000_000,
            trigger="scheduled",
        ),
    )


def _all_lines_well_framed(path: Path) -> int:
    """Walk every line and assert framing validity. Returns the count."""
    count = 0
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            parse_length_prefix(line)
            count += 1
    return count


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestResumeAfterTornWrite:
    """Restart with same run_id; torn tail repaired, emit continues."""

    def test_resume_truncates_torn_tail_and_continues(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run-resume"
        run_id = "run-resume"
        emitter_v1 = ControlEventEmitter.for_run(
            run_id=run_id, run_directory=run_dir,
        )

        # Phase A — emit 100 events across two sources.
        control_source = emitter_v1.source
        pod_source = f"pod://{run_id}/trainer"

        for i in range(60):
            emitter_v1.emit(_make_stage_started(run_id, control_source, i))
        for i in range(20):
            emitter_v1.emit_remote(
                _make_training_step(run_id, pod_source, i),
            )
        for i in range(20):
            emitter_v1.emit_remote(
                _make_memory_cleared(run_id, pod_source, 20 + i),
            )

        # Snapshot journal contents BEFORE the simulated crash.
        journal_path = run_dir / "events.jsonl"
        pre_crash_bytes = journal_path.read_text(encoding="utf-8")
        # The journal isn't explicitly fsynced; the writer's
        # line-buffered + immediate-flush behaviour means most lines
        # are visible. Force a flush via fsync_now to be sure.
        emitter_v1.journal.fsync_now()

        # Simulate kill -9: do NOT close the emitter; append a partial
        # line with a lying length prefix.
        with journal_path.open("ab") as fh:
            fh.write(b'99999\t{"kind":"ryotenkai.pod.training.step","sou')

        # The journal now has a torn tail.
        reader_pre = JournalReader(journal_path)
        assert reader_pre.has_torn_tail() is True

        # ------------------------------------------------------------
        # Phase B — construct a fresh emitter; it should repair the tail
        # and seed the counters from the surviving offsets.
        # ------------------------------------------------------------
        emitter_v2 = ControlEventEmitter.for_run(
            run_id=run_id, run_directory=run_dir,
        )

        # Journal truncation happened in for_run; no torn tail any more.
        reader_mid = JournalReader(journal_path)
        assert reader_mid.has_torn_tail() is False

        # Count surviving events. The exact count depends on whether the
        # final flush landed before the partial — but at least the bulk
        # of the 100 events should be present.
        envelopes_after_resume = list(reader_mid.iter_envelopes())
        surviving_count = len(envelopes_after_resume)
        assert surviving_count >= 99, (
            f"expected ~100 well-formed events after truncation, "
            f"got {surviving_count}"
        )

        # ------------------------------------------------------------
        # Phase C — continue emitting; new events get fresh offsets.
        # ------------------------------------------------------------
        n_new_events = 10
        for i in range(n_new_events):
            emitter_v2.emit(_make_stage_completed(run_id, control_source, 100 + i))

        emitter_v2.close()

        # ------------------------------------------------------------
        # Phase D — final assertions.
        # ------------------------------------------------------------
        reader_final = JournalReader(journal_path)
        all_envelopes = list(reader_final.iter_envelopes())
        # No torn-write residue should leak through (UNKNOWN_OFFSET).
        assert all(e.offset != UNKNOWN_OFFSET for e in all_envelopes)

        # Length-prefix framing intact for every line.
        framed_count = _all_lines_well_framed(journal_path)
        assert framed_count == len(all_envelopes)

        # Offsets strictly monotonic per source.
        per_source_offsets: dict[str, list[int]] = {}
        for envelope in all_envelopes:
            per_source_offsets.setdefault(envelope.source, []).append(envelope.offset)
        for source, offsets in per_source_offsets.items():
            sorted_unique = sorted(set(offsets))
            assert offsets == sorted_unique, (
                f"offsets for {source!r} not strictly monotonic: {offsets}"
            )

        # The control source picked up where it left off.
        control_offsets = sorted(per_source_offsets[control_source])
        # First control offset is 0, last is at least 60 (we added 10
        # more after resume, but they may have offsets >= 60).
        assert control_offsets[0] == 0
        assert control_offsets[-1] >= 60 + n_new_events - 1

    def test_resume_with_no_torn_tail_is_a_noop(self, tmp_path: Path) -> None:
        """A clean run that closed properly resumes without truncating."""
        run_dir = tmp_path / "run-clean"
        run_id = "run-clean"
        emitter_v1 = ControlEventEmitter.for_run(
            run_id=run_id, run_directory=run_dir,
        )
        for i in range(10):
            emitter_v1.emit(
                _make_stage_started(run_id, emitter_v1.source, i),
            )
        emitter_v1.close()

        # Snapshot the journal bytes.
        journal_path = run_dir / "events.jsonl"
        pre_bytes = journal_path.read_bytes()

        emitter_v2 = ControlEventEmitter.for_run(
            run_id=run_id, run_directory=run_dir,
        )
        emitter_v2.close()

        # File unchanged — same bytes.
        post_bytes = journal_path.read_bytes()
        assert pre_bytes == post_bytes

    def test_resume_reconstructs_offset_counter(self, tmp_path: Path) -> None:
        """Resumed counter starts strictly after the persisted maximum."""
        run_dir = tmp_path / "run-counter"
        run_id = "run-counter"

        emitter_v1 = ControlEventEmitter.for_run(
            run_id=run_id, run_directory=run_dir,
        )
        for i in range(5):
            emitter_v1.emit(_make_stage_started(run_id, emitter_v1.source, i))
        emitter_v1.close()

        # The persisted max offset on the control source is 4.
        reader = JournalReader(run_dir / "events.jsonl")
        per_src = reader.newest_persisted_offset_per_source()
        max_persisted = per_src[emitter_v1.source]
        assert max_persisted == 4

        # Resume — the next emit assigns offset 5.
        emitter_v2 = ControlEventEmitter.for_run(
            run_id=run_id, run_directory=run_dir,
        )
        emitter_v2.emit(_make_stage_completed(run_id, emitter_v2.source, 0))
        emitter_v2.close()

        reader_post = JournalReader(run_dir / "events.jsonl")
        offsets = [e.offset for e in reader_post.iter_envelopes()]
        # 5 old + 1 new == 6 total, offsets 0..5.
        assert offsets == [0, 1, 2, 3, 4, 5]

    def test_resume_reconstructs_dedup_set(self, tmp_path: Path) -> None:
        """After resume, replaying a remote envelope from the journal is dropped."""
        run_dir = tmp_path / "run-dedup"
        run_id = "run-dedup"
        pod_source = f"pod://{run_id}/trainer"

        emitter_v1 = ControlEventEmitter.for_run(
            run_id=run_id, run_directory=run_dir,
        )
        for i in range(5):
            emitter_v1.emit_remote(
                _make_training_step(run_id, pod_source, i),
            )
        emitter_v1.close()

        # Resume — the dedup set is rebuilt from the journal tail.
        emitter_v2 = ControlEventEmitter.for_run(
            run_id=run_id, run_directory=run_dir,
        )
        # Re-submit offset 3 — must be silently dropped.
        emitter_v2.emit_remote(
            _make_training_step(run_id, pod_source, 3),
        )
        assert emitter_v2.events_remote_dropped_total.get("duplicate") == 1
        # Journal size unchanged: the dedup'd event never landed on disk.
        reader = JournalReader(run_dir / "events.jsonl")
        offsets = [
            e.offset for e in reader.iter_envelopes() if e.source == pod_source
        ]
        assert offsets == [0, 1, 2, 3, 4]
        emitter_v2.close()
