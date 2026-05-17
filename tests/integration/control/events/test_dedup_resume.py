"""Dedup reconstruction after restart — closes R-13 (Phase 9).

Scenario:

1. Pod emits 101 events (offsets 0..100) via :meth:`emit_remote`.
2. Control journal contains all 101 events.
3. Control "restarts" — a fresh :class:`ControlEventEmitter` is built
   via :meth:`for_run` which reconstructs the dedup set from the
   journal tail (last 10k offsets per source).
4. Pod re-sends offsets 95..100 (network jitter scenario).
5. Verify:

   * Every re-sent envelope is detected as a duplicate and dropped.
   * The journal does NOT grow.
   * ``events_remote_dropped_total["duplicate"]`` reflects the count.
   * Live offsets continue from offset 101 on the next genuine emit.
"""

from __future__ import annotations

from pathlib import Path

from ryotenkai_control.events import ControlEventEmitter, JournalReader
from ryotenkai_shared.events.types.pod_training import (
    TrainingStepEvent,
    TrainingStepPayload,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_step(run_id: str, source: str, offset: int) -> TrainingStepEvent:
    return TrainingStepEvent(
        source=source,
        run_id=run_id,
        offset=offset,
        payload=TrainingStepPayload(
            step=offset,
            loss=2.0,
            learning_rate=1e-5,
            grad_norm=0.5,
            tokens_per_sec=1000.0,
            samples_per_sec=4.0,
        ),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDedupResume:
    """Dedup set survives a control restart via journal-tail reconstruction."""

    def test_resends_dropped_after_restart(self, tmp_path: Path) -> None:
        """Pod resends 95..100 after control restart — all silently dropped."""
        run_id = "run-dedup-resume"
        pod_source = f"pod://{run_id}/trainer"
        run_dir = tmp_path / run_id

        # ------------------------------------------------------------
        # Phase A — initial control instance accepts 0..100.
        # ------------------------------------------------------------
        emitter_v1 = ControlEventEmitter.for_run(
            run_id=run_id, run_directory=run_dir,
        )
        for offset in range(101):
            emitter_v1.emit_remote(_make_step(run_id, pod_source, offset))

        assert emitter_v1.events_remote_accepted_total == 101
        emitter_v1.close()

        journal_path = run_dir / "events.jsonl"
        pre_size = journal_path.stat().st_size
        pre_count = sum(
            1 for line in journal_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        assert pre_count == 101

        # ------------------------------------------------------------
        # Phase B — control "restart" — a fresh emitter reconstructs
        # the dedup set from the journal tail.
        # ------------------------------------------------------------
        emitter_v2 = ControlEventEmitter.for_run(
            run_id=run_id, run_directory=run_dir,
        )
        # The dedup set should contain at least the last 10k entries
        # per source (101 < 10k, so all of them).
        assert emitter_v2.dedup.size >= 101

        # ------------------------------------------------------------
        # Phase C — pod re-sends 95..100 (network jitter).
        # ------------------------------------------------------------
        for offset in range(95, 101):
            emitter_v2.emit_remote(_make_step(run_id, pod_source, offset))

        # All 6 re-sends should be silently dropped as duplicates.
        assert emitter_v2.events_remote_dropped_total.get("duplicate") == 6
        # And no new events were accepted.
        assert emitter_v2.events_remote_accepted_total == 0

        # Journal size unchanged.
        post_size = journal_path.stat().st_size
        post_count = sum(
            1 for line in journal_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        assert post_size == pre_size
        assert post_count == pre_count

        # ------------------------------------------------------------
        # Phase D — pod sends offset 101 (a genuinely new event).
        # That MUST land on disk.
        # ------------------------------------------------------------
        emitter_v2.emit_remote(_make_step(run_id, pod_source, 101))
        assert emitter_v2.events_remote_accepted_total == 1
        emitter_v2.close()

        # Verify the new event landed.
        post_phase_d_count = sum(
            1 for line in journal_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        assert post_phase_d_count == pre_count + 1

        # And read-back via JournalReader confirms the offset 101 exists.
        reader = JournalReader(journal_path)
        offsets = sorted({
            e.offset for e in reader.iter_envelopes()
            if e.source == pod_source
        })
        assert offsets == list(range(102))

    def test_resends_dropped_with_dedup_hits_counter(self, tmp_path: Path) -> None:
        """The dedup ``dedup_hits_total`` counter ticks on each duplicate."""
        run_id = "run-counter"
        pod_source = f"pod://{run_id}/trainer"
        run_dir = tmp_path / run_id

        emitter_v1 = ControlEventEmitter.for_run(
            run_id=run_id, run_directory=run_dir,
        )
        for offset in range(20):
            emitter_v1.emit_remote(_make_step(run_id, pod_source, offset))
        emitter_v1.close()

        # Restart.
        emitter_v2 = ControlEventEmitter.for_run(
            run_id=run_id, run_directory=run_dir,
        )
        pre_hits = emitter_v2.dedup.dedup_hits_total

        # Re-send the same range.
        for offset in range(20):
            emitter_v2.emit_remote(_make_step(run_id, pod_source, offset))
        # All 20 should be detected as duplicates → 20 hits.
        post_hits = emitter_v2.dedup.dedup_hits_total
        assert post_hits - pre_hits == 20
        emitter_v2.close()

    def test_partial_resend_drops_only_seen(self, tmp_path: Path) -> None:
        """Mixed batch: some seen + some new. Only the seen are dropped."""
        run_id = "run-mixed"
        pod_source = f"pod://{run_id}/trainer"
        run_dir = tmp_path / run_id

        emitter_v1 = ControlEventEmitter.for_run(
            run_id=run_id, run_directory=run_dir,
        )
        for offset in range(10):
            emitter_v1.emit_remote(_make_step(run_id, pod_source, offset))
        emitter_v1.close()

        # Restart.
        emitter_v2 = ControlEventEmitter.for_run(
            run_id=run_id, run_directory=run_dir,
        )

        # Submit 5..14 — overlap with original 5..9, new ones 10..14.
        for offset in range(5, 15):
            emitter_v2.emit_remote(_make_step(run_id, pod_source, offset))

        # 5 duplicates (5..9), 5 new (10..14).
        assert emitter_v2.events_remote_dropped_total.get("duplicate") == 5
        assert emitter_v2.events_remote_accepted_total == 5

        # Journal now has 15 events (0..14).
        emitter_v2.close()
        reader = JournalReader(run_dir / "events.jsonl")
        offsets = sorted({e.offset for e in reader.iter_envelopes()})
        assert offsets == list(range(15))

    def test_dedup_isolated_per_run_after_restart(self, tmp_path: Path) -> None:
        """Two different runs share neither dedup state nor journals."""
        # Run A — emit, close.
        run_a = "run-A"
        pod_source_a = f"pod://{run_a}/trainer"
        emitter_a = ControlEventEmitter.for_run(
            run_id=run_a, run_directory=tmp_path / run_a,
        )
        for offset in range(5):
            emitter_a.emit_remote(_make_step(run_a, pod_source_a, offset))
        emitter_a.close()

        # Run B — separate run dir, separate dedup set.
        run_b = "run-B"
        pod_source_b = f"pod://{run_b}/trainer"
        emitter_b = ControlEventEmitter.for_run(
            run_id=run_b, run_directory=tmp_path / run_b,
        )
        # Run B's dedup starts empty (different journal).
        assert emitter_b.dedup.size == 0

        # Emit the same offsets on B — they're new for B because the
        # source URI differs.
        for offset in range(5):
            emitter_b.emit_remote(_make_step(run_b, pod_source_b, offset))
        assert emitter_b.events_remote_accepted_total == 5
        assert emitter_b.events_remote_dropped_total.get("duplicate", 0) == 0
        emitter_b.close()
