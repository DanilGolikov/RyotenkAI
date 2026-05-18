"""E2E integration: emit→journal→manifest→sha256 verification (Phase 9).

This integration test wires together the production emitter, journal
writer/reader, in-memory bus, and MLflow finalizer to verify the
end-to-end invariants documented in
``docs/plans/ethereal-tumbling-patterson.md`` Phase 9:

1. Spin up a :class:`ControlEventEmitter` (the production emitter) for
   an in-memory simulated run.
2. Walk through a representative slice of 5 stages emitting realistic
   events (stage.started, stage.completed, training.*, memory.*).
3. Verify on disk:

   * Every expected event ``kind`` is present in ``events.jsonl``.
   * Offsets are strictly monotonic per ``source``.
   * Timestamps are monotonic per source.
   * Each line passes schema validation via :func:`from_jsonl`
     (``strict=True``).
4. Run :class:`MlflowFinalizer` against an in-memory recorder fake;
   verify the manifest's ``events_sha256`` matches the journal file's
   sha256 byte-for-byte.

Scope: the test is a *single-process* simulation of the full event
pipeline. It does NOT exercise the WebSocket pod-bridge (that surface
is covered by unit tests + the :mod:`test_dedup_resume` integration).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ryotenkai_control.events import (
    ControlEventEmitter,
    JournalReader,
)
from ryotenkai_shared.events import from_jsonl
from ryotenkai_shared.events.types.control_run import (
    RunCompletedEvent,
    RunCompletedPayload,
    RunStartedEvent,
    RunStartedPayload,
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
    TrainingCompletedEvent,
    TrainingCompletedPayload,
    TrainingStartedEvent,
    TrainingStartedPayload,
    TrainingStepEvent,
    TrainingStepPayload,
)
from ryotenkai_shared.events import UNKNOWN_OFFSET


# ---------------------------------------------------------------------------
# Event factories — produce a realistic stage timeline
# ---------------------------------------------------------------------------


_PIPELINE_STAGES: list[str] = [
    "gpu_deployer",
    "dataset_validator",
    "trainer",
    "evaluator",
    "inference_deployer",
]


def _emit_full_pipeline(emitter: ControlEventEmitter, run_id: str) -> list[type]:
    """Drive ``emitter`` through 5 stages worth of representative events.

    Returns the list of event class types emitted, in emit order. Tests
    use this to assert the journal contains every type after the round
    trip.
    """
    types_emitted: list[type] = []

    def _emit(ev) -> None:
        emitter.emit(ev)
        types_emitted.append(type(ev))

    # ------- RunStarted -------
    _emit(
        RunStartedEvent(
            source=emitter.source,
            run_id=run_id,
            offset=UNKNOWN_OFFSET,
            payload=RunStartedPayload(
                run_name="e2e-test",
                algorithm="sft",
                model_id="acme/test",
                dataset_id="default",
                config_hash="abc123",
            ),
        ),
    )

    # ------- 5 stages -------
    for idx, stage_name in enumerate(_PIPELINE_STAGES):
        with emitter.stage_scope(stage_name):
            _emit(
                StageStartedEvent(
                    source=emitter.source,
                    run_id=run_id,
                    offset=UNKNOWN_OFFSET,
                    payload=StageStartedPayload(
                        stage_name=stage_name,
                        stage_index=idx,
                        total_stages=len(_PIPELINE_STAGES),
                        inputs_summary={"idx": idx},
                    ),
                ),
            )
            # The trainer stage emits training events too — simulate that
            # the orchestrator forwards pod events through the same
            # emitter (in production this goes via emit_remote, but the
            # journal contract is identical).
            if stage_name == "trainer":
                pod_source = f"pod://{run_id}/trainer"
                _emit(
                    TrainingStartedEvent(
                        source=pod_source,
                        run_id=run_id,
                        offset=0,  # pre-numbered by pod (emit_remote)
                        payload=TrainingStartedPayload(
                            max_steps=100,
                            num_train_epochs=2,
                            per_device_batch_size=4,
                            gradient_accumulation_steps=1,
                            learning_rate=2e-5,
                            algorithm="sft",
                        ),
                    ),
                )
                # Step events.
                for step in range(3):
                    _emit(
                        TrainingStepEvent(
                            source=pod_source,
                            run_id=run_id,
                            offset=step + 1,
                            payload=TrainingStepPayload(
                                step=step,
                                loss=2.5 - 0.5 * step,
                                learning_rate=2e-5,
                                grad_norm=1.2,
                                tokens_per_sec=12345.6,
                                samples_per_sec=8.0,
                            ),
                        ),
                    )
                # Memory event mid-training.
                _emit(
                    MemoryCacheClearedEvent(
                        source=pod_source,
                        run_id=run_id,
                        offset=4,
                        payload=MemoryCacheClearedPayload(
                            device="cuda:0",
                            before_bytes=1_000_000_000,
                            after_bytes=500_000_000,
                            trigger="threshold",
                        ),
                    ),
                )
                _emit(
                    TrainingCompletedEvent(
                        source=pod_source,
                        run_id=run_id,
                        offset=5,
                        payload=TrainingCompletedPayload(
                            final_step=2,
                            mean_loss=2.0,
                            duration_s=60.0,
                            tokens_processed=10_000,
                        ),
                    ),
                )

            _emit(
                StageCompletedEvent(
                    source=emitter.source,
                    run_id=run_id,
                    offset=UNKNOWN_OFFSET,
                    payload=StageCompletedPayload(
                        stage_name=stage_name,
                        duration_s=1.5,
                        outputs_summary={"stage": stage_name},
                    ),
                ),
            )

    # ------- RunCompleted -------
    _emit(
        RunCompletedEvent(
            source=emitter.source,
            run_id=run_id,
            offset=UNKNOWN_OFFSET,
            payload=RunCompletedPayload(
                duration_s=12.3,
                final_status="success",
                mlflow_run_id="mlrun-abc",
            ),
        ),
    )
    return types_emitted


# ---------------------------------------------------------------------------
# E2E test cases
# ---------------------------------------------------------------------------


class TestE2EPipeline:
    """End-to-end: emit, journal, schema-check, sha256 verify."""

    def test_journal_contains_every_expected_kind(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run-e2e"
        emitter = ControlEventEmitter.for_run(
            run_id="run-e2e", run_directory=run_dir,
        )
        types_emitted = _emit_full_pipeline(emitter, "run-e2e")
        emitter.close()

        journal_path = run_dir / "events.jsonl"
        assert journal_path.exists()

        # Read back via the production reader; collect kinds seen.
        reader = JournalReader(journal_path)
        kinds_seen = {e.kind for e in reader.iter_envelopes()}

        expected_kinds = {
            "ryotenkai.control.run.started",
            "ryotenkai.control.stage.started",
            "ryotenkai.control.stage.completed",
            "ryotenkai.pod.training.started",
            "ryotenkai.pod.training.step",
            "ryotenkai.pod.training.completed",
            "ryotenkai.pod.memory.cache_cleared",
            "ryotenkai.control.run.completed",
        }
        # Every expected kind landed on disk.
        missing = expected_kinds - kinds_seen
        assert not missing, f"missing kinds in journal: {missing}"
        # And we observed at least len(_PIPELINE_STAGES) StageStarted/Completed pairs.
        assert types_emitted.count(StageStartedEvent) == len(_PIPELINE_STAGES)
        assert types_emitted.count(StageCompletedEvent) == len(_PIPELINE_STAGES)

    def test_offsets_strictly_monotonic_per_source(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run-mono"
        emitter = ControlEventEmitter.for_run(
            run_id="run-mono", run_directory=run_dir,
        )
        _emit_full_pipeline(emitter, "run-mono")
        emitter.close()

        reader = JournalReader(run_dir / "events.jsonl")
        per_source_offsets: dict[str, list[int]] = {}
        for envelope in reader.iter_envelopes():
            per_source_offsets.setdefault(envelope.source, []).append(envelope.offset)

        # Each source's offsets are strictly increasing (no duplicates,
        # no gaps allowed below — gaps are OK if other producers wrote
        # to disjoint sources).
        for source, offsets in per_source_offsets.items():
            assert offsets == sorted(set(offsets)), (
                f"source {source!r} offsets not strictly monotonic: {offsets}"
            )

    def test_timestamps_monotonic_per_source(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run-ts"
        emitter = ControlEventEmitter.for_run(
            run_id="run-ts", run_directory=run_dir,
        )
        _emit_full_pipeline(emitter, "run-ts")
        emitter.close()

        reader = JournalReader(run_dir / "events.jsonl")
        # Within a single source, timestamps should be non-decreasing.
        # Different sources can interleave any way they want.
        last_ts_per_source: dict[str, Any] = {}
        for envelope in reader.iter_envelopes():
            prev = last_ts_per_source.get(envelope.source)
            if prev is not None:
                assert envelope.time >= prev, (
                    f"non-monotonic time on {envelope.source}: "
                    f"prev={prev} now={envelope.time}"
                )
            last_ts_per_source[envelope.source] = envelope.time

    def test_every_line_passes_strict_schema_validation(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run-strict"
        emitter = ControlEventEmitter.for_run(
            run_id="run-strict", run_directory=run_dir,
        )
        _emit_full_pipeline(emitter, "run-strict")
        emitter.close()

        journal_path = run_dir / "events.jsonl"
        with journal_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                # ``strict=True`` re-raises ValidationError /
                # MalformedEventError; the test fails if either fires.
                from_jsonl(line, strict=True)

    # Phase M7.2 — manifest-related tests removed alongside the
    # legacy ``ryotenkai_control.events.mlflow_finalizer`` module.
    # Journal upload is now driven by the new
    # :class:`MlflowFinalizer` in
    # :mod:`ryotenkai_control.pipeline.mlflow.lifecycle.finalizer`
    # and covered by its own unit tests; the manifest contract no
    # longer exists.
