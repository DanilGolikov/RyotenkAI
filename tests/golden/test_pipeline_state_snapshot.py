"""Snapshot test for the FSM state machine after deterministic transitions.

The FSM is the canonical view of pipeline progress; this test captures
the on-disk ``job.json`` shape after a happy-path run.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ryotenkai_pod.runner.state import JobLifecycleFSM, JobState

pytestmark = [pytest.mark.golden]


def test_pipeline_state_snapshot(
    tmp_path_factory: pytest.TempPathFactory,
    snapshot_anchored,  # type: ignore[no-untyped-def]
    scrub,  # type: ignore[no-untyped-def]
) -> None:
    workspace = tmp_path_factory.mktemp("pipeline-snap")
    fsm = JobLifecycleFSM(workspace_dir=workspace)
    fsm.restore_or_init()
    fsm.submit("j-snap")
    fsm.transition(JobState.RUNNING, message="trainer_spawned")
    fsm.transition(JobState.COMPLETED, message="exit_code=0")

    snapshot = fsm.current()
    assert snapshot is not None
    payload = snapshot.to_dict()

    # Also include the on-disk JSONL trail for stage-by-stage visibility.
    jsonl_path: Path = workspace / "state" / "job.jsonl"
    jsonl_records = [
        json.loads(line) for line in jsonl_path.read_text().splitlines()
    ]

    artefact = {
        "final_state": payload,
        "audit_log": jsonl_records,
    }
    assert scrub(artefact) == snapshot_anchored
