from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.pipeline.domain import RunContext
from src.pipeline.state import (
    PipelineStateStore,
    PipelineStateLoadError,
    PipelineStateLockError,
    StageRunState,
    acquire_run_lock,
    build_attempt_id,
    build_attempt_state,
    update_lineage,
)


def test_state_store_roundtrip(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "logical_run_1"
    store = PipelineStateStore(run_dir)

    state = store.init_state(
        logical_run_id="logical_run_1",
        config_path="/path/to/config.yaml",
        training_critical_config_hash="train_hash",
        late_stage_config_hash="late_hash",
    )
    attempt = build_attempt_state(
        state=state,
        run_ctx=RunContext(name="runtime_attempt", created_at_utc=datetime.now(timezone.utc)),
        requested_action="fresh",
        effective_action="fresh",
        restart_from_stage="Dataset Validator",
        enabled_stage_names=["Dataset Validator"],
        training_critical_config_hash="train_hash",
        late_stage_config_hash="late_hash",
    )
    attempt.stage_runs["Dataset Validator"] = StageRunState(
        stage_name="Dataset Validator",
        status=StageRunState.STATUS_COMPLETED,
        execution_mode=StageRunState.MODE_EXECUTED,
        outputs={"sample_count": 10},
    )
    state.attempts.append(attempt)
    state.active_attempt_id = attempt.attempt_id
    store.save(state)

    loaded = store.load()

    assert loaded.logical_run_id == "logical_run_1"
    assert loaded.active_attempt_id == attempt.attempt_id
    assert loaded.attempts[0].stage_runs["Dataset Validator"].outputs["sample_count"] == 10


def test_acquire_run_lock_rejects_second_writer(tmp_path: Path) -> None:
    lock_path = tmp_path / "logical_run_1" / "run.lock"
    lock = acquire_run_lock(lock_path)

    with pytest.raises(PipelineStateLockError):
        acquire_run_lock(lock_path)

    lock.release()


def test_run_lock_can_be_reacquired_after_release(tmp_path: Path) -> None:
    lock_path = tmp_path / "logical_run_1" / "run.lock"
    first_lock = acquire_run_lock(lock_path)
    first_lock.release()

    second_lock = acquire_run_lock(lock_path)

    assert lock_path.exists()
    second_lock.release()
    assert not lock_path.exists()


def test_state_store_load_fails_on_corrupted_json(tmp_path: Path) -> None:
    store = PipelineStateStore(tmp_path / "runs" / "logical_run_1")
    store.state_path.parent.mkdir(parents=True, exist_ok=True)
    store.state_path.write_text("{broken json", encoding="utf-8")

    with pytest.raises(PipelineStateLoadError, match="Corrupted pipeline state"):
        store.load()


def test_state_store_load_fails_on_schema_version_mismatch(tmp_path: Path) -> None:
    store = PipelineStateStore(tmp_path / "runs" / "logical_run_1")
    store.state_path.parent.mkdir(parents=True, exist_ok=True)
    store.state_path.write_text(
        """
        {
          "schema_version": 999,
          "logical_run_id": "logical_run_1",
          "run_directory": "/tmp/logical_run_1",
          "active_attempt_id": null,
          "pipeline_status": "pending",
          "training_critical_config_hash": "train_hash",
          "late_stage_config_hash": "late_hash",
          "attempts": [],
          "current_output_lineage": {}
        }
        """,
        encoding="utf-8",
    )

    with pytest.raises(PipelineStateLoadError, match="Unsupported pipeline_state schema_version"):
        store.load()


def test_state_store_roundtrip_preserves_config_path(tmp_path: Path) -> None:
    store = PipelineStateStore(tmp_path / "runs" / "run_cfg")
    state = store.init_state(
        logical_run_id="run_cfg",
        config_path="/absolute/path/config.yaml",
        training_critical_config_hash="h1",
        late_stage_config_hash="h2",
    )
    assert state.config_path == "/absolute/path/config.yaml"

    loaded = store.load()
    assert loaded.config_path == "/absolute/path/config.yaml"


def test_state_store_load_succeeds_with_missing_config_path(tmp_path: Path) -> None:
    """Backward-compat: runs created before config_path tracking must load without error."""
    store = PipelineStateStore(tmp_path / "runs" / "run1")
    store.state_path.parent.mkdir(parents=True, exist_ok=True)
    store.state_path.write_text(
        '{"schema_version": 1, "logical_run_id": "run1", "run_directory": "/tmp/run1",'
        ' "config_path": "", "active_attempt_id": null, "pipeline_status": "pending",'
        ' "training_critical_config_hash": "h", "late_stage_config_hash": "h",'
        ' "attempts": [], "current_output_lineage": {}}',
        encoding="utf-8",
    )
    state = store.load()
    assert state.config_path == ""


def test_next_attempt_dir_is_attempt_scoped(tmp_path: Path) -> None:
    store = PipelineStateStore(tmp_path / "runs" / "logical_run_1")

    attempt_dir = store.next_attempt_dir(3)

    assert attempt_dir == tmp_path / "runs" / "logical_run_1" / "attempts" / "attempt_3"


def test_build_attempt_id_is_stable() -> None:
    assert build_attempt_id("logical_run_1", 7) == "logical_run_1:attempt:7"


def test_update_lineage_adds_removes_and_reuses_outputs() -> None:
    lineage = {}

    lineage = update_lineage(
        lineage,
        stage_name="GPU Deployer",
        attempt_id="logical_run_1:attempt:1",
        outputs={"resource_id": "pod-1"},
    )
    assert lineage["GPU Deployer"].outputs["resource_id"] == "pod-1"

    reused_ref = lineage["GPU Deployer"]
    lineage = update_lineage(
        lineage,
        stage_name="GPU Deployer",
        attempt_id="logical_run_1:attempt:2",
        source_ref=reused_ref,
    )
    assert lineage["GPU Deployer"].attempt_id == "logical_run_1:attempt:1"

    lineage = update_lineage(
        lineage,
        stage_name="GPU Deployer",
        attempt_id="logical_run_1:attempt:2",
        remove=True,
    )
    assert "GPU Deployer" not in lineage

