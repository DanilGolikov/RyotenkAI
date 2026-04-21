"""Tests for src.api.services.log_service — state-driven registry + legacy fallback."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.api.services.log_service import list_log_files, read_chunk, resolve_log_path
from src.pipeline.state import (
    PipelineAttemptState,
    PipelineState,
    PipelineStateStore,
    StageRunState,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers — build a run directory with/without state, in new or legacy layout.
# ---------------------------------------------------------------------------

def _write_state(run_dir: Path, attempt: PipelineAttemptState) -> None:
    store = PipelineStateStore(run_dir)
    state = PipelineState(
        schema_version=1,
        logical_run_id="run-x",
        run_directory=str(run_dir),
        config_path="/tmp/cfg.yaml",
        active_attempt_id=None,
        pipeline_status=StageRunState.STATUS_COMPLETED,
        training_critical_config_hash="h1",
        late_stage_config_hash="h2",
        model_dataset_config_hash="h3",
        attempts=[attempt],
        current_output_lineage={},
    )
    store.save(state)


def _make_attempt(stage_runs: dict[str, StageRunState]) -> PipelineAttemptState:
    return PipelineAttemptState(
        attempt_id="run-x:attempt:1",
        attempt_no=1,
        runtime_name="runpod",
        requested_action="fresh",
        effective_action="fresh",
        restart_from_stage=None,
        status=StageRunState.STATUS_COMPLETED,
        started_at="2026-01-01T00:00:00Z",
        stage_runs=stage_runs,
    )


@pytest.fixture
def new_layout_run(tmp_path: Path) -> Path:
    """Modern layout: logs/ directory with pipeline.log + per-stage files."""
    run_dir = tmp_path / "run_new"
    attempt_dir = run_dir / "attempts" / "attempt_1"
    (attempt_dir / "logs").mkdir(parents=True)
    (attempt_dir / "logs" / "pipeline.log").write_text("agg\n", encoding="utf-8")
    (attempt_dir / "logs" / "dataset_validator.log").write_text("dv\n", encoding="utf-8")
    (attempt_dir / "logs" / "training_monitor.log").write_text("tm\n", encoding="utf-8")
    (attempt_dir / "logs" / "training.log").write_text("remote-tm\n", encoding="utf-8")

    attempt = _make_attempt({
        "dataset_validator": StageRunState(
            stage_name="dataset_validator",
            status=StageRunState.STATUS_COMPLETED,
            log_paths={"stage": "logs/dataset_validator.log"},
        ),
        "training_monitor": StageRunState(
            stage_name="training_monitor",
            status=StageRunState.STATUS_COMPLETED,
            log_paths={
                "stage": "logs/training_monitor.log",
                "remote_training": "logs/training.log",
            },
        ),
    })
    _write_state(run_dir, attempt)
    return run_dir


@pytest.fixture
def legacy_run(tmp_path: Path) -> Path:
    """Legacy layout: logs at attempt root (pre-LogLayout runs)."""
    run_dir = tmp_path / "run_legacy"
    attempt_dir = run_dir / "attempts" / "attempt_1"
    attempt_dir.mkdir(parents=True)
    (attempt_dir / "pipeline.log").write_text("legacy-pipeline\n", encoding="utf-8")
    (attempt_dir / "training.log").write_text("legacy-training\n", encoding="utf-8")
    # State exists but attempts[0].stage_runs have no log_paths entries.
    attempt = _make_attempt({})
    _write_state(run_dir, attempt)
    return run_dir


# ---------------------------------------------------------------------------
# Positive — new layout, state-driven discovery
# ---------------------------------------------------------------------------

def test_list_log_files_new_layout_exposes_pipeline_and_per_stage(new_layout_run: Path) -> None:
    infos = list_log_files(new_layout_run, 1)
    names = {info.name: info for info in infos}

    assert names["pipeline.log"].exists is True
    assert names["dataset_validator.log"].exists is True
    assert names["training_monitor.log"].exists is True
    # Remote training log surfaces under its historical file name.
    assert names["training.log"].exists is True
    # Run-root file still advertised even when absent.
    assert "tui_launch.log" in names


def test_resolve_log_path_per_stage_points_to_logs_dir(new_layout_run: Path) -> None:
    path = resolve_log_path(new_layout_run, 1, "dataset_validator.log")
    assert path == (new_layout_run / "attempts" / "attempt_1" / "logs" / "dataset_validator.log").resolve()


def test_read_chunk_reads_per_stage_content(new_layout_run: Path) -> None:
    chunk = read_chunk(new_layout_run, 1, "training_monitor.log")
    assert "tm" in chunk.content


def test_remote_training_surfaced_under_historical_name(new_layout_run: Path) -> None:
    chunk = read_chunk(new_layout_run, 1, "training.log")
    assert "remote-tm" in chunk.content


# ---------------------------------------------------------------------------
# Positive — legacy fallback
# ---------------------------------------------------------------------------

def test_legacy_layout_falls_back_to_attempt_root(legacy_run: Path) -> None:
    infos = list_log_files(legacy_run, 1)
    names = {info.name: info for info in infos}

    assert names["pipeline.log"].exists is True
    assert names["training.log"].exists is True


def test_legacy_resolve_returns_attempt_root_path(legacy_run: Path) -> None:
    resolved = resolve_log_path(legacy_run, 1, "pipeline.log")
    assert resolved == (legacy_run / "attempts" / "attempt_1" / "pipeline.log").resolve()


def test_legacy_read_chunk_returns_content(legacy_run: Path) -> None:
    chunk = read_chunk(legacy_run, 1, "training.log")
    assert "legacy-training" in chunk.content


# ---------------------------------------------------------------------------
# Negative — unsupported / missing
# ---------------------------------------------------------------------------

def test_resolve_rejects_unknown_file(new_layout_run: Path) -> None:
    with pytest.raises(ValueError, match="unsupported log file"):
        resolve_log_path(new_layout_run, 1, "../../etc/passwd")


def test_resolve_rejects_unknown_file_in_legacy(legacy_run: Path) -> None:
    with pytest.raises(ValueError, match="unsupported log file"):
        resolve_log_path(legacy_run, 1, "model_evaluator.log")


def test_read_chunk_of_missing_file_returns_empty_eof(tmp_path: Path) -> None:
    """Missing run with tui_launch.log (run-root file): EOF chunk, no exception."""
    run_dir = tmp_path / "empty_run"
    run_dir.mkdir()
    chunk = read_chunk(run_dir, 1, "tui_launch.log")
    assert chunk.eof is True
    assert chunk.content == ""


# ---------------------------------------------------------------------------
# Dependency error — corrupted pipeline_state.json falls back, doesn't crash
# ---------------------------------------------------------------------------

def test_corrupted_state_falls_back_to_legacy_layout(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_corrupt"
    attempt_dir = run_dir / "attempts" / "attempt_1"
    attempt_dir.mkdir(parents=True)
    (attempt_dir / "pipeline.log").write_text("salvaged\n", encoding="utf-8")
    (run_dir / "pipeline_state.json").write_text("{not json", encoding="utf-8")

    infos = list_log_files(run_dir, 1)
    names = {info.name: info for info in infos}
    assert names["pipeline.log"].exists is True


def test_state_without_matching_attempt_falls_back(tmp_path: Path) -> None:
    """State exists but attempt_no is different — legacy fallback takes over."""
    run_dir = tmp_path / "run_no_attempt"
    attempt_dir = run_dir / "attempts" / "attempt_1"
    attempt_dir.mkdir(parents=True)
    (attempt_dir / "pipeline.log").write_text("x\n", encoding="utf-8")
    _write_state(run_dir, _make_attempt({}))

    # Ask for attempt_no=2 — state has only attempt_1; must fallback to disk.
    infos = list_log_files(run_dir, 2)
    names = {info.name: info for info in infos}
    # attempt_2 doesn't exist on disk → only run-root entries present.
    assert "tui_launch.log" in names


# ---------------------------------------------------------------------------
# Boundary — tui_launch.log handling, path traversal safety
# ---------------------------------------------------------------------------

def test_tui_launch_log_resolves_to_run_root(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_tui"
    run_dir.mkdir()
    (run_dir / "tui_launch.log").write_text("launch\n", encoding="utf-8")
    path = resolve_log_path(run_dir, 1, "tui_launch.log")
    assert path == (run_dir / "tui_launch.log").resolve()


def test_read_chunk_offset_beyond_file_resets_to_zero(new_layout_run: Path) -> None:
    """If offset > file size (truncation/rotation), read starts from 0 — no IndexError."""
    chunk = read_chunk(new_layout_run, 1, "pipeline.log", offset=10_000_000)
    assert chunk.offset == 0
    assert "agg" in chunk.content


# ---------------------------------------------------------------------------
# Regression — OCP: adding a stage requires NO changes in log_service
# ---------------------------------------------------------------------------

def test_new_stage_appears_in_listing_via_state_only(tmp_path: Path) -> None:
    """Add a made-up stage via log_paths; it must appear without touching log_service code."""
    run_dir = tmp_path / "run_ocp"
    attempt_dir = run_dir / "attempts" / "attempt_1"
    (attempt_dir / "logs").mkdir(parents=True)
    (attempt_dir / "logs" / "brand_new_stage.log").write_text("hi\n", encoding="utf-8")

    attempt = _make_attempt({
        "brand_new_stage": StageRunState(
            stage_name="brand_new_stage",
            status=StageRunState.STATUS_COMPLETED,
            log_paths={"stage": "logs/brand_new_stage.log"},
        ),
    })
    _write_state(run_dir, attempt)

    names = {info.name for info in list_log_files(run_dir, 1)}
    assert "brand_new_stage.log" in names


# ---------------------------------------------------------------------------
# Combinatorial — legacy + new layout mixed
# ---------------------------------------------------------------------------

def test_mixed_layout_exposes_new_files_and_preserves_legacy_root_fallback(
    tmp_path: Path,
) -> None:
    """When state has a registry, state wins: legacy root files stay hidden unless
    they match a registered logical name. This test asserts the current contract."""
    run_dir = tmp_path / "run_mixed"
    attempt_dir = run_dir / "attempts" / "attempt_1"
    (attempt_dir / "logs").mkdir(parents=True)
    (attempt_dir / "logs" / "pipeline.log").write_text("new\n", encoding="utf-8")
    (attempt_dir / "logs" / "dataset_validator.log").write_text("dv\n", encoding="utf-8")
    # Legacy leftover at root — should NOT override the new one.
    (attempt_dir / "pipeline.log").write_text("legacy\n", encoding="utf-8")

    attempt = _make_attempt({
        "dataset_validator": StageRunState(
            stage_name="dataset_validator",
            status=StageRunState.STATUS_COMPLETED,
            log_paths={"stage": "logs/dataset_validator.log"},
        ),
    })
    _write_state(run_dir, attempt)

    chunk = read_chunk(run_dir, 1, "pipeline.log")
    assert "new" in chunk.content
    assert "legacy" not in chunk.content


def test_stage_name_with_spaces_resolves_to_slug_filename(tmp_path: Path) -> None:
    """Real-world case: StageNames.DATASET_VALIDATOR = 'Dataset Validator'
    is used as the key in stage_runs, but log_paths stores a slug path.
    API must expose the slug filename — NOT reconstruct 'Dataset Validator.log'."""
    run_dir = tmp_path / "run_slug"
    attempt_dir = run_dir / "attempts" / "attempt_1"
    (attempt_dir / "logs").mkdir(parents=True)
    (attempt_dir / "logs" / "dataset_validator.log").write_text("dv\n", encoding="utf-8")

    attempt = _make_attempt({
        # Key matches StageNames enum value — with a SPACE.
        "Dataset Validator": StageRunState(
            stage_name="Dataset Validator",
            status=StageRunState.STATUS_COMPLETED,
            # Registry path is the slug, as written by LogLayout.
            log_paths={"stage": "logs/dataset_validator.log"},
        ),
    })
    _write_state(run_dir, attempt)

    names = {info.name for info in list_log_files(run_dir, 1)}
    # Slug name — this is the whole point.
    assert "dataset_validator.log" in names
    # Never should the raw StageNames value leak into the API as a filename.
    assert "Dataset Validator.log" not in names


def test_resolve_by_slug_filename_when_state_key_has_spaces(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_resolve_slug"
    attempt_dir = run_dir / "attempts" / "attempt_1"
    (attempt_dir / "logs").mkdir(parents=True)
    (attempt_dir / "logs" / "gpu_deployer.log").write_text("g\n", encoding="utf-8")

    attempt = _make_attempt({
        "GPU Deployer": StageRunState(
            stage_name="GPU Deployer",
            status=StageRunState.STATUS_COMPLETED,
            log_paths={"stage": "logs/gpu_deployer.log"},
        ),
    })
    _write_state(run_dir, attempt)

    resolved = resolve_log_path(run_dir, 1, "gpu_deployer.log")
    assert resolved == (attempt_dir / "logs" / "gpu_deployer.log").resolve()
