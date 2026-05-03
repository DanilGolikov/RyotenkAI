from __future__ import annotations

from pathlib import Path

import pytest

from src.cli.run_rendering import (
    RunInspectionRenderer,
    format_duration,
    render_run_diff_lines,
    render_run_inspection_lines,
    render_run_status_snapshot,
    render_runs_list_lines,
)
from src.pipeline.run_queries import RunInspectionData, RunSummaryRow
from src.pipeline.state import PipelineAttemptState, PipelineState, StageRunState


def _build_attempt(
    *,
    attempt_no: int = 1,
    status: str = StageRunState.STATUS_FAILED,
    execution_mode: str = StageRunState.MODE_EXECUTED,
    reuse_from: dict[str, object] | None = None,
    outputs: dict[str, object] | None = None,
    error: str | None = None,
    enabled_stage_names: list[str] | None = None,
) -> PipelineAttemptState:
    stage_name = (enabled_stage_names or ["Dataset Validator"])[0]
    return PipelineAttemptState(
        attempt_id=f"attempt-{attempt_no}",
        attempt_no=attempt_no,
        runtime_name="runtime",
        requested_action="resume",
        effective_action="resume",
        restart_from_stage=None,
        status=status,
        started_at="2026-03-30T00:00:00+00:00",
        completed_at="2026-03-30T00:01:05+00:00",
        error=error,
        enabled_stage_names=enabled_stage_names or [stage_name],
        stage_runs={
            stage_name: StageRunState(
                stage_name=stage_name,
                status=status,
                execution_mode=execution_mode,
                reuse_from=reuse_from,
                outputs=outputs or {},
                error=error,
                started_at="2026-03-30T00:00:00+00:00",
                completed_at="2026-03-30T00:01:05+00:00",
            )
        },
    )


def _build_state(*attempts: PipelineAttemptState) -> PipelineState:
    return PipelineState(
        schema_version=1,
        logical_run_id="logical_run_1",
        run_directory="runs/logical_run_1",
        config_path="/configs/pipeline.yaml",
        active_attempt_id=None,
        pipeline_status=StageRunState.STATUS_RUNNING,
        training_critical_config_hash="train_hash",
        late_stage_config_hash="late_hash",
        root_mlflow_run_id="1234567890abcdef",
        attempts=list(attempts),
    )


def test_format_duration_clamps_negative_delta_to_zero() -> None:
    """Clock drift / stale timestamps land on ``"0s"`` instead of an empty slot.

    Previously the helper returned ``""`` for negative deltas; after the
    rewrite through ``duration_seconds`` (which clamps via ``max(delta, 0)``)
    we render ``"0s"`` — readable and never confused with "no data".
    """
    assert format_duration("2026-03-30T00:02:00+00:00", "2026-03-30T00:01:00+00:00") == "0s"


def test_render_run_inspection_lines_includes_logs_verbose_outputs_and_reused_suffix() -> None:
    attempt = _build_attempt(
        execution_mode=StageRunState.MODE_REUSED,
        reuse_from={"attempt_id": "logical_run_1:attempt:7"},
        outputs={"long_output": "x" * 100},
        error="boom",
    )
    data = RunInspectionData(
        run_dir=Path("/tmp/run_render"),
        state=_build_state(attempt),
        log_tails={1: ["line 1", "line 2"]},
    )

    lines = render_run_inspection_lines(data, verbose=True, include_logs=True)
    rendered = "\n".join(lines)

    assert "Run: run_render" in rendered
    assert "MLflow  : 1234567890ab..." in rendered
    assert "reused (7)" in rendered
    assert "long_output = " in rendered
    assert "--- Attempt 1 / pipeline.log ---" in rendered
    assert "line 1" in rendered


def test_run_inspection_renderer_prints_exact_lines(capsys) -> None:
    attempt = _build_attempt(status=StageRunState.STATUS_COMPLETED)
    data = RunInspectionData(
        run_dir=Path("/tmp/run_exact"),
        state=_build_state(attempt),
        log_tails={},
    )
    renderer = RunInspectionRenderer()

    expected_output = "\n".join(render_run_inspection_lines(data)) + "\n"
    renderer.render(data)

    assert capsys.readouterr().out == expected_output


@pytest.mark.parametrize(
    ("training_changed", "late_changed", "expected_fragments"),
    [
        (False, False, ["No config changes between attempts."]),
        (True, False, ["critical", "A: hash_a_cri", "B: hash_b_cri"]),
        (False, True, ["late_stage", "A: hash_a_lat", "B: hash_b_lat"]),
        (True, True, ["critical", "late_stage"]),
    ],
)
def test_render_run_diff_lines_combinatorial(
    training_changed: bool,
    late_changed: bool,
    expected_fragments: list[str],
) -> None:
    diff = {
        "found_a": True,
        "found_b": True,
        "training_critical_changed": training_changed,
        "late_stage_changed": late_changed,
        "hash_a_critical": "hash_a_critical_value",
        "hash_b_critical": "hash_b_critical_value",
        "hash_a_late": "hash_a_late_value",
        "hash_b_late": "hash_b_late_value",
    }

    rendered = "\n".join(render_run_diff_lines(diff, 1, 2))

    for fragment in expected_fragments:
        assert fragment in rendered


def test_render_run_diff_lines_reports_missing_attempts() -> None:
    lines = render_run_diff_lines({"found_a": False, "found_b": True}, 1, 2)

    assert lines == ("Attempt 1 or 2 not found in state.",)


def test_render_runs_list_lines_empty_returns_single_message(tmp_path: Path) -> None:
    assert render_runs_list_lines(tmp_path / "runs", []) == (f"No runs found in {tmp_path / 'runs'}",)


def test_render_runs_list_lines_formats_duration_from_summary_rows(tmp_path: Path) -> None:
    row = RunSummaryRow(
        run_id="run_1",
        run_dir=tmp_path / "runs" / "run_1",
        created_at="2026-03-30 00:00",
        created_ts=1.0,
        status="completed",
        attempts=2,
        config_name="pipeline.yaml",
        mlflow_run_id=None,
        started_at="2026-03-30T00:00:00+00:00",
        completed_at="2026-03-30T00:01:05+00:00",
    )

    rendered = "\n".join(render_runs_list_lines(tmp_path / "runs", [row]))

    assert "run_1" in rendered
    assert "1m 5s" in rendered
    assert "pipeline.yaml" in rendered


def test_render_run_status_snapshot_marks_running_stage_and_pending_stage() -> None:
    attempt = PipelineAttemptState(
        attempt_id="attempt-1",
        attempt_no=1,
        runtime_name="runtime",
        requested_action="resume",
        effective_action="resume",
        restart_from_stage=None,
        status=StageRunState.STATUS_RUNNING,
        started_at="2026-03-30T00:00:00+00:00",
        enabled_stage_names=["Dataset Validator", "GPU Deployer"],
        stage_runs={
            "Dataset Validator": StageRunState(
                stage_name="Dataset Validator",
                status=StageRunState.STATUS_RUNNING,
                execution_mode=StageRunState.MODE_EXECUTED,
                started_at="2026-03-30T00:00:00+00:00",
            )
        },
    )
    state = _build_state(attempt)

    rendered = "\n".join(render_run_status_snapshot("logical_run_1", state))

    assert "attempt 1/1" in rendered
    assert "Dataset Validator" in rendered
    assert "<--" in rendered
    assert "GPU Deployer" in rendered
    assert "pending" in rendered


def test_render_run_status_snapshot_handles_run_without_attempts() -> None:
    state = _build_state()
    state.attempts = []

    lines = render_run_status_snapshot("logical_run_1", state)

    assert lines[0] == "Run: logical_run_1  status: running"
    assert lines[-1] == ""
