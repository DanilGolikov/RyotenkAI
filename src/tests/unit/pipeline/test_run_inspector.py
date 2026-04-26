"""
Unit tests for src/pipeline/run_inspector.py and related CLI commands.

Coverage:
- RunInspector: load state, graceful log tail
- RunInspectionRenderer: renders without exceptions for all flag combos
- scan_runs_dir: empty dir, single run, mixed valid/invalid entries
- diff_attempts: no diff, training-critical drift, late-stage drift, missing attempt
- config-validate CLI: pass / fail cases via main.app
- runs-list CLI: empty dir, with runs
- inspect-run CLI: smoke test
- logs CLI: graceful missing log
- run-diff CLI: smoke test
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from typer.testing import CliRunner

from src.cli.run_rendering import RunInspectionRenderer, format_duration
from src.main import app
from src.pipeline.state import RunContext
from src.pipeline.run_queries import (
    ROOT_GROUP,
    RunInspector,
    RunSummaryRow,
    build_run_summary_row,
    diff_attempts,
    effective_pipeline_status,
    scan_runs_dir,
    scan_runs_dir_grouped,
    tail_lines,
)
from src.pipeline.state import (
    PipelineState,
    PipelineStateStore,
    StageRunState,
    build_attempt_state,
)

# =============================================================================
# Helpers
# =============================================================================


def _make_state(
    tmp_path: Path,
    *,
    run_id: str = "run_test_001",
    status: str = StageRunState.STATUS_COMPLETED,
    n_attempts: int = 1,
    add_stage_runs: bool = True,
) -> tuple[PipelineStateStore, PipelineState]:
    run_dir = tmp_path / run_id
    store = PipelineStateStore(run_dir)
    state = store.init_state(
        logical_run_id=run_id,
        config_path="/configs/pipeline.yaml",
        training_critical_config_hash="abc123",
        late_stage_config_hash="def456",
    )

    for n in range(1, n_attempts + 1):
        attempt = build_attempt_state(
            state=state,
            run_ctx=RunContext(name=f"attempt_{n}", created_at_utc=datetime.now(timezone.utc)),
            requested_action="fresh" if n == 1 else "restart",
            effective_action="fresh" if n == 1 else "restart",
            restart_from_stage=None,
            enabled_stage_names=["Dataset Validator", "GPU Deployer"],
            training_critical_config_hash="abc123",
            late_stage_config_hash="def456",
        )
        if add_stage_runs:
            attempt.stage_runs["Dataset Validator"] = StageRunState(
                stage_name="Dataset Validator",
                status=StageRunState.STATUS_COMPLETED,
                execution_mode=StageRunState.MODE_EXECUTED,
                outputs={"sample_count": 100},
                started_at="2026-03-19T03:06:00+00:00",
                completed_at="2026-03-19T03:06:12+00:00",
            )
            attempt.stage_runs["GPU Deployer"] = StageRunState(
                stage_name="GPU Deployer",
                status=StageRunState.STATUS_FAILED if n == n_attempts and status == StageRunState.STATUS_FAILED else StageRunState.STATUS_COMPLETED,
                execution_mode=StageRunState.MODE_EXECUTED,
                error="SSH timeout" if n == n_attempts and status == StageRunState.STATUS_FAILED else None,
                started_at="2026-03-19T03:06:12+00:00",
                completed_at="2026-03-19T03:10:12+00:00",
            )
        attempt.status = (
            StageRunState.STATUS_FAILED
            if n == n_attempts and status == StageRunState.STATUS_FAILED
            else StageRunState.STATUS_COMPLETED
        )
        attempt.started_at = "2026-03-19T03:06:00+00:00"
        attempt.completed_at = "2026-03-19T05:20:03+00:00"
        state.attempts.append(attempt)

    state.pipeline_status = status
    state.active_attempt_id = state.attempts[-1].attempt_id if status == StageRunState.STATUS_RUNNING else None
    store.save(state)
    return store, state


# =============================================================================
# _fmt_duration
# =============================================================================


def test_fmt_duration_hours() -> None:
    result = format_duration("2026-03-19T03:06:00+00:00", "2026-03-19T05:20:03+00:00")
    assert "h" in result
    assert "m" in result


def test_fmt_duration_minutes_only() -> None:
    result = format_duration("2026-03-19T03:06:00+00:00", "2026-03-19T03:12:30+00:00")
    assert "m" in result
    assert "h" not in result


def test_fmt_duration_seconds_only() -> None:
    result = format_duration("2026-03-19T03:06:00+00:00", "2026-03-19T03:06:45+00:00")
    assert "45s" in result


def test_fmt_duration_missing_start_returns_empty() -> None:
    assert format_duration(None, "2026-03-19T03:06:00+00:00") == ""


def test_fmt_duration_missing_end_uses_now() -> None:
    result = format_duration("2026-03-19T03:06:00+00:00", None)
    assert isinstance(result, str)


def test_fmt_duration_invalid_timestamps_returns_empty() -> None:
    assert format_duration("not-a-date", "also-not") == ""


def test_tail_lines_zero_limit_returns_empty(tmp_path: Path) -> None:
    log_path = tmp_path / "pipeline.log"
    log_path.write_text("line 1\nline 2\n", encoding="utf-8")

    assert tail_lines(log_path, limit=0) == []


# =============================================================================
# RunInspector
# =============================================================================


def test_run_inspector_loads_state(tmp_path: Path) -> None:
    store, state = _make_state(tmp_path, run_id="run_001")
    inspector = RunInspector(store.run_directory)
    data = inspector.load()

    assert data.state.logical_run_id == "run_001"
    assert len(data.log_tails) == 0  # not requested


def test_run_inspector_includes_empty_log_tails_when_requested(tmp_path: Path) -> None:
    store, state = _make_state(tmp_path, run_id="run_002")
    inspector = RunInspector(store.run_directory)
    data = inspector.load(include_logs=True)

    # Log files don't exist → empty lists
    assert 1 in data.log_tails
    assert data.log_tails[1] == []


def test_run_inspector_reads_log_tail_when_file_exists(tmp_path: Path) -> None:
    store, state = _make_state(tmp_path, run_id="run_003")
    attempt_dir = store.next_attempt_dir(1)
    attempt_dir.mkdir(parents=True, exist_ok=True)
    log_file = attempt_dir / "pipeline.log"
    lines = [f"line {i}" for i in range(50)]
    log_file.write_text("\n".join(lines), encoding="utf-8")

    inspector = RunInspector(store.run_directory)
    data = inspector.load(include_logs=True)

    assert len(data.log_tails[1]) == 30  # _LOG_TAIL_LINES


def test_run_inspector_raises_on_missing_state(tmp_path: Path) -> None:
    from src.pipeline.state import PipelineStateLoadError

    missing_dir = tmp_path / "ghost_run"
    missing_dir.mkdir()
    inspector = RunInspector(missing_dir)
    with pytest.raises(PipelineStateLoadError):
        inspector.load()


# =============================================================================
# RunInspectionRenderer — smoke tests (renders without exception)
# =============================================================================


def test_renderer_base(tmp_path: Path, capsys) -> None:
    store, state = _make_state(tmp_path, run_id="run_r1")
    inspector = RunInspector(store.run_directory)
    data = inspector.load()
    renderer = RunInspectionRenderer()
    renderer.render(data)  # must not raise


def test_renderer_verbose(tmp_path: Path, capsys) -> None:
    store, state = _make_state(tmp_path, run_id="run_r2")
    inspector = RunInspector(store.run_directory)
    data = inspector.load()
    renderer = RunInspectionRenderer()
    renderer.render(data, verbose=True)


def test_renderer_with_logs(tmp_path: Path) -> None:
    store, state = _make_state(tmp_path, run_id="run_r3")
    inspector = RunInspector(store.run_directory)
    data = inspector.load(include_logs=True)
    renderer = RunInspectionRenderer()
    renderer.render(data, include_logs=True)


def test_renderer_verbose_and_logs(tmp_path: Path) -> None:
    store, state = _make_state(tmp_path, run_id="run_r4")
    inspector = RunInspector(store.run_directory)
    data = inspector.load(include_logs=True)
    renderer = RunInspectionRenderer()
    renderer.render(data, verbose=True, include_logs=True)


def test_renderer_failed_run_with_error(tmp_path: Path) -> None:
    store, state = _make_state(
        tmp_path, run_id="run_r5", status=StageRunState.STATUS_FAILED
    )
    inspector = RunInspector(store.run_directory)
    data = inspector.load()
    renderer = RunInspectionRenderer()
    renderer.render(data)


def test_renderer_multi_attempt(tmp_path: Path) -> None:
    store, state = _make_state(tmp_path, run_id="run_r6", n_attempts=3)
    inspector = RunInspector(store.run_directory)
    data = inspector.load()
    renderer = RunInspectionRenderer()
    renderer.render(data)


def test_renderer_empty_attempts(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_empty"
    store = PipelineStateStore(run_dir)
    state = store.init_state(
        logical_run_id="run_empty",
        config_path="/configs/pipeline.yaml",
        training_critical_config_hash="h1",
        late_stage_config_hash="h2",
    )

    from src.pipeline.run_queries import RunInspectionData

    data = RunInspectionData(run_dir=run_dir, state=state, log_tails={})
    renderer = RunInspectionRenderer()
    renderer.render(data)  # no attempts → must not crash


def test_effective_pipeline_status_prefers_latest_attempt_status(tmp_path: Path) -> None:
    _, state = _make_state(tmp_path, run_id="run_effective_status")
    state.pipeline_status = StageRunState.STATUS_RUNNING
    state.attempts[-1].status = StageRunState.STATUS_FAILED

    assert effective_pipeline_status(state) == StageRunState.STATUS_FAILED


def test_effective_pipeline_status_falls_back_to_root_when_latest_status_empty(tmp_path: Path) -> None:
    _, state = _make_state(tmp_path, run_id="run_effective_status_fallback")
    state.pipeline_status = StageRunState.STATUS_RUNNING
    state.attempts[-1].status = ""

    assert effective_pipeline_status(state) == StageRunState.STATUS_RUNNING


def test_run_summary_row_alias_contract_and_unknown_key() -> None:
    row = RunSummaryRow(
        run_id="run_alias",
        run_dir=Path("/tmp/run_alias"),
        created_at="2026-03-30 00:00",
        created_ts=123.0,
        status="completed",
        attempts=2,
        config_name="pipeline.yaml",
        mlflow_run_id="mlflow-123",
        started_at="2026-03-30T00:00:00+00:00",
        completed_at="2026-03-30T00:01:00+00:00",
        error=None,
    )

    assert "config" in row
    assert "config_name" in row
    assert row["config"] == row["config_name"] == "pipeline.yaml"
    assert row["run_id"] == "run_alias"

    with pytest.raises(KeyError):
        _ = row["missing"]


def test_build_run_summary_row_preserves_group_and_mlflow_id(tmp_path: Path) -> None:
    store, state = _make_state(tmp_path, run_id="run_grouped")
    state.root_mlflow_run_id = "mlflow-root-123"
    store.save(state)

    row = build_run_summary_row(store.run_directory, group="nested/group")

    assert row.group == "nested/group"
    assert row.mlflow_run_id == "mlflow-root-123"
    assert row["group"] == "nested/group"


# =============================================================================
# scan_runs_dir
# =============================================================================


def test_scan_runs_dir_empty(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    rows = scan_runs_dir(runs_dir)
    assert rows == []


def test_scan_runs_dir_nonexistent(tmp_path: Path) -> None:
    rows = scan_runs_dir(tmp_path / "does_not_exist")
    assert rows == []


def test_scan_runs_dir_returns_run(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    _make_state(runs_dir, run_id="run_001")

    rows = scan_runs_dir(runs_dir)
    assert len(rows) == 1
    assert rows[0]["run_id"] == "run_001"
    assert rows[0]["status"] == StageRunState.STATUS_COMPLETED
    assert rows[0]["attempts"] == 1
    assert rows[0]["config"] == "pipeline.yaml"


def test_scan_runs_dir_multiple_runs_sorted_newest_first(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    _make_state(runs_dir, run_id="run_a")
    _make_state(runs_dir, run_id="run_z")

    rows = scan_runs_dir(runs_dir)
    assert len(rows) == 2
    # Sorted by name descending → run_z first
    assert rows[0]["run_id"] == "run_z"
    assert rows[1]["run_id"] == "run_a"


def test_scan_runs_dir_skips_dirs_without_state(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    _make_state(runs_dir, run_id="run_good")
    (runs_dir / "run_bad").mkdir()  # no pipeline_state.json

    rows = scan_runs_dir(runs_dir)
    assert len(rows) == 1
    assert rows[0]["run_id"] == "run_good"


def test_scan_runs_dir_handles_corrupt_state(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    bad_run = runs_dir / "run_corrupt"
    bad_run.mkdir()
    (bad_run / "pipeline_state.json").write_text("{bad json", encoding="utf-8")

    rows = scan_runs_dir(runs_dir)
    assert len(rows) == 1
    assert rows[0]["status"] == "unknown"
    assert rows[0]["error"] is not None


# =============================================================================
# scan_runs_dir_grouped
# =============================================================================


def test_scan_runs_dir_grouped_empty(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    groups = scan_runs_dir_grouped(runs_dir)
    assert groups == {}


def test_scan_runs_dir_grouped_root_only(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    _make_state(runs_dir, run_id="run_001")

    groups = scan_runs_dir_grouped(runs_dir)
    assert ROOT_GROUP in groups
    assert len(groups[ROOT_GROUP]) == 1
    assert groups[ROOT_GROUP][0]["run_id"] == "run_001"
    assert groups[ROOT_GROUP][0]["group"] == ROOT_GROUP


def test_scan_runs_dir_grouped_subfolder(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    smoke = runs_dir / "smoke_abc12"
    smoke.mkdir()
    _make_state(smoke, run_id="run_in_smoke")

    groups = scan_runs_dir_grouped(runs_dir)
    assert ROOT_GROUP not in groups
    assert "smoke_abc12" in groups
    assert len(groups["smoke_abc12"]) == 1
    assert groups["smoke_abc12"][0]["run_id"] == "run_in_smoke"
    assert groups["smoke_abc12"][0]["group"] == "smoke_abc12"


def test_scan_runs_dir_grouped_mixed(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    _make_state(runs_dir, run_id="run_root")
    smoke = runs_dir / "smoke_xyz"
    smoke.mkdir()
    _make_state(smoke, run_id="run_nested")

    groups = scan_runs_dir_grouped(runs_dir)
    assert ROOT_GROUP in groups
    assert "smoke_xyz" in groups
    assert len(groups[ROOT_GROUP]) == 1
    assert len(groups["smoke_xyz"]) == 1
    assert groups[ROOT_GROUP][0]["run_id"] == "run_root"
    assert groups["smoke_xyz"][0]["run_id"] == "run_nested"


def test_scan_runs_dir_grouped_includes_created_ts(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    _make_state(runs_dir, run_id="run_ts")

    groups = scan_runs_dir_grouped(runs_dir)
    row = groups[ROOT_GROUP][0]
    assert "created_ts" in row
    assert isinstance(row["created_ts"], float)
    assert row["created_ts"] > 0


# =============================================================================
# diff_attempts
# =============================================================================


def _make_state_with_diff(tmp_path: Path) -> PipelineState:
    run_dir = tmp_path / "run_diff"
    store = PipelineStateStore(run_dir)
    state = store.init_state(
        logical_run_id="run_diff",
        config_path="/configs/pipeline.yaml",
        training_critical_config_hash="h1",
        late_stage_config_hash="l1",
    )
    for n, crit_hash, late_hash in [
        (1, "h1", "l1"),
        (2, "h1", "l2"),
        (3, "h2", "l3"),
    ]:
        attempt = build_attempt_state(
            state=state,
            run_ctx=RunContext(name=f"attempt_{n}", created_at_utc=datetime.now(timezone.utc)),
            requested_action="fresh",
            effective_action="fresh",
            restart_from_stage=None,
            enabled_stage_names=[],
            training_critical_config_hash=crit_hash,
            late_stage_config_hash=late_hash,
        )
        attempt.started_at = "2026-03-19T03:06:00+00:00"
        state.attempts.append(attempt)
    store.save(state)
    return state


def test_diff_attempts_no_changes(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_nd"
    store = PipelineStateStore(run_dir)
    state = store.init_state(
        logical_run_id="run_nd",
        config_path="/configs/pipeline.yaml",
        training_critical_config_hash="same",
        late_stage_config_hash="same",
    )
    for n in range(1, 3):
        attempt = build_attempt_state(
            state=state,
            run_ctx=RunContext(name=f"a{n}", created_at_utc=datetime.now(timezone.utc)),
            requested_action="fresh",
            effective_action="fresh",
            restart_from_stage=None,
            enabled_stage_names=[],
            training_critical_config_hash="same",
            late_stage_config_hash="same",
        )
        attempt.started_at = "2026-03-19T03:06:00+00:00"
        state.attempts.append(attempt)
    store.save(state)

    diff = diff_attempts(state, 1, 2)
    assert diff["training_critical_changed"] is False
    assert diff["late_stage_changed"] is False


def test_diff_attempts_late_stage_drift(tmp_path: Path) -> None:
    state = _make_state_with_diff(tmp_path)
    diff = diff_attempts(state, 1, 2)
    assert diff["training_critical_changed"] is False
    assert diff["late_stage_changed"] is True
    assert diff["hash_a_late"] != diff["hash_b_late"]


def test_diff_attempts_critical_and_late_drift(tmp_path: Path) -> None:
    state = _make_state_with_diff(tmp_path)
    diff = diff_attempts(state, 1, 3)
    assert diff["training_critical_changed"] is True
    assert diff["late_stage_changed"] is True


def test_diff_attempts_missing_attempt_a(tmp_path: Path) -> None:
    state = _make_state_with_diff(tmp_path)
    diff = diff_attempts(state, 99, 1)
    assert diff["found_a"] is False
    assert diff["training_critical_changed"] is False


def test_diff_attempts_missing_both(tmp_path: Path) -> None:
    state = _make_state_with_diff(tmp_path)
    diff = diff_attempts(state, 99, 100)
    assert diff["found_a"] is False
    assert diff["found_b"] is False


# =============================================================================
# CLI commands via CliRunner
# =============================================================================


@pytest.fixture
def cli_runner() -> CliRunner:
    return CliRunner()


# ── inspect-run ───────────────────────────────────────────────────────────────


def test_cli_inspect_run_success(tmp_path: Path, cli_runner: CliRunner) -> None:
    _make_state(tmp_path, run_id="run_inspect_01")
    result = cli_runner.invoke(app, ["inspect-run", str(tmp_path / "run_inspect_01")])
    assert result.exit_code == 0
    assert "run_inspect_01" in result.output


def test_cli_inspect_run_verbose(tmp_path: Path, cli_runner: CliRunner) -> None:
    # ``-v`` was replaced by explicit --outputs/--logs flags. Use both to
    # cover the verbose code path that this test originally exercised.
    _make_state(tmp_path, run_id="run_inspect_v")
    result = cli_runner.invoke(
        app,
        ["inspect-run", str(tmp_path / "run_inspect_v"), "--outputs", "--logs"],
    )
    assert result.exit_code == 0


def test_cli_inspect_run_with_logs(tmp_path: Path, cli_runner: CliRunner) -> None:
    _make_state(tmp_path, run_id="run_inspect_l")
    result = cli_runner.invoke(app, ["inspect-run", str(tmp_path / "run_inspect_l"), "--logs"])
    assert result.exit_code == 0


def test_cli_inspect_run_nonexistent_dir(tmp_path: Path, cli_runner: CliRunner) -> None:
    result = cli_runner.invoke(app, ["inspect-run", str(tmp_path / "ghost")])
    assert result.exit_code != 0


# ── runs-list ─────────────────────────────────────────────────────────────────


def test_cli_runs_list_empty(tmp_path: Path, cli_runner: CliRunner) -> None:
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    result = cli_runner.invoke(app, ["runs-list", str(runs_dir)])
    assert result.exit_code == 0
    assert "No runs found" in result.output


def test_cli_runs_list_with_runs(tmp_path: Path, cli_runner: CliRunner) -> None:
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    _make_state(runs_dir, run_id="run_abc")
    result = cli_runner.invoke(app, ["runs-list", str(runs_dir)])
    assert result.exit_code == 0
    assert "run_abc" in result.output


# ── logs ──────────────────────────────────────────────────────────────────────


def test_cli_logs_missing_log_file(tmp_path: Path, cli_runner: CliRunner) -> None:
    _make_state(tmp_path, run_id="run_log_01")
    result = cli_runner.invoke(app, ["logs", str(tmp_path / "run_log_01")])
    # Should exit non-zero with a message about missing log file
    assert result.exit_code != 0
    assert "not found" in result.output.lower() or "Log file" in result.output


def test_cli_logs_with_log_file(tmp_path: Path, cli_runner: CliRunner) -> None:
    store, state = _make_state(tmp_path, run_id="run_log_02")
    attempt_dir = store.next_attempt_dir(1)
    attempt_dir.mkdir(parents=True, exist_ok=True)
    (attempt_dir / "pipeline.log").write_text("line 1\nline 2\nline 3\n", encoding="utf-8")

    result = cli_runner.invoke(app, ["logs", str(tmp_path / "run_log_02")])
    assert result.exit_code == 0
    assert "line 1" in result.output


def test_cli_logs_specific_attempt(tmp_path: Path, cli_runner: CliRunner) -> None:
    store, state = _make_state(tmp_path, run_id="run_log_03", n_attempts=2)
    attempt_dir = store.next_attempt_dir(2)
    attempt_dir.mkdir(parents=True, exist_ok=True)
    (attempt_dir / "pipeline.log").write_text("attempt 2 log\n", encoding="utf-8")

    result = cli_runner.invoke(app, ["logs", str(tmp_path / "run_log_03"), "--attempt", "2"])
    assert result.exit_code == 0
    assert "attempt 2 log" in result.output


def test_cli_logs_no_attempts_exits_gracefully(tmp_path: Path, cli_runner: CliRunner) -> None:
    run_dir = tmp_path / "run_empty_log"
    store = PipelineStateStore(run_dir)
    store.init_state(
        logical_run_id="run_empty_log",
        config_path="/configs/pipeline.yaml",
        training_critical_config_hash="h",
        late_stage_config_hash="l",
    )
    result = cli_runner.invoke(app, ["logs", str(run_dir)])
    assert result.exit_code == 0
    assert "No attempts" in result.output


# ── run-diff ──────────────────────────────────────────────────────────────────


def test_cli_run_diff_no_changes(tmp_path: Path, cli_runner: CliRunner) -> None:
    _make_state(tmp_path, run_id="run_diff_nc", n_attempts=2)
    result = cli_runner.invoke(app, ["run-diff", str(tmp_path / "run_diff_nc")])
    assert result.exit_code == 0
    assert "No config changes" in result.output


def test_cli_run_diff_with_drift(tmp_path: Path, cli_runner: CliRunner) -> None:
    state = _make_state_with_diff(tmp_path)
    run_dir = tmp_path / "run_diff"
    result = cli_runner.invoke(app, ["run-diff", str(run_dir)])
    assert result.exit_code == 0
    # Should show some kind of change info (the renderer now groups hashes
    # under "training + model + datasets" / "inference + evaluation").
    assert (
        "training" in result.output
        or "inference" in result.output
        or "late_stage" in result.output
        or "critical" in result.output
    )


def test_cli_run_diff_only_one_attempt(tmp_path: Path, cli_runner: CliRunner) -> None:
    _make_state(tmp_path, run_id="run_diff_1a", n_attempts=1)
    result = cli_runner.invoke(app, ["run-diff", str(tmp_path / "run_diff_1a")])
    assert result.exit_code == 0
    assert "Only one attempt" in result.output


# ── config-validate ───────────────────────────────────────────────────────────


def test_cli_config_validate_missing_file(tmp_path: Path, cli_runner: CliRunner) -> None:
    result = cli_runner.invoke(app, ["config-validate", "--config", str(tmp_path / "nonexistent.yaml")])
    assert result.exit_code != 0


def test_cli_config_validate_invalid_yaml(tmp_path: Path, cli_runner: CliRunner) -> None:
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("{invalid: yaml: :", encoding="utf-8")
    result = cli_runner.invoke(app, ["config-validate", "--config", str(bad_yaml)])
    assert result.exit_code != 0
    assert "❌" in result.output or "schema" in result.output.lower()


def test_cli_config_validate_valid_config(tmp_path: Path, cli_runner: CliRunner) -> None:
    from unittest.mock import MagicMock, patch

    mock_cfg = MagicMock()
    mock_cfg.datasets = {}
    mock_cfg.get_active_provider_name.return_value = "single_node"
    mock_cfg.inference = MagicMock(enabled=False)
    mock_cfg.evaluation = MagicMock(enabled=False)

    config_path = tmp_path / "config.yaml"
    config_path.write_text("model:\n  name: test\n", encoding="utf-8")

    # HF_TOKEN is now part of the validator's readiness check; provide it so
    # the "valid-config" branch doesn't trip on a missing token in CI.
    with (
        patch("src.utils.config.load_config", return_value=mock_cfg),
        patch.dict("os.environ", {"HF_TOKEN": "hf_test_token"}, clear=False),
    ):
        result = cli_runner.invoke(app, ["config-validate", "--config", str(config_path)])

    assert result.exit_code == 0
    assert "ready to run" in result.output


def test_cli_config_validate_missing_hf_token(tmp_path: Path, cli_runner: CliRunner) -> None:
    import os
    from unittest.mock import MagicMock, patch

    mock_cfg = MagicMock()
    mock_cfg.datasets = {}
    mock_cfg.get_active_provider_name.return_value = "single_node"
    mock_cfg.inference = MagicMock(enabled=False)
    mock_cfg.evaluation = MagicMock(enabled=False)

    config_path = tmp_path / "config.yaml"
    config_path.write_text("model:\n  name: test\n", encoding="utf-8")

    env_without_hf = {k: v for k, v in os.environ.items() if k != "HF_TOKEN"}
    with patch("src.utils.config.load_config", return_value=mock_cfg), \
         patch.dict("os.environ", env_without_hf, clear=True):
        result = cli_runner.invoke(app, ["config-validate", "--config", str(config_path)])

    assert result.exit_code != 0
    assert "HF_TOKEN" in result.output


# ── run-status ─────────────────────────────────────────────────────────────────


def test_cli_run_status_exits_on_keyboard_interrupt(tmp_path: Path, cli_runner: CliRunner) -> None:
    """run-status loop stops on KeyboardInterrupt — test via patching time.sleep."""
    from unittest.mock import patch

    _make_state(tmp_path, run_id="run_status_test")
    run_dir = str(tmp_path / "run_status_test")

    with patch("time.sleep", side_effect=KeyboardInterrupt):
        result = cli_runner.invoke(app, ["run-status", run_dir])

    # Should exit cleanly (not crash)
    assert result.exit_code == 0
    assert "Stopped" in result.output
