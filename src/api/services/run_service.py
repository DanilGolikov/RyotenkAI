from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.api.schemas.attempt import AttemptDetail, StageRun, StagesResponse
from src.api.schemas.run import LineageRefSchema, RunDetail, RunsListResponse, RunSummary
from src.pipeline.launch import is_process_alive, read_lock_pid
from src.pipeline.presentation import (
    STATUS_COLORS,
    STATUS_ICONS,
    format_mode_label,
)
from src.pipeline.run_queries import (
    RunSummaryRow,
    effective_pipeline_status,
    scan_runs_dir_grouped,
)
from src.pipeline.state import PipelineState
from src.pipeline.state.cache import StateSnapshot, load_state_snapshot
from src.pipeline.state.queries import (
    find_running_attempt_no,
    get_attempt_by_no,
    predict_next_attempt_no,
)

_RUN_ID_RE = re.compile(r"^[A-Za-z0-9_.\-/]+$")


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed


def _duration_seconds(started_at: str | None, completed_at: str | None) -> float | None:
    start = _parse_iso(started_at)
    if start is None:
        return None
    end = _parse_iso(completed_at) or datetime.now(UTC)
    delta = (end - start).total_seconds()
    return delta if delta >= 0 else None


def _summary_row_to_schema(row: RunSummaryRow, runs_dir: Path) -> RunSummary:
    duration = _duration_seconds(row.started_at, row.completed_at)
    try:
        run_id = str(row.run_dir.relative_to(runs_dir))
    except ValueError:
        run_id = row.run_id
    return RunSummary(
        run_id=run_id,
        run_dir=str(row.run_dir),
        created_at=row.created_at,
        created_ts=row.created_ts,
        status=row.status,
        status_icon=STATUS_ICONS.get(row.status),
        status_color=STATUS_COLORS.get(row.status),
        attempts=row.attempts,
        config_name=row.config_name,
        mlflow_run_id=row.mlflow_run_id,
        started_at=row.started_at,
        completed_at=row.completed_at,
        duration_seconds=duration,
        error=row.error,
        group=row.group,
    )


def list_runs(runs_dir: Path) -> RunsListResponse:
    grouped = scan_runs_dir_grouped(runs_dir)
    groups: dict[str, list[RunSummary]] = {
        group: [_summary_row_to_schema(row, runs_dir) for row in rows] for group, rows in grouped.items()
    }
    return RunsListResponse(runs_dir=str(runs_dir), groups=groups)


def _stage_run_to_schema(stage_run: Any, started_fallback: str | None = None) -> StageRun:
    duration = _duration_seconds(stage_run.started_at or started_fallback, stage_run.completed_at)
    return StageRun(
        stage_name=stage_run.stage_name,
        status=stage_run.status,
        status_icon=STATUS_ICONS.get(stage_run.status),
        status_color=STATUS_COLORS.get(stage_run.status),
        execution_mode=stage_run.execution_mode,
        mode_label=format_mode_label(stage_run),
        outputs=dict(stage_run.outputs or {}),
        error=stage_run.error,
        failure_kind=stage_run.failure_kind,
        reuse_from=stage_run.reuse_from,
        skip_reason=stage_run.skip_reason,
        started_at=stage_run.started_at,
        completed_at=stage_run.completed_at,
        duration_seconds=duration,
    )


def _run_detail(state: PipelineState, run_dir: Path, runs_dir: Path) -> RunDetail:
    effective_status = effective_pipeline_status(state)
    config_path = state.config_path or ""
    config_abspath: str | None = None
    if config_path:
        try:
            config_abspath = str(Path(config_path).expanduser().resolve())
        except OSError:
            config_abspath = config_path

    lock_pid = read_lock_pid(run_dir)
    is_locked = lock_pid is not None and is_process_alive(lock_pid)

    return RunDetail(
        schema_version=state.schema_version,
        logical_run_id=state.logical_run_id,
        run_directory=state.run_directory,
        config_path=config_path,
        config_abspath=config_abspath,
        active_attempt_id=state.active_attempt_id,
        pipeline_status=state.pipeline_status,
        training_critical_config_hash=state.training_critical_config_hash,
        late_stage_config_hash=state.late_stage_config_hash,
        model_dataset_config_hash=state.model_dataset_config_hash,
        root_mlflow_run_id=state.root_mlflow_run_id,
        mlflow_runtime_tracking_uri=state.mlflow_runtime_tracking_uri,
        mlflow_ca_bundle_path=state.mlflow_ca_bundle_path,
        attempts=[attempt.to_dict() for attempt in state.attempts],
        current_output_lineage={
            name: LineageRefSchema.model_validate(ref.to_dict()) for name, ref in state.current_output_lineage.items()
        },
        status=effective_status,
        status_icon=STATUS_ICONS.get(effective_status),
        status_color=STATUS_COLORS.get(effective_status),
        running_attempt_no=find_running_attempt_no(state),
        next_attempt_no=predict_next_attempt_no(run_dir),
        is_locked=is_locked,
        lock_pid=lock_pid,
    )


def get_run_detail(run_dir: Path, runs_dir: Path) -> tuple[RunDetail, StateSnapshot]:
    """Read the run's state and return the view paired with its ``mtime_ns``.

    ``mtime_ns`` is needed by the router to emit ETag / Last-Modified headers
    and short-circuit polling with ``304 Not Modified``.
    """
    snapshot = load_state_snapshot(run_dir)
    return _run_detail(snapshot.state, run_dir, runs_dir), snapshot


def get_attempt_detail(run_dir: Path, attempt_no: int) -> tuple[AttemptDetail, StateSnapshot]:
    snapshot = load_state_snapshot(run_dir)
    return _attempt_detail(snapshot.state, attempt_no, run_dir), snapshot


def _attempt_detail(state: PipelineState, attempt_no: int, run_dir: Path) -> AttemptDetail:
    attempt = get_attempt_by_no(state, attempt_no)
    if attempt is None:
        raise FileNotFoundError(f"attempt {attempt_no} not found in {run_dir}")
    duration = _duration_seconds(attempt.started_at, attempt.completed_at)
    stage_runs = {
        name: _stage_run_to_schema(run, started_fallback=attempt.started_at) for name, run in attempt.stage_runs.items()
    }
    return AttemptDetail(
        attempt_id=attempt.attempt_id,
        attempt_no=attempt.attempt_no,
        runtime_name=attempt.runtime_name,
        requested_action=attempt.requested_action,
        effective_action=attempt.effective_action,
        restart_from_stage=attempt.restart_from_stage,
        status=attempt.status,
        status_icon=STATUS_ICONS.get(attempt.status),
        status_color=STATUS_COLORS.get(attempt.status),
        started_at=attempt.started_at,
        completed_at=attempt.completed_at,
        error=attempt.error,
        training_critical_config_hash=attempt.training_critical_config_hash,
        late_stage_config_hash=attempt.late_stage_config_hash,
        model_dataset_config_hash=attempt.model_dataset_config_hash,
        root_mlflow_run_id=attempt.root_mlflow_run_id,
        pipeline_attempt_mlflow_run_id=attempt.pipeline_attempt_mlflow_run_id,
        training_run_id=attempt.training_run_id,
        enabled_stage_names=list(attempt.enabled_stage_names),
        stage_runs=stage_runs,
        duration_seconds=duration,
    )


def get_attempt_stages(run_dir: Path, attempt_no: int) -> tuple[StagesResponse, StateSnapshot]:
    detail, snapshot = get_attempt_detail(run_dir, attempt_no)
    return _build_stages_response(detail), snapshot


def _build_stages_response(detail: AttemptDetail) -> StagesResponse:
    ordered_names = detail.enabled_stage_names or list(detail.stage_runs.keys())
    stages: list[StageRun] = []
    for name in ordered_names:
        stage = detail.stage_runs.get(name)
        if stage is None:
            # Not yet started — emit a minimal pending row so UI can render it.
            stages.append(
                StageRun(
                    stage_name=name,
                    status="pending",
                    status_icon=STATUS_ICONS.get("pending"),
                    status_color=STATUS_COLORS.get("pending"),
                )
            )
        else:
            stages.append(stage)
    # Preserve any extra stage_runs that are not in enabled_stage_names (shouldn't
    # usually happen but keeps UI resilient).
    for name, stage in detail.stage_runs.items():
        if name not in ordered_names:
            stages.append(stage)
    return StagesResponse(stages=stages)


def validate_run_id(run_id: str) -> str:
    if not _RUN_ID_RE.match(run_id):
        raise ValueError("run_id must match ^[A-Za-z0-9_.\\-/]+$")
    if ".." in run_id.split("/"):
        raise ValueError("run_id must not contain '..' segments")
    return run_id


def build_suggested_run_id() -> str:
    import secrets
    from datetime import datetime

    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    suffix = secrets.token_hex(4)
    return f"run_{stamp}_{suffix}"


def create_empty_run(runs_dir: Path, run_id: str | None, subgroup: str | None) -> RunSummary:
    effective_run_id = run_id or build_suggested_run_id()
    validate_run_id(effective_run_id)
    if subgroup:
        validate_run_id(subgroup)
        base = runs_dir / subgroup
    else:
        base = runs_dir
    target = (base / effective_run_id).resolve()
    # Ensure we stay inside runs_dir
    try:
        target.relative_to(runs_dir.resolve())
    except ValueError as exc:
        raise ValueError("run path escapes runs_dir") from exc
    if target.exists():
        raise ValueError(f"run already exists: {target}")
    target.mkdir(parents=True, exist_ok=False)

    # Derive created_at from filesystem stat (mirrors scan_runs_dir behaviour).
    stat = target.stat()
    created_ts = float(getattr(stat, "st_birthtime", None) or stat.st_ctime)
    created_at = datetime.fromtimestamp(created_ts).strftime("%Y-%m-%d %H:%M")
    group = subgroup or "(root)"
    try:
        rel = str(target.relative_to(runs_dir.resolve()))
    except ValueError:
        rel = effective_run_id
    return RunSummary(
        run_id=rel,
        run_dir=str(target),
        created_at=created_at,
        created_ts=created_ts,
        status="pending",
        status_icon=STATUS_ICONS.get("pending"),
        status_color=STATUS_COLORS.get("pending"),
        attempts=0,
        config_name="—",
        mlflow_run_id=None,
        started_at=None,
        completed_at=None,
        duration_seconds=None,
        error=None,
        group=group,
    )


__all__ = [
    "build_suggested_run_id",
    "create_empty_run",
    "get_attempt_detail",
    "get_attempt_stages",
    "get_run_detail",
    "list_runs",
    "validate_run_id",
]
