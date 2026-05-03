from __future__ import annotations

from collections.abc import Iterable

from src.cli.formatters import format_duration  # re-exported for back-compat
from src.pipeline.run_queries import RunInspectionData, RunSummaryRow, effective_pipeline_status
from src.pipeline.state import PipelineAttemptState, PipelineState, StageRunState

__all__ = (
    "RunInspectionRenderer",
    "format_duration",
    "render_run_diff_lines",
    "render_run_inspection_lines",
    "render_run_status_snapshot",
    "render_runs_list_lines",
)
_STATUS_ICONS: dict[str, str] = {
    StageRunState.STATUS_COMPLETED: "◉",
    StageRunState.STATUS_FAILED: "◉",
    StageRunState.STATUS_RUNNING: "▸",
    StageRunState.STATUS_INTERRUPTED: "◈",
    StageRunState.STATUS_STALE: "◌",
    StageRunState.STATUS_SKIPPED: "◇",
    StageRunState.STATUS_PENDING: "○",
}

_MLFLOW_ID_SHORT = 12
_ERROR_TRUNCATE = 120
_ATTEMPT_ERROR_TRUNCATE = 160
_MAX_OUTPUT_FIELDS = 8
_MAX_OUTPUT_VALUE_LEN = 60
_TIMESTAMP_DISPLAY_LEN = 19


class RunInspectionRenderer:
    """Render run query data to stdout-friendly plain text."""

    def render(
        self,
        data: RunInspectionData,
        *,
        verbose: bool = False,
        include_logs: bool = False,
    ) -> None:
        for line in render_run_inspection_lines(data, verbose=verbose, include_logs=include_logs):
            print(line)


def render_run_inspection_lines(
    data: RunInspectionData,
    *,
    verbose: bool = False,
    include_logs: bool = False,
) -> tuple[str, ...]:
    state = data.state
    lines: list[str] = []
    pipeline_status = effective_pipeline_status(state)
    pipeline_duration = format_duration(
        state.attempts[0].started_at if state.attempts else None,
        state.attempts[-1].completed_at if state.attempts else None,
    )
    config_name = state.config_path.split("/")[-1] if state.config_path else "-"
    mlflow_short = (state.root_mlflow_run_id or "-")[:_MLFLOW_ID_SHORT] + "..." if state.root_mlflow_run_id else "-"

    lines.append(f"Run: {data.run_dir.name}")
    lines.append(f"  Status  : {pipeline_status.upper()}")
    lines.append(f"  Config  : {config_name}")
    if state.attempts:
        lines.append(f"  Started : {state.attempts[0].started_at[:_TIMESTAMP_DISPLAY_LEN].replace('T', ' ')}")
    lines.append(f"  Duration: {pipeline_duration or '-'}")
    lines.append(f"  MLflow  : {mlflow_short}")
    lines.append(f"  Attempts: {len(state.attempts)}")
    lines.append(f"  Run ID  : {state.logical_run_id}")
    lines.append("")

    for attempt in state.attempts:
        lines.extend(_render_attempt_lines(attempt, verbose=verbose))
        lines.append("")

    if include_logs:
        for attempt in state.attempts:
            attempt_lines = data.log_tails.get(attempt.attempt_no, [])
            lines.append(f"--- Attempt {attempt.attempt_no} / pipeline.log ---")
            if not attempt_lines:
                lines.append("  (no log file)")
            else:
                lines.append(f"  (last {len(attempt_lines)} lines)")
                lines.extend(attempt_lines)
            lines.append("")

    return tuple(lines)


def render_runs_list_lines(runs_dir, rows: Iterable[RunSummaryRow]) -> tuple[str, ...]:
    rows = tuple(rows)
    if not rows:
        return (f"No runs found in {runs_dir}",)

    lines = [f"Runs in {runs_dir}/", ""]
    fmt = "{:<32} {:<13} {:>4}  {:<10} {}"
    lines.append(fmt.format("Run ID", "Status", "Att", "Duration", "Config"))
    lines.append("-" * 78)
    for row in rows:
        lines.append(
            fmt.format(
                row.run_id,
                row.status,
                str(row.attempts),
                format_duration(row.started_at, row.completed_at) or "-",
                row.config_name,
            )
        )
    return tuple(lines)


def render_run_diff_lines(diff: dict[str, object], attempt_a: int, attempt_b: int) -> tuple[str, ...]:
    if not diff["found_a"] or not diff["found_b"]:
        return (f"Attempt {attempt_a} or {attempt_b} not found in state.",)

    lines = ["", f"Config diff: attempt {attempt_a} -> attempt {attempt_b}", ""]
    no_change = not diff["training_critical_changed"] and not diff["late_stage_changed"]
    if no_change:
        lines.append("No config changes between attempts.")
        return tuple(lines)

    if diff["training_critical_changed"]:
        lines.append("training + model + datasets  [critical — blocks restart from early stages]")
        lines.append(f"  A: {str(diff['hash_a_critical'])[:10]}...")
        lines.append(f"  B: {str(diff['hash_b_critical'])[:10]}...")
    if diff["late_stage_changed"]:
        lines.append("inference + evaluation  [late_stage — restart allowed]")
        lines.append(f"  A: {str(diff['hash_a_late'])[:10]}...")
        lines.append(f"  B: {str(diff['hash_b_late'])[:10]}...")
    return tuple(lines)


def render_run_status_snapshot(run_name: str, state: PipelineState) -> tuple[str, ...]:
    current_attempt = state.attempts[-1] if state.attempts else None
    header = f"Run: {run_name}  status: {state.pipeline_status}"
    if current_attempt:
        header += f"  attempt {current_attempt.attempt_no}/{len(state.attempts)}"

    lines = [header, "-" * 60]
    if current_attempt:
        lines.extend(_render_status_attempt_lines(current_attempt))
    lines.append("")
    return tuple(lines)


def _render_attempt_lines(attempt: PipelineAttemptState, *, verbose: bool) -> list[str]:
    attempt_duration = format_duration(attempt.started_at, attempt.completed_at)
    action_label = attempt.restart_from_stage or attempt.effective_action
    lines = [f"Attempt {attempt.attempt_no}  {action_label}  {attempt.status}  {attempt_duration}", "-" * 70]

    fmt = "  {:3} {:<28} {:<13} {:<14} {}"
    for stage_name in attempt.enabled_stage_names or list(attempt.stage_runs):
        stage_run = attempt.stage_runs.get(stage_name)
        if stage_run is None:
            lines.append(fmt.format("-", stage_name, "pending", "-", ""))
            continue

        icon = _STATUS_ICONS.get(stage_run.status, "?")
        duration = format_duration(stage_run.started_at, stage_run.completed_at)
        lines.append(
            fmt.format(
                icon,
                stage_run.stage_name,
                stage_run.status,
                _format_mode_label(stage_run),
                duration,
            )
        )
        if stage_run.error:
            lines.append(f"      Error: {stage_run.error[:_ERROR_TRUNCATE]}")
        if verbose and stage_run.outputs:
            for key, value in list(stage_run.outputs.items())[:_MAX_OUTPUT_FIELDS]:
                value_str = str(value)[:_MAX_OUTPUT_VALUE_LEN] if value is not None else "-"
                lines.append(f"      {key} = {value_str}")

    if attempt.error and attempt.status not in (StageRunState.STATUS_COMPLETED,):
        lines.append(f"  Pipeline error: {attempt.error[:_ATTEMPT_ERROR_TRUNCATE]}")
    return lines


def _render_status_attempt_lines(attempt: PipelineAttemptState) -> list[str]:
    lines: list[str] = []
    for stage_name in attempt.enabled_stage_names or list(attempt.stage_runs):
        stage_run = attempt.stage_runs.get(stage_name)
        if stage_run is None:
            lines.append(f"  {'':3} {stage_name:<28} pending")
            continue
        icon = _STATUS_ICONS.get(stage_run.status, "?")
        duration = format_duration(stage_run.started_at, stage_run.completed_at)
        running_marker = " <--" if stage_run.status == StageRunState.STATUS_RUNNING else ""
        lines.append(f"  {icon:3} {stage_run.stage_name:<28} {stage_run.status:<13} {duration}{running_marker}")
    return lines


def _format_mode_label(stage_run: StageRunState) -> str:
    if stage_run.execution_mode == StageRunState.MODE_REUSED and stage_run.reuse_from:
        attempt_id = stage_run.reuse_from.get("attempt_id", "?")
        suffix = str(attempt_id).split(":")[-1] if ":" in str(attempt_id) else str(attempt_id)
        return f"reused ({suffix})"
    return stage_run.execution_mode or "—"
