"""``ryotenkai runs <verb>`` — read-only inspection + delete.

Read commands route through :mod:`src.api.services.run_service` so the
CLI and the FastAPI ``GET /runs`` endpoint share one source of truth
(see plan B contract-tests in :mod:`src.tests.contract`).

Live tail (``runs logs --follow``) reuses
:class:`src.api.ws.live_tail.LiveLogTail` — same offset-based reader the
WebSocket stream uses, no self-rolled file polling.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Annotated

import typer

from src.cli.common_options import DryRunOpt, RunDirArg, YesOpt
from src.cli.context import CLIContext
from src.cli.errors import die
from src.cli.renderer import get_renderer

runs_app = typer.Typer(
    no_args_is_help=True,
    help="Inspect, monitor, and delete pipeline runs.",
    rich_markup_mode=None,
    context_settings={"help_option_names": ["-h", "--help"]},
    pretty_exceptions_enable=False,
)


# ---------------------------------------------------------------------------
# ls
# ---------------------------------------------------------------------------


@runs_app.command("ls")
def ls_cmd(
    ctx: typer.Context,
    runs_dir: Annotated[
        Path,
        typer.Argument(
            help="Directory containing run subdirectories (default: ./runs).",
            file_okay=False,
            dir_okay=True,
        ),
    ] = Path("runs"),
) -> None:
    """List all runs with status summaries."""
    from src.cli.formatters import duration_seconds, format_duration
    from src.pipeline.run_queries import scan_runs_dir

    state = ctx.ensure_object(CLIContext)
    renderer = get_renderer(state)
    rows = list(scan_runs_dir(runs_dir))

    if state.is_machine_readable:
        renderer.emit([
            {
                "run_id": row.run_id,
                "status": row.status,
                "attempts": row.attempts,
                "started_at": row.started_at,
                "completed_at": row.completed_at,
                "duration_s": duration_seconds(row.started_at, row.completed_at),
                "config_name": row.config_name,
            }
            for row in rows
        ])
    elif not rows:
        renderer.text(f"No runs found in {runs_dir}")
    else:
        renderer.heading(f"Runs in {runs_dir}/")
        renderer.text("")
        renderer.table(
            headers=["Run ID", "Status", "Att", "Duration", "Config"],
            rows=[
                (row.run_id, row.status, row.attempts,
                 format_duration(row.started_at, row.completed_at) or "-",
                 row.config_name)
                for row in rows
            ],
        )
    renderer.flush()


# ---------------------------------------------------------------------------
# inspect
# ---------------------------------------------------------------------------


@runs_app.command("inspect")
def inspect_cmd(
    ctx: typer.Context,
    run_dir: RunDirArg,
    show_outputs: Annotated[
        bool, typer.Option(
            "--outputs", help="Show stage outputs and lineage (text mode only).",
        ),
    ] = False,
    logs: Annotated[
        bool, typer.Option("--logs", help="Show tail of pipeline.log per attempt."),
    ] = False,
) -> None:
    """Inspect a run — show structured info about attempts and stages."""
    from src.cli.formatters import duration_seconds
    from src.pipeline.run_queries import RunInspector, effective_pipeline_status

    state = ctx.ensure_object(CLIContext)
    renderer = get_renderer(state)

    try:
        data = RunInspector(run_dir).load(include_logs=logs)
    except Exception as exc:
        raise die(str(exc), hint="list available runs with `ryotenkai runs ls`")

    pipeline_state = data.state
    if state.is_machine_readable:
        renderer.emit({
            "run_id": data.run_dir.name,
            "logical_run_id": pipeline_state.logical_run_id,
            "status": effective_pipeline_status(pipeline_state),
            "config_path": pipeline_state.config_path,
            "mlflow_run_id": pipeline_state.root_mlflow_run_id,
            "attempts": [
                {
                    "attempt_no": attempt.attempt_no,
                    "status": attempt.status,
                    "action": attempt.restart_from_stage or attempt.effective_action,
                    "started_at": attempt.started_at,
                    "completed_at": attempt.completed_at,
                    "duration_s": duration_seconds(
                        attempt.started_at, attempt.completed_at,
                    ),
                    "error": attempt.error,
                    "stages": [
                        _stage_payload(attempt, stage_name)
                        for stage_name in (
                            attempt.enabled_stage_names or list(attempt.stage_runs)
                        )
                    ],
                }
                for attempt in pipeline_state.attempts
            ],
        })
    else:
        from src.cli.run_rendering import render_run_inspection_lines

        for line in render_run_inspection_lines(
            data, verbose=show_outputs, include_logs=logs,
        ):
            renderer.text(line)
    renderer.flush()


def _stage_payload(attempt, stage_name: str) -> dict:  # type: ignore[no-untyped-def]
    from src.cli.formatters import duration_seconds

    stage_run = attempt.stage_runs.get(stage_name)
    if stage_run is None:
        return {"name": stage_name, "status": "pending"}
    return {
        "name": stage_run.stage_name,
        "status": stage_run.status,
        "mode": stage_run.execution_mode,
        "started_at": stage_run.started_at,
        "completed_at": stage_run.completed_at,
        "duration_s": duration_seconds(
            stage_run.started_at, stage_run.completed_at,
        ),
        "error": stage_run.error,
        "outputs": stage_run.outputs,
    }


# ---------------------------------------------------------------------------
# logs
# ---------------------------------------------------------------------------


@runs_app.command("logs")
def logs_cmd(
    run_dir: RunDirArg,
    attempt: Annotated[
        int, typer.Option(
            "--attempt", help="Attempt number (default: last attempt).",
        ),
    ] = 0,
    follow: Annotated[
        bool, typer.Option("--follow", "-f", help="Stream log (tail -F mode)."),
    ] = False,
) -> None:
    """Show pipeline log for a run attempt."""
    from src.api.ws.live_tail import LiveLogTail
    from src.pipeline.state import PipelineStateLoadError, PipelineStateStore

    run_path = run_dir.expanduser().resolve()
    store = PipelineStateStore(run_path)

    try:
        state = store.load()
    except (PipelineStateLoadError, Exception) as exc:
        raise die(f"cannot load state: {exc}")

    if not state.attempts:
        typer.echo("No attempts found.")
        return

    target_no = attempt if attempt > 0 else state.attempts[-1].attempt_no
    log_path = store.next_attempt_dir(target_no) / "pipeline.log"
    if not log_path.exists():
        raise die(f"log file not found: {log_path}")

    if not follow:
        typer.echo(log_path.read_text(encoding="utf-8", errors="replace"))
        return

    typer.echo(f"Streaming {log_path}  (Ctrl+C to stop)")
    typer.echo("")
    tail = LiveLogTail(path=log_path)
    # Read everything currently in the file once, then poll for new lines.
    for line in tail.load_full(log_path):
        typer.echo(line)
    try:
        while True:
            for line in tail.read_new_lines():
                typer.echo(line)
            time.sleep(0.2)
    except KeyboardInterrupt:
        typer.echo("")
        typer.echo("Stopped.")


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------


@runs_app.command("status")
def status_cmd(
    ctx: typer.Context,
    run_dir: RunDirArg,
    interval: Annotated[
        float, typer.Option(
            "--interval", "-i",
            help="Polling interval in seconds (text mode live-poll only).",
        ),
    ] = 5.0,
    once: Annotated[
        bool, typer.Option(
            "--once",
            help="Print one snapshot and exit (implied by -o json/yaml).",
        ),
    ] = False,
) -> None:
    """Show pipeline status — one snapshot or live polling until Ctrl+C."""
    from src.cli.formatters import duration_seconds
    from src.cli.run_rendering import render_run_status_snapshot
    from src.pipeline.state import PipelineStateLoadError, PipelineStateStore

    state_ctx = ctx.ensure_object(CLIContext)
    renderer = get_renderer(state_ctx)
    run_path = run_dir.expanduser().resolve()
    store = PipelineStateStore(run_path)

    def _load_state():
        try:
            return store.load()
        except (PipelineStateLoadError, Exception) as exc:
            raise die(f"cannot read state: {exc}")

    def _status_payload(state) -> dict:
        current = state.attempts[-1] if state.attempts else None
        return {
            "run_id": run_path.name,
            "status": state.pipeline_status,
            "attempts_total": len(state.attempts),
            "current_attempt": current.attempt_no if current else None,
            "stages": [
                {
                    "name": stage_name,
                    "status": (
                        current.stage_runs.get(stage_name).status
                        if current and current.stage_runs.get(stage_name) else "pending"
                    ),
                    "duration_s": duration_seconds(
                        current.stage_runs.get(stage_name).started_at if current and current.stage_runs.get(stage_name) else None,
                        current.stage_runs.get(stage_name).completed_at if current and current.stage_runs.get(stage_name) else None,
                    ) if current and current.stage_runs.get(stage_name) else None,
                }
                for stage_name in (current.enabled_stage_names or list(current.stage_runs))
            ] if current else [],
        }

    if state_ctx.is_machine_readable:
        renderer.emit(_status_payload(_load_state()))
        renderer.flush()
        return

    if once:
        for line in render_run_status_snapshot(run_path.name, _load_state()):
            renderer.text(line)
        renderer.flush()
        return

    try:
        while True:
            print("\033[H\033[2J", end="")  # ANSI clear-screen
            for line in render_run_status_snapshot(run_path.name, _load_state()):
                renderer.text(line)
            renderer.text(f"Refreshing every {interval}s — Ctrl+C to stop")
            time.sleep(interval)
    except KeyboardInterrupt:
        renderer.text("")
        renderer.text("Stopped monitoring.")


# ---------------------------------------------------------------------------
# diff
# ---------------------------------------------------------------------------


@runs_app.command("diff")
def diff_cmd(
    ctx: typer.Context,
    run_dir: RunDirArg,
    attempt: Annotated[
        list[int], typer.Option(
            "--attempt",
            help="Attempt numbers to compare (use twice). Default: first vs last.",
        ),
    ] = [],  # noqa: B006 — Typer requires the literal default
) -> None:
    """Compare config hashes between attempts."""
    from src.cli.run_rendering import render_run_diff_lines
    from src.pipeline.run_queries import diff_attempts
    from src.pipeline.state import PipelineStateLoadError, PipelineStateStore

    state_ctx = ctx.ensure_object(CLIContext)
    renderer = get_renderer(state_ctx)
    run_path = run_dir.expanduser().resolve()

    try:
        state = PipelineStateStore(run_path).load()
    except (PipelineStateLoadError, Exception) as exc:
        raise die(f"cannot load state: {exc}")

    if len(state.attempts) < 2:
        if state_ctx.is_machine_readable:
            renderer.emit({"status": "single_attempt", "attempts": len(state.attempts)})
        else:
            renderer.text("Only one attempt — nothing to compare.")
        renderer.flush()
        return

    attempt_nos = list(attempt)
    if not attempt_nos:
        attempt_a = state.attempts[0].attempt_no
        attempt_b = state.attempts[-1].attempt_no
    elif len(attempt_nos) == 1:
        attempt_a = state.attempts[0].attempt_no
        attempt_b = attempt_nos[0]
    else:
        attempt_a, attempt_b = attempt_nos[0], attempt_nos[1]

    diff = diff_attempts(state, attempt_a, attempt_b)
    if not diff["found_a"] or not diff["found_b"]:
        raise die(f"attempt {attempt_a} or {attempt_b} not found in state")

    if state_ctx.is_machine_readable:
        renderer.emit({
            "attempt_a": attempt_a,
            "attempt_b": attempt_b,
            "training_critical_changed": bool(diff["training_critical_changed"]),
            "late_stage_changed": bool(diff["late_stage_changed"]),
            "hash_a_critical": diff.get("hash_a_critical"),
            "hash_b_critical": diff.get("hash_b_critical"),
            "hash_a_late": diff.get("hash_a_late"),
            "hash_b_late": diff.get("hash_b_late"),
        })
    else:
        for line in render_run_diff_lines(diff, attempt_a, attempt_b):
            renderer.text(line)
    renderer.flush()


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


@runs_app.command("report")
def report_cmd(
    run_dir: RunDirArg,
    output: Annotated[
        Path | None, typer.Option(
            "--output", "-o",
            help="Save report to this path (default: <run-dir>/report.md).",
            dir_okay=False,
        ),
    ] = None,
) -> None:
    """Generate a markdown MLflow report for a run."""
    from src.pipeline.state import PipelineStateLoadError, PipelineStateStore
    from src.reports.report_generator import ExperimentReportGenerator

    run_path = run_dir.expanduser().resolve()
    try:
        state = PipelineStateStore(run_path).load()
    except (PipelineStateLoadError, Exception) as exc:
        raise die(f"cannot load state: {exc}")

    if not state.root_mlflow_run_id:
        raise die("root_mlflow_run_id not found in pipeline_state.json")

    typer.echo(f"MLflow run: {state.root_mlflow_run_id}")
    save_path = output or (run_path / "report.md")
    try:
        generator = ExperimentReportGenerator()
        markdown = generator.generate(
            state.root_mlflow_run_id, local_logs_dir=run_path,
        )
        save_path.write_text(markdown, encoding="utf-8")
    except Exception as exc:
        raise die(f"report generation failed: {exc}")
    typer.echo(f"Report saved: {save_path}")


# ---------------------------------------------------------------------------
# rm
# ---------------------------------------------------------------------------


@runs_app.command("rm")
def rm_cmd(
    run_dir: RunDirArg,
    mode: Annotated[
        str, typer.Option(
            "--mode",
            help="Delete scope: local_and_mlflow | local_only | mlflow_only.",
        ),
    ] = "local_and_mlflow",
    yes: YesOpt = False,
    dry_run: DryRunOpt = False,
) -> None:
    """Delete a run's local directory and/or its MLflow record."""
    from src.api.services import delete_service

    if mode not in ("local_and_mlflow", "local_only", "mlflow_only"):
        raise die(
            f"invalid --mode: {mode!r}",
            hint="local_and_mlflow | local_only | mlflow_only",
        )

    if not yes and not dry_run:
        confirm = typer.confirm(
            f"Delete {run_dir.name} ({mode})? This cannot be undone.",
        )
        if not confirm:
            raise die("aborted by user", code=2)

    if dry_run:
        typer.echo(f"dry-run: would delete {run_dir} (mode={mode})")
        return

    result = delete_service.delete_run(run_dir, mode=mode)
    typer.echo(f"deleted: {result}")


__all__ = ["runs_app"]
