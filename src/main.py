"""Main entry point for RyotenkAI CLI.

Keep this module's *top-level* imports lean — every `ryotenkai` invocation
pays for them, even `--help`. Heavy dependencies (``src.utils.config``,
``src.pipeline.launch_queries``, the orchestrator, mlflow, torch, …) are
imported lazily from inside command bodies so the help screen renders in
<300 ms instead of ~1.7 s.

The only "noisy" top-level dependency we can't avoid is ``src.utils.logger``
(colorlog configuration). Warnings from transitively-loaded third-party
packages (e.g. torch's pynvml deprecation) are filtered here so users
don't get a wall of text before each command runs.
"""

from __future__ import annotations

import atexit
import os
import signal
import sys
import threading
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

# Hide noisy third-party deprecation warnings that surface at import time.
# Kept narrow so we don't swallow real signal from our own code.
# Filter must run before the downstream imports that trigger pynvml.
warnings.filterwarnings(
    "ignore",
    message=".*pynvml package is deprecated.*",
    category=FutureWarning,
)

import click  # noqa: E402 -- must come after the warnings filter above
import typer  # noqa: E402

from src.cli.run_rendering import (  # noqa: E402
    render_run_diff_lines,
    render_run_status_snapshot,
)
from src.utils.logger import logger  # noqa: E402

if TYPE_CHECKING:
    # PipelineConfig pulls the whole pydantic+torch+datasets cascade on
    # import (~200 ms + noisy warnings). We only use it as a type hint.
    from src.utils.config import PipelineConfig

# Global orchestrator reference for signal handler — typed loosely to avoid
# importing PipelineOrchestrator at module level (lazy import in train()).
_current_orchestrator: object | None = None


def _unregister_mlflow_atexit() -> None:
    """
    Unregister MLflow's atexit hook before sys.exit().

    MLflow registers _safe_end_run via atexit.register() when mlflow.start_run()
    is called. On sys.exit() this hook fires and calls MlflowClient().set_terminated(),
    which performs an HTTP request with default timeouts (120s × 7 retries ≈ 14 min).
    If the MLflow server is unreachable, the process hangs until all retries exhaust.

    We unregister this hook here so that our own cleanup (already done by the
    orchestrator's finally block) is not repeated and the process exits promptly.
    """
    try:
        import mlflow.tracking.fluent as _fluent

        atexit.unregister(_fluent._safe_end_run)
        logger.debug("[SIGNAL] MLflow atexit hook unregistered")
    except Exception:
        pass  # mlflow may not be imported; safe to ignore


def _signal_handler(signum: int, _frame: object) -> None:
    """
    Handle SIGINT/SIGTERM signals gracefully.

    Responsibilities:
      1. Notify orchestrator about the signal (sets _shutdown_signal_name so
         cleanup can adapt, e.g. skip GPU teardown on interrupt).
      2. Unregister the MLflow atexit hook to prevent the process from hanging
         after sys.exit() when the MLflow server is unreachable.
      3. Start a hard-deadline timer: if cleanup is still running after 30 s,
         force-exit with os._exit() to guarantee the process terminates.
      4. Call sys.exit() — this triggers the orchestrator's finally block which
         runs _cleanup_resources() exactly once (guarded by _cleanup_done flag).

    NOTE: cleanup is intentionally NOT called here. The orchestrator's finally
    block is the single owner of cleanup to avoid double-cleanup races.
    """
    signal_name = "SIGINT" if signum == signal.SIGINT else "SIGTERM"
    print(f"\nReceived {signal_name}, shutting down...", file=sys.stderr)
    logger.warning(f"Received {signal_name}, initiating graceful shutdown")

    if _current_orchestrator is not None:
        notify = getattr(_current_orchestrator, "notify_signal", None)
        if callable(notify):
            notify(signal_name=signal_name)

    _unregister_mlflow_atexit()

    exit_code = 130 if signum == signal.SIGINT else 143

    def _deadline_exit() -> None:
        logger.warning("[SIGNAL] Cleanup deadline exceeded, forcing exit")
        os._exit(exit_code)

    deadline = threading.Timer(30.0, _deadline_exit)
    deadline.daemon = True
    deadline.start()

    sys.exit(exit_code)


# Register signal handlers
signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


def _log_config_summary(config: PipelineConfig) -> None:
    """Log configuration summary at pipeline start."""
    from src.config.datasets.constants import SOURCE_TYPE_HUGGINGFACE

    logger.info("=" * 70)
    logger.info("📋 CONFIGURATION SUMMARY")
    logger.info("=" * 70)

    # Model config
    logger.info("🤖 Model:")
    logger.info(f"   Name: {config.model.name}")
    logger.info(f"   Training type: {config.training.type}")
    logger.info(
        f"   Quantization: {'4-bit (QLoRA)' if config.training.get_effective_load_in_4bit() else 'None (LoRA/Full)'}"
    )

    # Training config
    logger.info("🏋️ Training:")
    logger.info(f"   Type: {config.training.type}")
    logger.info(f"   Batch Size: {config.training.hyperparams.per_device_train_batch_size}")
    logger.info(f"   Learning Rate: {config.training.hyperparams.learning_rate}")

    # Strategy chain info
    strategies = config.training.get_strategy_chain()
    if strategies:
        logger.info(f"   Strategies: {' → '.join(s.strategy_type.upper() for s in strategies)}")
        for i, s in enumerate(strategies, 1):
            phase_epochs = s.hyperparams.epochs or config.training.hyperparams.epochs
            phase_lr = s.hyperparams.learning_rate or config.training.hyperparams.learning_rate
            lr_str = f", lr={phase_lr}" if phase_lr else ""
            logger.info(f"     Phase {i}: {s.strategy_type} ({phase_epochs} epochs{lr_str})")

    # Adapter config
    if config.training.type in ("qlora", "lora"):
        lora = config.get_adapter_config()
        logger.info("🔧 LoRA:")
        logger.info(f"   r: {lora.r}")
        logger.info(f"   alpha: {lora.lora_alpha}")
        logger.info(f"   dropout: {lora.lora_dropout}")
        if lora.target_modules:
            logger.info(f"   target_modules: {lora.target_modules}")
        else:
            logger.info("   target_modules: auto (adapter default)")
    elif config.training.type == "adalora":
        adalora = config.get_adapter_config()
        logger.info("🔧 AdaLoRA:")
        logger.info(f"   init_r: {adalora.init_r}")
        logger.info(f"   target_r: {adalora.target_r}")

    # Dataset config (use primary dataset from registry)
    default_dataset = config.get_primary_dataset()
    logger.info("📊 Dataset:")
    if default_dataset.get_source_type() == SOURCE_TYPE_HUGGINGFACE and default_dataset.source_hf is not None:
        logger.info(f"   Train (HF): {default_dataset.source_hf.train_id}")
        if default_dataset.source_hf.eval_id:
            logger.info(f"   Eval  (HF): {default_dataset.source_hf.eval_id}")
    elif default_dataset.source_local is not None:
        logger.info(f"   Train (local): {default_dataset.source_local.local_paths.train}")
        if default_dataset.source_local.local_paths.eval:
            logger.info(f"   Eval  (local): {default_dataset.source_local.local_paths.eval}")
    else:
        logger.info("   Dataset source not configured")
    if default_dataset.max_samples:
        logger.info(f"   Max Samples: {default_dataset.max_samples}")
    if default_dataset.adapter_type:
        logger.info(f"   Adapter: {default_dataset.adapter_type}")
    else:
        logger.info("   Adapter: auto-detect")

    # Provider config
    provider_name = config.get_active_provider_name()
    logger.info(f"☁️ Provider: {provider_name}")
    # NOTE: provider schemas may group training keys under `training:`.
    if hasattr(config, "get_provider_training_config"):
        provider_train_cfg_obj = config.get_provider_training_config()
    else:
        provider_train_cfg_obj = config.get_provider_config()
    provider_train_cfg = provider_train_cfg_obj if isinstance(provider_train_cfg_obj, dict) else {}
    gpu_type = provider_train_cfg.get("gpu_type") or "auto-detect"
    logger.info(f"   GPU: {gpu_type}")

    logger.info("=" * 70)


app = typer.Typer(
    name="ryotenkai",
    help="RyotenkAI - Automated CI/CD for LLM Training",
    # Completions enabled: users can `ryotenkai --install-completion`.
    add_completion=True,
    context_settings={"help_option_names": ["-h", "--help"]},
    # Plain Python tracebacks for unhandled errors; commands that expect
    # user mistakes print a single "error: ..." line via typer.Exit.
    pretty_exceptions_enable=False,
    pretty_exceptions_show_locals=False,
    pretty_exceptions_short=True,
)


# Root callback populates the shared CLIContext on ctx.obj. Downstream
# commands read `ctx.ensure_object(CLIContext)` to pick up the effective
# output format / colour / verbosity without redeclaring these flags.
from src.cli.context import CLIContext  # noqa: E402
from src.cli.errors import die  # noqa: E402
from src.cli.style import reconfigure as _reconfigure_style  # noqa: E402
from src.cli.version import collect_version_info  # noqa: E402


def _print_version(value: bool) -> None:
    if not value:
        return
    typer.echo(collect_version_info().format())
    raise typer.Exit()


@app.callback()
def _root(
    ctx: typer.Context,
    version: bool = typer.Option(  # noqa: ARG001 -- consumed by is_eager callback
        False, "-V", "--version",
        is_eager=True, callback=_print_version,
        help="Show version info and exit.",
    ),
    output: str = typer.Option(
        "text", "-o", "--output",
        envvar="RYOTENKAI_OUTPUT",
        help="Output format for read commands: text | json.",
    ),
    color: bool = typer.Option(
        True, "--color/--no-color",
        help="Colored output (honours NO_COLOR env).",
    ),
    verbose: int = typer.Option(
        0, "-v", "--verbose",
        count=True, help="Increase verbosity (-v, -vv).",
    ),
    quiet: bool = typer.Option(
        False, "-q", "--quiet",
        help="Suppress non-essential output.",
    ),
    log_level: str | None = typer.Option(
        None, "--log-level",
        help="Override log level (DEBUG / INFO / WARNING / ERROR).",
    ),
) -> None:
    """Populate ctx.obj with the shared CLI context before any sub-command runs."""
    if output not in ("text", "json"):
        raise die(
            f"invalid --output value: {output!r}",
            hint="choose one of: text, json",
        )
    state = CLIContext(
        output=output,  # type: ignore[arg-type]
        color=color,
        verbose=verbose,
        quiet=quiet,
        log_level=log_level,
    )
    _reconfigure_style(color=state.use_color)
    ctx.obj = state


@app.command("help", hidden=True)
def _help_cmd(ctx: typer.Context) -> None:
    """Alias for --help so `ryotenkai help` works alongside `ryotenkai --help`."""
    parent = ctx.parent
    typer.echo(parent.get_help() if parent is not None else ctx.get_help())


@app.command("version")
def version_cmd(ctx: typer.Context) -> None:
    """Show version info (ryotenkai / python / platform / git sha)."""
    from src.cli.renderer import get_renderer

    state = ctx.ensure_object(CLIContext)
    renderer = get_renderer(state)
    info = collect_version_info()
    if state.is_json:
        renderer.emit(
            {
                "ryotenkai": info.ryotenkai,
                "python": info.python,
                "platform": info.platform,
                "git_sha": info.git_sha,
            }
        )
    else:
        renderer.text(info.format())
    renderer.flush()


# Community manifest authoring toolchain (scaffold / sync).
from src.cli.community import community_app  # noqa: E402

app.add_typer(
    community_app,
    name="community",
    help="Scaffold, sync and pack community/ plugin and preset manifests.",
)

# Plugin authoring toolchain (bootstrap fresh plugin folders).
from src.cli.plugin_scaffold import plugin_app  # noqa: E402

app.add_typer(
    plugin_app,
    name="plugin",
    help="Bootstrap new plugin folders under community/.",
)


def _resolve_config(config: Path | None, run_dir: Path | None) -> Path:
    """
    Resolve the config path to use for this run.

    Priority:
    1. Explicit --config → use as-is (allows intentional config override for late-stage changes).
    2. --run-dir without --config → load config_path from pipeline_state.json inside the run dir.
    3. Neither provided → error.
    """
    if config is not None:
        return config

    if run_dir is not None:
        # Lazy import keeps `ryotenkai --help` / non-resume commands fast — the
        # state/models module pulls the whole pipeline-state type tree on import.
        from src.pipeline.state import PipelineStateLoadError, PipelineStateStore

        store = PipelineStateStore(run_dir.expanduser().resolve())
        if not store.exists():
            raise ValueError(f"pipeline_state.json not found in run directory: {run_dir}")
        try:
            state = store.load()
        except PipelineStateLoadError as exc:
            raise ValueError(f"Cannot load pipeline state for {run_dir}: {exc}") from exc
        config_path_str = state.config_path
        if not config_path_str:
            raise ValueError(
                f"Run '{run_dir.name}' was created before config tracking was added.\n"
                f"Pass the config explicitly:\n"
                f"  ./run.sh /path/to/config.yaml {run_dir}"
            )
        resolved = Path(config_path_str)
        if not resolved.exists():
            raise ValueError(
                f"Config from state no longer exists: {resolved}\nPass --config explicitly to override."
            )
        return resolved

    raise ValueError("Provide --config <path> for a fresh run, or --run-dir to resume/restart an existing run.")


@app.command()
def train(
    config: Path | None = typer.Option(
        None,
        "--config",
        "-c",
        help=(
            "Path to pipeline config YAML. "
            "Required for fresh runs; optional when --run-dir is set "
            "(config is loaded from pipeline_state.json)."
        ),
    ),
    run_dir: Path | None = typer.Option(
        None,
        "--run-dir",
        help="Path to an existing logical run directory for resume/restart",
    ),
    resume: bool = typer.Option(
        False,
        "--resume",
        help="Auto-resume from the first failed/interrupted stage in an existing logical run",
    ),
    restart_from_stage: str = typer.Option(
        None,
        "--restart-from-stage",
        help="Manual restart from a stage name or 1-based stage number (see: ryotenkai list-restart-points)",
    ),
):
    """
    Run the full training pipeline.

    Fresh run:
        ryotenkai train --config config.yaml

    Resume / restart (config is taken from pipeline_state.json automatically):
        ryotenkai train --run-dir runs/my-run --resume
        ryotenkai train --run-dir runs/my-run --restart-from-stage "Inference Deployer"
        ryotenkai train --run-dir runs/my-run --restart-from-stage 5

    Override config on resume (e.g. tweak inference settings):
        ryotenkai train --config config_v2.yaml --run-dir runs/my-run --resume
    """
    from src.pipeline.orchestrator import PipelineOrchestrator  # lazy: heavy init stack

    global _current_orchestrator

    try:
        if resume and restart_from_stage:
            raise ValueError("Use either --resume or --restart-from-stage, not both")

        resolved_config = _resolve_config(config, run_dir)
        if config is None and run_dir is not None:
            typer.echo(f"Config loaded from state: {resolved_config}")

        orchestrator = PipelineOrchestrator(resolved_config, run_directory=run_dir)
        _current_orchestrator = orchestrator  # Set for signal handler

        _log_config_summary(orchestrator.config)

        result = orchestrator.run(
            run_dir=run_dir,
            resume=resume,
            restart_from_stage=restart_from_stage,
        )

        if result.is_success():
            typer.echo("Pipeline completed successfully.")
            return
        else:
            typer.echo(f"Pipeline failed: {result.unwrap_err()}", err=True)
            raise typer.Exit(1)

    except (typer.Exit, click.exceptions.Exit):
        raise
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        typer.echo(f"Configuration error: {e}", err=True)
        raise typer.Exit(1)
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)
    finally:
        _current_orchestrator = None


@app.command()
def validate_dataset(
    config: Path = typer.Option(
        ...,
        "--config",
        "-c",
        help="Path to pipeline configuration file",
    ),
):
    """
    Run only dataset validation (Stage 0).

    Example:
        ryotenkai validate-dataset --config config.yaml
    """
    from src.pipeline.orchestrator import PipelineOrchestrator  # lazy: heavy init stack

    try:
        orchestrator = PipelineOrchestrator(config)
        stage = orchestrator.stages[0]
        result = stage.run(orchestrator.context)

        if result.is_success():
            typer.echo("Dataset validation passed.")
            return
        else:
            typer.echo(f"Validation failed: {result.unwrap_err()}", err=True)
            raise typer.Exit(1)

    except (typer.Exit, click.exceptions.Exit):
        raise
    except Exception as e:
        logger.exception(f"Error: {e}")
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@app.command("list-restart-points")
def list_restart_points_cmd(
    ctx: typer.Context,
    run_dir: Path = typer.Argument(..., help="Path to an existing logical run directory"),
    config: Path | None = typer.Option(
        None, "--config", "-c",
        help="Config YAML (optional if pipeline_state.json contains config_path).",
    ),
):
    """List available restart points for a run without starting the pipeline.

    \b
    Examples:
      ryotenkai list-restart-points runs/my-run
      ryotenkai list-restart-points runs/my-run -o json
    """
    from src.cli.context import CLIContext
    from src.cli.errors import die
    from src.cli.renderer import get_renderer
    from src.pipeline.launch_queries import load_restart_point_options  # heavy: lazy

    state = ctx.ensure_object(CLIContext)
    renderer = get_renderer(state)

    try:
        _, points = load_restart_point_options(run_dir, _resolve_config(config, run_dir))
    except Exception as exc:
        raise die(str(exc))

    if state.is_json:
        renderer.emit(
            [
                {
                    "index": idx,
                    "stage": item.stage,
                    "available": bool(item.available),
                    "mode": item.mode,
                    "reason": item.reason,
                }
                for idx, item in enumerate(points, start=1)
            ]
        )
    else:
        renderer.table(
            headers=["#", "Stage", "Available", "Mode", "Reason"],
            rows=[
                (idx, item.stage, "yes" if item.available else "no", item.mode, item.reason)
                for idx, item in enumerate(points, start=1)
            ],
        )
        renderer.text("")
        renderer.text("Use # or stage name with --restart-from-stage")
    renderer.flush()


@app.command()
def info(
    ctx: typer.Context,
    config: Path = typer.Option(
        ..., "--config", "-c",
        help="Path to pipeline configuration file.",
    ),
):
    """Show pipeline configuration and stage information.

    \b
    Examples:
      ryotenkai info --config config.yaml
      ryotenkai info -c config.yaml -o json | jq '.stages'
    """
    from src.cli.context import CLIContext
    from src.cli.errors import die
    from src.cli.renderer import get_renderer
    from src.config.datasets.constants import SOURCE_TYPE_HUGGINGFACE
    from src.pipeline.orchestrator import PipelineOrchestrator  # lazy: heavy init stack

    state = ctx.ensure_object(CLIContext)
    renderer = get_renderer(state)

    try:
        orchestrator = PipelineOrchestrator(config)
    except Exception as exc:
        logger.exception(f"Error: {exc}")
        raise die(str(exc))

    model_cfg = orchestrator.config.model
    training_cfg = orchestrator.config.training
    adapter_cfg = training_cfg.get_adapter_config()
    strategies = [s.strategy_type for s in training_cfg.get_strategy_chain()]

    # Dataset refs
    default_ds = orchestrator.config.get_primary_dataset()
    train_ref = default_ds.get_display_train_ref()
    if default_ds.get_source_type() == SOURCE_TYPE_HUGGINGFACE:
        assert default_ds.source_hf is not None
        eval_ref = default_ds.source_hf.eval_id or None
    else:
        assert default_ds.source_local is not None
        eval_ref = default_ds.source_local.local_paths.eval or None

    stages = list(orchestrator.list_stages())

    if state.is_json:
        renderer.emit(
            {
                "model": {"name": model_cfg.name, "training_type": training_cfg.type},
                "adapter": (
                    {"kind": training_cfg.type, "r": adapter_cfg.r}
                    if training_cfg.type in ("qlora", "lora")
                    else {"kind": training_cfg.type, "init_r": getattr(adapter_cfg, "init_r", None)}
                    if training_cfg.type == "adalora"
                    else None
                ),
                "strategies": strategies,
                "dataset": {"train": train_ref, "eval": eval_ref},
                "stages": stages,
            }
        )
    else:
        renderer.kv(
            {
                "Model": model_cfg.name,
                "Training": training_cfg.type,
                **(
                    {"LoRA r": adapter_cfg.r}
                    if training_cfg.type in ("qlora", "lora")
                    else {"AdaLoRA init_r": adapter_cfg.init_r}
                    if training_cfg.type == "adalora"
                    else {}
                ),
                **(
                    {"Strategies": " → ".join(s.upper() for s in strategies)}
                    if strategies else {}
                ),
            },
            title="Model",
        )
        renderer.text("")
        renderer.kv(
            {"Train": train_ref, "Eval": eval_ref or "-"},
            title="Dataset",
        )
        renderer.text("")
        renderer.table(
            headers=["#", "Stage"],
            rows=[(i, name) for i, name in enumerate(stages)],
            title="Stages",
        )
    renderer.flush()


@app.command(name="train-local")
def train_local(
    config: Path = typer.Option(
        ...,
        "--config",
        "-c",
        help="Path to pipeline configuration file",
    ),
    resume: bool = typer.Option(
        False,
        "--resume",
        "-r",
        help="Resume from last incomplete phase",
    ),
    run_id: str = typer.Option(
        None,
        "--run-id",
        help="Run ID for resume or reproducibility",
    ),
):
    """
    Run training LOCALLY with StrategyOrchestrator.

    This runs multi-phase training directly on your machine,
    NOT on RunPod. Useful for testing and development.

    For DEBUG logs, set LOG_LEVEL=DEBUG:
        LOG_LEVEL=DEBUG ryotenkai train-local --config config.yaml

    Debug log tags:
        [SO:]  - StrategyOrchestrator
        [SF:]  - StrategyFactory
        [TF:]  - TrainerFactory
        [DB:]  - DataBuffer
        [MM:]  - MemoryManager
        [CFG:] - Config

    Example:
        ryotenkai train-local --config config/53_tests/test_1_single_sft.yaml
        ryotenkai train-local --config config.yaml --resume --run-id run_xxx
    """
    from src.training.run_training import run_training  # lazy: loads strategy factory

    try:
        output_path = run_training(
            config_path=str(config),
            resume=resume,
            run_id=run_id,
        )
        typer.echo(f"Training completed: {output_path}")
        return

    except (typer.Exit, click.exceptions.Exit):
        raise
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        typer.echo(f"Configuration error: {e}", err=True)
        raise typer.Exit(1)
    except Exception as e:
        logger.exception(f"Training error: {e}")
        typer.echo(f"Training failed: {e}", err=True)
        raise typer.Exit(1)


@app.command(name="inspect-run")
def inspect_run(
    ctx: typer.Context,
    run_dir: Path = typer.Argument(..., help="Path to an existing logical run directory"),
    show_outputs: bool = typer.Option(
        False, "--outputs",
        help="Show stage outputs and lineage (text mode only; json always includes them).",
    ),
    logs: bool = typer.Option(False, "--logs", help="Show tail of pipeline.log for each attempt"),
):
    """Inspect a run — show structured info about the run and its attempts.

    \b
    Examples:
      ryotenkai inspect-run runs/run_xxx
      ryotenkai inspect-run runs/run_xxx --outputs --logs
      ryotenkai inspect-run runs/run_xxx -o json | jq '.attempts[0].stages'
    """
    from src.cli.context import CLIContext
    from src.cli.errors import die
    from src.cli.renderer import get_renderer
    from src.pipeline.run_queries import RunInspector, effective_pipeline_status

    state = ctx.ensure_object(CLIContext)
    renderer = get_renderer(state)

    try:
        data = RunInspector(run_dir).load(include_logs=logs)
    except Exception as exc:
        raise die(str(exc), hint="list available runs with `ryotenkai runs-list`")

    if state.is_json:
        renderer.emit(_inspect_run_json(data))
    else:
        from src.cli.run_rendering import render_run_inspection_lines

        for line in render_run_inspection_lines(data, verbose=show_outputs, include_logs=logs):
            renderer.text(line)
    # Silence unused-import warning for effective_pipeline_status when json path
    # doesn't trigger (it's used inside _inspect_run_json via re-import).
    _ = effective_pipeline_status
    renderer.flush()


def _inspect_run_json(data) -> dict:  # type: ignore[no-untyped-def]
    """Build the JSON payload for ``inspect-run -o json`` (data: RunInspectionData)."""
    from src.cli.formatters import duration_seconds
    from src.pipeline.run_queries import effective_pipeline_status

    pipeline_state = data.state
    return {
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
                "duration_s": duration_seconds(attempt.started_at, attempt.completed_at),
                "error": attempt.error,
                "stages": [
                    _stage_json(attempt, stage_name)
                    for stage_name in (attempt.enabled_stage_names or list(attempt.stage_runs))
                ],
            }
            for attempt in pipeline_state.attempts
        ],
    }


def _stage_json(attempt, stage_name: str) -> dict:  # type: ignore[no-untyped-def]
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
        "duration_s": duration_seconds(stage_run.started_at, stage_run.completed_at),
        "error": stage_run.error,
        "outputs": stage_run.outputs,
    }


@app.command(name="runs-list")
def runs_list(
    ctx: typer.Context,
    runs_dir: Path = typer.Argument(
        Path("runs"),
        help="Directory containing run subdirectories (default: ./runs)",
    ),
):
    """List all runs with summary info.

    \b
    Examples:
      ryotenkai runs-list
      ryotenkai runs-list /path/to/runs
      ryotenkai runs-list -o json | jq '.[] | select(.status=="failed")'
    """
    from src.cli.context import CLIContext
    from src.cli.formatters import duration_seconds, format_duration
    from src.cli.renderer import get_renderer
    from src.pipeline.run_queries import scan_runs_dir

    state = ctx.ensure_object(CLIContext)
    renderer = get_renderer(state)
    rows = list(scan_runs_dir(runs_dir))

    if state.is_json:
        renderer.emit(
            [
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
            ]
        )
    elif not rows:
        renderer.text(f"No runs found in {runs_dir}")
    else:
        renderer.heading(f"Runs in {runs_dir}/")
        renderer.text("")
        renderer.table(
            headers=["Run ID", "Status", "Att", "Duration", "Config"],
            rows=[
                (
                    row.run_id,
                    row.status,
                    row.attempts,
                    format_duration(row.started_at, row.completed_at) or "-",
                    row.config_name,
                )
                for row in rows
            ],
        )
    renderer.flush()


@app.command(name="logs")
def show_logs(
    run_dir: Path = typer.Argument(..., help="Path to the run directory"),
    attempt: int = typer.Option(
        0,
        "--attempt",
        help="Attempt number (default: last attempt)",
    ),
    follow: bool = typer.Option(False, "--follow", "-f", help="Stream log (tail -f mode)"),
):
    """
    Show pipeline log for a run attempt.

    Examples:
        ryotenkai logs runs/run_xxx               # last attempt
        ryotenkai logs runs/run_xxx --attempt 2   # specific attempt
        ryotenkai logs runs/run_xxx --follow       # stream mode
    """
    import time

    from src.pipeline.state import PipelineStateLoadError, PipelineStateStore

    run_path = run_dir.expanduser().resolve()
    store = PipelineStateStore(run_path)

    try:
        state = store.load()
    except (PipelineStateLoadError, Exception) as exc:
        typer.echo(f"Cannot load state: {exc}", err=True)
        raise typer.Exit(1)

    if not state.attempts:
        typer.echo("No attempts found.")
        return

    target_no = attempt if attempt > 0 else state.attempts[-1].attempt_no
    log_path = store.next_attempt_dir(target_no) / "pipeline.log"

    if not log_path.exists():
        typer.echo(f"Log file not found: {log_path}", err=True)
        raise typer.Exit(1)

    if follow:
        typer.echo(f"Streaming {log_path}  (Ctrl+C to stop)\n")
        try:
            with log_path.open(encoding="utf-8", errors="replace") as fh:
                while True:
                    line = fh.readline()
                    if line:
                        typer.echo(line.rstrip())
                    else:
                        time.sleep(0.5)
        except KeyboardInterrupt:
            typer.echo("\nStopped.")
    else:
        typer.echo(log_path.read_text(encoding="utf-8", errors="replace"))


@app.command(name="run-diff")
def run_diff(
    ctx: typer.Context,
    run_dir: Path = typer.Argument(..., help="Path to the run directory"),
    attempt: list[int] = typer.Option(
        [], "--attempt",
        help="Attempt numbers to compare (use twice: --attempt 1 --attempt 3). Default: first vs last.",
    ),
):
    """Compare config hashes between attempts.

    \b
    Examples:
      ryotenkai run-diff runs/run_xxx
      ryotenkai run-diff runs/run_xxx --attempt 1 --attempt 3
      ryotenkai run-diff runs/run_xxx -o json
    """
    from src.cli.context import CLIContext
    from src.cli.errors import die
    from src.cli.renderer import get_renderer
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
        if state_ctx.is_json:
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

    if state_ctx.is_json:
        renderer.emit(
            {
                "attempt_a": attempt_a,
                "attempt_b": attempt_b,
                "training_critical_changed": bool(diff["training_critical_changed"]),
                "late_stage_changed": bool(diff["late_stage_changed"]),
                "hash_a_critical": diff.get("hash_a_critical"),
                "hash_b_critical": diff.get("hash_b_critical"),
                "hash_a_late": diff.get("hash_a_late"),
                "hash_b_late": diff.get("hash_b_late"),
            }
        )
    else:
        for line in render_run_diff_lines(diff, attempt_a, attempt_b):
            renderer.text(line)
    renderer.flush()


@app.command(name="run-status")
def run_status(
    ctx: typer.Context,
    run_dir: Path = typer.Argument(..., help="Path to the run directory"),
    interval: float = typer.Option(5.0, "--interval", "-i", help="Polling interval in seconds (text mode only)."),
    once: bool = typer.Option(False, "--once", help="Print one snapshot and exit (implied by -o json)."),
):
    """Show pipeline status — one snapshot or live-polling until Ctrl+C.

    \b
    Examples:
      ryotenkai run-status runs/run_xxx
      ryotenkai run-status runs/run_xxx --once
      ryotenkai run-status runs/run_xxx -o json
    """
    import time

    from src.cli.context import CLIContext
    from src.cli.errors import die
    from src.cli.formatters import duration_seconds
    from src.cli.renderer import get_renderer
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

    def _status_json(state) -> dict:
        current = state.attempts[-1] if state.attempts else None
        return {
            "run_id": run_path.name,
            "status": state.pipeline_status,
            "attempts_total": len(state.attempts),
            "current_attempt": current.attempt_no if current else None,
            "stages": [
                {
                    "name": stage_name,
                    "status": (current.stage_runs.get(stage_name).status
                               if current and current.stage_runs.get(stage_name) else "pending"),
                    "duration_s": duration_seconds(
                        current.stage_runs.get(stage_name).started_at if current and current.stage_runs.get(stage_name) else None,
                        current.stage_runs.get(stage_name).completed_at if current and current.stage_runs.get(stage_name) else None,
                    ) if current and current.stage_runs.get(stage_name) else None,
                }
                for stage_name in (current.enabled_stage_names or list(current.stage_runs))
            ] if current else [],
        }

    # JSON mode is always a single snapshot.
    if state_ctx.is_json:
        renderer.emit(_status_json(_load_state()))
        renderer.flush()
        return

    if once:
        for line in render_run_status_snapshot(run_path.name, _load_state()):
            renderer.text(line)
        renderer.flush()
        return

    try:
        while True:
            print("\033[H\033[2J", end="")  # clear screen
            for line in render_run_status_snapshot(run_path.name, _load_state()):
                renderer.text(line)
            renderer.text(f"Refreshing every {interval}s — Ctrl+C to stop")
            time.sleep(interval)
    except KeyboardInterrupt:
        renderer.text("")
        renderer.text("Stopped monitoring.")


@app.command(name="config-validate")
def config_validate(
    config: Path = typer.Option(
        ...,
        "--config",
        "-c",
        help="Path to pipeline config YAML",
    ),
):
    """
    Static pre-flight checks for a pipeline config (no network calls).

    Checks:
      - YAML schema valid (Pydantic)
      - Dataset paths exist (for local source)
      - Required env vars present (HF_TOKEN, RUNPOD_API_KEY when needed)
      - Evaluation plugins importable
      - Stage consistency (inference enabled → evaluator requirements)

    Example:
        ryotenkai config-validate --config config/pipeline.yaml
    """
    import importlib

    from src.config.datasets.constants import SOURCE_TYPE_LOCAL

    checks: list[tuple[bool | None, str, str]] = []  # (ok/warn/None, label, detail)

    def _ok(label: str, detail: str = "") -> None:
        checks.append((True, label, detail))

    def _fail(label: str, detail: str = "") -> None:
        checks.append((False, label, detail))

    def _warn(label: str, detail: str = "") -> None:
        checks.append((None, label, detail))

    # 1. Pydantic schema
    cfg = None
    try:
        from src.utils.config import load_config

        cfg = load_config(config)
        _ok("YAML schema valid (Pydantic)")
    except Exception as exc:
        _fail("YAML schema", str(exc))

    if cfg is not None:
        # 2. Dataset paths
        for ds_name, ds_cfg in cfg.datasets.items():
            if ds_cfg.get_source_type() == SOURCE_TYPE_LOCAL and ds_cfg.source_local:
                train_path = ds_cfg.source_local.local_paths.train
                eval_path = getattr(ds_cfg.source_local.local_paths, "eval", None)
                if Path(train_path).exists():
                    _ok(f"Dataset '{ds_name}' train path exists", train_path)
                else:
                    _fail(f"Dataset '{ds_name}' train path not found", train_path)
                if eval_path:
                    if Path(eval_path).exists():
                        _ok(f"Dataset '{ds_name}' eval path exists", eval_path)
                    else:
                        _fail(f"Dataset '{ds_name}' eval path not found", eval_path)
            else:
                _ok(f"Dataset '{ds_name}' (HuggingFace — path check skipped)")

        # 3. Env vars
        hf_token = os.environ.get("HF_TOKEN", "").strip()
        if hf_token:
            _ok("HF_TOKEN found")
        else:
            _fail("HF_TOKEN not set")

        runpod_key = os.environ.get("RUNPOD_API_KEY", "").strip()
        active_provider = cfg.get_active_provider_name() if hasattr(cfg, "get_active_provider_name") else None
        if active_provider and "runpod" in (active_provider or "").lower():
            if runpod_key:
                _ok("RUNPOD_API_KEY found")
            else:
                _fail("RUNPOD_API_KEY not set (required for runpod provider)")
        else:
            if runpod_key:
                _ok("RUNPOD_API_KEY found (optional)")
            else:
                _warn("RUNPOD_API_KEY not set (optional)")

        # 4. Evaluation plugins importable
        eval_cfg = getattr(cfg, "evaluation", None)
        if eval_cfg and getattr(eval_cfg, "enabled", False):
            plugins_cfg = getattr(getattr(eval_cfg, "evaluators", None), "plugins", [])
            for plug_cfg in plugins_cfg:
                plugin_type = getattr(plug_cfg, "type", None)
                if plugin_type:
                    try:
                        importlib.import_module(f"src.evaluation.plugins.builtins.{plugin_type}")
                        _ok(f"Eval plugin '{plugin_type}' importable")
                    except ImportError:
                        _warn(f"Eval plugin '{plugin_type}' not found (custom?)")
        else:
            _ok("Evaluation disabled — plugin check skipped")

        # 5. Stage consistency
        inference_enabled = getattr(getattr(cfg, "inference", None), "enabled", False)
        eval_enabled = eval_cfg and getattr(eval_cfg, "enabled", False)
        if eval_enabled and not inference_enabled:
            _warn(
                "Stage consistency",
                "evaluation.enabled=true but inference.enabled=false — ModelEvaluator needs a live runtime",
            )
        else:
            _ok("Stage consistency")

    from src.cli.context import CLIContext
    from src.cli.renderer import get_renderer
    from src.cli.style import COLOR_ERR, COLOR_OK, COLOR_WARN, ICONS

    ctx = click.get_current_context()
    state = ctx.ensure_object(CLIContext)
    renderer = get_renderer(state)

    def _status(ok: bool | None) -> str:
        if ok is True:
            return "ok"
        if ok is False:
            return "fail"
        return "warn"

    any_fail = any(ok is False for ok, _, _ in checks)

    if state.is_json:
        renderer.emit(
            {
                "ok": not any_fail,
                "checks": [
                    {"label": label, "status": _status(ok), "detail": detail}
                    for ok, label, detail in checks
                ],
            }
        )
    else:
        marker_for = {
            "ok":   f"[{COLOR_OK}]{ICONS.ok}[/{COLOR_OK}]",
            "fail": f"[{COLOR_ERR}]{ICONS.err}[/{COLOR_ERR}]",
            "warn": f"[{COLOR_WARN}]{ICONS.warn}[/{COLOR_WARN}]",
        }
        for ok, label, detail in checks:
            line = f"  {marker_for[_status(ok)]}  {label}"
            if detail:
                line += f"  [dim]{detail}[/dim]"
            renderer.text(line)
        renderer.text("")
        if any_fail:
            renderer.text(f"[{COLOR_ERR}]Result: not ready — fix errors above[/{COLOR_ERR}]")
        else:
            renderer.text(f"[{COLOR_OK}]Result: ready to run[/{COLOR_OK}]")
    renderer.flush()
    if any_fail:
        raise typer.Exit(1)


@app.command(name="report")
def generate_report(
    run_dir: Path = typer.Argument(..., help="Path to an existing logical run directory"),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Save report to this path (default: <run-dir>/report.md)",
    ),
):
    """
    Generate an MLflow experiment report for a run.

    Reads root_mlflow_run_id from pipeline_state.json and generates
    a markdown report via ExperimentReportGenerator.

    Example:
        ryotenkai report runs/run_20260319_030619_x9so8
    """
    from src.pipeline.state import PipelineStateLoadError, PipelineStateStore
    from src.reports.report_generator import ExperimentReportGenerator

    run_path = run_dir.expanduser().resolve()
    store = PipelineStateStore(run_path)

    try:
        state = store.load()
    except (PipelineStateLoadError, Exception) as exc:
        typer.echo(f"Cannot load state: {exc}", err=True)
        raise typer.Exit(1)

    run_id = state.root_mlflow_run_id
    if not run_id:
        typer.echo("root_mlflow_run_id not found in pipeline_state.json", err=True)
        raise typer.Exit(1)

    typer.echo(f"MLflow run: {run_id}")

    save_path = output or (run_path / "report.md")
    try:
        generator = ExperimentReportGenerator()
        markdown = generator.generate(run_id, local_logs_dir=run_path)
        save_path.write_text(markdown, encoding="utf-8")
        typer.echo(f"Report saved: {save_path}")
    except Exception as exc:
        typer.echo(f"Report generation failed: {exc}", err=True)
        raise typer.Exit(1)


@app.command(name="serve")
def serve(
    host: str = typer.Option("127.0.0.1", "--host", help="Bind host (use 0.0.0.0 for remote access)"),
    port: int = typer.Option(8000, "--port", help="Bind port"),
    runs_dir: Path = typer.Option(Path("runs"), "--runs-dir", help="Runs directory served by the API"),
    cors_origins: str = typer.Option(
        "http://localhost:5173",
        "--cors-origins",
        help="Comma-separated CORS origins (Vite dev server by default)",
    ),
    reload: bool = typer.Option(False, "--reload", help="Dev auto-reload"),
    log_level: str = typer.Option("info", "--log-level", help="uvicorn log level"),
):
    """Run the FastAPI web backend (CLI and web share the runs/ directory)."""
    from src.api.cli import run_server

    run_server(
        host=host,
        port=port,
        runs_dir=runs_dir,
        cors_origins=[origin.strip() for origin in cors_origins.split(",") if origin.strip()],
        reload=reload,
        log_level=log_level,
    )


if __name__ == "__main__":
    app()
