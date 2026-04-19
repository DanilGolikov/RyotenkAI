"""
Main entry point for RyotenkAI CLI.
"""

import atexit
import json
import os
import signal
import sys
import threading
from pathlib import Path

import click
import typer

from src.cli.run_rendering import (
    RunInspectionRenderer,
    render_run_diff_lines,
    render_run_status_snapshot,
    render_runs_list_lines,
)
from src.config.datasets.constants import SOURCE_TYPE_HUGGINGFACE, SOURCE_TYPE_LOCAL
from src.pipeline.launch_queries import load_restart_point_options
from src.utils.config import PipelineConfig, load_config
from src.utils.logger import logger

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
    add_completion=False,
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
        state_file = run_dir.expanduser().resolve() / "pipeline_state.json"
        if not state_file.exists():
            raise ValueError(f"pipeline_state.json not found in run directory: {run_dir}")
        try:
            raw = json.loads(state_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Corrupt pipeline_state.json in {run_dir}: {exc}") from exc
        config_path_str = raw.get("config_path", "")
        if not config_path_str:
            raise ValueError(
                f"Run '{run_dir.name}' was created before config tracking was added.\n"
                f"Pass the config explicitly:\n"
                f"  ./run.sh /path/to/config.yaml {run_dir}"
            )
        resolved = Path(config_path_str)
        if not resolved.exists():
            raise ValueError(
                f"Config from state no longer exists: {resolved}\n" f"Pass --config explicitly to override."
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
    run_dir: Path = typer.Argument(..., help="Path to an existing logical run directory"),
    config: Path | None = typer.Option(
        None,
        "--config",
        "-c",
        help="Config YAML (optional if pipeline_state.json contains config_path)",
    ),
):
    """
    List available restart points for a run without starting the pipeline.

    Example:
        ryotenkai list-restart-points runs/my-run
        ryotenkai list-restart-points runs/my-run --config config.yaml
    """
    try:
        _, points = load_restart_point_options(run_dir, _resolve_config(config, run_dir))

        typer.echo(f"{'#':>3}  {'Stage':<30} {'Available':<10} {'Mode':<12} Reason")
        typer.echo("-" * 80)
        for idx, item in enumerate(points, start=1):
            avail = "yes" if item.available else "no"
            typer.echo(f"{idx:>3}  {item.stage:<30} {avail:<10} {item.mode:<12} {item.reason}")
        typer.echo("\nUse # or stage name with --restart-from-stage")
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def info(
    config: Path = typer.Option(
        ...,
        "--config",
        "-c",
        help="Path to pipeline configuration file",
    ),
):
    """
    Show pipeline configuration and stage information.

    Example:
        ryotenkai info --config config.yaml
    """
    from src.pipeline.orchestrator import PipelineOrchestrator  # lazy: heavy init stack

    try:
        orchestrator = PipelineOrchestrator(config)

        typer.echo("Pipeline Stages:")
        for i, stage_name in enumerate(orchestrator.list_stages()):
            typer.echo(f"  {i}. {stage_name}")

        typer.echo("\nModel Configuration:")
        typer.echo(f"  Model         : {orchestrator.config.model.name}")
        typer.echo(f"  Training type : {orchestrator.config.training.type}")
        adapter_cfg = orchestrator.config.training.get_adapter_config()
        if orchestrator.config.training.type in ("qlora", "lora"):
            typer.echo(f"  LoRA r        : {adapter_cfg.r}")
        elif orchestrator.config.training.type == "adalora":
            typer.echo(f"  AdaLoRA init_r: {adapter_cfg.init_r}")
        strategies = orchestrator.config.training.get_strategy_chain()
        if strategies:
            typer.echo(f"  Strategies    : {' -> '.join(s.strategy_type.upper() for s in strategies)}")

        typer.echo("\nDataset Configuration:")
        default_ds = orchestrator.config.get_primary_dataset()
        train_ref = default_ds.get_display_train_ref()

        if default_ds.get_source_type() == SOURCE_TYPE_HUGGINGFACE:
            assert default_ds.source_hf is not None
            eval_ref = default_ds.source_hf.eval_id or "N/A"
        else:
            assert default_ds.source_local is not None
            eval_ref = default_ds.source_local.local_paths.eval or "N/A"

        typer.echo(f"  Train: {train_ref}")
        typer.echo(f"  Eval : {eval_ref}")

    except Exception as e:
        logger.exception(f"Error: {e}")
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


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
    run_dir: Path = typer.Argument(..., help="Path to an existing logical run directory"),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Show stage outputs and lineage"),
    logs: bool = typer.Option(False, "--logs", help="Show tail of pipeline.log for each attempt"),
):
    """
    Inspect a run directory — show structured info about the run and its attempts.

    Examples:
        ryotenkai inspect-run runs/run_20260319_030619_x9so8
        ryotenkai inspect-run runs/run_20260319_030619_x9so8 -v
        ryotenkai inspect-run runs/run_20260319_030619_x9so8 --logs
        ryotenkai inspect-run runs/run_20260319_030619_x9so8 -v --logs
    """
    from src.pipeline.run_queries import RunInspector

    try:
        inspector = RunInspector(run_dir)
        data = inspector.load(include_logs=logs)
        renderer = RunInspectionRenderer()
        renderer.render(data, verbose=verbose, include_logs=logs)
    except Exception as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1)


@app.command(name="runs-list")
def runs_list(
    runs_dir: Path = typer.Argument(
        Path("runs"),
        help="Directory containing run subdirectories (default: ./runs)",
    ),
):
    """
    List all runs with summary info.

    Example:
        ryotenkai runs-list
        ryotenkai runs-list /path/to/runs
    """
    from src.pipeline.run_queries import scan_runs_dir

    rows = scan_runs_dir(runs_dir)
    for line in render_runs_list_lines(runs_dir, rows):
        typer.echo(line)


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
    run_dir: Path = typer.Argument(..., help="Path to the run directory"),
    attempt: list[int] = typer.Option(
        [],
        "--attempt",
        help="Attempt numbers to compare (use twice: --attempt 1 --attempt 3). Default: first vs last.",
    ),
):
    """
    Compare config hashes between attempts.

    Examples:
        ryotenkai run-diff runs/run_xxx
        ryotenkai run-diff runs/run_xxx --attempt 1 --attempt 3
    """
    from src.pipeline.run_queries import diff_attempts
    from src.pipeline.state import PipelineStateLoadError, PipelineStateStore

    run_path = run_dir.expanduser().resolve()
    store = PipelineStateStore(run_path)

    try:
        state = store.load()
    except (PipelineStateLoadError, Exception) as exc:
        typer.echo(f"Cannot load state: {exc}", err=True)
        raise typer.Exit(1)

    if len(state.attempts) < 2:  # noqa: WPS432
        typer.echo("Only one attempt — nothing to compare.")
        return

    attempt_nos = list(attempt)
    if len(attempt_nos) == 0:
        attempt_a = state.attempts[0].attempt_no
        attempt_b = state.attempts[-1].attempt_no
    elif len(attempt_nos) == 1:
        attempt_a = state.attempts[0].attempt_no
        attempt_b = attempt_nos[0]
    else:
        attempt_a, attempt_b = attempt_nos[0], attempt_nos[1]

    diff = diff_attempts(state, attempt_a, attempt_b)

    if not diff["found_a"] or not diff["found_b"]:
        typer.echo(f"Attempt {attempt_a} or {attempt_b} not found in state.", err=True)
        raise typer.Exit(1)
    for line in render_run_diff_lines(diff, attempt_a, attempt_b):
        typer.echo(line)


@app.command(name="run-status")
def run_status(
    run_dir: Path = typer.Argument(..., help="Path to the run directory"),
    interval: float = typer.Option(5.0, "--interval", "-i", help="Polling interval in seconds"),
):
    """
    Live monitoring of a running pipeline (polls pipeline_state.json).

    Press Ctrl+C to stop monitoring.

    Example:
        ryotenkai run-status runs/run_xxx
    """
    import time

    from src.pipeline.state import PipelineStateLoadError, PipelineStateStore

    run_path = run_dir.expanduser().resolve()
    store = PipelineStateStore(run_path)

    def _print_status() -> None:
        try:
            state = store.load()
        except (PipelineStateLoadError, Exception) as exc:
            typer.echo(f"Error reading state: {exc}")
            return
        for line in render_run_status_snapshot(run_path.name, state):
            typer.echo(line)

    try:
        while True:
            print("\033[H\033[2J", end="")  # clear screen
            _print_status()
            typer.echo(f"Refreshing every {interval}s — Ctrl+C to stop")
            time.sleep(interval)
    except KeyboardInterrupt:
        typer.echo("\nStopped monitoring.")


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

    any_fail = False
    for ok, label, detail in checks:
        if ok is True:
            marker = "[ OK ]"
        elif ok is False:
            marker = "[FAIL]"
            any_fail = True
        else:
            marker = "[WARN]"
        detail_str = f"  {detail}" if detail else ""
        typer.echo(f"{marker} {label}{detail_str}")

    typer.echo("")
    if any_fail:
        typer.echo("Result: not ready — fix errors above", err=True)
        raise typer.Exit(1)
    else:
        typer.echo("Result: ready to run")


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


@app.command(name="tui")
def ryotenkai_tui(
    run_dir: Path | None = typer.Argument(
        None,
        help="Optional run directory. When provided, opens live-monitor for that run directly.",
    ),
    interval: float = typer.Option(
        5.0,
        "--interval",
        "-i",
        help="Auto-refresh interval in seconds for the live monitor.",
    ),
    log_level: str = typer.Option(
        "INFO",
        "--log-level",
        help="TUI log level: INFO or DEBUG.",
    ),
):
    """
    Interactive TUI for pipeline run inspection.

    Without arguments — opens the runs browser:
        Navigable table of all runs. Arrow keys to move, Enter to inspect,
        M to open live monitor, R to refresh, ? for help, Q to quit.

    With a run directory — opens the live monitor for that run:
        Auto-refreshing dashboard showing stage statuses.

    Looks for runs in ./runs relative to the current working directory.

    Examples:
        ryotenkai tui
        ryotenkai tui runs/run_20260319_030619_x9so8
        ryotenkai tui runs/run_20260319_030619_x9so8 --interval 10
    """
    normalized_log_level = log_level.upper()
    if normalized_log_level not in {"INFO", "DEBUG"}:
        raise typer.BadParameter("log level must be INFO or DEBUG", param_hint="--log-level")

    from src.tui.apps import RyotenkaiApp
    from src.tui.runtime import (
        TuiRuntimeConfig,
        default_errors_log_path,
        default_tui_log_path,
        run_tui_with_restart,
    )
    from src.utils.logger import set_log_level

    set_log_level(normalized_log_level)

    runs_dir = Path("runs").resolve()
    resolved_run_dir = run_dir.expanduser().resolve() if run_dir is not None else None
    run_tui_with_restart(
        lambda: RyotenkaiApp(
            runs_dir=runs_dir,
            initial_run_dir=resolved_run_dir,
            interval=interval,
        ),
        config=TuiRuntimeConfig(
            errors_log_path=default_errors_log_path(runs_dir),
            tui_log_path=default_tui_log_path(runs_dir),
        ),
    )


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
    """Run the FastAPI web backend (CLI/TUI/web share the runs/ directory)."""
    from src.api.cli import run_server

    run_server(
        host=host,
        port=port,
        runs_dir=runs_dir,
        cors_origins=[origin.strip() for origin in cors_origins.split(",") if origin.strip()],
        reload=reload,
        log_level=log_level,
    )


@app.command()
def version():
    """Show version information."""
    typer.echo("RyotenkAI")
    typer.echo("Version: v0.1.0")
    typer.echo("Author: Golikov Daniil")


if __name__ == "__main__":
    app()
