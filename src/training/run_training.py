"""
Multi-Phase LLM Training Entry Point.

Main entry point for running LLM training with StrategyOrchestrator.
Supports single-phase and multi-phase training pipelines.

Components used:
- MemoryManager: OOM protection and GPU management
- StrategyOrchestrator: Multi-phase training coordination
- StrategyFactory: Strategy creation (CPT, SFT, CoT, DPO, ORPO)
- TrainerFactory: TRL trainer creation
- DataBuffer: Checkpoint and state management

Features:
- Single-phase training (backward compatible)
- Multi-phase training: CPT → SFT → CoT → DPO
- Resume from failed/interrupted phases
- Automatic checkpoint cleanup
- Dependency Injection via TrainingContainer
- MLflow experiment tracking with event logging

Usage:
    # Run training
    python -m src.training.run_training --config config/pipeline_config.yaml

    # With DEBUG logs
    LOG_LEVEL=DEBUG python -m src.training.run_training --config config/pipeline_config.yaml

    # Resume interrupted training
    python -m src.training.run_training --config config/pipeline_config.yaml --resume --run-id run_xxx
"""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from src.training.constants import (
    CATEGORY_TRAINING,
    EXIT_KEYBOARD_INTERRUPT,
    SOURCE_RUN_TRAINING,
    TRUNCATE_ERROR_MSG,
    TRUNCATE_SHA_DISPLAY,
)
from src.training.managers.mlflow_manager import MLflowManager
from src.utils.config import load_config
from src.utils.container import TrainingContainer
from src.utils.environment import EnvironmentReporter
from src.utils.logger import logger
from src.utils.run_naming import generate_run_name

if TYPE_CHECKING:
    from src.utils.config import PipelineConfig


def _extract_model_size(model_name: str) -> str:
    """
    Extract model size from model name.

    Examples:
        "Qwen/Qwen2.5-0.5B-Instruct" -> "0.5B"
        "meta-llama/Llama-3.2-7B" -> "7B"
        "HuggingFaceTB/SmolLM2-1.7B-Instruct" -> "1.7B"
        "unsloth/Qwen2.5-14B" -> "14B"

    Args:
        model_name: HuggingFace model name

    Returns:
        Model size string (e.g., "0.5B", "7B") or "unknown" if not found
    """
    import re

    # Pattern: digits followed by optional decimal, then B (billion)
    pattern = r"(\d+\.?\d*)[Bb]"
    match = re.search(pattern, model_name)

    if match:
        size = match.group(1)
        return f"{size}B"

    logger.warning(f"Could not extract model size from model name: {model_name}")
    return "unknown"


def _setup_mlflow(config: PipelineConfig) -> MLflowManager | None:
    """
    Initialize MLflow tracking.

    Args:
        config: Pipeline configuration

    Returns:
        MLflowManager instance when tracking is available, otherwise None
    """
    try:
        manager = MLflowManager(config, runtime_role="training")
        if not manager.setup():
            logger.warning("MLflow setup failed or tracking backend is unreachable; continuing without MLflow")
            return None
        return manager
    except Exception as e:
        logger.warning(f"MLflow setup failed: {e}; continuing without MLflow")
        return None


def run_training(
    config_path: str,
    *,
    resume: bool = False,
    run_id: str | None = None,
    container: TrainingContainer | None = None,
) -> Path:
    """
    Main training function using StrategyOrchestrator.

    Executes LLM training with support for:
    - Multi-phase training (CPT → SFT → CoT → DPO)
    - Resume from failed/interrupted phases
    - OOM protection via MemoryManager
    - Checkpoint management via DataBuffer
    - Dependency Injection via TrainingContainer
    - MLflow event logging and summary generation

    Args:
        config_path: Path to pipeline configuration file
        resume: If True, resume from last incomplete phase
        run_id: Optional run ID for resume or reproducibility
        container: Optional pre-configured TrainingContainer (for testing)

    Returns:
        Path to final model checkpoint

    Raises:
        RuntimeError: If training fails

    Example:
        # Production usage
        output_path = run_training("config/pipeline_config.yaml")

        # Testing with mocks
        container = TrainingContainer.for_testing(config, memory_manager=mock_mm)
        output_path = run_training("config/test.yaml", container=container)
    """
    notifier = None
    mlflow_mgr: MLflowManager | None = None
    mlflow_run_context = None
    memory_manager = None  # For finally block memory snapshot
    training_success = False
    failure_notified = False

    try:
        logger.info("Starting LLM Training")
        logger.debug(f"[RUN_TRAINING:START] config={config_path}, resume={resume}, run_id={run_id}")

        # =====================================================================
        # 1. LOAD CONFIGURATION
        # =====================================================================
        # DEBUG: Check config consistency on remote
        try:
            from src.utils import config as config_module

            logger.info(f"DEBUG: Config module: {config_module.__file__}")
            if hasattr(config_module, "VALID_START_STRATEGIES"):
                logger.info(f"DEBUG: Valid strategies: {config_module.VALID_START_STRATEGIES}")
            else:
                logger.info("DEBUG: VALID_START_STRATEGIES not found in config module")
        except Exception as e:
            logger.warning(f"DEBUG: Failed to inspect config module: {e}")

        config = load_config(Path(config_path))
        strategies = config.training.get_strategy_chain()

        logger.info("Training config loaded")
        logger.info(f"   Model: {config.model.name}")
        logger.info(f"   Training type: {config.training.type}")
        logger.info(f"   4-bit quantization: {config.training.get_effective_load_in_4bit()}")
        logger.info(f"   Strategies: {' -> '.join(s.strategy_type.upper() for s in strategies)}")
        logger.info(f"   Multi-phase: {config.training.is_multi_phase()}")

        # =====================================================================
        # 2. SETUP EXPERIMENT TRACKING (MLflow) - EARLY
        # =====================================================================
        mlflow_mgr = _setup_mlflow(config)

        if mlflow_mgr and mlflow_mgr.is_active:
            # Enable autologging for Transformers
            mlflow_mgr.enable_autolog(log_models=False)

            # Ensure system metrics are ENABLED for the Provider (GPU)
            # This is critical for monitoring GPU usage during training
            # Note: accessing protected member for config check as specific property not exposed in interface
            mlflow_config = getattr(mlflow_mgr, "_mlflow_config", None)
            if mlflow_config and not mlflow_config.system_metrics_callback_enabled:
                logger.warning("⚠️ System metrics callback is disabled in config! GPU monitoring will be missing.")
                logger.info("i To fix: set experiment_tracking.mlflow.system_metrics_callback_enabled = true")
            else:
                logger.info("✅ GPU System Metrics monitoring enabled for this Provider process")

        # =====================================================================
        # 3. START MLFLOW RUN (to log all events from start)
        # =====================================================================
        strategy_chain = "_".join(s.strategy_type for s in strategies)
        run_name = f"{config.model.name.split('/')[-1]}_{strategy_chain}_{datetime.now().strftime('%Y%m%d_%H%M')}"

        if mlflow_mgr and mlflow_mgr.is_active:
            mlflow_run_context = mlflow_mgr.start_run(run_name=run_name)
            mlflow_run_context.__enter__()

            # Set parent run ID tag for nested run structure (from Mac pipeline)
            parent_run_id = os.environ.get("MLFLOW_PARENT_RUN_ID")
            if parent_run_id:
                mlflow_mgr.set_tags({"mlflow.parentRunId": parent_run_id})
                logger.info(f"✅ Linked to parent run: {parent_run_id}")

            # Log Docker image SHA for reproducibility (Layer Caching Strategy)
            docker_image_sha = os.environ.get("DOCKER_IMAGE_SHA")
            if docker_image_sha:
                mlflow_mgr.set_tags(
                    {"docker.image.sha": docker_image_sha, "docker.strategy": "layer_caching_immutable"}
                )
                logger.info(f"📌 Docker image SHA: {docker_image_sha[:TRUNCATE_SHA_DISPLAY]}...")

            # Log initial event
            mlflow_mgr.log_event_start(
                "Training pipeline started",
                category=CATEGORY_TRAINING,
                source=SOURCE_RUN_TRAINING,
            )

            # Log config info as event
            mlflow_mgr.log_event_info(
                f"Config loaded: {config.model.name}",
                category=CATEGORY_TRAINING,
                source=SOURCE_RUN_TRAINING,
                model=config.model.name,
                training_type=config.training.type,
                strategies=[s.strategy_type for s in strategies],
            )

            # Log training config
            mlflow_mgr.log_training_config(config)
            mlflow_mgr.log_artifact(config_path)

        # =====================================================================
        # 4. LOG ENVIRONMENT (for reproducibility)
        # =====================================================================
        env_reporter = EnvironmentReporter.collect()
        env_reporter.log_summary()
        logger.debug("[RUN_TRAINING:ENV] Environment snapshot collected")

        # Log environment to MLflow
        if mlflow_mgr and mlflow_mgr.is_active:
            mlflow_mgr.log_environment(env_reporter.snapshot.to_dict())

        # =====================================================================
        # 5. CREATE OR USE CONTAINER (Dependency Injection)
        # =====================================================================
        if container is None:
            container = TrainingContainer(config)
            logger.debug("[RUN_TRAINING:CONTAINER] Created new TrainingContainer")
        else:
            logger.debug("[RUN_TRAINING:CONTAINER] Using injected TrainingContainer")

        notifier = container.completion_notifier

        # =====================================================================
        # 6. GET MEMORY MANAGER WITH CALLBACKS + LOG GPU DETECTION
        # =====================================================================
        # Create MemoryManager with MLflow callbacks for event logging
        memory_manager = container.create_memory_manager_with_callbacks(mlflow_mgr)

        # Log GPU Info
        if memory_manager.gpu_info:
            gpu = memory_manager.gpu_info
            logger.info(f"GPU: {gpu.name}")
            logger.info(f"   Tier: {gpu.tier.value}")
            logger.info(f"   VRAM: {gpu.total_memory_gb:.1f} GB")
        else:
            logger.info("GPU: Unknown/CPU")

        # Log recommendations to MLflow with config comparison
        if mlflow_mgr and mlflow_mgr.is_active and memory_manager.gpu_info:
            gpu = memory_manager.gpu_info
            # Explicitly log GPU detection event with structured data for Report Generator
            # Note: memory_manager.gpu_info is typed as Any in protocol due to circular imports,
            # but at runtime it is a GPUInfo object.
            mlflow_mgr.log_gpu_detection(
                name=gpu.name,
                vram_gb=gpu.total_memory_gb,
                tier=gpu.tier.value,
            )

            # Extract actual model size from model name (e.g., "Qwen2.5-0.5B-Instruct" -> "0.5B")
            model_size = _extract_model_size(config.model.name)

            # Log actual model size as param for report filtering
            mlflow_mgr.log_params({"mm.actual_model_size": model_size})

            # Log MemoryManager configuration parameters for report display
            if memory_manager.preset:
                preset = memory_manager.preset
                mlflow_mgr.log_params(
                    {
                        "mm.memory_margin_mb": preset.memory_margin_mb,
                        "mm.critical_threshold": preset.critical_threshold,
                        "mm.warning_threshold": preset.warning_threshold,
                        "mm.max_retries": preset.max_retries,
                    }
                )
                logger.debug(
                    f"[MM:CONFIG] memory_margin={preset.memory_margin_mb}MB, "
                    f"thresholds={preset.critical_threshold}/{preset.warning_threshold}%, "
                    f"max_retries={preset.max_retries}"
                )

        # Log initial memory state (before model loading)
        if mlflow_mgr and mlflow_mgr.is_active:
            mem_stats = memory_manager.get_memory_stats()
            if mem_stats:
                mlflow_mgr.log_memory_snapshot(
                    phase="pre_model_load",
                    used_mb=mem_stats.used_mb,
                    free_mb=mem_stats.free_mb,
                    total_mb=mem_stats.total_mb,
                    utilization_percent=mem_stats.utilization_percent,
                )

        # =====================================================================
        # 7. LOAD MODEL AND TOKENIZER
        # =====================================================================
        import time

        model_load_start = time.time()
        model, tokenizer = container.load_model_and_tokenizer()
        model_load_duration = time.time() - model_load_start

        # Calculate model parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        trainable_percent = (trainable_params / total_params * 100) if total_params > 0 else 0

        logger.info(f"Model loaded in {model_load_duration:.1f}s")
        logger.info(f"   Total parameters: {total_params:,}")
        logger.info(f"   Trainable parameters: {trainable_params:,} ({trainable_percent:.2f}%)")

        # Log memory state after model loading
        if mlflow_mgr and mlflow_mgr.is_active:
            mem_stats = memory_manager.get_memory_stats()
            if mem_stats:
                mlflow_mgr.log_memory_snapshot(
                    phase="post_model_load",
                    used_mb=mem_stats.used_mb,
                    free_mb=mem_stats.free_mb,
                    total_mb=mem_stats.total_mb,
                    utilization_percent=mem_stats.utilization_percent,
                )

            # Log model info as params and event
            mlflow_mgr.log_params(
                {
                    "model.loading_time_seconds": round(model_load_duration, 2),
                    "model.total_parameters": total_params,
                    "model.trainable_parameters": trainable_params,
                    "model.trainable_percent": round(trainable_percent, 2),
                }
            )

        if mlflow_mgr and mlflow_mgr.is_active:
            mlflow_mgr.log_event_info(
                f"Model loaded: {trainable_params:,} trainable params ({model_load_duration:.1f}s)",
                category="training",
                source="run_training",
                model_loading_time_seconds=model_load_duration,
                total_parameters=total_params,
                trainable_parameters=trainable_params,
                trainable_percent=trainable_percent,
            )

        # =====================================================================
        # 8. CREATE ORCHESTRATOR (with MLflow manager)
        # =====================================================================
        logger.info("Creating StrategyOrchestrator...")
        orchestrator = container.create_orchestrator(
            model,
            tokenizer,
            mlflow_manager=mlflow_mgr,
        )
        logger.info("StrategyOrchestrator ready")

        # Log additional params
        if mlflow_mgr and mlflow_mgr.is_active:
            mlflow_mgr.set_tags(
                {
                    "run_id": run_id or "none",
                    "resume": str(resume),
                }
            )

        # =====================================================================
        # 9. RUN TRAINING CHAIN
        # =====================================================================
        logger.info(f"Running training chain: {' -> '.join(s.strategy_type.upper() for s in strategies)}")

        if mlflow_mgr and mlflow_mgr.is_active:
            mlflow_mgr.log_pipeline_initialized(
                run_id=run_id or generate_run_name()[0],
                total_phases=len(strategies),
                strategy_chain=[s.strategy_type for s in strategies],
            )

        result = orchestrator.run_chain(
            strategies=strategies,
            resume=resume,
            run_id=run_id,
        )

        # =====================================================================
        # 10. HANDLE RESULT
        # =====================================================================
        if result.is_failure():
            error_msg = result.unwrap_err()  # type: ignore[union-attr]
            logger.error(f"Training failed: {error_msg}")

            if mlflow_mgr and mlflow_mgr.is_active:
                mlflow_mgr.set_tag("status", "failed")
                mlflow_mgr.log_params({"error": str(error_msg)[:TRUNCATE_ERROR_MSG]})
                mlflow_mgr.log_event_error(
                    f"Training failed: {error_msg}",
                    category=CATEGORY_TRAINING,
                    source=SOURCE_RUN_TRAINING,
                )

            notifier.notify_failed(
                str(error_msg),
                {
                    "error_type": "TrainingError",
                    "model": config.model.name,
                    "strategies": [s.strategy_type for s in strategies],
                },
            )
            failure_notified = True

            raise RuntimeError(f"Training failed: {error_msg}")

        _ = result.unwrap()
        training_success = True
        logger.info("Training chain completed successfully!")

        # =====================================================================
        # 11. GET OUTPUT PATH + REGISTER MODEL
        # =====================================================================
        if orchestrator.buffer:
            last_phase_idx = len(strategies) - 1
            output_path = Path(orchestrator.buffer.get_phase_output_dir(last_phase_idx)) / "checkpoint-final"
        else:
            # Fallback: derive expected last phase output dir (same naming as DataBuffer).
            last_phase_idx = len(strategies) - 1
            last_phase = strategies[last_phase_idx]
            output_path = Path("output") / f"phase_{last_phase_idx}_{last_phase.strategy_type}" / "checkpoint-final"

        if mlflow_mgr and mlflow_mgr.is_active:
            mlflow_mgr.set_tag("status", "completed")
            mlflow_mgr.log_params({"output_path": str(output_path)})
            mlflow_mgr.log_event_complete(
                "Training completed successfully",
                category="training",
                source="run_training",
                output_path=str(output_path),
            )

            model_name = config.model.name.split("/")[-1].replace(".", "-").lower()
            mlflow_mgr.register_model(
                model_name=f"helix-{model_name}",
                alias="latest",
                tags={"strategy_chain": strategy_chain},
            )

        # =====================================================================
        # 12. NOTIFY SUCCESS
        # =====================================================================
        logger.info("Training completed successfully!")
        logger.info(f"Output: {output_path}")
        logger.info(f"Strategies: {' -> '.join(s.strategy_type for s in strategies)}")
        logger.info(f"Total phases: {len(strategies)}")

        notifier.notify_complete(
            {
                "output_path": str(output_path),
                "model_name": config.model.name,
                "strategies": [s.strategy_type for s in strategies],
                "total_phases": len(strategies),
                "run_id": orchestrator.buffer.run_id if orchestrator.buffer else None,
            }
        )

        return output_path

    except Exception as e:
        logger.exception(f"Unexpected error during training: {e}")

        # Log error event to MLflow
        if mlflow_mgr and mlflow_mgr.is_active:
            mlflow_mgr.log_event_error(
                f"Training failed: {e!s}",
                category="training",
                source="run_training",
                error_type=type(e).__name__,
            )

        if notifier is not None and not failure_notified:
            with contextlib.suppress(Exception):
                notifier.notify_failed(
                    f"Unexpected error: {e!s}",
                    {"error_type": type(e).__name__},
                )

        raise

    finally:
        # =====================================================================
        # GENERATE TRAINING SUMMARY (always runs)
        # =====================================================================
        # Log final memory state
        if mlflow_mgr and mlflow_mgr.is_active and memory_manager:
            mem_stats = memory_manager.get_memory_stats()
            if mem_stats:
                mlflow_mgr.log_memory_snapshot(
                    phase="training_complete",
                    used_mb=mem_stats.used_mb,
                    free_mb=mem_stats.free_mb,
                    total_mb=mem_stats.total_mb,
                    utilization_percent=mem_stats.utilization_percent,
                )

        if mlflow_mgr and mlflow_mgr.is_active:
            try:
                # Log training_events.json to PARENT run (pipeline_* on Mac)
                # All artifacts should be centralized in parent for easy access
                mac_parent_run_id = os.environ.get("MLFLOW_PARENT_RUN_ID")
                mlflow_mgr.log_summary_artifact(
                    events_artifact_name="training_events.json",
                    parent_run_id=mac_parent_run_id,  # Log to parent (Mac pipeline run)
                )
                if mac_parent_run_id:
                    logger.info(f"Training summary logged to parent run: {mac_parent_run_id[:8]}...")
                else:
                    logger.info("Training summary logged to MLflow (current run)")

            except Exception as summary_error:
                logger.warning(f"Failed to generate training summary: {summary_error}")

            # End run with explicit status
            run_status = "FINISHED" if training_success else "FAILED"
            mlflow_mgr.end_run(status=run_status)

        # Cleanup MLflow context
        if mlflow_run_context:
            with contextlib.suppress(Exception):
                mlflow_run_context.__exit__(None, None, None)

        if mlflow_mgr:
            mlflow_mgr.cleanup()


# =============================================================================
# BACKWARD COMPATIBILITY
# =============================================================================

# Alias for backward compatibility with existing code
train_v2 = run_training


def _install_crash_observability() -> None:
    """
    Install faulthandler + atexit logging flush so silent deaths leave a trace.

    Why this exists
    ---------------
    On remote training (RunPod), training crashes have repeatedly left zero
    evidence in ``training.log``: no Python traceback, no error message, just
    a truncated progress bar. Three failure modes bypass normal logging:

    1. **Native crash** (SEGV/ABRT/BUS/FPE/ILL) from a C extension
       (bitsandbytes, flash-attn, torch, CUDA kernels). Python never gets a
       chance to raise — only the OS knows.
    2. **Signal kill** (SIGTERM from OOM-killer-adjacent actors, SIGKILL from
       cgroup, driver-initiated aborts).
    3. **Block-buffered stderr** on ``exec >file 2>&1`` shell redirect. Any
       tail of stderr that wasn't flushed before the crash is simply lost.

    What we install
    ---------------
    - ``faulthandler.enable(file=..., all_threads=True)``: CPython's built-in
      native-crash handler. On fatal signals it writes Python + C stack frames
      of *all* threads directly via ``write(2)`` — it survives a Python
      runtime crash because it doesn't go through the logging stack.

      We try a persistent sibling file (``training.faulthandler.log`` next to
      ``training.log``) so the monitor can fetch it post-mortem. Path is taken
      from ``PYTHONFAULTHANDLER_PATH`` env var (set by the bash wrapper in
      ``deployment_manager._start_training_cloud``). If opening the file
      fails, we fall back to stderr — ``faulthandler`` remains active either
      way.

    - ``atexit`` flush of all logging handlers: on *any* normal exit path
      (including ``sys.exit``, ``return``, or a Python exception that reaches
      ``main()``), ensure the tail of ``training.log`` is written to disk
      before Python tears down. Prevents "last 5 log lines lost" on crash.

    Best-effort: this function never raises. Observability must never
    prevent training from starting.
    """
    import atexit
    import faulthandler

    fault_log_path = os.environ.get("PYTHONFAULTHANDLER_PATH", "training.faulthandler.log")
    try:
        # Line-buffered so each write hits disk as it happens. Kept open for the
        # lifetime of the process — faulthandler writes to this fd on SIGSEGV.
        fault_log = Path(fault_log_path).open("w", buffering=1)  # noqa: SIM115
        faulthandler.enable(file=fault_log, all_threads=True)
        logger.debug(f"[RUN_TRAINING:OBSERVABILITY] faulthandler enabled → {fault_log_path}")
    except OSError as exc:
        # Fall back to stderr — still captures native crashes.
        with contextlib.suppress(Exception):  # pragma: no cover — faulthandler is stdlib
            faulthandler.enable(all_threads=True)
        logger.warning(
            f"[RUN_TRAINING:OBSERVABILITY] could not open {fault_log_path} "
            f"({exc}); faulthandler redirected to stderr",
        )
    except Exception as exc:  # pragma: no cover — defensive
        logger.warning(f"[RUN_TRAINING:OBSERVABILITY] faulthandler.enable failed: {exc}")

    def _flush_logging_handlers() -> None:
        """Flush every handler attached to the training logger on exit."""
        with contextlib.suppress(Exception):
            for handler in list(logger.handlers):
                with contextlib.suppress(Exception):
                    handler.flush()

    try:
        atexit.register(_flush_logging_handlers)
    except Exception as exc:  # pragma: no cover
        logger.warning(f"[RUN_TRAINING:OBSERVABILITY] atexit.register failed: {exc}")


def main() -> int:
    """CLI entry point."""
    # Crash observability MUST be installed before argparse / any heavy import
    # that may itself segfault (bitsandbytes, flash-attn). See
    # _install_crash_observability() docstring.
    _install_crash_observability()

    parser = argparse.ArgumentParser(
        description="Multi-Phase LLM Training with StrategyOrchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Single-phase training
    python -m src.training.run_training --config config/pipeline_config.yaml

    # With DEBUG logs
    LOG_LEVEL=DEBUG python -m src.training.run_training --config config/pipeline_config.yaml

    # Resume interrupted training
    python -m src.training.run_training --config config/pipeline_config.yaml --resume --run-id run_xxx

Debug log tags:
    [RUN_TRAINING:] - This script
    [SO:]           - StrategyOrchestrator
    [SF:]           - StrategyFactory
    [TF:]           - TrainerFactory
    [DB:]           - DataBuffer
    [MM:]           - MemoryManager
    [CFG:]          - Config
    [CONTAINER:]    - TrainingContainer
        """,
    )

    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="Path to configuration file",
    )

    parser.add_argument(
        "--resume",
        "-r",
        action="store_true",
        help="Resume from last incomplete phase",
    )

    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Run ID for resume or reproducibility",
    )

    args = parser.parse_args()

    # Propagate HF_HUB_* keys from secrets.env to os.environ before training starts.
    # pydantic-settings does not write env-file values to os.environ automatically, so
    # variables like HF_HUB_DISABLE_XET=1 would be silently ignored without this step.
    # setdefault() preserves any values already set in the environment.
    try:
        from src.config.secrets import load_secrets as _load_secrets
        _secrets = _load_secrets()
        for _key, _val in (_secrets.model_extra or {}).items():
            if _key.startswith("HF_HUB_") and isinstance(_val, str):
                os.environ.setdefault(_key, _val)
    except Exception:
        pass  # never block training due to secrets propagation

    try:
        output_path = run_training(
            config_path=args.config,
            resume=args.resume,
            run_id=args.run_id,
        )
        logger.info(f"Training completed: {output_path}")
        return 0
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
        return EXIT_KEYBOARD_INTERRUPT
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
