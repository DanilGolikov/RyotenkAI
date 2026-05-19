"""
Multi-Phase LLM Training Entry Point (Pattern A, post Phase M4).

Main entry point for running LLM training with StrategyOrchestrator.
Supports single-phase and multi-phase training pipelines.

Pattern A migration (Phase M4)
------------------------------
The trainer subprocess NO LONGER opens its own top-level MLflow run.
Instead, the control plane opens the parent run and exports the
following env vars before launching this process (see
``ryotenkai_control.pipeline.stages.managers.deployment.training_launcher``):

* ``MLFLOW_TRACKING_URI``
* ``MLFLOW_RUN_ID``    -- the parent attempt run id to adopt
* ``MLFLOW_NESTED_RUN`` -- literal ``"TRUE"`` (R-29)
* ``MLFLOW_EXPERIMENT_NAME``
* ``MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING`` -- ``"true"``

The HF :class:`transformers.integrations.MLflowCallback` adopts those
env vars and creates a structurally-nested child of the parent. The
trainer:

* Validates the env up-front via :meth:`HFMlflowWiring.validate_env`.
* Configures :class:`TrainingArguments` via
  :meth:`HFMlflowWiring.configure_training_args`.
* (M5-followup) Registers the trained model via :class:`ModelPublisher`
  after the chain completes.

Strong assets preserved:

* :class:`RunnerEventCallback` continues emitting typed envelopes onto
  the runner bus (ADR-0009 SSOT journal).
* ``_flush_helper.py`` is kept and still consulted by the cancellation /
  completion callbacks.

Components used:

* MemoryManager: OOM protection and GPU management
* StrategyOrchestrator: Multi-phase training coordination
* StrategyFactory: Strategy creation (CPT, SFT, CoT, DPO, ORPO)
* TrainerFactory: TRL trainer creation
* DataBuffer: Checkpoint and state management

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
from pathlib import Path
from typing import TYPE_CHECKING

from ryotenkai_pod.trainer.constants import (
    EXIT_KEYBOARD_INTERRUPT,
)
from ryotenkai_pod.trainer.container import TrainingContainer
from ryotenkai_pod.trainer.mlflow.hf_wiring import HFMlflowWiring
from ryotenkai_shared.config.loader import load_pipeline_config
from ryotenkai_shared.errors import ConfigInvalidError, RyotenkAIError
from ryotenkai_shared.utils.environment import EnvironmentReporter
from ryotenkai_shared.utils.logger import logger

if TYPE_CHECKING:
    pass


def _derive_model_family(model_name: str) -> str:
    """Derive a short, registry-safe family slug from an HF model id.

    The template ``ryotenkai/{experiment}/{model_family}`` needs a
    second placeholder that is both human-readable and stable across
    fine-tunes of the same base. We keep it simple: take everything
    after the last ``/`` in the HF id, lowercase, and replace any
    character outside ``[a-z0-9._-]`` with a dash. Empty / pathological
    inputs degrade to ``"model"``.

    Examples:

        ``"Qwen/Qwen2.5-0.5B-Instruct"`` -> ``"qwen2.5-0.5b-instruct"``
        ``"meta-llama/Llama-3.2-1B"``    -> ``"llama-3.2-1b"``
    """
    import re

    if not model_name:
        return "model"
    tail = model_name.rsplit("/", 1)[-1].lower()
    cleaned = re.sub(r"[^a-z0-9._-]+", "-", tail).strip("-")
    return cleaned or "model"


def _publish_trained_model(
    *,
    config: object,
    trainer_model: object,
    tokenizer: object,
    output_path: Path,
) -> None:
    """Log the trained transformer and register it under an alias.

    Two-step Pattern A publish:

    1. :func:`mlflow.transformers.log_model` uploads the model +
       tokenizer to ``runs:/{run_id}/model`` with
       ``save_pretrained=True`` (R-21), so the artifact contains the
       Hugging Face directory layout consumers expect.
    2. :meth:`ModelPublisher.publish` registers that URI and attaches
       the success alias (default ``challenger``, configurable via
       ``MLFLOW_ALIAS_ON_SUCCESS``).

    All failures are caught and logged at WARNING — training succeeded
    by the time we reach this helper; a registry hiccup must not flip
    the pipeline status to FAILED. Operators can re-publish manually
    via ``ryotenkai model promote`` once the issue is resolved.

    :param config: Loaded :class:`PipelineConfig` (carrying
        ``model.name`` and ``integrations.mlflow``).
    :param trainer_model: Final trained model returned by the
        orchestrator (passed to :func:`mlflow.transformers.log_model`).
    :param tokenizer: Tokenizer paired with ``trainer_model``.
    :param output_path: Local checkpoint directory (informational only;
        the artifact comes from the live model handle).
    """
    try:
        import mlflow  # noqa: PLC0415 — heavy, lazy
    except Exception as exc:  # pragma: no cover -- defensive
        logger.warning(
            "[RUN_TRAINING:PUBLISH] mlflow not importable, skipping publish: %s",
            exc,
        )
        return

    # Pull the live nested run id the HF MLflowCallback opened. Fall
    # back to ``MLFLOW_RUN_ID`` only when there is no active run --
    # using the env value directly when the callback is still open
    # would register against the parent attempt run instead of the
    # nested child where the metrics live.
    active = mlflow.active_run()
    nested_run_id: str | None
    if active is not None:
        nested_run_id = active.info.run_id
    else:
        nested_run_id = os.environ.get("MLFLOW_RUN_ID") or None
    if not nested_run_id:
        logger.warning(
            "[RUN_TRAINING:PUBLISH] no active mlflow run; skipping publish",
        )
        return

    # Resolve registry name from the project template.
    mlflow_cfg = getattr(getattr(config, "integrations", None), "mlflow", None)
    if mlflow_cfg is None:
        logger.warning(
            "[RUN_TRAINING:PUBLISH] integrations.mlflow not configured; "
            "skipping registry publish",
        )
        return

    template = getattr(
        mlflow_cfg, "model_registry_name_template",
        "ryotenkai/{experiment}/{model_family}",
    )
    experiment = getattr(mlflow_cfg, "experiment_name", "default")
    model_family = _derive_model_family(getattr(config.model, "name", ""))
    registered_name = template.format(
        experiment=experiment, model_family=model_family,
    )

    # Caller may override the alias via env (training_launcher passes
    # ``MLFLOW_ALIAS_ON_SUCCESS`` derived from
    # ``mlflow_cfg.alias_on_success`` in M6). Default to the project
    # config value, with a final ``challenger`` fallback.
    alias = (
        os.environ.get("MLFLOW_ALIAS_ON_SUCCESS")
        or getattr(mlflow_cfg, "alias_on_success", "challenger")
        or "challenger"
    )

    artifact_path = "model"

    # Step 1: log the transformer + tokenizer under the active run.
    # ``mlflow.transformers.log_model`` honours the active fluent run
    # (no need to pass run_id explicitly). ``save_pretrained=True``
    # writes the HF on-disk format (R-21) so consumers can load with
    # ``AutoModel.from_pretrained(<artifact_uri>)``.
    try:
        from mlflow import transformers as mlflow_transformers  # noqa: PLC0415

        mlflow_transformers.log_model(
            transformers_model={"model": trainer_model, "tokenizer": tokenizer},
            artifact_path=artifact_path,
            save_pretrained=True,
        )
        logger.info(
            "[RUN_TRAINING:PUBLISH] artifact logged run_id=%s path=%s",
            nested_run_id, artifact_path,
        )
    except Exception as exc:
        logger.warning(
            "[RUN_TRAINING:PUBLISH] mlflow.transformers.log_model failed: %s "
            "(output_path=%s)",
            exc, output_path,
        )
        return

    # Step 2: register + alias via :class:`ModelPublisher`. The
    # registry is constructed against the same tracking URI MlflowTransport
    # stamped at start-up — we read it back from ``mlflow.get_tracking_uri``
    # rather than re-resolving the project config.
    try:
        from ryotenkai_pod.trainer.mlflow.model_publisher import ModelPublisher
        from ryotenkai_shared.infrastructure.mlflow.registry import (
            MlflowModelRegistry,
        )

        tracking_uri = mlflow.get_tracking_uri()
        registry = MlflowModelRegistry(tracking_uri=tracking_uri)
        publisher = ModelPublisher(registry=registry)
        version = publisher.publish(
            run_id=nested_run_id,
            artifact_path=artifact_path,
            registered_name=registered_name,
            alias_on_success=alias,
        )
        logger.info(
            "[RUN_TRAINING:PUBLISH] registered name=%s version=%s alias=%s",
            registered_name, version.version, alias,
        )
    except Exception as exc:
        logger.warning(
            "[RUN_TRAINING:PUBLISH] ModelPublisher.publish failed for "
            "name=%s alias=%s: %s",
            registered_name, alias, exc,
        )


def _emit_training_failed(
    *,
    exc: BaseException,
    step: int | None = None,
    error_type: str | None = None,
) -> None:
    """Best-effort fan-out of a :class:`TrainingFailedEvent` to the runner.

    Creates a short-lived :class:`RunnerEventCallback` purely to push a
    typed ``training.failed`` envelope onto the runner bus. The callback
    no-ops when ``RYOTENKAI_RUNNER_URL`` is unset (standalone trainer
    runs), so this helper is safe to call unconditionally.

    All disk / network errors are swallowed: a broken event-push must
    not mask the original training failure.
    """
    try:
        from ryotenkai_pod.trainer.callbacks.runner_event_callback import (
            RunnerEventCallback,
        )

        cb = RunnerEventCallback()
        try:
            cb.emit_training_failed(
                exc=exc,
                step=step,
                error_type=error_type,
            )
        finally:
            # Drain whatever was enqueued, then signal the worker to
            # stop and close the httpx client. The 5s deadline is
            # generous for a single envelope on a loopback HTTP path.
            with contextlib.suppress(Exception):
                cb._drain_with_deadline(deadline_s=5.0)
            cb._stop_evt.set()
            if cb._worker is not None:
                with contextlib.suppress(Exception):
                    cb._worker.join(timeout=1.0)
            cb._close_client()
    except Exception as emit_exc:  # pragma: no cover -- defensive
        logger.warning(
            f"[RUN_TRAINING:FAILED-EVENT] emit_training_failed suppressed: {emit_exc}",
        )


def run_training(
    config_path: str,
    *,
    resume: bool = False,
    run_id: str | None = None,
    container: TrainingContainer | None = None,
) -> Path:
    """
    Main training function using StrategyOrchestrator under Pattern A.

    Pattern A: the trainer subprocess does NOT open its own MLflow run.
    The control plane has already opened the parent (attempt) run; we
    validate env, configure HF wiring, and let the HF MLflowCallback
    create the nested child automatically.

    Args:
        config_path: Path to pipeline configuration file.
        resume: If True, resume from last incomplete phase.
        run_id: Optional run ID for resume or reproducibility.
        container: Optional pre-configured TrainingContainer (for testing).

    Returns:
        Path to final model checkpoint.

    Raises:
        RuntimeError: If training fails.
        ConfigInvalidError: If Pattern A env vars are missing.
    """
    memory_manager = None  # For finally block memory snapshot
    training_success = False
    # Phase 3 (pre-Phase-3 fix): ensures the typed
    # ``TrainingFailedEvent`` is emitted at most once per ``run_training``
    # invocation even though the inner ``except RyotenkAIError`` rewraps
    # to ``RuntimeError`` and is then re-caught by the outer
    # ``except Exception``.
    training_failed_emitted = False

    # Determine whether Pattern A env is fully wired. We do NOT
    # fail-fast in standalone trainer runs (RYOTENKAI_RUNNER_URL absent
    # or MLFLOW_RUN_ID absent) -- the trainer must still run from a
    # bare CLI invocation for local debugging. The validate_env call
    # below guards only the cloud / control-plane-launched path.
    pattern_a_active = bool(os.environ.get("MLFLOW_RUN_ID", "").strip())

    try:
        logger.info("Starting LLM Training")
        logger.debug(
            f"[RUN_TRAINING:START] config={config_path}, resume={resume}, "
            f"run_id={run_id}, pattern_a_active={pattern_a_active}",
        )

        # =====================================================================
        # 1. LOAD CONFIGURATION
        # =====================================================================
        config = load_pipeline_config(Path(config_path))
        print(
            "[TRAINER:M3] Config loaded, entering heavy-init chain",
            file=sys.stderr,
            flush=True,
        )
        strategies = config.training.get_strategy_chain()

        logger.info("Training config loaded")
        logger.info(f"   Model: {config.model.name}")
        logger.info(f"   Training type: {config.training.adapter.kind}")
        logger.info(
            f"   4-bit quantization: {config.training.get_effective_load_in_4bit()}",
        )
        logger.info(
            "   Strategies: "
            + " -> ".join(s.strategy_type.upper() for s in strategies),
        )
        logger.info(f"   Multi-phase: {config.training.is_multi_phase()}")

        # =====================================================================
        # 2. PATTERN A: VALIDATE MLFLOW ENV (control plane sets these)
        # =====================================================================
        # When the trainer is launched by the control plane, every
        # ``MLFLOW_*`` env var must be present. Standalone trainer
        # invocations (local dev) skip this check.
        if pattern_a_active:
            try:
                HFMlflowWiring.validate_env()
                logger.info(
                    "Pattern A active: parent run id=%s, experiment=%s",
                    os.environ["MLFLOW_RUN_ID"],
                    os.environ["MLFLOW_EXPERIMENT_NAME"],
                )
            except ConfigInvalidError:
                logger.exception(
                    "Pattern A env validation failed; trainer cannot "
                    "proceed because the control plane expected a "
                    "nested MLflow child.",
                )
                raise

        # =====================================================================
        # 3. LOG ENVIRONMENT (for reproducibility)
        # =====================================================================
        env_reporter = EnvironmentReporter.collect()
        env_reporter.log_summary()
        logger.debug("[RUN_TRAINING:ENV] Environment snapshot collected")

        # =====================================================================
        # 4. CREATE OR USE CONTAINER (Dependency Injection)
        # =====================================================================
        if container is None:
            container = TrainingContainer(config)
            logger.debug("[RUN_TRAINING:CONTAINER] Created new TrainingContainer")
        else:
            logger.debug("[RUN_TRAINING:CONTAINER] Using injected TrainingContainer")

        # =====================================================================
        # 5. GET MEMORY MANAGER + LOG GPU DETECTION (no MLflow callbacks)
        # =====================================================================
        # Pattern A: no MLflow callbacks attached -- HF MLflowCallback
        # writes the metrics to the nested child it owns. MemoryManager
        # is constructed with ``mlflow_manager=None`` so any internal
        # mlflow-wiring code paths short-circuit.
        memory_manager = container.create_memory_manager_with_callbacks()

        if memory_manager.gpu_info:
            gpu = memory_manager.gpu_info
            logger.info(f"GPU: {gpu.name}")
            logger.info(f"   Tier: {gpu.tier.value}")
            logger.info(f"   VRAM: {gpu.total_memory_gb:.1f} GB")
        else:
            logger.info("GPU: Unknown/CPU")

        # =====================================================================
        # 6. LOAD MODEL AND TOKENIZER
        # =====================================================================
        import time

        model_load_start = time.time()
        model, tokenizer = container.load_model_and_tokenizer()
        model_load_duration = time.time() - model_load_start

        # Calculate model parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        trainable_percent = (
            (trainable_params / total_params * 100) if total_params > 0 else 0
        )

        logger.info(f"Model loaded in {model_load_duration:.1f}s")
        logger.info(f"   Total parameters: {total_params:,}")
        logger.info(
            f"   Trainable parameters: {trainable_params:,} "
            f"({trainable_percent:.2f}%)",
        )

        # =====================================================================
        # 7. CREATE ORCHESTRATOR (no MLflow manager under Pattern A)
        # =====================================================================
        logger.info("Creating StrategyOrchestrator...")
        # Phase M5: ``HFMlflowWiring.configure_training_args`` is invoked
        # inside :class:`TrainerFactory.create` per-phase, right after
        # the ``TrainingArguments`` object is built. This guarantees
        # ``report_to=["mlflow"]`` and the per-rank
        # ``mlflow.set_system_metrics_node_id`` are applied to every
        # strategy in the chain, not only the first. The hook short-
        # circuits when ``MLFLOW_RUN_ID`` is unset (standalone trainer
        # runs / local dev), so the same trainer binary covers both
        # control-plane-launched and bare CLI paths.
        orchestrator = container.create_orchestrator(
            model,
            tokenizer,
        )
        logger.info("StrategyOrchestrator ready")

        # =====================================================================
        # 8. RUN TRAINING CHAIN
        # =====================================================================
        logger.info(
            "Running training chain: "
            + " -> ".join(s.strategy_type.upper() for s in strategies),
        )

        try:
            trained_model = orchestrator.run_chain(
                strategies=strategies,
                resume=resume,
                run_id=run_id,
            )
        except RyotenkAIError as exc:
            error_msg = exc.detail or str(exc)
            logger.error(f"Training failed: {error_msg}")

            # Typed envelope on the bus is the SSOT.
            _emit_training_failed(exc=exc)
            training_failed_emitted = True

            raise RuntimeError(f"Training failed: {error_msg}") from exc

        training_success = True
        logger.info("Training chain completed successfully!")

        # =====================================================================
        # 9. GET OUTPUT PATH
        # =====================================================================
        if orchestrator.buffer:
            last_phase_idx = len(strategies) - 1
            output_path = (
                Path(orchestrator.buffer.get_phase_output_dir(last_phase_idx))
                / "checkpoint-final"
            )
        else:
            # Fallback: derive expected last phase output dir.
            last_phase_idx = len(strategies) - 1
            last_phase = strategies[last_phase_idx]
            output_path = (
                Path("output")
                / f"phase_{last_phase_idx}_{last_phase.strategy_type}"
                / "checkpoint-final"
            )

        # =====================================================================
        # 10. PUBLISH MODEL VIA ALIASES (Pattern A, Phase M5)
        # =====================================================================
        # Two-step sequence:
        #
        #   a) ``mlflow.transformers.log_model(..., save_pretrained=True)``
        #      uploads the trained transformer + tokenizer under
        #      ``runs:/{run_id}/model`` (R-21).
        #   b) :meth:`ModelPublisher.publish` calls ``register_model``
        #      against that URI and sets ``alias_on_success`` (default
        #      ``challenger``). Promotion to ``champion`` is a manual
        #      operator action via ``ryotenkai model promote``.
        #
        # The HF MLflowCallback opens a nested child of the parent
        # attempt run; ``mlflow.active_run().info.run_id`` is that
        # child. We use it (rather than ``MLFLOW_RUN_ID``) so the
        # registered model URI points at the artifact the callback
        # actually wrote.
        #
        # All failures here are non-fatal -- training succeeded; the
        # operator can re-publish manually. Logs are loud so failures
        # are caught in CI smoke runs.
        if pattern_a_active and training_success:
            _publish_trained_model(
                config=config,
                trainer_model=trained_model if trained_model is not None else model,
                tokenizer=tokenizer,
                output_path=output_path,
            )

        # =====================================================================
        # 11. NOTIFY SUCCESS
        # =====================================================================
        logger.info("Training completed successfully!")
        logger.info(f"Output: {output_path}")
        logger.info(
            "Strategies: " + " -> ".join(s.strategy_type for s in strategies),
        )
        logger.info(f"Total phases: {len(strategies)}")

        return output_path

    except Exception as e:
        logger.exception(f"Unexpected error during training: {e}")

        # Phase 3 / Phase 6.b: emit typed ``TrainingFailedEvent`` for
        # unexpected exceptions. Skip if the inner ``RyotenkAIError``
        # branch already emitted -- that branch rewraps to
        # ``RuntimeError`` which is then caught here, so without the
        # flag we would emit twice.
        if not training_failed_emitted:
            _emit_training_failed(exc=e)
            training_failed_emitted = True

        raise

    finally:
        # =====================================================================
        # No MLflow teardown -- the HF MLflowCallback closes its own
        # nested child on ``Trainer`` exit, and the control plane
        # finalizes the parent run via ``MlflowFinalizer``.
        # =====================================================================
        if memory_manager is not None and hasattr(memory_manager, "cleanup"):
            with contextlib.suppress(Exception):
                memory_manager.cleanup()
        # Signal end of training_success state for logs.
        logger.info(
            "[RUN_TRAINING:DONE] training_success=%s pattern_a_active=%s",
            training_success,
            pattern_a_active,
        )


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
    evidence: no Python traceback, no error message, just a truncated
    progress bar. Three failure modes bypass normal logging:

    1. **Native crash** (SEGV/ABRT/BUS/FPE/ILL) from a C extension
       (bitsandbytes, flash-attn, torch, CUDA kernels). Python never gets a
       chance to raise — only the OS knows.
    2. **Signal kill** (SIGTERM from OOM-killer-adjacent actors, SIGKILL from
       cgroup, driver-initiated aborts).
    3. **Block-buffered stderr** on ``exec >file 2>&1`` shell redirect. Any
       tail of stderr that wasn't flushed before the crash is simply lost.

    What we install
    ---------------
    - ``faulthandler.enable(all_threads=True)``: CPython's built-in
      native-crash handler.
    - ``atexit`` flush of all logging handlers.

    Best-effort: this function never raises.
    """
    import atexit
    import faulthandler

    try:
        # Default ``file=sys.stderr`` — Supervisor pump captures it.
        faulthandler.enable(all_threads=True)
        logger.debug(
            "[RUN_TRAINING:OBSERVABILITY] faulthandler enabled (writes to stderr)",
        )
    except Exception as exc:  # pragma: no cover -- defensive
        logger.warning(
            f"[RUN_TRAINING:OBSERVABILITY] faulthandler.enable failed: {exc}",
        )

    def _flush_logging_handlers() -> None:
        """Flush every handler attached to the training logger on exit."""
        with contextlib.suppress(Exception):
            for handler in list(logger.handlers):
                with contextlib.suppress(Exception):
                    handler.flush()

    try:
        atexit.register(_flush_logging_handlers)
    except Exception as exc:  # pragma: no cover
        logger.warning(
            f"[RUN_TRAINING:OBSERVABILITY] atexit.register failed: {exc}",
        )


def main() -> int:
    """CLI entry point."""
    print(
        "[TRAINER:M1] Python interpreter started, argv parsed",
        file=sys.stderr,
        flush=True,
    )

    import time
    _trainer_started_at = time.monotonic()

    _install_crash_observability()

    parser = argparse.ArgumentParser(
        description="Multi-Phase LLM Training with StrategyOrchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
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
    try:
        from ryotenkai_shared.config.secrets import load_secrets as _load_secrets
        _secrets = _load_secrets()
        for _key, _val in (_secrets.model_extra or {}).items():
            if _key.startswith("HF_HUB_") and isinstance(_val, str):
                os.environ.setdefault(_key, _val)
    except Exception:
        pass  # never block training due to secrets propagation

    print(f"[TRAINER:M2] Loading config from {args.config}", file=sys.stderr, flush=True)

    from pathlib import Path as _Path
    _exit_workdir = _Path.cwd()

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
        try:
            from ryotenkai_pod.trainer.exit_reporter import write_failure_payload
            write_failure_payload(
                _exit_workdir, e,
                started_at=_trainer_started_at, exit_code=1,
            )
        except Exception:  # pragma: no cover -- defensive
            pass
        return 1


if __name__ == "__main__":
    sys.exit(main())
