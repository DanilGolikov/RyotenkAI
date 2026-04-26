"""One-shot wiring of every collaborator the orchestrator needs.

``PipelineOrchestrator.__init__`` used to carry ~120 LOC of instance-field
assignment in a very specific order (stages depend on validation_mgr which
depends on collectors, stage_planner depends on finalised stages, etc.).
That wiring is pure composition work — no state mutation, no lifecycle —
so it belongs in a dedicated bootstrap module where it can be
test-in-isolation.

:meth:`PipelineBootstrap.build` loads config + secrets, validates them via
:class:`StartupValidator`, then constructs every downstream collaborator
in dependency order. It returns a frozen :class:`BootstrapResult` — a
value object the orchestrator copies onto its own ``self.*`` fields.

Why a frozen result instead of constructing the orchestrator directly?
The orchestrator owns per-run mutable state (``_state_store``,
``_run_lock_guard``, ``attempt_directory``) that the bootstrap has no
business touching. Returning a plain data bag keeps the lifecycle
boundary clean: bootstrap produces a snapshot, orchestrator takes
ownership.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.pipeline.bootstrap.startup_validator import StartupValidator
from src.pipeline.config_drift import ConfigDriftValidator
from src.pipeline.context import ContextPropagator, PipelineContext, StageInfoLogger
from src.pipeline.execution import (
    RestartPointsInspector,
    StageExecutionLoop,
    StagePlanner,
    StageRegistry,
)
from src.pipeline.launch import LaunchPreparator
from src.pipeline.mlflow_attempt import MLflowAttemptManager
from src.pipeline.reporting import ExecutionSummaryReporter
from src.pipeline.stages import PipelineContextKeys
from src.pipeline.validation.artifact_manager import ValidationArtifactManager
from src.utils.config import load_config, load_secrets
from src.utils.logger import logger

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from src.config.runtime import RuntimeSettings
    from src.pipeline.artifacts import StageArtifactCollector
    from src.pipeline.domain import RunContext
    from src.pipeline.stages.base import PipelineStage
    from src.pipeline.state import AttemptController
    from src.utils.config import PipelineConfig, Secrets


@dataclass(frozen=True, slots=True)
class BootstrapResult:
    """Frozen snapshot of everything the orchestrator needs post-construction.

    All fields are constructed in :meth:`PipelineBootstrap.build`. The
    orchestrator's ``__init__`` unpacks this result onto its own instance
    fields — keeping backward compatibility for callers that read
    ``orch.config`` / ``orch.stages`` / etc.
    """

    config_path: Path
    config: PipelineConfig
    secrets: Secrets
    settings: RuntimeSettings
    run_ctx: RunContext
    context: PipelineContext
    collectors: dict[str, StageArtifactCollector]
    stages: list[PipelineStage]
    # Extracted collaborators — listed in dependency order matching
    # construction inside :meth:`PipelineBootstrap.build`.
    validation_artifact_mgr: ValidationArtifactManager
    context_propagator: ContextPropagator
    stage_info_logger: StageInfoLogger
    config_drift: ConfigDriftValidator
    summary_reporter: ExecutionSummaryReporter
    mlflow_attempt: MLflowAttemptManager
    registry: StageRegistry
    stage_planner: StagePlanner
    attempt_controller: AttemptController
    launch_preparator: LaunchPreparator
    restart_inspector: RestartPointsInspector
    stage_execution_loop: StageExecutionLoop


class PipelineBootstrap:
    """Factory for :class:`BootstrapResult` — runs exactly once per orchestrator.

    Side effects are limited to logging and env-var assignment (via
    :class:`StartupValidator.set_hf_token_env`). No pipeline state is
    created here — that happens per-run inside the orchestrator.
    """

    @classmethod
    def build(
        cls,
        *,
        config_path: Path,
        run_ctx: RunContext,
        settings: RuntimeSettings,
        attempt_controller: AttemptController,
        on_stage_completed: Callable[[str], None],
        on_shutdown_signal: Callable[[str], None],
    ) -> BootstrapResult:
        """Load + validate config, then wire every collaborator.

        ``attempt_controller`` is passed in rather than constructed here
        because its ``save_fn`` must close over orchestrator-owned per-run
        state (``_state_store`` reference). The two hooks
        (``on_stage_completed``, ``on_shutdown_signal``) have the same
        requirement — they read/write orchestrator-owned state.
        """
        logger.info("Initializing Pipeline Orchestrator")

        # Step 1: Load config + secrets, fail-fast validation.
        try:
            config = load_config(config_path)
            secrets = load_secrets()
            StartupValidator.validate(config=config, secrets=secrets)
            logger.info("Configuration loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise

        # Step 1.5: Preflight gate — refuse to spin up the rest of the
        # orchestrator if a non-optional ``[[required_env]]`` is unset
        # OR if any plugin instance's params/thresholds violate the
        # manifest schema. Catches both classes of mistake at second 0
        # instead of at minute 4 mid-stage. Process env at this point
        # already has the project's env.json merged in (the launcher
        # merges before fork), so we don't pass project_env explicitly.
        from src.community.preflight import LaunchAbortedError, run_preflight

        report = run_preflight(config, secrets=secrets)
        if not report.ok:
            for m in report.missing_envs:
                logger.error(
                    "[PREFLIGHT] %s plugin %r (instance %r) needs env %r%s",
                    m.plugin_kind,
                    m.plugin_name,
                    m.plugin_instance_id,
                    m.name,
                    f" — {m.description}" if m.description else "",
                )
            for e in report.instance_errors:
                logger.error(
                    "[PREFLIGHT] %s plugin %r (instance %r) shape error at %s: %s",
                    e.plugin_kind,
                    e.plugin_name,
                    e.plugin_instance_id,
                    e.location,
                    e.message,
                )
            raise LaunchAbortedError(
                missing=report.missing_envs,
                instance_errors=report.instance_errors,
            )

        # Step 2: Pipeline context seeded with run-scope keys.
        # PipelineContext inherits from dict — existing stages keep working.
        context = PipelineContext(
            {
                PipelineContextKeys.CONFIG_PATH: str(config_path),
                PipelineContextKeys.RUN: run_ctx,
            }
        )

        # Step 3: Collectors — the canonical mapping every downstream
        # component watches. Built FIRST so ValidationArtifactManager and
        # StageRegistry see the exact same dict.
        collectors = StageRegistry._build_collectors()

        # Step 4: Collaborators that depend on collectors/context.
        validation_artifact_mgr = ValidationArtifactManager(
            collectors=collectors,
            context=context,
        )
        context_propagator = ContextPropagator(validation_artifact_mgr)
        stage_info_logger = StageInfoLogger()
        config_drift = ConfigDriftValidator(config)
        summary_reporter = ExecutionSummaryReporter(config)
        mlflow_attempt = MLflowAttemptManager(config, config_path)

        # Step 5: Stages + registry. Stages need validation_artifact_mgr
        # for DatasetValidator's event callbacks, so build them after.
        stages_list = StageRegistry._build_stages(
            config=config,
            secrets=secrets,
            validation_artifact_mgr=validation_artifact_mgr,
        )
        registry = StageRegistry(
            config=config, stages=stages_list, collectors=collectors
        )
        logger.info(f"Initialized {len(registry.stages)} pipeline stages")

        # Step 6: Pure stage-ordering logic — needs finalised stages + config.
        stage_planner = StagePlanner(registry.stages, config)

        # Step 7: Per-run orchestration components (stateless between runs).
        launch_preparator = LaunchPreparator(
            config_path=config_path,
            run_ctx=run_ctx,
            settings=settings,
            stages=registry.stages,
            stage_planner=stage_planner,
            config_drift=config_drift,
            attempt_controller=attempt_controller,
        )
        restart_inspector = RestartPointsInspector(
            stages=registry.stages, config_drift=config_drift
        )
        stage_execution_loop = StageExecutionLoop(
            stages=registry.stages,
            collectors=collectors,
            attempt_controller=attempt_controller,
            stage_planner=stage_planner,
            context_propagator=context_propagator,
            stage_info_logger=stage_info_logger,
            validation_artifact_mgr=validation_artifact_mgr,
            summary_reporter=summary_reporter,
            on_stage_completed=on_stage_completed,
            on_shutdown_signal=on_shutdown_signal,
        )

        return BootstrapResult(
            config_path=config_path,
            config=config,
            secrets=secrets,
            settings=settings,
            run_ctx=run_ctx,
            context=context,
            collectors=collectors,
            stages=registry.stages,
            validation_artifact_mgr=validation_artifact_mgr,
            context_propagator=context_propagator,
            stage_info_logger=stage_info_logger,
            config_drift=config_drift,
            summary_reporter=summary_reporter,
            mlflow_attempt=mlflow_attempt,
            registry=registry,
            stage_planner=stage_planner,
            attempt_controller=attempt_controller,
            launch_preparator=launch_preparator,
            restart_inspector=restart_inspector,
            stage_execution_loop=stage_execution_loop,
        )


__all__ = ["BootstrapResult", "PipelineBootstrap"]
