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

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ryotenkai_control.events import ControlEventEmitter
from ryotenkai_control.pipeline.bootstrap.startup_validator import StartupValidator
from ryotenkai_control.pipeline.config_drift import ConfigDriftValidator
from ryotenkai_control.pipeline.context import ContextPropagator, PipelineContext, StageInfoLogger
from ryotenkai_control.pipeline.execution import (
    RestartPointsInspector,
    StageExecutionLoop,
    StagePlanner,
    StageRegistry,
)
from ryotenkai_control.pipeline.launch import LaunchPreparator
from ryotenkai_control.pipeline.mlflow_attempt import MLflowAttemptManager
from ryotenkai_control.pipeline.reporting import ExecutionSummaryReporter
from ryotenkai_control.pipeline.stages import PipelineContextKeys
from ryotenkai_control.pipeline.stages.dataset_validator.artifact_manager import ValidationArtifactManager
from ryotenkai_shared.config import load_secrets
from ryotenkai_shared.utils.logger import logger

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

    from ryotenkai_shared.config.runtime import RuntimeSettings
    from ryotenkai_control.pipeline.artifacts import StageArtifactCollector
    from ryotenkai_control.pipeline.stages.base import PipelineStage
    from ryotenkai_control.pipeline.state import AttemptController
    from ryotenkai_shared.pipeline_context import RunContext
    from ryotenkai_shared.config import PipelineConfig, Secrets


# Env-var contract: launcher (CLI / API) sets these before spawning the
# worker; bootstrap reads them and stamps onto ``PipelineState.metadata``,
# which downstream MLflow tags ``meta.*`` mirror. See
# ``docs/plans/task-notification-task-id-b6y40vnmp-tas-majestic-stream.md``.
_METADATA_ENV_KEYS: tuple[tuple[str, str], ...] = (
    ("project_id", "RYOTENKAI_PROJECT_ID"),
    ("actor", "RYOTENKAI_ACTOR"),
    ("config_version_hash", "RYOTENKAI_CONFIG_VERSION_HASH"),
    ("config_override_path", "RYOTENKAI_CONFIG_OVERRIDE_PATH"),
)


def read_metadata_from_env() -> dict[str, str]:
    """Read run-level metadata that the launcher injected via env vars.

    Returns a dict suitable for :class:`PipelineState`'s ``metadata``
    slot — only keys whose env var is set and non-empty are included.
    Anonymous runs (no enclosing project) return ``{}``.

    Single source of truth for "where does ``meta.project_id`` come
    from" — both CLI and Web API pass it through the same env-var
    mechanism.
    """
    out: dict[str, str] = {}
    for key, env_var in _METADATA_ENV_KEYS:
        value = os.environ.get(env_var)
        if value and value.strip():
            out[key] = value.strip()
    return out


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
    # Optional: control-side event emitter. ``None`` when bootstrap
    # was called without a ``run_directory`` — the orchestrator builds
    # it lazily inside :meth:`_prepare_stateful_attempt` once
    # ``LaunchPreparator`` has resolved the canonical path. See
    # :meth:`PipelineOrchestrator._ensure_event_emitter`.
    emitter: ControlEventEmitter | None = None


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
        config: PipelineConfig,
        secrets: Secrets | None = None,
        run_ctx: RunContext,
        settings: RuntimeSettings,
        attempt_controller: AttemptController,
        on_stage_completed: Callable[[str], None],
        on_shutdown_signal: Callable[[str], None],
        stages_override: Sequence[PipelineStage] | None = None,
        run_directory: Path | None = None,
    ) -> BootstrapResult:
        """Wire every collaborator from a pre-loaded config.

        Caller responsibility: load the YAML via
        :func:`src.workspace.integrations.loader.load_pipeline_config`
        before constructing the orchestrator. By the time bootstrap
        runs, ``config`` is a validated ``PipelineConfig`` with
        ``_source_path`` set.

        ``secrets`` is optional — defaults to ``load_secrets()`` (which
        reads ``os.environ`` plus ``secrets.env``).

        Run-level metadata (``project_id``, ``actor``,
        ``config_version_hash``) is read from ``RYOTENKAI_*`` env vars
        via :func:`read_metadata_from_env` — set by the launcher (CLI
        ``run start --project X`` or Web API). Anonymous runs (no
        enclosing project) get an empty metadata dict.

        ``attempt_controller`` is passed in rather than constructed here
        because its ``save_fn`` must close over orchestrator-owned per-run
        state (``_state_store`` reference). The two hooks
        (``on_stage_completed``, ``on_shutdown_signal``) have the same
        requirement — they read/write orchestrator-owned state.

        ``stages_override`` is the additive test seam introduced in
        Phase 4-followup. When provided, the bootstrap installs the
        caller-supplied stage list onto the :class:`StageRegistry` instead
        of calling :meth:`StageRegistry._build_stages` — which means
        production-time concrete stages are never constructed. Production
        callers always omit this kwarg; tests can pass a list of
        ``MagicMock`` stages (or any duck-typed ``PipelineStage``) to
        replace the legacy ``patch.object(StageRegistry, "_build_stages")``
        scaffold.
        """
        logger.info("Initializing Pipeline Orchestrator")

        # Step 1: Validate the pre-loaded config + load secrets.
        try:
            source_path = getattr(config, "_source_path", None)
            if source_path is None:
                raise ValueError(
                    "PipelineBootstrap.build: pre-loaded config has no "
                    "_source_path. Use "
                    "``src.workspace.integrations.loader.load_pipeline_config(path)`` "
                    "instead of constructing PipelineConfig directly — that "
                    "loader stamps _source_path so downstream consumers "
                    "(state_store, config_drift, …) work."
                )
            config_path: Path = source_path

            if secrets is None:
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
        from ryotenkai_community.preflight import LaunchAbortedError, run_preflight

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
            for inst_err in report.instance_errors:
                logger.error(
                    "[PREFLIGHT] %s plugin %r (instance %r) shape error at %s: %s",
                    inst_err.plugin_kind,
                    inst_err.plugin_name,
                    inst_err.plugin_instance_id,
                    inst_err.location,
                    inst_err.message,
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
        # When ``stages_override`` is supplied (tests / advanced callers),
        # skip the canonical stage construction entirely — the override
        # owns the stage list verbatim.
        if stages_override is not None:
            stages_list = list(stages_override)
        else:
            stages_list = StageRegistry._build_stages(
                config=config,
                secrets=secrets,
                validation_artifact_mgr=validation_artifact_mgr,
            )
        registry = StageRegistry(config=config, stages=stages_list, collectors=collectors)
        logger.info(f"Initialized {len(registry.stages)} pipeline stages")

        # Step 6: Pure stage-ordering logic — needs finalised stages + config.
        stage_planner = StagePlanner(registry.stages, config)

        # Step 7: Per-run orchestration components (stateless between runs).
        # Run-level metadata is sourced exclusively from RYOTENKAI_* env
        # vars set by the launcher. ``LaunchPreparator`` stamps it onto
        # :class:`PipelineState` at fresh-run init time; downstream
        # MLflow tags ``meta.*`` mirror it. Empty dict for anonymous
        # runs (no enclosing project).
        launch_preparator = LaunchPreparator(
            config_path=config_path,
            run_ctx=run_ctx,
            settings=settings,
            stages=registry.stages,
            stage_planner=stage_planner,
            config_drift=config_drift,
            attempt_controller=attempt_controller,
            metadata=read_metadata_from_env(),
        )
        restart_inspector = RestartPointsInspector(stages=registry.stages, config_drift=config_drift)
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

        # Optional Phase 3 event emitter — built up-front only when the
        # caller already knows the run directory (rare; the usual flow
        # has the orchestrator resolve it via ``LaunchPreparator`` and
        # then lazily build the emitter). The emitter is the SSOT for
        # the events.jsonl journal and the in-memory bus that SSE/WS
        # adapters subscribe to in Phase 6.
        emitter: ControlEventEmitter | None = None
        if run_directory is not None:
            emitter = ControlEventEmitter.for_run(
                run_id=run_ctx.name,
                run_directory=run_directory,
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
            emitter=emitter,
        )


__all__ = ["BootstrapResult", "PipelineBootstrap"]
