"""Ownership and lifecycle management for pipeline stages + collectors.

The registry bundles together everything about the stage roster that the
orchestrator used to own directly:

* Construction — each stage gets its config, secrets, and (for
  DatasetValidator) event callbacks.
* Collectors — one :class:`StageArtifactCollector` per stage, keyed by
  :class:`StageNames` value.
* Lifecycle:
    - :meth:`flush_pending_collectors` at teardown,
    - :meth:`cleanup_in_reverse` after the stage loop (reverse order;
      respects SIGINT policy),
    - :meth:`maybe_early_release_gpu` after MODEL_RETRIEVER.
* Lookups — :meth:`get_stage_by_name`, :meth:`list_stage_names`.

After this extraction, the orchestrator holds a single ``self._registry``
instead of the ``stages`` list, ``collectors`` dict, three cleanup methods,
and four factory helpers. Tests for cleanup ordering / SIGINT policy work
with a direct :class:`StageRegistry` instead of orchestrator mocks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from ryotenkai_control.pipeline.artifacts import StageArtifactCollector
from ryotenkai_shared.utils.cancellation import PipelineCancelled
from ryotenkai_control.pipeline.stages import StageNames
from ryotenkai_control.pipeline.stages.dataset_validator import DatasetValidator
from ryotenkai_control.pipeline.stages.gpu_deployer import GPUDeployer, IEarlyReleasable
from ryotenkai_control.pipeline.stages.inference_deployer import InferenceDeployer
from ryotenkai_control.pipeline.stages.model_evaluator import ModelEvaluator
from ryotenkai_control.pipeline.stages.model_retriever import ModelRetriever
from ryotenkai_control.pipeline.stages.training_monitor import TrainingMonitor
from ryotenkai_shared.utils.logger import logger

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ryotenkai_control.pipeline.context import PipelineContext
    from ryotenkai_control.pipeline.stages.base import PipelineStage
    from ryotenkai_control.pipeline.stages.dataset_validator.artifact_manager import ValidationArtifactManager
    from ryotenkai_shared.config import PipelineConfig, Secrets


class StageRegistry:
    """Immutable-after-construction owner of stages + collectors + cleanup policy.

    Once ``build`` returns, the stage list never changes. Lifecycle methods
    (``flush_pending_collectors``, ``cleanup_in_reverse``,
    ``maybe_early_release_gpu``) mutate stage-internal state but don't alter
    the registry's identity.
    """

    __slots__ = (
        "_cleanup_done",
        "_collectors",
        "_config",
        "_stages",
    )

    def __init__(
        self,
        *,
        config: PipelineConfig,
        stages: Sequence[PipelineStage],
        collectors: dict[str, StageArtifactCollector],
    ) -> None:
        self._config = config
        self._stages = list(stages)
        # Store the passed dict by reference (no copy) — downstream
        # components (ValidationArtifactManager) watch the exact same
        # mapping, so flush-state changes stay visible.
        self._collectors = collectors
        # Cleanup is idempotent — second invocation is a logged no-op.
        self._cleanup_done = False

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def build(
        cls,
        *,
        config: PipelineConfig,
        secrets: Secrets,
        validation_artifact_mgr: ValidationArtifactManager,
    ) -> StageRegistry:
        """Factory: produce the canonical stage list + per-stage collectors."""
        return cls(
            config=config,
            stages=cls._build_stages(
                config=config,
                secrets=secrets,
                validation_artifact_mgr=validation_artifact_mgr,
            ),
            collectors=cls._build_collectors(),
        )

    @staticmethod
    def _build_stages(
        *,
        config: PipelineConfig,
        secrets: Secrets,
        validation_artifact_mgr: ValidationArtifactManager,
    ) -> list[PipelineStage]:
        """Initialise every stage in execution order.

        Phase 4 (event-system unification, 2026-05-16): the legacy
        ``*EventCallbacks`` dataclasses are gone. Stages now take
        ``emitter`` keyword for typed event emission; the
        :class:`ValidationArtifactManager` is passed to
        :class:`DatasetValidator` as a direct collaborator
        (``artifact_recorder``). The emitter is wired in lazily by the
        orchestrator via :meth:`PipelineStage.set_emitter` once the
        canonical run directory is resolved.
        """
        return [
            DatasetValidator(
                config,
                secrets=secrets,
                artifact_recorder=validation_artifact_mgr,
            ),
            GPUDeployer(config, secrets),
            TrainingMonitor(config, secrets=secrets),
            ModelRetriever(config, secrets),
            InferenceDeployer(config, secrets),
            ModelEvaluator(config, secrets),
        ]

    @staticmethod
    def _build_collectors() -> dict[str, StageArtifactCollector]:
        """One StageArtifactCollector per pipeline stage."""
        return {
            StageNames.DATASET_VALIDATOR: StageArtifactCollector(
                stage="dataset_validator",
                artifact_name="dataset_validator_results.json",
            ),
            StageNames.GPU_DEPLOYER: StageArtifactCollector(
                stage="gpu_deployer",
                artifact_name="gpu_deployer_results.json",
            ),
            StageNames.TRAINING_MONITOR: StageArtifactCollector(
                stage="training_monitor",
                artifact_name="training_monitor_results.json",
            ),
            StageNames.MODEL_RETRIEVER: StageArtifactCollector(
                stage="model_retriever",
                artifact_name="model_retriever_results.json",
            ),
            StageNames.INFERENCE_DEPLOYER: StageArtifactCollector(
                stage="inference_deployer",
                artifact_name="inference_deployer_results.json",
            ),
            StageNames.MODEL_EVALUATOR: StageArtifactCollector(
                stage="model_evaluator",
                artifact_name="evaluation_results.json",
            ),
        }

    # ------------------------------------------------------------------
    # Read-only accessors
    # ------------------------------------------------------------------

    @property
    def stages(self) -> list[PipelineStage]:
        """The stage list (live reference — ordering matters for the loop)."""
        return self._stages

    @property
    def collectors(self) -> dict[str, StageArtifactCollector]:
        """Collector mapping (live reference)."""
        return self._collectors

    def list_stage_names(self) -> list[str]:
        """Return all stage names in execution order."""
        return [stage.stage_name for stage in self._stages]

    def get_stage_by_name(self, name: str) -> PipelineStage | None:
        """Look up a stage by its ``stage_name`` attribute."""
        for stage in self._stages:
            if stage.stage_name == name:
                return stage
        return None

    # ------------------------------------------------------------------
    # Lifecycle operations
    # ------------------------------------------------------------------

    def flush_pending_collectors(self, context: PipelineContext) -> None:
        """Flush still-open collectors with an INTERRUPTED artifact.

        Called from the orchestrator's finally block so every stage that
        *started* produces a terminal artifact even when the pipeline is
        killed. Stages that never started (``set_started_at`` was never
        called) are intentionally skipped — an empty interrupted artifact
        is just noise in the report.
        """
        for stage_name, collector in self._collectors.items():
            if collector.is_flushed:
                continue
            if collector._started_at is None:
                logger.debug("[ARTIFACT] skip flush for not-started stage %s", stage_name)
                continue
            try:
                collector.flush_interrupted(
                    started_at=collector._started_at,
                    duration_seconds=0.0,
                    context=context,
                )
                logger.debug("[ARTIFACT] flush_interrupted for %s", stage_name)
            except Exception as exc:
                logger.warning("[ARTIFACT] flush_interrupted failed for %s: %s", stage_name, exc)

    def cleanup_in_reverse(
        self,
        *,
        success: bool,
        shutdown_signal_name: str | None,
    ) -> None:
        """Call ``stage.cleanup()`` in reverse order with SIGINT policy.

        Idempotent — a second invocation is a logged no-op. If the pipeline
        failed (``success=False``), every stage that exposes
        ``notify_pipeline_failure`` is called first so long-running workers
        can stop cleanly before cleanup destroys their state.

        The SIGINT policy: if the active provider sets
        ``cleanup.on_interrupt=false`` and the orchestrator reports
        ``shutdown_signal_name == "SIGINT"``, the GPU deployer's cleanup is
        skipped (keeps the pod alive for user debugging).
        """
        if self._cleanup_done:
            logger.debug("Cleanup already done, skipping duplicate call")
            return
        self._cleanup_done = True
        logger.info("Cleaning up pipeline resources...")

        if not success:
            for stage in self._stages:
                if hasattr(stage, "notify_pipeline_failure"):
                    try:
                        stage.notify_pipeline_failure()
                    except Exception as e:
                        logger.debug(f"Error notifying pipeline failure to {stage.stage_name}: {e}")

        skip_gpu_deployer_cleanup = self._should_skip_gpu_cleanup_on_sigint(shutdown_signal_name)

        for stage in reversed(self._stages):
            try:
                if hasattr(stage, "cleanup"):
                    if skip_gpu_deployer_cleanup and getattr(stage, "stage_name", None) == StageNames.GPU_DEPLOYER:
                        continue
                    stage.cleanup()
                    logger.debug(f"Cleanup complete: {stage.stage_name}")
            except KeyboardInterrupt:
                logger.warning(f"[CLEANUP] Interrupted during cleanup of {stage.stage_name}, continuing...")
            except PipelineCancelled:
                # Cancel signal arrived mid-cleanup. Don't abort the
                # cleanup chain — keep tearing down later stages so the
                # user doesn't end up with leaked pods. The deadline
                # timer in the cancel handler is the hard fallback if
                # cleanup itself wedges.
                logger.warning(f"[CLEANUP] cancellation received during cleanup of {stage.stage_name}, continuing...")
            except Exception as e:
                logger.warning(f"Error during cleanup of {stage.stage_name}: {e}")

        logger.info("Pipeline cleanup complete")

    def maybe_early_release_gpu(self) -> None:
        """Release the training pod after MODEL_RETRIEVER when policy allows.

        Invoked from :class:`StageExecutionLoop` via the
        ``on_stage_completed`` hook. Uses the :class:`IEarlyReleasable`
        protocol so any future GPU-bound stage can opt in without changes
        here. Silently no-ops if the active provider doesn't declare
        ``cleanup.terminate_after_retrieval=true``.
        """
        try:
            provider_cfg = self._config.get_provider_config()
        except Exception:
            return
        if not _read_provider_cleanup_flag(provider_cfg, "terminate_after_retrieval", default=False):
            return

        for stage in self._stages:
            if isinstance(stage, IEarlyReleasable):
                logger.info("[ORCHESTRATOR] terminate_after_retrieval=true: releasing training pod early.")
                stage.release()
                return

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _should_skip_gpu_cleanup_on_sigint(self, shutdown_signal_name: str | None) -> bool:
        """Read provider config to decide whether SIGINT skips GPU cleanup.

        Best-effort: any config-inspection failure yields ``False`` so cleanup
        errs on the side of releasing provider resources.
        """
        if shutdown_signal_name != "SIGINT":
            return False
        try:
            provider_name = self._config.get_active_provider_name()
            provider_cfg = self._config.get_provider_config()
        except Exception:
            return False
        on_interrupt = _read_provider_cleanup_flag(provider_cfg, "on_interrupt", default=True)
        if on_interrupt is False:
            logger.warning(
                f"[CLEANUP] Skipping GPU provider disconnect on SIGINT "
                f"(providers.{provider_name}.cleanup.on_interrupt=false)"
            )
            return True
        return False


def _read_provider_cleanup_flag(provider_cfg: Any, flag_name: str, *, default: bool) -> bool:
    """Read ``providers.<id>.cleanup.<flag_name>`` from either typed
    Pydantic schema or raw dict block.

    Returns ``default`` if the cleanup section is absent or malformed.
    """
    if isinstance(provider_cfg, BaseModel):
        cleanup_obj = getattr(provider_cfg, "cleanup", None)
        if cleanup_obj is None:
            return default
        return bool(getattr(cleanup_obj, flag_name, default))
    if isinstance(provider_cfg, dict):
        cleanup_cfg = provider_cfg.get("cleanup")
        if isinstance(cleanup_cfg, dict):
            return bool(cleanup_cfg.get(flag_name, default))
    return default


__all__ = ["StageRegistry"]
