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

from typing import TYPE_CHECKING

from src.pipeline.artifacts import StageArtifactCollector
from src.pipeline.stages import StageNames
from src.pipeline.stages.dataset_validator import DatasetValidator
from src.pipeline.stages.gpu_deployer import GPUDeployer, IEarlyReleasable
from src.pipeline.stages.inference_deployer import InferenceDeployer
from src.pipeline.stages.model_evaluator import ModelEvaluator
from src.pipeline.stages.model_retriever import ModelRetriever
from src.pipeline.stages.training_monitor import TrainingMonitor
from src.utils.logger import logger

if TYPE_CHECKING:
    from collections.abc import Sequence

    from src.pipeline.context import PipelineContext
    from src.pipeline.stages.base import PipelineStage
    from src.pipeline.stages.dataset_validator.artifact_manager import ValidationArtifactManager
    from src.utils.config import PipelineConfig, Secrets


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

        Imported lazily so a partially-configured test orchestrator can skip
        this path with ``patch.object(StageRegistry, "_build_stages", ...)``.
        """
        from src.pipeline.stages.dataset_validator import DatasetValidatorEventCallbacks

        vam = validation_artifact_mgr
        validator_callbacks = DatasetValidatorEventCallbacks(
            on_dataset_scheduled=vam.on_dataset_scheduled,
            on_dataset_loaded=vam.on_dataset_loaded,
            on_validation_completed=vam.on_validation_completed,
            on_validation_failed=vam.on_validation_failed,
            on_plugin_start=vam.on_plugin_start,
            on_plugin_complete=vam.on_plugin_complete,
            on_plugin_failed=vam.on_plugin_failed,
        )
        return [
            DatasetValidator(config, secrets=secrets, callbacks=validator_callbacks),
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
                logger.warning(
                    "[ARTIFACT] flush_interrupted failed for %s: %s", stage_name, exc
                )

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
                        logger.debug(
                            f"Error notifying pipeline failure to {stage.stage_name}: {e}"
                        )

        skip_gpu_deployer_cleanup = self._should_skip_gpu_cleanup_on_sigint(
            shutdown_signal_name
        )

        for stage in reversed(self._stages):
            try:
                if hasattr(stage, "cleanup"):
                    if (
                        skip_gpu_deployer_cleanup
                        and getattr(stage, "stage_name", None) == StageNames.GPU_DEPLOYER
                    ):
                        continue
                    stage.cleanup()
                    logger.debug(f"Cleanup complete: {stage.stage_name}")
            except KeyboardInterrupt:
                logger.warning(
                    f"[CLEANUP] Interrupted during cleanup of {stage.stage_name}, continuing..."
                )
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
            cleanup_cfg = (
                provider_cfg.get("cleanup") if isinstance(provider_cfg, dict) else None
            )
            if not (
                isinstance(cleanup_cfg, dict)
                and cleanup_cfg.get("terminate_after_retrieval") is True
            ):
                return
        except Exception:
            return

        for stage in self._stages:
            if isinstance(stage, IEarlyReleasable):
                logger.info(
                    "[ORCHESTRATOR] terminate_after_retrieval=true: releasing training pod early."
                )
                stage.release()
                return

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _should_skip_gpu_cleanup_on_sigint(
        self, shutdown_signal_name: str | None
    ) -> bool:
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
        cleanup_cfg = (
            provider_cfg.get("cleanup") if isinstance(provider_cfg, dict) else None
        )
        if isinstance(cleanup_cfg, dict) and cleanup_cfg.get("on_interrupt") is False:
            logger.warning(
                f"[CLEANUP] Skipping GPU provider disconnect on SIGINT "
                f"(providers.{provider_name}.cleanup.on_interrupt=false)"
            )
            return True
        return False


__all__ = ["StageRegistry"]
