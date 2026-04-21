"""Stage-context propagation helpers.

Collects four closely-related concerns previously tangled inside
PipelineOrchestrator:

* ``sync_root_from_stage`` — promote selected keys from a stage's outputs into
  the root context so downstream stages can read them.
* ``extract_restart_outputs`` — project a stage's context dict into the exact
  subset persisted as state lineage on restart.
* ``fill_collector_from_context`` — mirror that subset into the stage's
  StageArtifactCollector for non-callback stages.
* ``get_stage_skip_reason`` — derive a skip-reason from stage context when the
  stage self-reports ``inference_skipped`` / ``evaluation_skipped``.

The propagator holds no mutable state; the orchestrator owns the context dict
and passes it in explicitly. ValidationArtifactManager is injected once (it's
the only cross-collaborator required, for DatasetValidator outputs).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.pipeline.constants import CTX_TRAINING_DURATION
from src.pipeline.stages import StageNames

if TYPE_CHECKING:
    from src.pipeline.artifacts import StageArtifactCollector
    from src.pipeline.validation.artifact_manager import ValidationArtifactManager


_KEY_UPLOAD_DURATION = "upload_duration_seconds"


class ContextPropagator:
    """Stateless helper: move data between stage context, lineage, and collectors."""

    def __init__(self, validation_artifact_mgr: ValidationArtifactManager) -> None:
        self._validation_artifact_mgr = validation_artifact_mgr

    def sync_root_from_stage(
        self,
        *,
        context: dict[str, Any],
        stage_name: str,
        outputs: dict[str, Any],
    ) -> None:
        """Copy selected keys from a stage's outputs into the root context.

        Only InferenceDeployer promotes keys today; other stages keep their
        outputs scoped under their own stage_name entry.
        """
        if stage_name == StageNames.INFERENCE_DEPLOYER:
            if "endpoint_url" in outputs:
                context["endpoint_url"] = outputs["endpoint_url"]
            if "inference_model_name" in outputs:
                context["inference_model_name"] = outputs["inference_model_name"]

    def extract_restart_outputs(
        self,
        *,
        context: dict[str, Any],
        stage_name: str,
    ) -> dict[str, Any]:
        """Project a stage's context into the dict persisted as restart lineage."""
        stage_ctx = context.get(stage_name, {})
        if not isinstance(stage_ctx, dict):
            return {}
        if stage_name == StageNames.DATASET_VALIDATOR:
            return self._validation_artifact_mgr.build_dataset_validation_state_outputs(stage_ctx=stage_ctx)
        if stage_name == StageNames.GPU_DEPLOYER:
            keys: tuple[str, ...] = (
                "resource_id",
                "ssh_host",
                "ssh_port",
                "ssh_user",
                "ssh_key_path",
                "workspace_path",
                "provider_name",
                "provider_type",
                "provider_info",
                "is_alias_mode",
            )
            return {key: stage_ctx.get(key) for key in keys if stage_ctx.get(key) is not None}
        if stage_name == StageNames.TRAINING_MONITOR:
            outputs = {
                "status": stage_ctx.get("status"),
                "training_duration_seconds": stage_ctx.get("training_duration_seconds"),
                "training_info": stage_ctx.get("training_info"),
            }
            gpu_ctx = context.get(StageNames.GPU_DEPLOYER, {})
            if isinstance(gpu_ctx, dict) and gpu_ctx.get("workspace_path"):
                outputs["remote_output_dir"] = f"{gpu_ctx['workspace_path']}/output"
            return {key: value for key, value in outputs.items() if value is not None}
        if stage_name == StageNames.MODEL_RETRIEVER:
            keys = ("hf_repo_id", "local_model_path", "hf_uploaded", "model_size_mb", "upload_duration_seconds")
            return {key: stage_ctx.get(key) for key in keys if stage_ctx.get(key) is not None}
        if stage_name == StageNames.INFERENCE_DEPLOYER:
            endpoint_info = stage_ctx.get("endpoint_info")
            outputs = {
                "endpoint_url": stage_ctx.get("endpoint_url"),
                "inference_endpoint_url": stage_ctx.get("inference_endpoint_url"),
                "inference_model_name": stage_ctx.get("inference_model_name"),
                "endpoint_info": endpoint_info if isinstance(endpoint_info, dict) else {},
                "inference_manifest_path": stage_ctx.get("inference_manifest_path"),
                "inference_scripts": stage_ctx.get("inference_scripts"),
            }
            return {key: value for key, value in outputs.items() if value is not None}
        if stage_name == StageNames.MODEL_EVALUATOR:
            keys = ("eval_passed", "eval_summary", "evaluation_completed_at")
            return {key: stage_ctx.get(key) for key in keys if stage_ctx.get(key) is not None}
        return dict(stage_ctx)

    def fill_collector_from_context(
        self,
        *,
        context: dict[str, Any],
        stage_name: str,
        collector: StageArtifactCollector,
    ) -> None:
        """Populate collector with stage-specific data read from pipeline context.

        Called for simple stages where all data is available after stage.run()
        completes. Complex stages (DatasetValidator) populate the collector
        themselves via callbacks, so this method is never called for them.
        """
        ctx = context.get(stage_name, {})
        if not isinstance(ctx, dict):
            return

        if stage_name == StageNames.GPU_DEPLOYER:
            collector.put(
                upload_duration_seconds=ctx.get(_KEY_UPLOAD_DURATION),
                deps_duration_seconds=ctx.get("deps_duration_seconds"),
                provider_name=ctx.get("provider_name"),
                provider_type=ctx.get("provider_type"),
                gpu_type=ctx.get("gpu_type"),
                resource_id=ctx.get("resource_id"),
            )

        elif stage_name == StageNames.TRAINING_MONITOR:
            collector.put(
                training_duration_seconds=ctx.get(CTX_TRAINING_DURATION),
            )

        elif stage_name == StageNames.MODEL_RETRIEVER:
            collector.put(
                model_size_mb=ctx.get("model_size_mb"),
                hf_repo_id=ctx.get("hf_repo_id"),
                upload_duration_seconds=ctx.get(_KEY_UPLOAD_DURATION),
            )

        elif stage_name == StageNames.INFERENCE_DEPLOYER:
            collector.put(
                endpoint_url=ctx.get("endpoint_url"),
                model_name=ctx.get("model_name"),
                provider=ctx.get("provider"),
            )

        elif stage_name == StageNames.MODEL_EVALUATOR:
            eval_summary = ctx.get("eval_summary", {})
            if isinstance(eval_summary, dict):
                collector.put(**eval_summary)
            else:
                collector.put(eval_summary=str(eval_summary))

    def get_stage_skip_reason(
        self,
        *,
        context: dict[str, Any],
        stage_name: str,
    ) -> str | None:
        """Derive a skip-reason from stage context when a stage self-reports skipping."""
        stage_ctx = context.get(stage_name, {})
        if not isinstance(stage_ctx, dict):
            return None
        if stage_ctx.get("inference_skipped"):
            return str(stage_ctx.get("reason", "inference_skipped"))
        if stage_ctx.get("evaluation_skipped"):
            return str(stage_ctx.get("reason", "evaluation_skipped"))
        return None


__all__ = ["ContextPropagator"]
