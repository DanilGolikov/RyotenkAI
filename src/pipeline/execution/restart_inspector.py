"""Runtime-aware restart-point inspection.

The orchestrator exposes :meth:`PipelineOrchestrator.list_restart_points` so
callers (CLI, TUI, web API) can ask "which stages of this saved run can I
restart from?". The answer depends on:

* what outputs the persisted ``pipeline_state.json`` has,
* whether the stages between restart-point and current state are still
  valid (config hashes match),
* for ``model_evaluator``, whether the inference runtime is still live
  (network probe).

This module owns that logic as a standalone class so the orchestrator's
public API is a 3-line delegate and the inspector is test-in-isolation.

Note
----
A parallel, lighter-weight :func:`src.pipeline.restart_points.list_restart_points`
exists for callers that only have ``(config, run_dir)`` and don't need the
health probe. Both share :func:`is_inference_runtime_healthy` from the
``executor`` package.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.pipeline.executor import is_inference_runtime_healthy
from src.pipeline.stages import StageNames
from src.pipeline.state import PipelineStateStore

if TYPE_CHECKING:
    from pathlib import Path

    from src.pipeline.config_drift import ConfigDriftValidator
    from src.pipeline.stages.base import PipelineStage

# Late-stage config changes don't invalidate runtime-only restarts of the
# inference/eval stages — those only depend on runtime hashes.
_LATE_STAGE_RUNTIME_EXEMPT = frozenset({
    StageNames.INFERENCE_DEPLOYER,
    StageNames.MODEL_EVALUATOR,
})


class RestartPointsInspector:
    """Produce restart-point availability dicts for a saved run.

    Kept stateless-per-call: the inspector holds config-drift + stages
    references but ``inspect(run_dir)`` is the only public method. Runtime
    health probe is encapsulated so tests can inject a stub by patching
    :func:`is_inference_runtime_healthy`.
    """

    __slots__ = ("_config_drift", "_stages")

    def __init__(
        self,
        *,
        stages: list[PipelineStage],
        config_drift: ConfigDriftValidator,
    ) -> None:
        self._stages = stages
        self._config_drift = config_drift

    def inspect(self, run_dir: Path) -> list[dict[str, Any]]:
        """Load the state at ``run_dir`` and return per-stage availability dicts.

        Dict shape: ``{stage, available, mode, reason}``. Each stage in
        orchestrator stage order produces exactly one entry.
        """
        store = PipelineStateStore(run_dir.expanduser().resolve())
        state = store.load()
        config_hashes = self._config_drift.build_config_hashes()

        points: list[dict[str, Any]] = []
        for stage in self._stages:
            stage_name = stage.stage_name
            available = True
            reason = "restart_allowed"
            mode = "fresh_only"

            if stage_name == StageNames.TRAINING_MONITOR:
                mode = "reconnect_only"
                ref = state.current_output_lineage.get(StageNames.GPU_DEPLOYER)
                gpu_outputs = ref.outputs if ref else {}
                if not all(
                    gpu_outputs.get(key)
                    for key in ("ssh_host", "ssh_port", "workspace_path")
                ):
                    available = False
                    reason = "missing_gpu_deployer_outputs"
            elif stage_name == StageNames.MODEL_RETRIEVER:
                mode = "fresh_or_resume"
                ref = state.current_output_lineage.get(StageNames.GPU_DEPLOYER)
                if ref is None:
                    available = False
                    reason = "missing_gpu_deployer_outputs"
            elif stage_name == StageNames.INFERENCE_DEPLOYER:
                mode = "fresh_or_resume"
                ref = state.current_output_lineage.get(StageNames.MODEL_RETRIEVER)
                outputs = ref.outputs if ref else {}
                if not (outputs.get("hf_repo_id") or outputs.get("local_model_path")):
                    available = False
                    reason = "missing_model_retriever_outputs"
            elif stage_name == StageNames.MODEL_EVALUATOR:
                mode = "live_runtime_only"
                ref = state.current_output_lineage.get(StageNames.INFERENCE_DEPLOYER)
                if ref is None:
                    available = False
                    reason = "missing_inference_outputs"
                elif not self._is_inference_runtime_healthy(dict(ref.outputs)):
                    available = False
                    reason = "inference_runtime_not_healthy"

            # Training-critical drift supersedes stage-specific reasons.
            if state.model_dataset_config_hash:
                if state.model_dataset_config_hash != config_hashes["model_dataset"]:
                    available = False
                    reason = "training_critical_config_changed"
            elif state.training_critical_config_hash != config_hashes["training_critical"]:
                available = False
                reason = "training_critical_config_changed"

            if (
                state.late_stage_config_hash != config_hashes["late_stage"]
                and stage_name not in _LATE_STAGE_RUNTIME_EXEMPT
            ):
                available = False
                reason = "late_stage_config_changed"

            points.append(
                {
                    "stage": stage_name,
                    "available": available,
                    "mode": mode,
                    "reason": reason,
                }
            )
        return points

    @staticmethod
    def _is_inference_runtime_healthy(inference_ctx: dict[str, Any] | None) -> bool:
        """Probe the inference endpoint; returns ``False`` on any error."""
        return is_inference_runtime_healthy(
            inference_ctx if isinstance(inference_ctx, dict) else None
        )


__all__ = ["RestartPointsInspector"]
