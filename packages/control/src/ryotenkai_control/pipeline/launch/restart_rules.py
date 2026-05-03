"""Pure decision rules for restart-point availability.

Both :func:`src.pipeline.launch.restart_options.list_restart_points` (lightweight,
config-driven, no health probe) and
:class:`src.pipeline.execution.restart_inspector.RestartPointsInspector`
(orchestrator-driven, includes runtime probe) call into this module.

Keeping the rules in one file prevents silent drift when restart semantics
change — both consumers must keep working against the same truth table.

The caller supplies ``inference_health_checker`` — a function
``(inference_ctx) -> bool``. Passing a no-op ``lambda _ctx: True`` gives
the "skip network probe" behaviour used by the lightweight path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.pipeline.stages import StageNames

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from src.pipeline.state.models import PipelineState

# Late-stage config changes don't invalidate runtime-only restarts of the
# inference/eval stages — those only depend on runtime hashes.
_LATE_STAGE_RUNTIME_EXEMPT = frozenset({
    StageNames.INFERENCE_DEPLOYER,
    StageNames.MODEL_EVALUATOR,
})


def compute_restart_points(
    *,
    state: PipelineState,
    stage_names: Iterable[str],
    config_hashes: dict[str, str],
    inference_health_checker: Callable[[dict[str, Any] | None], bool],
) -> list[dict[str, Any]]:
    """Return per-stage availability dicts.

    Dict shape: ``{stage, available, mode, reason}``. Each stage in
    ``stage_names`` order produces exactly one entry.

    Parameters
    ----------
    state:
        The loaded :class:`PipelineState` (contains lineage + persisted hashes).
    stage_names:
        Stage-name iteration order. Typically the canonical pipeline order.
    config_hashes:
        Fresh hashes for the current config (keys: ``training_critical``,
        ``late_stage``, ``model_dataset``).
    inference_health_checker:
        Callable that receives ``inference_deployer`` outputs (or ``None``)
        and returns whether the runtime is still live. Pass
        ``lambda _: True`` when the caller doesn't want to probe the network.
    """
    points: list[dict[str, Any]] = []
    for stage_name in stage_names:
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
            elif not inference_health_checker(
                dict(ref.outputs) if isinstance(ref.outputs, dict) else None
            ):
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


__all__ = ["compute_restart_points"]
