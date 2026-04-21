"""Stage planning: ordering, enablement, resume derivation, prerequisite checks.

Extracted from PipelineOrchestrator to isolate pure stage-ordering logic from
orchestration side effects. StagePlanner is a stateless helper whose methods
read from stages + config + state; mutations live in the orchestrator.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from urllib.request import urlopen

from src.pipeline.stages import StageNames
from src.pipeline.state import PipelineState, StageRunState
from src.utils.result import AppError

if TYPE_CHECKING:
    from src.pipeline.stages.base import PipelineStage
    from src.utils.config import PipelineConfig


_HTTP_OK_MIN = 200
_HTTP_ERROR_MIN = 400
_HEALTH_CHECK_TIMEOUT_SECONDS = 5


def is_inference_runtime_healthy(inference_ctx: dict[str, Any] | None) -> bool:
    """Probe inference endpoint health via HTTP.

    Accepts the DatasetValidator-shaped context dict for the InferenceDeployer
    stage; falls back to ``endpoint_url`` when ``endpoint_info.health_url`` is
    absent. Treats any non-2xx/3xx response or network failure as unhealthy.
    """
    if not isinstance(inference_ctx, dict):
        return False
    endpoint_info = inference_ctx.get("endpoint_info")
    if not isinstance(endpoint_info, dict):
        endpoint_info = {}
    health_url = endpoint_info.get("health_url") or inference_ctx.get("endpoint_url")
    if not isinstance(health_url, str) or not health_url:
        return False
    try:
        with urlopen(health_url, timeout=_HEALTH_CHECK_TIMEOUT_SECONDS) as response:
            status = int(getattr(response, "status", _HTTP_OK_MIN))
            return _HTTP_OK_MIN <= status < _HTTP_ERROR_MIN
    except Exception:
        return False


class StagePlanner:
    """Pure stage-ordering helper.

    Resolves stage references, computes the list of enabled stages given
    config, derives the resume stage from persisted state, and validates
    prerequisites before a restart. Holds no mutable state.
    """

    def __init__(self, stages: list[PipelineStage], config: PipelineConfig) -> None:
        self._stages = stages
        self._config = config

    @property
    def stages(self) -> list[PipelineStage]:
        return self._stages

    def get_stage_index(self, stage_name: str) -> int:
        for idx, stage in enumerate(self._stages):
            if stage.stage_name == stage_name:
                return idx
        raise ValueError(f"Unknown stage name: {stage_name}")

    def normalize_stage_ref(self, stage_ref: str | int | None) -> str:
        """Resolve stage ref to canonical stage name.

        Accepts:
        - str name: "Inference Deployer", "inference_deployer" (case/underscore-insensitive)
        - int or str digit: 1-based human index (1 = first stage, N = last stage)
        """
        if stage_ref is None:
            raise ValueError("Stage reference is required")
        n = len(self._stages)
        if isinstance(stage_ref, int):
            if 1 <= stage_ref <= n:
                return self._stages[stage_ref - 1].stage_name
            raise ValueError(f"Stage index {stage_ref} out of range 1–{n}")
        stage_value = str(stage_ref).strip()
        if stage_value == "":
            raise ValueError("Stage reference is empty")
        if stage_value.isdigit():
            idx = int(stage_value)
            if 1 <= idx <= n:
                return self._stages[idx - 1].stage_name
            raise ValueError(f"Stage index {stage_value} out of range 1–{n}")
        lowered = stage_value.casefold()
        aliases = {stage.stage_name.casefold(): stage.stage_name for stage in self._stages}
        normalized_aliases = {
            stage.stage_name.casefold().replace(" ", "_"): stage.stage_name for stage in self._stages
        }
        if lowered in aliases:
            return aliases[lowered]
        if lowered.replace(" ", "_") in normalized_aliases:
            return normalized_aliases[lowered.replace(" ", "_")]
        raise ValueError(f"Unknown stage reference: {stage_ref!r}. Use a name or 1–{n}")

    def forced_stage_names(self, *, start_stage_name: str) -> set[str]:
        """Stages forced into the enabled list when the user explicitly targeted them."""
        forced: set[str] = set()
        if start_stage_name == StageNames.INFERENCE_DEPLOYER and not self._config.inference.enabled:
            forced.add(StageNames.INFERENCE_DEPLOYER)
        if start_stage_name == StageNames.MODEL_EVALUATOR and not self._config.evaluation.enabled:
            forced.add(StageNames.MODEL_EVALUATOR)
        return forced

    def compute_enabled_stage_names(self, *, start_stage_name: str) -> list[str]:
        """Return the ordered list of stages enabled for this run."""
        enabled = [stage.stage_name for stage in self._stages[:4]]
        if self._config.inference.enabled:
            enabled.append(StageNames.INFERENCE_DEPLOYER)
        if self._config.evaluation.enabled:
            enabled.append(StageNames.MODEL_EVALUATOR)
        for forced_name in self.forced_stage_names(start_stage_name=start_stage_name):
            if forced_name not in enabled:
                enabled.append(forced_name)
        return enabled

    def derive_resume_stage(self, state: PipelineState) -> str | None:
        """Find the first stage that is not successfully completed in the latest attempt."""
        if not state.attempts:
            return self._stages[0].stage_name
        latest = state.attempts[-1]
        for stage in self._stages:
            stage_state = latest.stage_runs.get(stage.stage_name)
            if stage_state is None:
                return stage.stage_name
            if stage_state.status in {
                StageRunState.STATUS_FAILED,
                StageRunState.STATUS_INTERRUPTED,
                StageRunState.STATUS_PENDING,
                StageRunState.STATUS_RUNNING,
                StageRunState.STATUS_STALE,
            }:
                return stage.stage_name
        return None

    def validate_stage_prerequisites(
        self,
        *,
        stage_name: str,
        start_stage_name: str,
        context: dict[str, Any],
    ) -> AppError | None:
        """Check that upstream outputs required by a restart target are present.

        The orchestrator supplies ``context`` so the planner stays stateless
        and testable without a live pipeline run.
        """
        if stage_name == StageNames.TRAINING_MONITOR and start_stage_name == StageNames.TRAINING_MONITOR:
            gpu_ctx = context.get(StageNames.GPU_DEPLOYER, {})
            if not isinstance(gpu_ctx, dict) or not all(
                gpu_ctx.get(key) for key in ("ssh_host", "ssh_port", "workspace_path")
            ):
                return AppError(
                    message="Training Monitor restart requires persisted GPU deploy outputs and workspace_path",
                    code="MISSING_TRAINING_MONITOR_PREREQUISITES",
                )
        if stage_name == StageNames.INFERENCE_DEPLOYER and start_stage_name == StageNames.INFERENCE_DEPLOYER:
            retriever_ctx = context.get(StageNames.MODEL_RETRIEVER, {})
            if not isinstance(retriever_ctx, dict) or not (
                retriever_ctx.get("hf_repo_id") or retriever_ctx.get("local_model_path")
            ):
                return AppError(
                    message="Inference Deployer restart requires Model Retriever outputs",
                    code="MISSING_INFERENCE_PREREQUISITES",
                )
        if (
            stage_name == StageNames.MODEL_EVALUATOR
            and start_stage_name == StageNames.MODEL_EVALUATOR
            and not is_inference_runtime_healthy(context.get(StageNames.INFERENCE_DEPLOYER, {}))
        ):
            return AppError(
                message="Model Evaluator restart requires a live inference runtime; restart from Inference Deployer",
                code="INFERENCE_RUNTIME_NOT_HEALTHY",
            )
        return None


__all__ = [
    "StagePlanner",
    "is_inference_runtime_healthy",
]
