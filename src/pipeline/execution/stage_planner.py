"""Stage planning: ordering, enablement, resume derivation, prerequisite checks.

Extracted from PipelineOrchestrator to isolate pure stage-ordering logic from
orchestration side effects. StagePlanner is a stateless helper whose methods
read from stages + config + state; mutations live in the orchestrator.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from urllib.request import urlopen

from src.pipeline.stages import StageNames
from src.pipeline.state import PipelineState
from src.pipeline.state.queries import first_unfinished_stage
from src.utils.result import AppError

if TYPE_CHECKING:
    from src.pipeline.stages.base import PipelineStage
    from src.utils.config import PipelineConfig


_HTTP_OK_MIN = 200
_HTTP_ERROR_MIN = 400
_HEALTH_CHECK_TIMEOUT_SECONDS = 5


# Canonical pipeline stage order. Listing by name (rather than slicing
# self._stages) decouples enabled-stage computation from how many/which
# stage instances happen to populate ``self._stages`` — especially
# relevant in tests that patch stage classes with MagicMock.
_CANONICAL_STAGE_ORDER: tuple[str, ...] = (
    StageNames.DATASET_VALIDATOR,
    StageNames.GPU_DEPLOYER,
    StageNames.TRAINING_MONITOR,
    StageNames.MODEL_RETRIEVER,
    StageNames.INFERENCE_DEPLOYER,
    StageNames.MODEL_EVALUATOR,
)

# Stages that always run regardless of config toggles.
_MANDATORY_STAGES: frozenset[str] = frozenset(
    {
        StageNames.DATASET_VALIDATOR,
        StageNames.GPU_DEPLOYER,
        StageNames.TRAINING_MONITOR,
        StageNames.MODEL_RETRIEVER,
    }
)


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
        # ``type() is int`` rather than ``isinstance(..., int)`` so that a
        # stray ``True`` / ``False`` (which subclass int) is rejected cleanly.
        if type(stage_ref) is int:
            if 1 <= stage_ref <= n:
                return self._stages[stage_ref - 1].stage_name
            raise ValueError(f"Stage index {stage_ref} out of range 1–{n}")
        if isinstance(stage_ref, bool):
            raise TypeError(f"Stage reference must be str|int, not bool (got {stage_ref!r})")
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
        """Return the ordered list of stages enabled for this run.

        The result is built from a canonical stage order (below) rather than
        ``self._stages``: it's a pipeline-level contract that does not depend
        on how many or which mock objects happen to populate ``self._stages``.

        Membership in ``_MANDATORY_STAGES`` guarantees inclusion regardless of
        config; inference/evaluation are toggleable; ``forced_stage_names``
        covers "user explicitly targeted a config-disabled stage".
        """
        forced = self.forced_stage_names(start_stage_name=start_stage_name)
        enabled: list[str] = []
        # Kept as three branches (not combined) for readability: each branch
        # documents a distinct inclusion rule.
        for name in _CANONICAL_STAGE_ORDER:
            if name in _MANDATORY_STAGES:  # noqa: SIM114 — keep branches separate
                enabled.append(name)
            elif name == StageNames.INFERENCE_DEPLOYER and (  # noqa: SIM114
                self._config.inference.enabled or name in forced
            ):
                enabled.append(name)
            elif name == StageNames.MODEL_EVALUATOR and (
                self._config.evaluation.enabled or name in forced
            ):
                enabled.append(name)
        return enabled

    def derive_resume_stage(self, state: PipelineState) -> str | None:
        """Find the first stage that is not successfully completed in the latest attempt.

        Walks the *live* ``self._stages`` order, not the attempt's saved
        ``enabled_stage_names`` — the orchestrator wants to know "what
        stage should I run next given the current stage roster", which
        differs from the launch-options view (saved intent) when stages
        get added or removed between attempts.

        Returns the first stage's name when ``state.attempts`` is empty
        (no attempt has run yet, so we resume from the start). The shared
        :func:`first_unfinished_stage` helper handles the per-attempt
        loop; the empty-attempts fallback is callers' choice.
        """
        if not state.attempts:
            return self._stages[0].stage_name
        latest = state.attempts[-1]
        return first_unfinished_stage(
            (stage.stage_name for stage in self._stages),
            latest.stage_runs,
        )

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
