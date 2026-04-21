"""Runtime-aware restart-point inspection.

Delegates rule evaluation to :func:`src.pipeline.restart_rules.compute_restart_points`
and adds the runtime health probe. The lighter-weight
:func:`src.pipeline.restart_points.list_restart_points` shares the same
rule engine but passes a no-op health checker.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.pipeline.executor import is_inference_runtime_healthy
from src.pipeline.restart_rules import compute_restart_points
from src.pipeline.state import PipelineStateStore

if TYPE_CHECKING:
    from pathlib import Path

    from src.pipeline.config_drift import ConfigDriftValidator
    from src.pipeline.stages.base import PipelineStage


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
        stage_names = [s.stage_name for s in self._stages]
        return compute_restart_points(
            state=state,
            stage_names=stage_names,
            config_hashes=config_hashes,
            inference_health_checker=self._is_inference_runtime_healthy,
        )

    @staticmethod
    def _is_inference_runtime_healthy(inference_ctx: dict[str, Any] | None) -> bool:
        """Probe the inference endpoint; returns ``False`` on any error."""
        return is_inference_runtime_healthy(
            inference_ctx if isinstance(inference_ctx, dict) else None
        )


__all__ = ["RestartPointsInspector"]
