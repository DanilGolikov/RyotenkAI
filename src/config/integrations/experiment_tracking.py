"""Project-level experiment-tracking block.

Core schema. No UX/registry concepts: by the time a
``ExperimentTrackingConfig`` is being validated, the
``integration: <id>`` shorthand has already been expanded inline by
:func:`src.workspace.integrations.resolver.resolve_yaml_integrations`
(if it was used). That's why ``mlflow`` is just ``MLflowConfig | None``
and not the old ``MLflowTrackingRef | MLflowConfig`` Union.

Legacy guard kept: catastrophic mis-edits to the nested
``system_metrics:`` block surface as a migration hint instead of the
generic ``extra_forbidden`` Pydantic error.
"""

from __future__ import annotations

from typing import Any

from pydantic import Field, model_validator

from ..base import StrictBaseModel
from .huggingface import HuggingFaceHubConfig  # noqa: TC001  — required at runtime for Pydantic
from .mlflow import MLflowConfig  # noqa: TC001  — required at runtime for Pydantic

# Fields that used to live INSIDE the nested
# ``experiment_tracking.mlflow.system_metrics:`` block but were removed
# in the second refactor pass:
#  - ``sampling_interval`` / ``samples_before_logging`` configured the
#    native MLflow background sampler, which the codebase no longer
#    enables (it bypassed ``ResilientMLflowTransport`` and dropped
#    samples on offline windows).
#  - ``callback_interval`` was a step-throttle for the HF Trainer
#    callback that's no longer needed (we log every step on our typical
#    1-3 s/step training; on tiny test models batching belongs in
#    ``ResilientMLflowTransport``, not the callback).
# Surface a targeted migration hint for the rare YAML that already
# carried the nested form.
_REMOVED_SYSTEM_METRICS_KEYS: set[str] = {
    "sampling_interval",
    "samples_before_logging",
    "callback_interval",
}

_SYSTEM_METRICS_REMOVED_HINT = (
    "These fields were removed: native MLflow sampler is no longer "
    "enabled (it bypassed our offline buffer), and the HF Trainer "
    "callback now logs every step. Only ``callback_enabled: bool`` "
    "remains in the ``system_metrics:`` block."
)


class ExperimentTrackingConfig(StrictBaseModel):
    """Project-level experiment-tracking block.

    Both ``mlflow`` and ``huggingface`` are core runtime types — by
    the time validation runs, the UX-layer resolver has already
    inlined any ``integration: <id>`` shorthand into the corresponding
    fields. Pipeline stages read
    ``cfg.experiment_tracking.mlflow.tracking_uri`` etc. directly.
    """

    mlflow: MLflowConfig | None = Field(None)
    huggingface: HuggingFaceHubConfig | None = Field(None)

    @model_validator(mode="before")
    @classmethod
    def _reject_removed_system_metrics_keys(cls, data: Any) -> Any:
        """Targeted migration hint for fields removed from the
        ``system_metrics:`` block."""
        if not isinstance(data, dict):
            return data
        mlflow = data.get("mlflow")
        if isinstance(mlflow, dict):
            sm_block = mlflow.get("system_metrics")
            if isinstance(sm_block, dict):
                removed = sorted(set(sm_block.keys()) & _REMOVED_SYSTEM_METRICS_KEYS)
                if removed:
                    raise ValueError(
                        "experiment_tracking.mlflow.system_metrics no longer "
                        f"accepts {removed!r}. {_SYSTEM_METRICS_REMOVED_HINT}"
                    )
        return data

    def get_report_to(self) -> list[str]:
        """Get list of trackers for HuggingFace Trainer.

        ``mlflow`` is considered active iff a tracking URI is set (we
        don't bother distinguishing "user wrote inline" from "resolver
        inlined" — both end up the same).
        """
        if self.mlflow is None:
            return ["none"]
        if not (self.mlflow.tracking_uri or self.mlflow.local_tracking_uri):
            return ["none"]
        return ["mlflow"]


__all__ = [
    "ExperimentTrackingConfig",
]
