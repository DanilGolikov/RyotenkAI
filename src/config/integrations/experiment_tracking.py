"""Project-level experiment-tracking block.

PR3 change: legacy inline fields (``tracking_uri``, ``local_tracking_uri``,
``ca_bundle_path``, ``system_metrics_*``, ``enabled``) are moved to
Settings → Integrations. The project YAML only references an integration
by id plus project-local knobs (``experiment_name`` / ``repo_id``).

StrictBaseModel already rejects unknown fields (``extra='forbid'``). We
add a ``model_validator(mode='before')`` that turns that default
low-signal Pydantic message into a user-friendly migration hint so
``tracking_uri: …`` on the old path points at the exact solution.
"""

from __future__ import annotations

from typing import Any

from pydantic import Field, model_validator

from ..base import StrictBaseModel
from .huggingface import HuggingFaceHubConfig  # noqa: TC001  — required at runtime for Pydantic
from .mlflow import MLflowConfig, MLflowTrackingRef  # noqa: TC001, F401


# Legacy fields that used to live at ``experiment_tracking.mlflow.*``.
# Now live on the integration side.
_LEGACY_MLFLOW_KEYS: set[str] = {
    "tracking_uri",
    "local_tracking_uri",
    "ca_bundle_path",
    "system_metrics_sampling_interval",
    "system_metrics_samples_before_logging",
    "system_metrics_callback_enabled",
    "system_metrics_callback_interval",
}

# Legacy fields that used to live at ``experiment_tracking.huggingface.*``.
_LEGACY_HF_KEYS: set[str] = {"enabled"}


_MIGRATION_HINT = (
    "Migration: move these keys to Settings → Integrations (one shared "
    "account per team). See docs/migration/integrations.md. Keep only "
    "``integration`` + ``experiment_name`` on mlflow, ``integration`` + "
    "``repo_id`` + ``private`` on huggingface."
)


class ExperimentTrackingConfig(StrictBaseModel):
    """Project-level references to reusable tracking integrations.

    ``mlflow`` now holds an integration id + project-local fields only;
    every runtime knob (URI / CA bundle / system metrics) comes from the
    integration. Same for ``huggingface``.
    """

    mlflow: MLflowTrackingRef | None = Field(None)
    huggingface: HuggingFaceHubConfig | None = Field(None)

    @model_validator(mode="before")
    @classmethod
    def _reject_legacy_keys(cls, data: Any) -> Any:
        """Turn the generic ``extra_forbidden`` error into a migration hint."""
        if not isinstance(data, dict):
            return data

        mlflow = data.get("mlflow")
        if isinstance(mlflow, dict):
            legacy = sorted(set(mlflow.keys()) & _LEGACY_MLFLOW_KEYS)
            if legacy:
                raise ValueError(
                    "experiment_tracking.mlflow no longer accepts "
                    f"{legacy!r}. These fields moved to Settings → Integrations → "
                    f"MLflow. {_MIGRATION_HINT}"
                )

        hf = data.get("huggingface")
        if isinstance(hf, dict):
            legacy = sorted(set(hf.keys()) & _LEGACY_HF_KEYS)
            if legacy:
                raise ValueError(
                    "experiment_tracking.huggingface no longer accepts "
                    f"{legacy!r}. The enable/disable state is now derived "
                    f"from the ``integration`` field. {_MIGRATION_HINT}"
                )

        return data

    def get_report_to(self) -> list[str]:
        """Get list of trackers for HuggingFace Trainer.

        ``mlflow`` is considered active iff an integration id is set.
        """
        if self.mlflow is None or not self.mlflow.integration:
            return ["none"]
        return ["mlflow"]


__all__ = [
    "ExperimentTrackingConfig",
]
