"""Core MLflow runtime config.

Project YAMLs configure MLflow inline: ``tracking_uri`` /
``local_tracking_uri`` / ``ca_bundle_path`` / ``experiment_name`` are
written directly in the ``integrations.mlflow`` block.
"""

from __future__ import annotations

from pydantic import Field, field_validator, model_validator

from ..base import StrictBaseModel
from .system_metrics import SystemMetricsConfig


class MLflowConfig(StrictBaseModel):
    """Runtime view of an MLflow tracker for a single project.

    Either ``tracking_uri`` or ``local_tracking_uri`` must be set.
    """

    tracking_uri: str | None = Field(None)
    local_tracking_uri: str | None = Field(None)
    ca_bundle_path: str | None = Field(None)
    experiment_name: str = Field(..., description="MLflow experiment name")
    run_description_file: str | None = Field(None)

    system_metrics: SystemMetricsConfig = Field(
        default_factory=SystemMetricsConfig,
        description=(
            "MLflow system-metric collection settings (CPU / GPU / "
            "RAM). Optional — sensible defaults if omitted."
        ),
    )

    @field_validator(
        "tracking_uri",
        "local_tracking_uri",
        "ca_bundle_path",
        "run_description_file",
        mode="before",
    )
    @classmethod
    def _normalize_optional_strings(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = str(value).strip()
        return normalized or None

    @model_validator(mode="after")
    def _run_model_validators(self) -> MLflowConfig:
        if not (self.tracking_uri or self.local_tracking_uri):
            raise ValueError(
                "integrations.mlflow needs either ``tracking_uri`` "
                "or ``local_tracking_uri``."
            )
        return self


__all__ = [
    "MLflowConfig",
]
