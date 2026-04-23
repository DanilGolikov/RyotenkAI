"""Project-scoped MLflow reference + resolved runtime view.

Starting with PR3, ``experiment_tracking.mlflow`` in the project YAML is
a *reference* to a reusable integration managed in Settings → Integrations.
Only fields that are truly project-local stay inline:

- ``integration``: id of the integration that carries tracking URI,
  token, system-metrics knobs.
- ``experiment_name``: project's experiment name inside that tracker.
- ``run_description_file``: optional path to a run description .md file.

All other MLflow knobs live on the integration side
(``MLflowIntegrationConfig`` in ``mlflow_integration.py``) so multiple
projects can share a single tracker account.

``MLflowConfig`` remains as the **resolved runtime type** (project ref +
integration config merged) for backend code that reads
``config.experiment_tracking.mlflow``. The resolver in
``src/config/integrations/resolver.py`` produces it at config-load time;
pipeline stages keep reading ``cfg.experiment_tracking.mlflow.tracking_uri``
as they did before.

Legacy YAML — `tracking_uri` etc. declared directly under
`experiment_tracking.mlflow` — is rejected with a user-friendly error;
see ``ExperimentTrackingConfig._reject_legacy_keys``.
"""

from __future__ import annotations

from pydantic import Field, field_validator, model_validator

from ..base import StrictBaseModel


class MLflowTrackingRef(StrictBaseModel):
    """Project-scoped reference to a Settings MLflow integration."""

    integration: str | None = Field(
        None,
        description=(
            "Id of the MLflow integration in Settings → Integrations. "
            "When empty, MLflow tracking is disabled for this project."
        ),
    )
    experiment_name: str | None = Field(
        None,
        description="MLflow experiment name inside the selected tracker.",
    )
    run_description_file: str | None = Field(
        None,
        description="Optional path to a run description .md file.",
    )

    @field_validator("integration", "experiment_name", "run_description_file", mode="before")
    @classmethod
    def _normalize(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = str(value).strip()
        return normalized or None

    @model_validator(mode="after")
    def _require_experiment_when_integration_set(self) -> MLflowTrackingRef:
        if self.integration and not self.experiment_name:
            raise ValueError(
                "experiment_tracking.mlflow.experiment_name is required when "
                "experiment_tracking.mlflow.integration is set."
            )
        return self


class MLflowConfig(StrictBaseModel):
    """Resolved MLflow runtime view (project ref + integration).

    Produced by the resolver at config-load time. Runtime code reads this
    object unchanged — ``tracking_uri`` / ``ca_bundle_path`` / system-metrics
    knobs come from the integration; ``experiment_name`` /
    ``run_description_file`` stay from the project.
    """

    tracking_uri: str | None = Field(None)
    local_tracking_uri: str | None = Field(None)
    ca_bundle_path: str | None = Field(None)
    experiment_name: str = Field(..., description="MLflow experiment name")
    run_description_file: str | None = Field(None)

    system_metrics_sampling_interval: int = Field(5, ge=1, le=60)
    system_metrics_samples_before_logging: int = Field(1, ge=1, le=10)
    system_metrics_callback_enabled: bool = Field(False)
    system_metrics_callback_interval: int = Field(10, ge=1, le=100)

    @field_validator(
        "tracking_uri",
        "local_tracking_uri",
        "ca_bundle_path",
        mode="before",
    )
    @classmethod
    def _normalize_optional_strings(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None

    @model_validator(mode="after")
    def _run_model_validators(self) -> MLflowConfig:
        if not (self.tracking_uri or self.local_tracking_uri):
            raise ValueError(
                "At least one of 'tracking_uri' or 'local_tracking_uri' must be set for MLflow"
            )
        return self


__all__ = [
    "MLflowConfig",
    "MLflowTrackingRef",
]
