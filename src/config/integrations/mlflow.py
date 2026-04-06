from __future__ import annotations

from pydantic import Field, field_validator, model_validator

from ..base import StrictBaseModel


class MLflowConfig(StrictBaseModel):
    """
    MLflow - production-ready experiment tracking with PostgreSQL + MinIO.

    Setup:
        ./docker/mlflow/start.sh
        # http://localhost:5002
        # MinIO UI: http://localhost:9001

    Env vars (optional):
        MLFLOW_TRACKING_TOKEN - for authenticated servers
    """

    tracking_uri: str | None = Field(
        None,
        description="Primary MLflow tracking URI for training/runtime and external access",
    )
    local_tracking_uri: str | None = Field(
        None,
        description="Optional MLflow tracking URI for the local control plane/orchestrator",
    )
    ca_bundle_path: str | None = Field(
        None,
        description="Optional CA bundle path for HTTPS verification against MLflow",
    )
    experiment_name: str = Field(..., description="MLflow experiment name")
    run_description_file: str | None = Field(None, description="Path to run description .md file")

    # System metrics (GPU/CPU/RAM monitoring) - MLflow built-in
    system_metrics_sampling_interval: int = Field(
        5, ge=1, le=60, description="System metrics sampling interval in seconds"
    )
    system_metrics_samples_before_logging: int = Field(
        1, ge=1, le=10, description="Number of samples to collect before logging to MLflow"
    )

    # SystemMetricsCallback - manual GPU/CPU tracking via pynvml/psutil
    # Disabled by default - may cause hangs on some cloud GPU images (RunPod)
    # Enable only if MLflow built-in system metrics are not working
    system_metrics_callback_enabled: bool = Field(
        False,
        description="Enable SystemMetricsCallback for manual GPU/CPU tracking (may hang on some systems)",
    )
    system_metrics_callback_interval: int = Field(
        10, ge=1, le=100, description="Log system metrics every N training steps (if callback enabled)"
    )

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
            raise ValueError("At least one of 'tracking_uri' or 'local_tracking_uri' must be set for MLflow")
        return self


__all__ = [
    "MLflowConfig",
]
