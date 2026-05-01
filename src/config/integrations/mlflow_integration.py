"""Settings-level MLflow integration schema.

This is the schema a reusable MLflow integration is validated against
in Settings → Integrations → MLflow. It carries only fields that are
shared across projects — tracking URIs, TLS bundle, system-metrics
cadence. Project-scoped fields (``experiment_name``,
``run_description_file``) live on the project's own ``MLflowConfig``
inside the pipeline YAML.

Security note: this schema deliberately declares no ``token`` /
``tracking_token`` field. The MLflow bearer token is stored encrypted
in ``token.enc`` and fetched at the point of use.
"""

from __future__ import annotations

from pydantic import Field, field_validator

from ..base import StrictBaseModel
from .system_metrics import SystemMetricsConfig


class MLflowIntegrationConfig(StrictBaseModel):
    """MLflow integration (tracking server + system metrics cadence)."""

    tracking_uri: str = Field(
        ...,
        min_length=1,
        description=(
            "Primary MLflow tracking URI for training/runtime and external "
            "access. Required — an integration without a tracker is a dead ref."
        ),
    )
    local_tracking_uri: str | None = Field(
        None,
        description="Optional MLflow tracking URI for the local control plane/orchestrator.",
    )
    ca_bundle_path: str | None = Field(
        None,
        description="Optional CA bundle path for HTTPS verification against MLflow.",
    )

    # Phase 14 follow-up — was four flat ``system_metrics_*`` fields,
    # collapsed into a single nested block with hardcoded defaults
    # (see :class:`SystemMetricsConfig`). Operators tune via YAML;
    # omitting the block yields working defaults.
    system_metrics: SystemMetricsConfig = Field(
        default_factory=SystemMetricsConfig,
        description=(
            "MLflow system-metric collection settings (CPU / GPU / "
            "RAM). Optional — sensible defaults if omitted."
        ),
    )

    @field_validator("tracking_uri", mode="before")
    @classmethod
    def _require_tracking_uri(cls, value: str | None) -> str:
        if value is None:
            raise ValueError("tracking_uri is required")
        normalized = str(value).strip()
        if not normalized:
            raise ValueError("tracking_uri is required")
        return normalized

    @field_validator("local_tracking_uri", "ca_bundle_path", mode="before")
    @classmethod
    def _normalize_optional_strings(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = str(value).strip()
        return normalized or None


__all__ = [
    "MLflowIntegrationConfig",
]
