"""Core MLflow runtime config.

This module is **core schema only** — no knowledge of integrations
(the registry that lives at ``~/.ryotenkai/integrations/``), no
resolver, no UX-layer types.

Project YAMLs may use the convenience shorthand
``integrations.mlflow.integration: <id>`` to pull values from a
saved Settings integration. That substitution happens **before**
core validation, in
:func:`src.workspace.integrations.resolver.resolve_yaml_integrations`
— by the time a ``MLflowConfig`` instance is constructed, the
``integration`` reference has been expanded into the corresponding
``tracking_uri`` / ``local_tracking_uri`` / ``ca_bundle_path`` fields.

The ``integration`` field on this model is deliberately retained as
an optional **secrets-tag**: pipeline stages call
``secrets.get_hf_token(cfg.integration)`` etc. to look up encrypted
tokens stored under the integration's workspace. That's runtime
semantics, not a UX-registry pointer.
"""

from __future__ import annotations

from pydantic import Field, field_validator, model_validator

from ..base import StrictBaseModel
from .system_metrics import SystemMetricsConfig


class MLflowConfig(StrictBaseModel):
    """Runtime view of an MLflow tracker for a single project.

    Either ``tracking_uri`` or ``local_tracking_uri`` must be set;
    project YAMLs typically inherit them via the
    ``integration: <id>`` shorthand which the UX-layer resolver
    expands inline before this class is validated.
    """

    tracking_uri: str | None = Field(None)
    local_tracking_uri: str | None = Field(None)
    ca_bundle_path: str | None = Field(None)
    experiment_name: str = Field(..., description="MLflow experiment name")
    run_description_file: str | None = Field(None)

    # Optional secrets-tag — set by the UX-layer resolver when the
    # YAML used the ``integration:`` shorthand. Runtime code reads
    # this to look up the integration's encrypted token. Not the same
    # thing as a UX entity reference; by the time core sees this, the
    # tracking_uri / local_tracking_uri / ca_bundle_path have already
    # been inlined.
    integration: str | None = Field(
        None,
        description=(
            "Identifier under ``~/.ryotenkai/integrations/<id>/`` for "
            "secrets lookup (read by ``secrets.get_provider_token``). "
            "Set automatically by the UX resolver when a project YAML "
            "uses the ``integration: <id>`` shorthand; users do not "
            "typically write this field directly."
        ),
    )

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
        "integration",
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
                "or ``local_tracking_uri``. Tip: use the ``integration: "
                "<id>`` shorthand to inherit them from a saved Settings "
                "integration."
            )
        return self


__all__ = [
    "MLflowConfig",
]
