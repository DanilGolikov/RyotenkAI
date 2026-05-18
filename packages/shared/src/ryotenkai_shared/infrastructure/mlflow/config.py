"""Base MLflow reachability config (settings-scoped, no project fields).

This is the base shared by:

* :class:`~ryotenkai_shared.config.integrations.mlflow_project.MLflowProjectConfig`
  (project-scoped; adds ``experiment_name``, ``run_description_file``,
  ``system_metrics``, ``model_registry_name_template``,
  ``alias_on_success``).
* The Settings/Integrations UI form (consumed as-is — needs only
  reachability + auth, never an experiment name).

Replaces the asymmetric pair ``MLflowConfig`` / ``MLflowIntegrationConfig``
(removed in Phase M4 atomically with the pod-trainer rewire).

URI semantics — kept from the audit:

* ``local_tracking_uri`` is the same MLflow server addressed via
  local loopback or docker bridge (control-plane on Mac).
* ``tracking_uri`` is the public/funnel-exposed address (pod-trainer
  on cloud GPU).

Both fields are optional individually, but at least one must be set.
Resolution is performed by :func:`~.uri.RuntimeUriResolver`, not by
this config.

Per ``docs/plans/vectorized-fluttering-mist.md`` §Configuration.
"""

from __future__ import annotations

from typing import Self
from urllib.parse import urlparse

from pydantic import Field, field_validator, model_validator

from ryotenkai_shared.config.base import StrictBaseModel
from ryotenkai_shared.infrastructure.mlflow.auth import MLflowAuthConfig, _AuthNone


class MLflowConnectionConfig(StrictBaseModel):
    """Base reachability + transport config.

    Project-scoped extension lives in
    :class:`~ryotenkai_shared.config.integrations.mlflow_project.MLflowProjectConfig`.
    """

    tracking_uri: str | None = Field(
        default=None,
        description=(
            "Public/funnel-exposed MLflow tracking URI. Used by pod-trainer "
            "subprocess (which cannot reach Mac loopback). May be omitted if "
            "``local_tracking_uri`` is set; one of the two is required."
        ),
    )
    local_tracking_uri: str | None = Field(
        default=None,
        description=(
            "Local-loopback / docker-bridge MLflow URI. Used by control-plane "
            "on the orchestrator host (Mac). May be omitted if "
            "``tracking_uri`` is set; one of the two is required."
        ),
    )
    ca_bundle_path: str | None = Field(
        default=None,
        description=(
            "Path to a CA bundle for verifying the MLflow server's TLS cert. "
            "Passed to ``MlflowTransport`` via ``verify=`` rather than via "
            "process-wide ``REQUESTS_CA_BUNDLE`` mutation."
        ),
    )
    connect_timeout_s: float = Field(default=5.0, gt=0)
    request_timeout_s: float = Field(default=30.0, gt=0)
    retry_total_budget_s: float = Field(default=30.0, gt=0)
    auth: MLflowAuthConfig = Field(
        default_factory=_AuthNone,
        description=(
            "Discriminated-union auth config. Default ``{kind: 'none'}`` "
            "is appropriate for loopback / Tailscale-mesh setups. For "
            "publicly exposed servers, configure ``basic`` or ``bearer`` "
            "with secrets referenced by env-var name."
        ),
    )

    @field_validator("tracking_uri", "local_tracking_uri", "ca_bundle_path", mode="before")
    @classmethod
    def _normalize_blank(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = str(value).strip()
        return normalized or None

    @field_validator("tracking_uri", "local_tracking_uri")
    @classmethod
    def _reject_userinfo(cls, value: str | None) -> str | None:
        """R-07 mitigation: never accept userinfo embedded in the URI.

        Credentials live in ``auth`` (env-var-referenced); inline
        userinfo (``http://user:pass@host``) leaks into logs and tags.
        """
        if value is None:
            return None
        parsed = urlparse(value)
        if parsed.username or parsed.password:
            msg = (
                "MLflow URI must not contain userinfo (user/password). "
                "Use the ``auth`` field with env-var-referenced secrets."
            )
            raise ValueError(msg)
        return value

    @model_validator(mode="after")
    def _at_least_one_uri(self) -> Self:
        if not (self.tracking_uri or self.local_tracking_uri):
            msg = (
                "MLflowConnectionConfig requires at least one of "
                "``tracking_uri`` or ``local_tracking_uri``."
            )
            raise ValueError(msg)
        return self


__all__ = ["MLflowConnectionConfig"]
