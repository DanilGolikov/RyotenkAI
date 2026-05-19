"""Project-scoped MLflow config (extends MLflowConnectionConfig).

Adds the fields the pipeline-YAML needs (``experiment_name``,
``run_description_file``, ``system_metrics``, registry naming, alias
on success) on top of the reachability + auth base shared with the
Settings/Integrations UI form.

Replaces the asymmetric pair ``MLflowConfig`` /
``MLflowIntegrationConfig`` (slated for removal in Phase M4
atomically with the pod-trainer rewire — see
``docs/plans/vectorized-fluttering-mist.md`` migration plan).

The base ``MLflowConnectionConfig`` lives in
``ryotenkai_shared.infrastructure.mlflow.config``. The split keeps
the leaf-only ``shared.infrastructure.mlflow`` package free of
project-domain knowledge (experiment names, run descriptions) while
the project extension lives where other pipeline-config models do.
"""

from __future__ import annotations

import re
from typing import Self

from pydantic import Field, field_validator, model_validator

from ryotenkai_shared.config.integrations.system_metrics import SystemMetricsConfig
from ryotenkai_shared.infrastructure.mlflow.config import MLflowConnectionConfig

# Required pattern: env__team__purpose (R-09 mitigation — prevents
# dev/prod experiment-name collisions on a shared server). Exactly
# three lowercase kebab-/snake-segments separated by double
# underscores. Each segment is ``[a-z][a-z0-9-]*(?:_[a-z0-9-]+)*``
# (lowercase alnum + dashes, with single underscores allowed inside).
_EXPERIMENT_SEGMENT_RE: re.Pattern[str] = re.compile(
    r"^[a-z][a-z0-9-]*(?:_[a-z0-9-]+)*$",
)


class MLflowProjectConfig(MLflowConnectionConfig):
    """Project-scoped MLflow config (pipeline YAML).

    Inherits reachability + auth fields from
    :class:`~ryotenkai_shared.infrastructure.mlflow.config.MLflowConnectionConfig`.
    """

    experiment_name: str = Field(
        ...,
        min_length=1,
        description=(
            "MLflow experiment name. Must follow the pattern "
            "``<env>__<team>__<purpose>`` (lowercase, double-underscore "
            "separator) to avoid dev/prod collisions on a shared server."
        ),
    )
    run_description_file: str | None = Field(
        default=None,
        description=(
            "Optional path to a markdown file whose content is set as the "
            "``mlflow.note.content`` of the parent run. When omitted, a "
            "bundled default template is used."
        ),
    )
    system_metrics: SystemMetricsConfig = Field(
        default_factory=SystemMetricsConfig,
        description=(
            "System-metrics (CPU/GPU/RAM) collection settings. The new "
            "architecture uses MLflow's native sampler via "
            "``MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING=true``; this field "
            "stays as a feature toggle for parity with the pre-refactor "
            "callback."
        ),
    )
    model_registry_name_template: str = Field(
        default="ryotenkai/{experiment}/{model_family}",
        description=(
            "Template for the registered-model name. Two placeholders are "
            "substituted at publish-time: ``{experiment}`` from "
            "``experiment_name`` and ``{model_family}`` from the trainer."
        ),
    )
    alias_on_success: str = Field(
        default="challenger",
        description=(
            "Alias assigned to a freshly-published model version on a "
            "successful pipeline. Promotion to ``champion`` is a separate "
            "manual CLI step (``ryotenkai model promote``)."
        ),
    )

    @field_validator("experiment_name")
    @classmethod
    def _validate_experiment_name(cls, value: str) -> str:
        """Validate experiment_name is a non-empty string.

        The R-09 recommendation is the ``env__team__purpose`` three-segment
        pattern (e.g. ``dev__alignment__sft_smoke``) to keep dev/prod
        runs from colliding on a shared MLflow server. That pattern is
        a *guideline* — enforced loosely so existing single-segment
        experiment names (``my_run``, ``sft-baseline``) keep working.

        For new projects use the recommended ``env__team__purpose``
        layout — described in ``docs/codestyle/CODE_MLFLOW.md``.
        """
        normalized = value.strip()
        if not normalized:
            msg = "experiment_name must be a non-empty string."
            raise ValueError(msg)
        return normalized

    @field_validator("run_description_file", "model_registry_name_template", "alias_on_success", mode="before")
    @classmethod
    def _normalize_blank(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = str(value).strip()
        return normalized or None

    @model_validator(mode="after")
    def _validate_template(self) -> Self:
        # Must contain both placeholders for the publisher to substitute.
        tpl = self.model_registry_name_template or ""
        for required in ("{experiment}", "{model_family}"):
            if required not in tpl:
                msg = (
                    "model_registry_name_template must contain both "
                    "'{experiment}' and '{model_family}' placeholders. "
                    f"Got: {tpl!r}"
                )
                raise ValueError(msg)
        return self


__all__ = ["MLflowProjectConfig"]
