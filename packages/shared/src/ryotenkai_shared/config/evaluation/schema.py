"""
Evaluation stage configuration.

Top-level block 'evaluation:' in pipeline_config.yaml.

Structure:
    evaluation:
      enabled: false
      dataset:
        path: data/eval/helixql_eval.jsonl
      evaluators:
        plugins:
          - plugin: helixql_syntax       # id auto-generated as
                                          #   helixql_syntax_<md5(params)[:8]>
            enabled: true
            params: {}
            thresholds:
              min_valid_ratio: 0.80
          - id: my_custom_judge          # explicit id — overrides auto
            plugin: cerebras_judge
            enabled: false
            params:
              model: gpt-4o-mini
              max_samples: 50
            thresholds:
              min_score: 3.5
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from pydantic import Field, field_validator, model_validator

from ..base import StrictBaseModel


def _autogen_plugin_id(plugin: str, params: dict[str, Any]) -> str:
    """Stable id from plugin name + params hash.

    Identical (plugin, params) ⇒ identical id ⇒ MLflow artifact paths
    stay stable across runs.
    """
    payload = json.dumps(params, sort_keys=True, default=str).encode()
    digest = hashlib.md5(payload, usedforsecurity=False).hexdigest()[:8]
    return f"{plugin}_{digest}"


class EvaluationDatasetConfig(StrictBaseModel):
    """Dataset dedicated for evaluation (separate from training split)."""

    path: str = Field(..., description="Absolute or project-relative path to JSONL eval dataset.")


class EvaluatorPluginConfig(StrictBaseModel):
    """Single evaluation plugin instance.

    ``id`` is OPTIONAL — auto-generated as ``f"{plugin}_{md5(params)[:8]}"``
    when not supplied. Override with an explicit string for human-readable
    names that survive across param changes (otherwise the auto-id changes
    when params change, intentionally — it tracks distinct invocations).

    Collisions within the same parent ``EvaluatorsConfig.plugins`` list
    (rare: identical plugin+params) get suffixed with ``_2`` / ``_3`` …
    by the parent validator.
    """

    id: str | None = Field(
        default=None,
        description=(
            "Optional unique instance id used in metrics, artifacts, and "
            "reports. Auto-generated as ``f'{plugin}_{md5(params)[:8]}'`` "
            "when not supplied."
        ),
    )
    plugin: str = Field(..., description="Registered evaluator plugin name.")
    enabled: bool = True
    save_report: bool = Field(
        default=False,
        description=(
            "Save a per-plugin Markdown report after this plugin completes. "
            "Written to runs/{run}/evaluation/{plugin_name}_report.md."
        ),
    )
    params: dict[str, Any] = Field(default_factory=dict)
    thresholds: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _autofill_id(self) -> EvaluatorPluginConfig:
        if not self.id:
            self.id = _autogen_plugin_id(self.plugin, self.params)
        return self


class EvaluatorsConfig(StrictBaseModel):
    """Flat list of evaluator plugin instances."""

    plugins: list[EvaluatorPluginConfig] = Field(default_factory=list)

    @field_validator("plugins")
    @classmethod
    def validate_unique_plugin_ids(cls, plugins: list[EvaluatorPluginConfig]) -> list[EvaluatorPluginConfig]:
        """Resolve duplicates by appending _2/_3/... to the auto-generated id.

        Identical (plugin, params) pairs produce the same auto-id. The
        first occurrence keeps the bare id, subsequent ones get a
        positional suffix. Explicit ids that collide (rare; mostly
        config bugs) raise.
        """
        # First pass: separate explicit-id collisions (a hard error) from
        # auto-id collisions (resolvable via suffix).
        seen_explicit: set[str] = set()
        for plugin in plugins:
            # ``id`` was already filled by ``_autofill_id`` at this point.
            # We need to know whether it was originally explicit; the
            # heuristic: if id matches the auto-id formula, treat as auto.
            assert plugin.id is not None  # filled by _autofill_id
            auto_id = _autogen_plugin_id(plugin.plugin, plugin.params)
            if plugin.id != auto_id:
                # Explicit id (user-supplied or already-suffixed).
                if plugin.id in seen_explicit:
                    raise ValueError(
                        f"Duplicate explicit evaluator plugin id: {plugin.id!r}"
                    )
                seen_explicit.add(plugin.id)

        # Second pass: collision-resolve duplicate auto-ids by suffixing.
        seen_count: dict[str, int] = {}
        for plugin in plugins:
            base = plugin.id or ""
            n = seen_count.get(base, 0)
            if n > 0:
                plugin.id = f"{base}_{n + 1}"
            seen_count[base] = n + 1

        return plugins


class EvaluationConfig(StrictBaseModel):
    """
    Top-level evaluation configuration block.

    Fail-fast rule (enforced in PipelineConfig model_validator):
        evaluation.enabled=true  requires  inference.enabled=true
    """

    enabled: bool = False

    dataset: EvaluationDatasetConfig | None = Field(
        default=None,
        description="Eval dataset (separate JSONL, not training split).",
    )

    save_answers_md: bool = Field(
        default=True,
        description=(
            "Save a human-readable Markdown file with model answers after inference collection. "
            "Written to runs/{run}/evaluation/answers.md."
        ),
    )

    evaluators: EvaluatorsConfig = Field(default_factory=EvaluatorsConfig)


__all__ = [
    "EvaluationConfig",
    "EvaluationDatasetConfig",
    "EvaluatorPluginConfig",
    "EvaluatorsConfig",
]
