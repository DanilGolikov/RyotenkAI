"""
Evaluation stage configuration.

Top-level block 'evaluation:' in pipeline_config.yaml.

Structure:
    evaluation:
      enabled: false
      dataset:
        path: data/eval/helixql_eval.jsonl
      evaluators:
        - id: syntax_check
          plugin: helixql_syntax
          enabled: true
          params: {}
          thresholds:
            min_valid_ratio: 0.80
        - id: answer_quality
          plugin: cerebras_judge
          enabled: false
          params:
            model: gpt-4o-mini
            max_samples: 50
          thresholds:
            min_score: 3.5
"""

from __future__ import annotations

from typing import Any

from pydantic import Field, field_validator

from ..base import StrictBaseModel


class EvaluationDatasetConfig(StrictBaseModel):
    """Dataset dedicated for evaluation (separate from training split)."""

    path: str = Field(..., description="Absolute or project-relative path to JSONL eval dataset.")


class EvaluatorPluginConfig(StrictBaseModel):
    """Single evaluation plugin instance."""

    id: str = Field(..., description="Unique instance id used in metrics, artifacts, and reports.")
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


class EvaluatorsConfig(StrictBaseModel):
    """Flat list of evaluator plugin instances."""

    plugins: list[EvaluatorPluginConfig] = Field(default_factory=list)

    @field_validator("plugins")
    @classmethod
    def validate_unique_plugin_ids(cls, plugins: list[EvaluatorPluginConfig]) -> list[EvaluatorPluginConfig]:
        seen: set[str] = set()
        for plugin in plugins:
            if plugin.id in seen:
                raise ValueError(f"Duplicate evaluation plugin id: {plugin.id!r}")
            seen.add(plugin.id)
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
