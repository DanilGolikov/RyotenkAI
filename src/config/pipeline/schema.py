from __future__ import annotations

# NOTE: Runtime imports are required for Pydantic field types and PrivateAttr annotations.
from pathlib import Path  # noqa: TC003
from typing import Any

from pydantic import Field, PrivateAttr, model_validator

from ..base import StrictBaseModel
from ..datasets import DatasetConfig  # noqa: TC001
from ..evaluation import EvaluationConfig
from ..inference import InferenceConfig
from ..integrations import ExperimentTrackingConfig, HuggingFaceHubConfig
from ..model import ModelConfig  # noqa: TC001
from ..training import AdaLoraConfig, LoraConfig, TrainingOnlyConfig
from .datasets import PipelineDatasetMixin
from .io import PipelineIOMixin
from .providers import PipelineProviderMixin


class PipelineConfig(
    PipelineIOMixin,
    PipelineProviderMixin,
    PipelineDatasetMixin,
    StrictBaseModel,
):
    """
    Main configuration - with multi-provider support.

    Structure (v4 - with providers):
        model:
          name: "Qwen/Qwen2.5-7B-Instruct"
          torch_dtype: bfloat16

        providers:                    # NEW: Provider registry
          single_node:
            ...
          runpod:
            ...

        training:
          provider: single_node       # NEW: Reference to provider
          type: qlora
          ...

        datasets:
          default: {...}
    """

    model: ModelConfig

    # =========================================================================
    # PROVIDERS REGISTRY (NEW!)
    # =========================================================================
    providers: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description="Named providers registry. Each provider has 'type' field.",
    )

    # =========================================================================
    # DATASETS REGISTRY
    # =========================================================================
    datasets: dict[str, DatasetConfig] = Field(
        ...,
        description="Named datasets registry. Must contain at least one entry.",
    )

    training: TrainingOnlyConfig = Field(default_factory=TrainingOnlyConfig)  # type: ignore[arg-type]

    # All integrations in one place: MLflow, HuggingFace (all fields required)
    # If not specified, integrations are disabled (all blocks optional)
    experiment_tracking: ExperimentTrackingConfig = Field(default_factory=ExperimentTrackingConfig)  # type: ignore[arg-type]

    # Optional inference deployment stage config (NEW)
    inference: InferenceConfig = Field(
        default_factory=InferenceConfig,  # pyright: ignore[reportCallIssue]  # type: ignore[call-arg]
        description="Inference deployment settings (used by InferenceDeployer stage).",
    )

    # Evaluation stage config (NEW)
    evaluation: EvaluationConfig = Field(
        default_factory=EvaluationConfig,  # pyright: ignore[reportCallIssue]  # type: ignore[call-arg]
        description="Evaluation settings (used by ModelEvaluator stage).",
    )

    # =========================================================================
    # SOURCE CONTEXT (NOT PART OF YAML)
    # =========================================================================
    _source_path: Path | None = PrivateAttr(default=None)
    _source_root: Path | None = PrivateAttr(default=None)

    @property
    def huggingface(self) -> HuggingFaceHubConfig | None:
        """Shortcut to experiment_tracking.huggingface."""
        return self.experiment_tracking.huggingface

    def get_adapter_config(self) -> LoraConfig | AdaLoraConfig:
        """
        Get adapter configuration matching the training type.

        Returns:
            LoraConfig for type="qlora" or type="lora"
            AdaLoraConfig for type="adalora"
        """
        return self.training.get_adapter_config()

    @model_validator(mode="after")
    def _run_model_validators(self) -> PipelineConfig:
        """
        Centralized fail-fast validation for this config.

        IMPORTANT:
        - Schema-only checks only (no FS, no env, no network).
        - Environment/secrets/FS checks belong to validate_environment() stage.
        """
        # Local import to avoid circular imports.
        from ..validators.pipeline import validate_pipeline_config_references

        validate_pipeline_config_references(self)
        return self


__all__ = [
    "PipelineConfig",
]
