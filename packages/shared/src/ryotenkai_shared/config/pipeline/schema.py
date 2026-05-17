from __future__ import annotations

# NOTE: Runtime imports are required for Pydantic field types and PrivateAttr annotations.
from pathlib import Path  # noqa: TC003
from typing import Any

from pydantic import Field, PrivateAttr, model_validator

from ..base import StrictBaseModel
from ..datasets import DatasetConfig  # noqa: TC001
from ..evaluation import EvaluationConfig
from ..inference import InferenceConfig
from ..integrations import IntegrationsConfig, HuggingFaceHubConfig
from ..model import ModelConfig  # noqa: TC001
from ..pod_lifecycle import PodLifecycleConfig
from ..reports import ReportsConfig
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
    # NOTE: declared as ``dict[str, Any]`` because each block is post-
    # validated into the provider-specific Pydantic schema declared in
    # ``provider.toml`` (see ``_validate_provider_blocks_against_manifests``).
    # YAML loads as raw dict; the validator promotes each entry to the
    # provider's typed config class (RunPodProviderConfig /
    # SingleNodeProviderConfig). Downstream code reads typed attributes.
    providers: dict[str, Any] = Field(
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
    integrations: IntegrationsConfig = Field(default_factory=IntegrationsConfig)  # type: ignore[arg-type]

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

    # Reports section control — single source of truth for section order.
    reports: ReportsConfig = Field(
        default_factory=ReportsConfig,
        description=(
            "Which report plugins appear in the post-run Markdown report and in "
            "what order. Default (null) uses the built-in section list."
        ),
    )

    # Optional pod-lifecycle thresholds. Translates to env vars at
    # job-submission time so the in-pod :class:`IdleDetector` no longer
    # has to hard-code 48h / 20m (E-СРЕД fix, post-Phase-10 research).
    pod_lifecycle: PodLifecycleConfig | None = Field(
        default=None,
        description=(
            "Optional auto-shutdown thresholds for the pod runner's "
            "IdleDetector. When omitted, the pod-side defaults "
            "(48h max-lifetime, 20m idle window) apply."
        ),
    )

    # =========================================================================
    # SOURCE CONTEXT (NOT PART OF YAML)
    # =========================================================================
    _source_path: Path | None = PrivateAttr(default=None)
    _source_root: Path | None = PrivateAttr(default=None)

    @property
    def huggingface(self) -> HuggingFaceHubConfig | None:
        """Shortcut to integrations.huggingface."""
        return self.integrations.huggingface

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
        self._validate_provider_blocks_against_manifests()
        return self

    def _validate_provider_blocks_against_manifests(self) -> None:
        """Validate-and-typify every ``providers.<id>`` block against the
        manifest's Pydantic schema (Phase 14.D+F follow-up — concurrent-
        gathering-hippo plan §4.4).

        Each provider declares a ``[entry_points.config_schema]`` Pydantic
        class in its ``provider.toml``; this validator runs
        ``ConfigCls.model_validate(<block>)`` so YAML typos
        ("cloud_typo: ALL" instead of "cloud_type: ALL") surface AT
        CONFIG LOAD, not 30 seconds into the pipeline. The validated
        instance REPLACES the raw dict in ``self.providers[id]`` —
        downstream code reads typed attributes (``block.training.gpu_type``)
        with full mypy coverage.

        Best-effort: when the registry hasn't loaded a provider's
        manifest (e.g. modular runtimes that don't ship the providers
        package, or unit tests with MagicMock configs) the block is
        left as-is. The cross-validator
        ``validate_pipeline_active_provider_is_registered`` already
        rejects unknown provider names earlier in the chain.
        """
        try:
            # Import lazily — modular runtimes (slim CLI without the
            # providers package) skip this whole branch via ImportError.
            import importlib

            registry_mod = importlib.import_module("ryotenkai_providers.registry")
            registry = registry_mod.get_registry()
        except ImportError:
            return

        from pydantic import BaseModel

        for provider_id, raw_block in list(self.providers.items()):
            if isinstance(raw_block, BaseModel):
                # Already typed (e.g. tests construct PipelineConfig
                # programmatically with typed blocks).
                continue
            if provider_id not in registry.list():
                # Unknown provider — surfaced by another validator.
                continue
            try:
                config_cls = registry.get_config_class(provider_id)
            except Exception:  # noqa: BLE001 — ignore registry plumbing errors here
                continue
            try:
                typed_block = config_cls.model_validate(raw_block)
            except Exception as exc:
                raise ValueError(
                    f"providers.{provider_id} block failed validation against "
                    f"{config_cls.__name__}: {exc}"
                ) from exc
            self.providers[provider_id] = typed_block  # type: ignore[assignment]


__all__ = [
    "PipelineConfig",
]
