from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, cast

from pydantic import Field, model_validator

from ..base import StrictBaseModel
from .constants import (
    SOURCE_TYPE_HUGGINGFACE,
    SOURCE_TYPE_LOCAL,
    SOURCE_URI_HUGGINGFACE_PREFIX,
)

# NOTE: Runtime imports are required for Pydantic field types.
from .sources import DatasetSourceHF, DatasetSourceLocal  # noqa: TC001
from .validation import DatasetValidationsConfig


class DatasetConfig(StrictBaseModel):
    """
    Dataset configuration (v6.0).

    Design goals:
    - Explicit source type (local or huggingface)
    - training_paths auto-generated: data/{strategy_type}/{basename(local_paths.train)}
    - No hidden config patching

    Supported sources:
    - local: files on the machine where pipeline is launched
    - huggingface: datasets on HuggingFace Hub (train_id / eval_id)
    """

    # =========================================================================
    # SOURCE TYPE
    # =========================================================================
    source_type: Literal["local", "huggingface"] | None = Field(
        None,
        description="Data source type: 'local' for files, 'huggingface' for HF Hub",
    )

    # === Processing options ===
    max_samples: int | None = Field(default=None, description="Limit samples for experiments")
    adapter_type: str | None = Field(default=None)
    adapter_config: dict[str, Any] = Field(default_factory=dict)

    # === UI / lifecycle metadata ===
    # `auto_created=True` is set by the web UI when a dataset entry is
    # generated together with a strategy (1:1 strict pairing). Removing
    # the strategy then removes the dataset automatically. Manually-
    # added datasets keep this False, so the UI can confirm with the
    # user before deleting them. Backend treats it as opaque metadata.
    auto_created: bool = Field(
        default=False,
        description="True when this dataset was auto-created together with a strategy (1:1 lifecycle).",
    )

    validations: DatasetValidationsConfig = Field(
        default_factory=lambda: DatasetValidationsConfig(critical_failures=0, mode="fast", plugins=[]),
    )

    # =========================================================================
    # SOURCE OBJECTS (mutually exclusive)
    # =========================================================================
    source_local: DatasetSourceLocal | None = Field(
        default=None,
        description="Local dataset source configuration (used when source_type=local).",
    )
    source_hf: DatasetSourceHF | None = Field(
        default=None,
        description="HuggingFace dataset source configuration (used when source_type=huggingface).",
    )

    @model_validator(mode="after")
    def _run_model_validators(self) -> DatasetConfig:
        """
        Centralized cross-field validators for this config.

        Convention:
        - Keep ONE `@model_validator(mode="after")` method per config model.
        - Delegate validation logic to `src.config.validators.*`.
        """
        # Local import to avoid circular imports.
        from ..validators.datasets import validate_dataset_source_blocks

        validate_dataset_source_blocks(self)
        return self

    def get_source_type(self) -> Literal["local", "huggingface"]:
        """
        Get the data source type.

        Auto-detects if not explicitly set:
        - If source_hf is set → huggingface
        - Otherwise → local

        Returns:
            Source type: "local" or "huggingface"
        """
        if self.source_type:
            return self.source_type
        if self.source_hf is not None:
            return cast("Literal['local', 'huggingface']", SOURCE_TYPE_HUGGINGFACE)
        return cast("Literal['local', 'huggingface']", SOURCE_TYPE_LOCAL)

    def get_source_uri(self) -> str:
        """
        Get source URI for MLflow tracking.

        For v6.0: Always uses local_paths.train (training_paths removed).

        Returns:
            URI string like "file:///abs/path" or "huggingface://dataset_id"
        """
        source_type = self.get_source_type()
        if source_type == SOURCE_TYPE_HUGGINGFACE:
            # Prefer train_id as stable identifier
            assert self.source_hf is not None
            return f"{SOURCE_URI_HUGGINGFACE_PREFIX}{self.source_hf.train_id}"

        # Local: use local_paths.train (training_paths removed in v6.0)
        assert self.source_local is not None
        return str(Path(self.source_local.local_paths.train).resolve())

    def is_huggingface(self) -> bool:
        """Check if this config uses HuggingFace source."""
        return self.get_source_type() == SOURCE_TYPE_HUGGINGFACE

    def get_display_train_ref(self) -> str:
        """
        Human-friendly reference for logging/debugging.

        For v6.0: Always uses local_paths.train (training_paths removed).
        """
        if self.get_source_type() == SOURCE_TYPE_HUGGINGFACE:
            assert self.source_hf is not None
            return self.source_hf.train_id
        assert self.source_local is not None
        return self.source_local.local_paths.train


__all__ = [
    "DatasetConfig",
]
