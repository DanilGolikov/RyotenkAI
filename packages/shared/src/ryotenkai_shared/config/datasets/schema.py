from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import Field

from ..base import StrictBaseModel
from .constants import (
    SOURCE_TYPE_HUGGINGFACE,
    SOURCE_URI_HUGGINGFACE_PREFIX,
)
from .source import DatasetSourceUnion

# NOTE: Runtime imports are required for Pydantic field types.
from .sources import DatasetSourceHF, DatasetSourceLocal  # noqa: TC001
from .validation import DatasetValidationsConfig


class DatasetConfig(StrictBaseModel):
    """Dataset configuration with discriminated source union.

    Architecture (post-discriminated-unions refactor):
      * ``source: DatasetSourceUnion`` — Tag-based discriminated union over
        ``DatasetSourceLocal`` (``kind="local"``) /
        ``DatasetSourceHF`` (``kind="huggingface"``). The discriminator
        narrows the type at YAML load.

    YAML example (local)::

        datasets:
          default:
            source:
              kind: local
              local_paths:
                train: data/train.jsonl

    YAML example (huggingface)::

        datasets:
          default:
            source:
              kind: huggingface
              train_id: my-org/dataset
    """

    # =========================================================================
    # SOURCE (Tag-based discriminated union)
    # =========================================================================
    source: DatasetSourceUnion = Field(  # type: ignore[valid-type]
        ...,
        description=(
            "Dataset source. Pydantic dispatches on ``kind`` "
            "(``local`` | ``huggingface``) to the matching source class."
        ),
    )

    # === Processing options ===
    max_samples: int | None = Field(default=None, description="Limit samples for experiments")
    adapter_type: str | None = Field(default=None)
    adapter_config: dict[str, Any] = Field(default_factory=dict)

    # === UI / lifecycle metadata ===
    auto_created: bool = Field(
        default=False,
        description="True when this dataset was auto-created together with a strategy (1:1 lifecycle).",
    )

    validations: DatasetValidationsConfig = Field(
        default_factory=lambda: DatasetValidationsConfig(critical_failures=0, mode="fast", plugins=[]),
    )

    # =========================================================================
    # =========================================================================
    # ACCESSORS
    # =========================================================================
    def get_source_uri(self) -> str:
        """Get source URI for MLflow tracking.

        Returns:
            URI string like "file:///abs/path" or "huggingface://dataset_id"
        """
        if isinstance(self.source, DatasetSourceHF):
            return f"{SOURCE_URI_HUGGINGFACE_PREFIX}{self.source.train_id}"
        return str(Path(self.source.local_paths.train).resolve())

    def is_huggingface(self) -> bool:
        """Check if this config uses HuggingFace source."""
        return self.source.kind == SOURCE_TYPE_HUGGINGFACE

    def get_display_train_ref(self) -> str:
        """Human-friendly reference for logging/debugging."""
        if isinstance(self.source, DatasetSourceHF):
            return self.source.train_id
        return self.source.local_paths.train


__all__ = [
    "DatasetConfig",
]
