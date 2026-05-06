from __future__ import annotations

from typing import Literal

from pydantic import Field

from ..base import StrictBaseModel


class DatasetLocalPaths(StrictBaseModel):
    """
    Local filesystem paths (on control plane / pipeline launch machine).

    REQUIRED fields (explicit config v6.0):
    - train: Path to training dataset (REQUIRED)

    OPTIONAL fields:
    - eval: Path to evaluation dataset (optional, can be None)
    """

    train: str = Field(
        ...,
        description="REQUIRED: Local training dataset path (absolute or relative to config source_root).",
    )
    eval: str | None = Field(
        None,
        description="Optional: Local evaluation dataset path. Absolute or relative to config source_root.",
    )


class DatasetSourceLocal(StrictBaseModel):
    """
    Local source config: local_paths only.

    Used with ``DatasetConfig.source.kind == "local"`` (Tag-based discriminated
    union — see ``..source._union:DatasetSourceUnion``).

    training_paths removed in v6.0 - auto-generated as:
        data/{strategy_type}/{basename(local_paths.train)}
    """

    kind: Literal["local"] = "local"

    local_paths: DatasetLocalPaths = Field(
        ...,
        description="REQUIRED: Local dataset paths (train required, eval optional)",
    )


class DatasetSourceHF(StrictBaseModel):
    """HuggingFace source config: train_id + optional eval_id.

    Used with ``DatasetConfig.source.kind == "huggingface"`` (Tag-based
    discriminated union — see ``..source._union:DatasetSourceUnion``).
    """

    kind: Literal["huggingface"] = "huggingface"

    train_id: str = Field(..., description="HuggingFace dataset identifier for training set.")
    eval_id: str | None = Field(None, description="Optional HuggingFace dataset identifier for eval set.")


__all__ = [
    "DatasetLocalPaths",
    "DatasetSourceHF",
    "DatasetSourceLocal",
]
