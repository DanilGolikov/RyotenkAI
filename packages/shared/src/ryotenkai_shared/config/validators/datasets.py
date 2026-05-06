"""Dataset config validators (post-discriminated-unions).

The legacy ``validate_dataset_source_blocks`` rule (source_type=X requires
source_X block) is gone — Pydantic's Tag-based discriminated union
enforces structural correctness at YAML load. Module preserved as a
placeholder for future cross-field rules; today there are none.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..datasets.schema import DatasetConfig


def validate_dataset_source_blocks(cfg: DatasetConfig) -> None:
    """No-op (kept by name for any external callers until PR-9).

    The legacy "source_type=X requires source_X block" rules are
    redundant — Pydantic's discriminated union enforces structurally
    at YAML load time.
    """
    _ = cfg  # silence linter


__all__ = ["validate_dataset_source_blocks"]
