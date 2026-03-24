from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..datasets.schema import DatasetConfig


def validate_dataset_source_blocks(cfg: DatasetConfig) -> None:
    """Validate source_* blocks against source_type and enforce required fields."""

    st = cfg.get_source_type()
    if st == "huggingface":
        # If source_type == 'huggingface' → require source_hf block.
        if cfg.source_hf is None:
            raise ValueError("source_type='huggingface' requires 'source_hf:' block")
        return

    # local
    if st == "local":
        # If source_type == 'local' → require source_local block.
        if cfg.source_local is None:
            raise ValueError("source_type='local' requires 'source_local:' block")
        return

    raise ValueError(f"non supported source_type='{st}'")


__all__ = ["validate_dataset_source_blocks"]
