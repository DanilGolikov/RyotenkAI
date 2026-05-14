"""Per-strategy column-format checks for a loaded dataset.

Wraps the pure helper :func:`src.data.validation.standalone.check_dataset_format`
so the HTTP API (``POST /projects/.../datasets/.../validate``) and the
DatasetValidator stage answer from the same code path.

Behaviour: O(1) — only reads column metadata, no dataset iteration.
Called BEFORE quality plugins to fail fast before GPU spin-up. Returns
the first failure with code ``DATASET_FORMAT_ERROR``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ryotenkai_shared.utils.result import AppError, DatasetError, Err, Ok, Result

if TYPE_CHECKING:
    from datasets import Dataset, IterableDataset

    from ryotenkai_shared.config import PipelineConfig


class FormatChecker:
    """Runs per-strategy ``check_dataset_format`` against a loaded dataset."""

    def __init__(self, config: PipelineConfig) -> None:
        self._config = config

    def check(
        self,
        dataset: Dataset | IterableDataset,
        dataset_name: str,
        strategy_phases: list,
    ) -> Result[None, AppError]:
        """Run format checks for every strategy phase. Early-fails on first error."""
        from ryotenkai_control.data.validation.standalone import check_dataset_format
        from ryotenkai_shared.errors import DatasetValidationFailedError

        # Phase A2 Batch 7: check_dataset_format now returns the bare list
        # and raises ``DatasetValidationFailedError`` on the
        # "unknown strategy type" branch. The Batch 8 dataset_validator
        # migration drops this adapter; for now we translate back to the
        # legacy ``Result`` shape its callers still expect.
        try:
            items = check_dataset_format(dataset, dataset_name, strategy_phases, self._config)
        except DatasetValidationFailedError as exc:
            return Err(DatasetError(message=exc.detail or str(exc), code="DATASET_FORMAT_ERROR"))

        for item in items:
            if not item.ok:
                return Err(
                    DatasetError(
                        message=(
                            f"[{dataset_name}] Format check failed for "
                            f"'{item.strategy_type}': {item.message}"
                        ),
                        code="DATASET_FORMAT_ERROR",
                    )
                )
        return Ok(None)


__all__ = ["FormatChecker"]
