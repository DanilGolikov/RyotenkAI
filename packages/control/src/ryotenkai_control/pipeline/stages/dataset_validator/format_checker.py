"""Per-strategy column-format checks for a loaded dataset.

Wraps the pure helper :func:`src.data.validation.standalone.check_dataset_format`
so the HTTP API (``POST /projects/.../datasets/.../validate``) and the
DatasetValidator stage answer from the same code path.

Behaviour: O(1) — only reads column metadata, no dataset iteration.
Called BEFORE quality plugins to fail fast before GPU spin-up. Raises
:class:`DatasetValidationFailedError` on the first failure.

Phase A2 Batch 8 — raise-based migration. The legacy ``Result`` adapter
is gone; the underlying ``check_dataset_format`` already raises, so the
"unknown strategy" branch propagates verbatim and per-strategy failures
are translated into the same exception type.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ryotenkai_shared.errors import DatasetValidationFailedError

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
    ) -> None:
        """Run format checks for every strategy phase. Early-fails on first error.

        Raises:
            DatasetValidationFailedError: on the first failed strategy. The
                "unknown strategy type" branch raised by
                :func:`check_dataset_format` propagates unchanged.
        """
        from ryotenkai_control.data.validation.standalone import check_dataset_format

        items = check_dataset_format(dataset, dataset_name, strategy_phases, self._config)

        for item in items:
            if not item.ok:
                raise DatasetValidationFailedError(
                    detail=(
                        f"[{dataset_name}] Format check failed for "
                        f"'{item.strategy_type}': {item.message}"
                    ),
                    context={
                        "dataset_name": dataset_name,
                        "strategy_type": item.strategy_type,
                        "legacy_code": "DATASET_FORMAT_ERROR",
                    },
                )


__all__ = ["FormatChecker"]
