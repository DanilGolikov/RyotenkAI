"""Load train/eval splits of a dataset for validation.

Two source types:

* **HuggingFace** — uses ``load_dataset(..., streaming=True)`` so the
  dataset is never fully downloaded; ``fast`` mode further caps the
  iterator with ``.take(max_samples)``.
* **Local files** — goes through the project's
  :class:`DatasetLoaderFactory`; ``fast`` mode caps ``.select(range(N))``.

Owns the :class:`DatasetLoaderFactory` ref via DI; exposes
``load_train`` / ``try_load_eval`` / ``get_size`` / ``get_train_ref``
to the facade.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from src.config.datasets.constants import SOURCE_TYPE_HUGGINGFACE
from src.pipeline.stages.dataset_validator.constants import (
    SPLIT_TRAIN,
    VALIDATION_MAX_SAMPLES_FAST,
    VALIDATION_MODE_ATTR,
    VALIDATION_MODE_FAST,
    VALIDATIONS_ATTR,
)
from src.utils.logger import logger

if TYPE_CHECKING:
    from datasets import Dataset, IterableDataset

    from src.data.loaders.factory import DatasetLoaderFactory


class DatasetSplitLoader:
    """Load HF / local train/eval splits with validation-mode awareness."""

    def __init__(self, loader_factory: DatasetLoaderFactory) -> None:
        self._loader_factory = loader_factory

    def load_train(self, dataset_config: Any) -> Dataset | IterableDataset | None:
        """Load the train split for validation. Returns None on failure."""
        loader = self._loader_factory.create_for_dataset(dataset_config)
        return self._load(dataset_config, loader, split_name=SPLIT_TRAIN)  # type: ignore[arg-type]

    def try_load_eval(
        self, dataset_config: Any
    ) -> tuple[Dataset | IterableDataset | None, str | None]:
        """Load eval split if configured; returns ``(dataset, ref)``.

        Returns ``(None, None)`` when no eval is configured for the
        dataset, when load fails, or on unexpected exceptions.
        """
        try:
            loader = self._loader_factory.create_for_dataset(dataset_config)
            if dataset_config.get_source_type() == SOURCE_TYPE_HUGGINGFACE:
                if dataset_config.source_hf is None or not dataset_config.source_hf.eval_id:
                    return None, None
                ds = self._load(dataset_config, loader, split_name="eval")
                return ds, dataset_config.source_hf.eval_id

            if dataset_config.source_local is None or not dataset_config.source_local.local_paths.eval:
                return None, None
            ds = self._load(dataset_config, loader, split_name="eval")
            return ds, dataset_config.source_local.local_paths.eval
        except Exception:
            return None, None

    @staticmethod
    def get_size(dataset: Dataset | IterableDataset) -> int:
        """Number of samples; ``-1`` for streaming (unknown)."""
        from datasets import IterableDataset as HFIterableDataset

        if isinstance(dataset, HFIterableDataset):
            return -1
        return len(dataset)

    @staticmethod
    def get_train_ref(dataset_config: Any) -> str:
        """Stable train reference string for logging / events."""
        try:
            if dataset_config.get_source_type() == SOURCE_TYPE_HUGGINGFACE and dataset_config.source_hf is not None:
                return dataset_config.source_hf.train_id
            if dataset_config.source_local is not None:
                return dataset_config.source_local.local_paths.train
        except Exception:
            pass
        return "unknown"

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _load(
        dataset_config: Any,
        loader: Any,
        *,
        split_name: Literal["train", "eval"] = SPLIT_TRAIN,  # type: ignore[assignment]
    ) -> Dataset | IterableDataset | None:
        """Load a split: HF streaming or local-file path."""
        from datasets import IterableDataset as HFIterableDataset

        source_type = dataset_config.get_source_type()
        validation_mode = getattr(
            getattr(dataset_config, VALIDATIONS_ATTR, None), VALIDATION_MODE_ATTR, VALIDATION_MODE_FAST
        )

        if source_type == SOURCE_TYPE_HUGGINGFACE:
            try:
                from datasets import load_dataset

                token = None
                if hasattr(loader, "token"):
                    token = loader.token

                src = (
                    dataset_config.source_hf.train_id if split_name == SPLIT_TRAIN else dataset_config.source_hf.eval_id
                )
                if not src:
                    return None

                dataset = load_dataset(
                    src,
                    split="train",
                    streaming=True,  # KEY: don't download full dataset
                    token=token,
                    trust_remote_code=True,
                )

                if not isinstance(dataset, HFIterableDataset):
                    logger.warning(f"Expected IterableDataset, got {type(dataset)}")
                    return None

                if validation_mode == VALIDATION_MODE_FAST:
                    max_samples = dataset_config.max_samples or VALIDATION_MAX_SAMPLES_FAST
                    dataset = dataset.take(max_samples)
                    logger.info(f"Loaded HF dataset (streaming, fast mode): {src}")
                    logger.info(f"  Validation sample size: {max_samples}")
                else:
                    logger.info(f"Loaded HF dataset (streaming, full mode): {src}")
                    logger.warning("  Full validation mode - this may take a while for large datasets")

                return dataset

            except Exception as e:
                logger.error(f"Failed to load HF dataset: {e}")
                return None

        # Local files
        source_local = dataset_config.source_local
        if source_local is None:
            return None
        local_path_str = (
            source_local.local_paths.train if split_name == SPLIT_TRAIN else source_local.local_paths.eval
        )
        if not local_path_str:
            return None

        local_path = Path(local_path_str)
        if not local_path.exists():
            logger.error(f"Dataset not found: {local_path}")
            return None

        try:
            dataset = loader.load(str(local_path), split="train")
            total_samples = len(dataset)

            if validation_mode == VALIDATION_MODE_FAST and total_samples > VALIDATION_MAX_SAMPLES_FAST:
                dataset = dataset.select(range(VALIDATION_MAX_SAMPLES_FAST))
                logger.info(f"Loaded local dataset (fast mode): {local_path}")
                logger.info(f"  Validation sample: {VALIDATION_MAX_SAMPLES_FAST} / {total_samples}")
            else:
                logger.info(f"Loaded local dataset (full mode): {local_path}")
                logger.info(f"  Total samples: {total_samples}")

            return dataset

        except Exception as e:
            logger.error(f"Failed to load local dataset: {e}")
            return None


__all__ = ["DatasetSplitLoader"]
