"""
DatasetLoader - Load and validate datasets for training phases.

Single Responsibility: Dataset loading and validation from config paths.

Supports:
- Direct file paths
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast

from datasets import Dataset, load_dataset

from ryotenkai_shared.constants import STRATEGY_SFT
from ryotenkai_pod.trainer.managers.constants import HF_SPLIT_TRAIN
from ryotenkai_shared.errors import DatasetLoadFailedError
from ryotenkai_shared.utils.logger import logger

if TYPE_CHECKING:
    from ryotenkai_shared.config import (
        DatasetSourceHF,
        DatasetSourceLocal,
        PipelineConfig,
        StrategyPhaseConfig,
    )


class DatasetLoader:
    """
    Handles dataset loading and validation for training phases.

    Responsibilities:
    - Load dataset from config paths
    - Apply max_samples limit if configured
    - Validate dataset exists

    Example:
        loader = DatasetLoader(config)
        train, eval_ds = loader.load_for_phase(phase_config)

    Raises:
        DatasetLoadFailedError: on any loader failure (missing dataset
            config, source kind unsupported, file missing, parse error).
            Legacy error codes preserved on ``context["legacy_code"]``.
    """

    def __init__(
        self,
        config: PipelineConfig,
    ):
        """
        Initialize DatasetLoader.

        Args:
            config: Pipeline configuration with dataset paths
        """
        self.config = config
        logger.debug("[DL:INIT] DatasetLoader initialized")

    def load_for_phase(
        self,
        phase: StrategyPhaseConfig,
    ) -> tuple[Dataset, Dataset | None]:
        """
        Load dataset for a training phase.

        Args:
            phase: Strategy phase configuration with dataset reference

        Returns:
            ``(train_dataset, eval_dataset)`` tuple. ``eval_dataset`` is
            ``None`` when not configured.

        Raises:
            DatasetLoadFailedError: dataset config missing, source kind
                unsupported, file not found, or underlying ``load_dataset``
                raised.
        """
        from ryotenkai_shared.config import DatasetSourceHF, DatasetSourceLocal

        try:
            dataset_config = self.config.get_dataset_for_strategy(phase)
            source = dataset_config.source

            # Discriminated dispatch — typed source binds via isinstance, no string match.
            if isinstance(source, DatasetSourceHF):
                train_dataset, eval_dataset = self._load_hf_datasets(source)
            elif isinstance(source, DatasetSourceLocal):
                # Pass strategy_type for auto-generating training paths
                train_dataset, eval_dataset = self._load_local_datasets(
                    source, strategy_type=phase.strategy_type
                )
            else:
                raise DatasetLoadFailedError(
                    detail=f"Unsupported dataset source kind: {source.kind!r}",
                    context={
                        "legacy_code": "DATA_LOADER_UNKNOWN_SOURCE",
                        "source_kind": source.kind,
                    },
                )

            # Apply max_samples if configured
            if dataset_config.max_samples:
                original_train = len(train_dataset)
                train_dataset = train_dataset.select(range(min(len(train_dataset), dataset_config.max_samples)))
                logger.info(f"   Limited train to {len(train_dataset)} samples (from {original_train})")

                if eval_dataset is not None:
                    eval_limit = max(dataset_config.max_samples // 10, 1)
                    original_eval = len(eval_dataset)
                    eval_dataset = eval_dataset.select(range(min(len(eval_dataset), eval_limit)))
                    logger.info(f"   Limited eval to {len(eval_dataset)} samples (from {original_eval})")

            logger.debug(
                f"[DL:LOADED] train={len(train_dataset)}, eval={len(eval_dataset) if eval_dataset is not None else 0}"
            )
            return train_dataset, eval_dataset

        except DatasetLoadFailedError:
            # Already-typed exception from a helper — surface as-is.
            raise

        except KeyError as e:
            error_msg = f"Dataset '{phase.dataset}' not found in config: {e}"
            logger.error(f"[DL:ERROR] {error_msg}")
            raise DatasetLoadFailedError(
                detail=error_msg,
                context={
                    "legacy_code": "DATA_LOADER_DATASET_NOT_FOUND",
                    "phase_dataset": getattr(phase, "dataset", None),
                },
                cause=e,
            )

        except Exception as e:
            error_msg = f"Failed to load dataset: {e}"
            logger.error(f"[DL:ERROR] {error_msg}")
            raise DatasetLoadFailedError(
                detail=error_msg,
                context={"legacy_code": "DATA_LOADER_LOAD_FAILED"},
                cause=e,
            )

    @staticmethod
    def _load_hf_datasets(source: DatasetSourceHF) -> tuple[Dataset, Dataset | None]:
        """Load HuggingFace datasets (train + optional eval)."""
        train_id = source.train_id
        eval_id = source.eval_id

        logger.info(f"   Loading HF train dataset: {train_id}")
        train_dataset = cast(
            "Dataset",  # noqa: WPS226
            load_dataset(train_id, split=HF_SPLIT_TRAIN, trust_remote_code=True),
        )
        eval_dataset: Dataset | None = None
        if eval_id:
            logger.info(f"   Loading HF eval dataset: {eval_id}")
            eval_dataset = cast(
                "Dataset",
                load_dataset(eval_id, split=HF_SPLIT_TRAIN, trust_remote_code=True),
            )

        return train_dataset, eval_dataset

    @staticmethod
    def _load_local_datasets(
        source: DatasetSourceLocal,
        strategy_type: str = STRATEGY_SFT,  # noqa: ARG004 — kept for API stability
    ) -> tuple[Dataset, Dataset | None]:
        """
        Load local datasets from the pod-side ``data/`` directory.

        Args:
            source: Local dataset source configuration (typed via discriminator)
            strategy_type: Current strategy type (UNUSED — see note). Kept in
                the signature so existing callers don't churn.

        Raises:
            DatasetLoadFailedError: if the resolved train file does not exist.

        Path resolution contract (post-bugfix 2026-05-07):
            The Mac-side :class:`FileUploader` uploads dataset files via
            ``POST /api/v1/files/upload`` with ``target=DATASET``. The pod's
            handler (:func:`ryotenkai_pod.runner.api.files.upload_file`)
            stores them at ``<run_dir>/data/<basename>`` — basename only,
            no strategy-type subdirectory. So the trainer reads from the
            same flat layout: ``data/<basename(local_paths.*)>``.

            Earlier code looked at ``data/{strategy_type}/<basename>``,
            which never matched the actual upload layout — that mismatch
            shipped silently because nobody ran a multi-strategy chain
            against a fresh pod after the upload contract was finalised.
            The ``strategy_type`` parameter is kept for backward signature
            compatibility but is no longer used in path construction.
        """
        # Pod-side flat layout: data/<basename>. Match upload contract.
        local_train = source.local_paths.train
        train_basename = Path(local_train).name
        train_rel = f"data/{train_basename}"

        train_path = Path(train_rel)
        logger.debug(f"[DL:LOADING] train_path={train_path} (resolved from local_paths basename)")
        if not train_path.exists():
            error_msg = f"Dataset file not found: {train_path}"
            logger.error(f"[DL:ERROR] {error_msg}")
            raise DatasetLoadFailedError(
                detail=error_msg,
                context={
                    "legacy_code": "DATA_LOADER_FILE_NOT_FOUND",
                    "train_path": str(train_path),
                },
            )

        logger.info(f"   Loading dataset: {train_path}")
        train_dataset = cast(
            "Dataset",
            load_dataset("json", data_files=str(train_path), split=HF_SPLIT_TRAIN),
        )

        eval_dataset: Dataset | None = None
        local_eval = source.local_paths.eval
        if local_eval:
            eval_basename = Path(local_eval).name
            eval_rel = f"data/{eval_basename}"
            eval_path = Path(eval_rel)
            if eval_path.exists():
                logger.info(f"   Loading eval dataset: {eval_path}")
                eval_dataset = cast(
                    "Dataset",
                    load_dataset("json", data_files=str(eval_path), split=HF_SPLIT_TRAIN),
                )
            else:
                logger.warning(f"[DL:EVAL_MISSING] eval dataset not found: {eval_path} (skipping)")

        return train_dataset, eval_dataset

    @staticmethod
    def validate_exists(train_path: str) -> bool:
        """
        Check if dataset file exists.

        Args:
            train_path: Path to training data file

        Returns:
            True if file exists
        """
        exists = Path(train_path).exists()
        logger.debug(f"[DL:VALIDATE] path={train_path}, exists={exists}")
        return exists

    @staticmethod
    def load(
        source: str,
        split: str = HF_SPLIT_TRAIN,
        max_samples: int | None = None,
    ) -> Dataset:
        """
        Load dataset from source.

        Implements IDatasetLoader interface.

        Args:
            source: File path to dataset
            split: Dataset split (default: "train")
            max_samples: Maximum samples to load

        Returns:
            Loaded dataset
        """
        logger.debug(f"[DL:LOAD] source={source}")

        dataset = load_dataset("json", data_files=source, split=split)
        dataset = cast("Dataset", dataset)

        if max_samples:
            dataset = dataset.select(range(min(len(dataset), max_samples)))

        return dataset

    @staticmethod
    def validate_source(source: str) -> bool:
        """
        Validate dataset source exists.

        Implements IDatasetLoader interface.

        Args:
            source: File path to dataset

        Returns:
            True if source is valid
        """
        return Path(source).exists()


__all__ = ["DatasetLoader"]
