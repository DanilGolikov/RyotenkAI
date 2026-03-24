"""
Multi-source Dataset Loader.

Purpose:
- Support training configs where datasets can be either local files or HuggingFace Hub datasets.
- Route each phase's dataset loading to the correct concrete loader based on DatasetConfig.source_type.

Why:
- In multi-phase training, different phases can reference different datasets.
- A single fixed loader (e.g., JSON only) breaks HuggingFace datasets in training.

Implementation:
- Delegates to JsonDatasetLoader for local files
- Delegates to HuggingFaceDatasetLoader for HF Hub
- Chooses per-phase via config.get_dataset_for_strategy(phase).get_source_type()
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.utils.logger import logger

if TYPE_CHECKING:
    from datasets import Dataset

    from src.utils.config import PipelineConfig, StrategyPhaseConfig


class MultiSourceDatasetLoader:
    """Route dataset loading to the correct loader per phase."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self._loaders: dict[str, Any] = {}

    @property
    def _log_prefix(self) -> str:
        return "MULTI_DL"

    def _get_loader(self, source_type: str) -> Any:
        if source_type in self._loaders:
            return self._loaders[source_type]

        loader: Any
        if source_type == "huggingface":
            from src.data.loaders.hf_loader import HuggingFaceDatasetLoader

            loader = HuggingFaceDatasetLoader(self.config)
        else:
            from src.data.loaders.json_loader import JsonDatasetLoader

            loader = JsonDatasetLoader(self.config)

        self._loaders[source_type] = loader
        logger.debug(f"[{self._log_prefix}:CREATED] source_type={source_type}, loader={type(loader).__name__}")
        return loader

    def load(
        self,
        source: str,
        split: str = "train",
        max_samples: int | None = None,
    ) -> Dataset:
        # Heuristic routing for direct load() usage
        from src.data.loaders.hf_loader import HuggingFaceDatasetLoader

        source_type = "huggingface" if HuggingFaceDatasetLoader.is_hf_dataset_id(source) else "local"
        loader = self._get_loader(source_type)
        logger.debug(f"[{self._log_prefix}:LOAD] source={source}, split={split}, source_type={source_type}")
        return loader.load(source, split=split, max_samples=max_samples)

    def validate_source(self, source: str) -> bool:
        from src.data.loaders.hf_loader import HuggingFaceDatasetLoader

        source_type = "huggingface" if HuggingFaceDatasetLoader.is_hf_dataset_id(source) else "local"
        loader = self._get_loader(source_type)
        return bool(loader.validate_source(source))

    def load_for_phase(self, phase: StrategyPhaseConfig) -> Any:
        dataset_config = self.config.get_dataset_for_strategy(phase)
        source_type = dataset_config.get_source_type()
        loader = self._get_loader(source_type)

        logger.debug(
            f"[{self._log_prefix}:PHASE_LOAD] phase={getattr(phase, 'strategy_type', None)}, "
            f"dataset={getattr(phase, 'dataset', None)}, source_type={source_type}, loader={type(loader).__name__}"
        )

        return loader.load_for_phase(phase)


__all__ = ["MultiSourceDatasetLoader"]
