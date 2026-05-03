from __future__ import annotations

from typing import Any


class PipelineDatasetMixin:
    """
    Dataset-related helper methods for PipelineConfig.

    NOTE: This mixin must be used with a Pydantic model that defines:
    - self.datasets: dict[str, DatasetConfig]
    - self.training.get_strategy_chain(): list[StrategyPhaseConfig]
    - self.training.strategies: list[StrategyPhaseConfig]
    """

    datasets: dict
    training: Any

    def get_dataset(self, name: str | None = None):
        """
        Get dataset config by name.

        Args:
            name: Dataset name from registry. None = use primary dataset

        Returns:
            DatasetConfig for the requested dataset

        Raises:
            KeyError: If dataset name not found in registry
        """
        # Local import to avoid heavy side-effects at module import time.
        from src.utils.logger import logger

        if name is None:
            return self.get_primary_dataset()
        if name not in self.datasets:
            available = list(self.datasets.keys())
            logger.debug(f"[CFG:DATASET_NOT_FOUND] name={name}, available={available}")
            raise KeyError(f"Dataset '{name}' not found. Available: {available}")
        logger.debug(f"[CFG:DATASET_RESOLVED] name={name}, ref={self.datasets[name].get_display_train_ref()}")
        return self.datasets[name]

    def get_primary_dataset(self):
        """
        Get the primary dataset for validation/logging.

        Resolution order:
        1. "default" dataset if exists
        2. First strategy's dataset
        3. First available dataset

        Returns:
            DatasetConfig

        Raises:
            KeyError: If no datasets configured
        """
        # 1. Try "default"
        if "default" in self.datasets:
            return self.datasets["default"]

        # 2. Try first strategy's dataset
        strategies = self.training.get_strategy_chain()
        if strategies and strategies[0].dataset:
            dataset_name = strategies[0].dataset
            if isinstance(dataset_name, str) and dataset_name in self.datasets:
                return self.datasets[dataset_name]

        # 3. First available
        available = list(self.datasets.keys())
        if available:
            return self.datasets[available[0]]

        raise KeyError("No datasets configured")

    def get_dataset_for_strategy(self, strategy):
        """
        Get the dataset config for a strategy phase.

        Resolution order:
        1. If strategy.dataset is set → lookup in registry
        2. Otherwise → use primary dataset

        Args:
            strategy: Strategy phase config

        Returns:
            DatasetConfig for this strategy
        """
        # Local import to avoid heavy side-effects at module import time.
        from src.utils.logger import logger

        if strategy.dataset:
            dataset = self.get_dataset(strategy.dataset)
            logger.debug(
                f"[CFG:DATASET_FOR_STRATEGY] strategy={strategy.strategy_type}, "
                f"dataset={strategy.dataset}, ref={dataset.get_display_train_ref()}"
            )
            return dataset
        dataset = self.get_primary_dataset()
        logger.debug(
            f"[CFG:DATASET_FOR_STRATEGY] strategy={strategy.strategy_type}, dataset=primary, ref={dataset.get_display_train_ref()}"
        )
        return dataset


__all__ = [
    "PipelineDatasetMixin",
]
