"""
MinSamples validation plugin.

Checks if dataset has minimum number of samples.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, ClassVar

from src.data.validation.base import ValidationPlugin, ValidationResult
from src.data.validation.registry import ValidationPluginRegistry
from src.utils.logger import logger

if TYPE_CHECKING:
    from datasets import Dataset, IterableDataset

_DEFAULT_SAMPLE_SIZE = 10_000


@ValidationPluginRegistry.register
class MinSamplesValidator(ValidationPlugin):
    """
    Validates minimum sample count in dataset.

    Params:
        threshold (int): Minimum number of samples (default: 100)
        sample_size (int): For streaming datasets, how many to load (default: 10000)

    Recommendations on failure:
        - Add more samples to dataset
        - Use data augmentation
        - Lower threshold if testing
    """

    name = "min_samples"
    priority = 10  # Run early (basic check)
    expensive = False
    supports_streaming = True

    MANIFEST: ClassVar[dict[str, Any]] = {
        "description": "Checks that the dataset has at least N examples.",
        "category": "basic",
        "stability": "stable",
        "params_schema": {
            "sample_size": {
                "type": "integer",
                "min": 1,
                "default": _DEFAULT_SAMPLE_SIZE,
            },
        },
        "thresholds_schema": {
            "threshold": {"type": "integer", "min": 1, "default": 100},
        },
        "suggested_params": {"sample_size": _DEFAULT_SAMPLE_SIZE},
        "suggested_thresholds": {"threshold": 100},
    }

    @classmethod
    def get_description(cls) -> str:
        return "Checks the minimum number of examples in the dataset"

    def validate(
        self,
        dataset: Dataset | IterableDataset,
    ) -> ValidationResult:
        """Check minimum sample count."""
        from datasets import IterableDataset

        start_time = time.time()
        threshold = self._threshold("threshold", 100)
        errors = []
        warnings = []

        # For streaming, we need to count samples (limited)
        if isinstance(dataset, IterableDataset):
            sample_size = self._param("sample_size", _DEFAULT_SAMPLE_SIZE)
            logger.debug(f"[{self.name}] Counting samples from IterableDataset (max: {sample_size})")
            samples = list(dataset.take(sample_size))
            count = len(samples)

            # Streaming: exact size is unknown
            if count == sample_size:
                warnings.append(f"Dataset might have more than {sample_size} samples (streaming mode, limited check)")
        else:
            count = len(dataset)

        passed = count >= threshold

        if not passed:
            errors.append(f"Not enough examples: {count} < {threshold}")

        execution_time = (time.time() - start_time) * 1000

        return ValidationResult(
            plugin_name=self.name,
            passed=passed,
            params=dict(self.params),
            thresholds={"threshold": float(threshold)},
            metrics={
                "sample_count": float(count),
            },
            warnings=warnings,
            errors=errors,
            execution_time_ms=execution_time,
        )

    def get_recommendations(self, result: ValidationResult) -> list[str]:
        """Generate recommendations on failure."""
        count = int(result.metrics["sample_count"])
        threshold = int(result.thresholds["threshold"])
        deficit = threshold - count

        recommendations = [
            f"Minimum {threshold} examples required, found {count}",
            "Recommendations:",
            f"  1. Add {deficit} more examples to the dataset",
            "  2. Use data augmentation to increase size",
            "  3. If this is a test run, lower the threshold in config",
        ]

        if count < threshold * 0.5:
            recommendations.append(
                f"  ⚠️ CRITICAL: Dataset too small ({count} < {threshold // 2}). " "Training may be ineffective."
            )

        return recommendations


__all__ = ["MinSamplesValidator"]
