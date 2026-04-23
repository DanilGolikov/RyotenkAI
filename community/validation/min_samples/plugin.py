"""MinSamples validation plugin — checks dataset has minimum number of examples."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from src.data.validation.base import ValidationPlugin, ValidationResult
from src.utils.logger import logger

if TYPE_CHECKING:
    from datasets import Dataset, IterableDataset

_DEFAULT_SAMPLE_SIZE = 10_000


class MinSamplesValidator(ValidationPlugin):
    expensive = False
    supports_streaming = True

    def validate(
        self,
        dataset: Dataset | IterableDataset,
    ) -> ValidationResult:
        from datasets import IterableDataset

        start_time = time.time()
        threshold = self._threshold("threshold", 100)
        errors: list[str] = []
        warnings: list[str] = []

        if isinstance(dataset, IterableDataset):
            sample_size = self._param("sample_size", _DEFAULT_SAMPLE_SIZE)
            logger.debug(f"[{self.name}] Counting samples from IterableDataset (max: {sample_size})")
            samples = list(dataset.take(sample_size))
            count = len(samples)

            if count == sample_size:
                warnings.append(
                    f"Dataset might have more than {sample_size} samples (streaming mode, limited check)"
                )
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
            metrics={"sample_count": float(count)},
            warnings=warnings,
            errors=errors,
            execution_time_ms=execution_time,
        )

    def get_recommendations(self, result: ValidationResult) -> list[str]:
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
                f"  ⚠️ CRITICAL: Dataset too small ({count} < {threshold // 2}). Training may be ineffective."
            )

        return recommendations


__all__ = ["MinSamplesValidator"]
