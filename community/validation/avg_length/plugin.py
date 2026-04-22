"""AvgLength validation plugin — checks average text length of samples."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from src.data.validation.base import ValidationPlugin, ValidationResult

if TYPE_CHECKING:
    from datasets import Dataset, IterableDataset

_PLUGIN_NAME = "avg_length"
_DEFAULT_MIN_LENGTH = 50
_DEFAULT_MAX_LENGTH = 8192
_NEAR_WARNING_FACTOR = 1.2


class AvgLengthValidator(ValidationPlugin):
    expensive = False
    supports_streaming = True

    def validate(
        self,
        dataset: Dataset | IterableDataset,
    ) -> ValidationResult:
        start_time = time.time()
        min_length = self._threshold("min", _DEFAULT_MIN_LENGTH)
        max_length = self._threshold("max", _DEFAULT_MAX_LENGTH)
        errors, warnings = self._new_issue_lists(_PLUGIN_NAME)

        samples = self._get_samples_for_validation(dataset)

        lengths = []
        for sample in samples:
            text = self._extract_text(sample)
            lengths.append(len(text))

        avg_length = sum(lengths) / len(lengths) if lengths else 0
        passed = min_length <= avg_length <= max_length

        if avg_length < min_length:
            errors.append(f"Average length too low: {avg_length:.1f} < {min_length}")
        elif avg_length > max_length:
            errors.append(f"Average length too high: {avg_length:.1f} > {max_length}")

        if avg_length < min_length * _NEAR_WARNING_FACTOR:
            warnings.append(
                f"Average length is close to the minimum: {avg_length:.1f} (min: {min_length})"
            )

        execution_time = (time.time() - start_time) * 1000

        return ValidationResult(
            plugin_name=self.name,
            passed=passed,
            params=dict(self.params),
            thresholds={
                "min": float(min_length),
                "max": float(max_length),
            },
            metrics={
                "avg_length": round(avg_length, 2),
                "samples_checked": float(len(samples)),
            },
            warnings=warnings,
            errors=errors,
            execution_time_ms=execution_time,
        )

    def get_recommendations(self, result: ValidationResult) -> list[str]:
        avg = result.metrics[_PLUGIN_NAME]
        min_thresh = result.thresholds["min"]
        max_thresh = result.thresholds["max"]

        recommendations = []

        if avg < min_thresh:
            recommendations.extend(
                [
                    f"Average length {avg:.1f} is below the minimum {min_thresh}",
                    "Recommendations:",
                    "  1. Check whether texts were truncated on load",
                    "  2. Ensure the correct field is used (text/messages/input)",
                    "  3. Consider filtering out overly short examples",
                    "  4. If the dataset is correct, lower the min threshold",
                ]
            )
        else:
            recommendations.extend(
                [
                    f"Average length {avg:.1f} is above the maximum {max_thresh}",
                    "Recommendations:",
                    "  1. Check for duplicated text in examples",
                    "  2. Consider truncating long examples",
                    "  3. Increase max_length in hyperparameters if long texts are needed",
                    "  4. If the dataset is correct, raise the max threshold",
                ]
            )

        return recommendations


__all__ = ["AvgLengthValidator"]
