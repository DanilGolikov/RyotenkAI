"""EmptyRatio validation plugin — checks ratio of empty or very short samples."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from src.data.validation.base import ValidationPlugin, ValidationResult

if TYPE_CHECKING:
    from datasets import Dataset, IterableDataset

_PLUGIN_NAME = "empty_ratio"
_CRITICAL_THRESHOLD = 0.2


class EmptyRatioValidator(ValidationPlugin):
    expensive = False
    supports_streaming = True

    def validate(
        self,
        dataset: Dataset | IterableDataset,
    ) -> ValidationResult:
        start_time = time.time()
        max_ratio = self._threshold("max_ratio", 0.1)
        min_chars = self._param("min_chars", 10)
        errors, warnings = self._new_issue_lists(_PLUGIN_NAME)

        samples = self._get_samples_for_validation(dataset)

        empty_count = 0
        for sample in samples:
            text = self._extract_text(sample)
            if len(text.strip()) < min_chars:
                empty_count += 1

        empty_ratio = self._safe_ratio(empty_count, len(samples), _PLUGIN_NAME)
        passed = self._check_max_ratio(
            ratio=empty_ratio,
            max_ratio=max_ratio,
            errors=errors,
            warnings=warnings,
            error_message=f"Too many empty examples: {empty_ratio:.2%} > {max_ratio:.2%}",
            near_message=f"Empty-example ratio is close to the threshold: {empty_ratio:.2%} (max: {max_ratio:.2%})",
            near_fraction=0.5,
        )

        execution_time = (time.time() - start_time) * 1000

        return ValidationResult(
            plugin_name=self.name,
            passed=passed,
            params=dict(self.params),
            thresholds={"max_ratio": max_ratio},
            metrics={
                _PLUGIN_NAME: round(empty_ratio, 4),
                "empty_count": float(empty_count),
                "total_checked": float(len(samples)),
            },
            warnings=warnings,
            errors=errors,
            execution_time_ms=execution_time,
        )

    def get_recommendations(self, result: ValidationResult) -> list[str]:
        empty_ratio = result.metrics[_PLUGIN_NAME]
        empty_count = int(result.metrics["empty_count"])
        total = int(result.metrics["total_checked"])
        max_threshold = result.thresholds["max_ratio"]

        recommendations = [
            f"Found {empty_count} empty examples out of {total} ({empty_ratio:.2%})",
            f"Threshold: {max_threshold:.2%}",
            "Recommendations:",
            "  1. Remove empty or overly short examples from the dataset",
            "  2. Check the data processing pipeline for errors",
            "  3. Ensure the correct text extraction field is used",
            "  4. Check source data quality",
        ]

        if empty_ratio > _CRITICAL_THRESHOLD:
            recommendations.append("  ⚠️ CRITICAL: >20% empty examples. Serious data quality issues!")

        return recommendations


__all__ = ["EmptyRatioValidator"]
