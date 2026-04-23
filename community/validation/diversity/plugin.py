"""Diversity validation plugin — checks vocabulary diversity (lexical richness)."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from src.data.validation.base import ValidationPlugin, ValidationResult

if TYPE_CHECKING:
    from datasets import Dataset, IterableDataset

_PLUGIN_NAME = "diversity_score"
_DEFAULT_MIN_SCORE = 0.7
_NEAR_WARNING_FACTOR = 1.1
_CRITICAL_THRESHOLD = 0.3


class DiversityValidator(ValidationPlugin):
    expensive = True
    supports_streaming = True

    def validate(
        self,
        dataset: Dataset | IterableDataset,
    ) -> ValidationResult:
        start_time = time.time()
        min_score = self._threshold("min_score", _DEFAULT_MIN_SCORE)
        errors, warnings = self._new_issue_lists(_PLUGIN_NAME)

        samples = self._get_samples_for_validation(dataset)

        all_tokens: set[str] = set()
        total_tokens = 0

        for sample in samples:
            text = self._extract_text(sample)
            tokens = text.lower().split()
            all_tokens.update(tokens)
            total_tokens += len(tokens)

        raw_diversity = len(all_tokens) / total_tokens if total_tokens > 0 else 0
        diversity_score = min(raw_diversity * 10, 1.0)

        passed = diversity_score >= min_score

        if not passed:
            errors.append(f"Diversity score too low: {diversity_score:.2f} < {min_score}")

        if diversity_score < min_score * _NEAR_WARNING_FACTOR:
            warnings.append(
                f"Diversity score is close to the minimum: {diversity_score:.2f} (min: {min_score})"
            )

        execution_time = (time.time() - start_time) * 1000

        return ValidationResult(
            plugin_name=self.name,
            passed=passed,
            params=dict(self.params),
            thresholds={"min_score": min_score},
            metrics={
                _PLUGIN_NAME: round(diversity_score, 4),
                "unique_tokens": float(len(all_tokens)),
                "total_tokens": float(total_tokens),
                "samples_checked": float(len(samples)),
            },
            warnings=warnings,
            errors=errors,
            execution_time_ms=execution_time,
        )

    def get_recommendations(self, result: ValidationResult) -> list[str]:
        score = result.metrics[_PLUGIN_NAME]
        min_score = result.thresholds["min_score"]
        unique = int(result.metrics["unique_tokens"])
        total = int(result.metrics["total_tokens"])

        recommendations = [
            f"Diversity score: {score:.2f} < {min_score} (minimum)",
            f"Unique tokens: {unique:,} of {total:,}",
            "Recommendations:",
            "  1. Increase example variety (different topics/styles)",
            "  2. Ensure the dataset does not contain many repeated phrases",
            "  3. Consider using synthetic data",
            "  4. Check for duplication in source data",
        ]

        if score < _CRITICAL_THRESHOLD:
            recommendations.append("  ⚠️ CRITICAL: Very low diversity. The model may overfit!")

        return recommendations


__all__ = ["DiversityValidator"]
