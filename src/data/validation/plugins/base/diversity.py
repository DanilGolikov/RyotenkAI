"""
Diversity validation plugin.

Checks vocabulary diversity (lexical richness).
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from src.data.validation.base import ValidationPlugin, ValidationResult
from src.data.validation.registry import ValidationPluginRegistry

if TYPE_CHECKING:
    from datasets import Dataset, IterableDataset

_PLUGIN_NAME = "diversity_score"
_DEFAULT_MIN_SCORE = 0.7
_NEAR_WARNING_FACTOR = 1.1
_CRITICAL_THRESHOLD = 0.3


@ValidationPluginRegistry.register
class DiversityValidator(ValidationPlugin):
    """
    Validates vocabulary diversity (lexical richness).

    Measures unique_tokens / total_tokens ratio (normalized).

    Params:
        min_score (float): Minimum diversity score (default: 0.7)
        sample_size (int): For large/streaming datasets (default: 10000)

    Recommendations on failure:
        - Increase topic/style variety
        - Remove repetitive phrases
        - Use synthetic data for diversity
    """

    name = _PLUGIN_NAME
    priority = 30
    expensive = True  # Can be slow for large datasets
    supports_streaming = True

    @classmethod
    def get_description(cls) -> str:
        return "Checks dataset vocabulary diversity (lexical richness)"

    def validate(
        self,
        dataset: Dataset | IterableDataset,
    ) -> ValidationResult:
        """Check vocabulary diversity."""
        start_time = time.time()
        min_score = self._threshold("min_score", _DEFAULT_MIN_SCORE)
        errors, warnings = self._new_issue_lists(_PLUGIN_NAME)

        # Get samples
        samples = self._get_samples_for_validation(dataset)

        # Calculate diversity
        all_tokens: set[str] = set()
        total_tokens = 0

        for sample in samples:
            text = self._extract_text(sample)
            tokens = text.lower().split()
            all_tokens.update(tokens)
            total_tokens += len(tokens)

        # Calculate diversity score (unique / total, normalized)
        raw_diversity = len(all_tokens) / total_tokens if total_tokens > 0 else 0
        # Normalize to 0-1 range (typical values: 0.01-0.1)
        diversity_score = min(raw_diversity * 10, 1.0)

        passed = diversity_score >= min_score

        if not passed:
            errors.append(f"Diversity score too low: {diversity_score:.2f} < {min_score}")

        # Warnings
        if diversity_score < min_score * _NEAR_WARNING_FACTOR:
            warnings.append(f"Diversity score is close to the minimum: {diversity_score:.2f} (min: {min_score})")

        execution_time = (time.time() - start_time) * 1000

        return ValidationResult(
            plugin_name=self.name,
            passed=passed,
            params=dict(self.params),
            thresholds={
                "min_score": min_score,
            },
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
        """Generate recommendations on failure."""
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
