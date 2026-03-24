"""
Preference Format validation plugin (DPO-specific).

Validates DPO dataset format (chosen/rejected pairs).
"""

from __future__ import annotations

import time
from collections import Counter, defaultdict
from collections.abc import Mapping
from typing import TYPE_CHECKING, ClassVar

from src.data.validation.base import ValidationPlugin, ValidationResult
from src.data.validation.registry import ValidationPluginRegistry

if TYPE_CHECKING:
    from datasets import Dataset, IterableDataset

_PLUGIN_NAME = "preference_format"
_DEFAULT_MIN_VALID_RATIO = 0.95
_CRITICAL_THRESHOLD = 0.8


@ValidationPluginRegistry.register
class PreferenceFormatValidator(ValidationPlugin):
    """
    Validates DPO preference format.

    Checks that dataset has required fields: chosen, rejected.
    Optionally: prompt field.

    Params:
        required_fields (list): Required fields (default: ["chosen", "rejected"])
        min_valid_ratio (float): Minimum ratio of valid samples (default: 0.95)
        sample_size (int): For large/streaming datasets (default: 10000)

    Recommendations on failure:
        - Check dataset format
        - Ensure chosen/rejected fields exist
        - Verify field types (str/dict)
    """

    name = "preference_format"
    priority = 15  # Run early (format check)
    expensive = False
    required_fields: ClassVar[list[str]] = ["chosen", "rejected"]
    supports_streaming = True

    @classmethod
    def get_description(cls) -> str:
        return "Validates DPO dataset format (chosen/rejected pairs)"

    def _validate_contract(self) -> None:
        max_error_examples = self._max_error_examples()
        if max_error_examples < -1:
            raise ValueError("preference_format requires params.max_error_examples >= -1")

    def validate(
        self,
        dataset: Dataset | IterableDataset,
    ) -> ValidationResult:
        """Check DPO preference format."""
        start_time = time.time()
        required_fields = self._param("required_fields", ["chosen", "rejected"])
        min_valid_ratio = self._threshold("min_valid_ratio", _DEFAULT_MIN_VALID_RATIO)
        errors, warnings = self._new_issue_lists(_PLUGIN_NAME)

        # Get samples
        samples = self._get_samples_for_validation(dataset)

        # Check format
        valid_count = 0
        taxonomy: Counter[str] = Counter()
        indices_by_error: dict[str, list[int]] = defaultdict(list)

        for i, sample in enumerate(samples):
            if not isinstance(sample, Mapping):
                continue
            is_valid = True

            # Check required fields
            for field in required_fields:
                if field not in sample:
                    is_valid = False
                    issue = f"missing_field.{field}"
                    taxonomy[issue] += 1
                    if self._should_collect_error_example(len(indices_by_error[issue])):
                        indices_by_error[issue].append(i)
                    break

            # Check field types (should be str or dict)
            if is_valid:
                for field in required_fields:
                    value = sample[field]
                    if not isinstance(value, str | dict | list):
                        is_valid = False
                        issue = f"invalid_type.{field}"
                        taxonomy[issue] += 1
                        if self._should_collect_error_example(len(indices_by_error[issue])):
                            indices_by_error[issue].append(i)
                        break

            if is_valid:
                valid_count += 1

        valid_ratio = self._safe_ratio(valid_count, len(samples), _PLUGIN_NAME)
        passed = valid_ratio >= min_valid_ratio

        if not passed:
            errors.append(f"Too many invalid examples: {valid_ratio:.2%} < {min_valid_ratio:.2%}")

        # Warnings
        if valid_ratio < 1.0:
            invalid_count = len(samples) - valid_count
            warnings.append(f"Found {invalid_count} invalid examples ({(1 - valid_ratio):.2%})")

        execution_time = (time.time() - start_time) * 1000
        error_groups = self._build_error_groups(indices_by_error, counts_by_error=taxonomy)

        return ValidationResult(
            plugin_name=self.name,
            passed=passed,
            params=dict(self.params),
            thresholds={
                "min_valid_ratio": min_valid_ratio,
            },
            metrics={
                "valid_ratio": round(valid_ratio, 4),
                "valid_count": float(valid_count),
                "invalid_count": float(len(samples) - valid_count),
                "total_checked": float(len(samples)),
            },
            warnings=warnings,
            errors=errors,
            execution_time_ms=execution_time,
            error_groups=error_groups,
        )

    def get_recommendations(self, result: ValidationResult) -> list[str]:
        """Generate recommendations on failure."""
        valid_ratio = result.metrics["valid_ratio"]
        invalid_count = int(result.metrics["invalid_count"])
        min_threshold = result.thresholds["min_valid_ratio"]

        recommendations = [
            f"Valid ratio: {valid_ratio:.2%} < {min_threshold:.2%} (minimum)",
            f"Invalid examples: {invalid_count}",
            "Recommendations:",
            "  1. Check DPO dataset format:",
            "     - Each example must have fields: 'chosen', 'rejected'",
            "     - Optional: 'prompt' field",
            "  2. Ensure field values have the correct type (str/dict)",
            "  3. Remove or fix invalid examples",
            "  4. Check the data processing pipeline",
        ]

        if valid_ratio < _CRITICAL_THRESHOLD:
            recommendations.append("  ⚠️ CRITICAL: Fewer than 80% valid examples. Dataset is unsuitable for DPO!")

        return recommendations


__all__ = ["PreferenceFormatValidator"]
