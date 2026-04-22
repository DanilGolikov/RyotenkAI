"""Identical Pairs validation plugin (DPO) — detects identical chosen/rejected pairs."""

from __future__ import annotations

import time
from collections import Counter, defaultdict
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, ClassVar

from src.data.validation.base import ValidationPlugin, ValidationResult

if TYPE_CHECKING:
    from datasets import Dataset, IterableDataset

_DEFAULT_MAX_RATIO = 0.01
_NEAR_WARNING_THRESHOLD = 0.05
_KEY_CONTENT = "content"
_PLUGIN_NAME = "identical_pairs"


class IdenticalPairsValidator(ValidationPlugin):
    expensive = False
    required_fields: ClassVar[list[str]] = ["chosen", "rejected"]
    supports_streaming = True

    def _validate_contract(self) -> None:
        max_error_examples = self._max_error_examples(default=3)
        if max_error_examples < -1:
            raise ValueError("identical_pairs requires params.max_error_examples >= -1")

    def validate(
        self,
        dataset: Dataset | IterableDataset,
    ) -> ValidationResult:
        start_time = time.time()
        max_ratio = self._threshold("max_identical_ratio", _DEFAULT_MAX_RATIO)
        errors, warnings = self._new_issue_lists(_PLUGIN_NAME)

        samples = self._get_samples_for_validation(dataset)

        identical_count = 0
        indices_by_error: dict[str, list[int]] = defaultdict(list)
        taxonomy: Counter[str] = Counter()

        for i, sample in enumerate(samples):
            if not isinstance(sample, Mapping):
                continue
            if "chosen" not in sample or "rejected" not in sample:
                continue

            chosen = self._normalize_text(sample["chosen"])
            rejected = self._normalize_text(sample["rejected"])

            if chosen == rejected:
                identical_count += 1
                taxonomy["identical_pair"] += 1
                if self._should_collect_error_example(
                    len(indices_by_error["identical_pair"]), default=3
                ):
                    indices_by_error["identical_pair"].append(i)

        identical_ratio = self._safe_ratio(identical_count, len(samples), _PLUGIN_NAME)
        passed = identical_ratio <= max_ratio

        if not passed:
            errors.append(
                f"Too many identical pairs: {identical_ratio:.2%} > {max_ratio:.2%}"
            )

        if identical_count > 0 and passed:
            warnings.append(
                f"Found {identical_count} identical pairs ({identical_ratio:.2%})"
            )

        execution_time = (time.time() - start_time) * 1000
        error_groups = self._build_error_groups(indices_by_error, counts_by_error=taxonomy)

        return ValidationResult(
            plugin_name=self.name,
            passed=passed,
            params=dict(self.params),
            thresholds={"max_identical_ratio": max_ratio},
            metrics={
                "identical_ratio": round(identical_ratio, 4),
                "identical_count": float(identical_count),
                "total_checked": float(len(samples)),
            },
            warnings=warnings,
            errors=errors,
            execution_time_ms=execution_time,
            error_groups=error_groups,
        )

    @staticmethod
    def _normalize_text(value: Any) -> str:
        if isinstance(value, str):
            return value.strip().lower()
        elif isinstance(value, dict):
            if _KEY_CONTENT in value:
                return str(value[_KEY_CONTENT]).strip().lower()
            return str(value).strip().lower()
        elif isinstance(value, list):
            texts = []
            for item in value:
                if isinstance(item, dict) and _KEY_CONTENT in item:
                    texts.append(str(item[_KEY_CONTENT]))
                else:
                    texts.append(str(item))
            return " ".join(texts).strip().lower()
        else:
            return str(value).strip().lower()

    def get_recommendations(self, result: ValidationResult) -> list[str]:
        identical_ratio = result.metrics["identical_ratio"]
        identical_count = int(result.metrics["identical_count"])
        total = int(result.metrics["total_checked"])
        max_threshold = result.thresholds["max_identical_ratio"]

        recommendations = [
            f"Found {identical_count} identical pairs out of {total} ({identical_ratio:.2%})",
            f"Threshold: {max_threshold:.2%}",
            "Recommendations:",
            "  1. Remove examples where chosen == rejected",
            "  2. Review the data labeling process:",
            "     - Ensure annotators actually chose different answers",
            "     - There may be a bug in the processing pipeline",
            "  3. For DPO training, chosen and rejected MUST differ",
            "  4. Check source data quality",
        ]

        if identical_ratio > _NEAR_WARNING_THRESHOLD:
            recommendations.append(
                "  ⚠️ CRITICAL: >5% identical pairs. DPO training will be ineffective!"
            )

        return recommendations


__all__ = ["IdenticalPairsValidator"]
