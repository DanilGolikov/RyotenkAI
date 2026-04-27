"""HelixQL Preference Semantics validation plugin (DPO)."""

from __future__ import annotations

import time
from collections import Counter, defaultdict
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from community_libs.helixql import (
    extract_query_text,
    extract_schema_block,
    get_compiler,
    semantic_match_details,
)
from src.data.validation.base import ValidationPlugin, ValidationResult

_MIN_VALID_RATIO_DEFAULT = 0.95

if TYPE_CHECKING:
    from datasets import Dataset, IterableDataset


class HelixQLPreferenceSemanticsValidator(ValidationPlugin):
    REQUIRED_LIBS = ("helixql",)
    supports_streaming = True

    def _validate_contract(self) -> None:
        timeout_seconds = self.params.get("timeout_seconds")
        if timeout_seconds is None or int(timeout_seconds) <= 0:
            raise ValueError(
                "helixql_preference_semantics requires positive params.timeout_seconds"
            )
        max_error_examples = self._max_error_examples()
        if max_error_examples < -1:
            raise ValueError(
                "helixql_preference_semantics requires params.max_error_examples >= -1"
            )

    def _get_compiler(self):
        # Pass-through to the shared community_libs.helixql cache so multiple
        # plugin instances configured with the same timeout share one
        # compiler — and one validate() result cache.
        return get_compiler(timeout_seconds=int(self.params["timeout_seconds"]))

    def validate(
        self,
        dataset: Dataset | IterableDataset,
    ) -> ValidationResult:
        start_time = time.time()
        samples = self._get_samples_for_validation(dataset)
        min_valid_ratio = float(self._threshold("min_valid_ratio", _MIN_VALID_RATIO_DEFAULT))

        checked = 0
        valid_count = 0
        taxonomy: Counter[str] = Counter()
        errors: list[str] = []
        indices_by_error: dict[str, list[int]] = defaultdict(list)

        for idx, sample in enumerate(samples):
            if not isinstance(sample, Mapping):
                taxonomy["malformed_sample"] += 1
                if self._should_collect_error_example(len(indices_by_error["malformed_sample"])):
                    indices_by_error["malformed_sample"].append(idx)
                continue

            checked += 1
            issue = self._validate_pair(sample=sample)
            if issue is None:
                valid_count += 1
                continue

            taxonomy[issue] += 1
            if self._should_collect_error_example(len(indices_by_error[issue])):
                indices_by_error[issue].append(idx)

        valid_ratio = self._safe_ratio(valid_count, checked, self.name)
        passed = valid_ratio >= min_valid_ratio
        if not passed:
            errors.append(f"valid_ratio={valid_ratio:.4f} is below threshold {min_valid_ratio:.4f}")

        metrics: dict[str, float] = {
            "valid_ratio": round(valid_ratio, 4),
            "valid_count": float(valid_count),
            "checked_samples": float(checked),
        }
        for name, count in sorted(taxonomy.items()):
            metrics[f"error_taxonomy.{name}"] = float(count)

        error_groups = self._build_error_groups(indices_by_error, counts_by_error=taxonomy)

        return ValidationResult(
            plugin_name=self.name,
            passed=passed,
            params=dict(self.params),
            thresholds={"min_valid_ratio": min_valid_ratio},
            metrics=metrics,
            warnings=[],
            errors=errors,
            execution_time_ms=(time.time() - start_time) * 1000,
            error_groups=error_groups,
        )

    def get_recommendations(self, result: ValidationResult) -> list[str]:
        if result.passed:
            return []
        return [
            "Ensure chosen and rejected refer to the same task and the same schema context.",
            "Synthetic rejected should be worse than chosen not only in format but also in compilation/semantics.",
            "If synthetic rejected is too weak, add a second rejected layer from a weak model later.",
        ]

    def _validate_pair(self, *, sample: Mapping[str, Any]) -> str | None:
        prompt_text = extract_query_text(sample.get("prompt") or sample.get("question"))
        schema_text = extract_query_text(sample.get("schema_context")) or extract_schema_block(
            prompt_text
        )
        reference_answer = extract_query_text(
            sample.get("reference_answer") or sample.get("expected_answer")
        )
        chosen = extract_query_text(sample.get("chosen"))
        rejected = extract_query_text(sample.get("rejected"))

        if not schema_text.strip():
            return "missing_schema_context"
        if not chosen.strip():
            return "missing_chosen"
        if not rejected.strip():
            return "missing_rejected"

        compiler = self._get_compiler()
        chosen_result = compiler.validate(schema=schema_text, query=chosen)
        rejected_result = compiler.validate(schema=schema_text, query=rejected)

        if not chosen_result.ok:
            return f"chosen_{chosen_result.error_type}"
        if chosen_result.ok and not rejected_result.ok:
            return None

        chosen_score = semantic_match_details(
            candidate=chosen, expected=reference_answer or chosen, user_text=prompt_text
        )["score"]
        rejected_score = semantic_match_details(
            candidate=rejected,
            expected=reference_answer or chosen,
            user_text=prompt_text,
        )["score"]

        if float(chosen_score) <= float(rejected_score):
            return "chosen_not_better_than_rejected"
        return None


__all__ = ["HelixQLPreferenceSemanticsValidator"]
