"""HelixQL SAPO Prompt Contract validation plugin."""

from __future__ import annotations

import time
from collections import Counter, defaultdict
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from src.data.validation.base import ValidationPlugin, ValidationResult
from src.training.constants import COL_MESSAGES, COL_PROMPT
from src.utils.domains.helixql import extract_query_text, extract_schema_block

if TYPE_CHECKING:
    from datasets import Dataset, IterableDataset


class HelixQLSAPOPromptContractValidator(ValidationPlugin):
    supports_streaming = True

    def _validate_contract(self) -> None:
        max_error_examples = self._max_error_examples()
        if max_error_examples < -1:
            raise ValueError(
                "helixql_sapo_prompt_contract requires params.max_error_examples >= -1"
            )

    def validate(
        self,
        dataset: Dataset | IterableDataset,
    ) -> ValidationResult:
        start_time = time.time()
        samples = self._get_samples_for_validation(dataset)
        min_pass_rate = float(self._threshold("min_pass_rate", 1.0))

        checked = 0
        passed_count = 0
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
            issues = self._sample_issues(sample)
            if not issues:
                passed_count += 1
                continue

            taxonomy.update(issues)
            for issue in issues:
                if self._should_collect_error_example(len(indices_by_error[issue])):
                    indices_by_error[issue].append(idx)

        pass_rate = self._safe_ratio(passed_count, checked, self.name)
        passed = pass_rate >= min_pass_rate
        if not passed:
            errors.append(f"pass_rate={pass_rate:.4f} is below threshold {min_pass_rate:.4f}")

        metrics: dict[str, float] = {
            "pass_rate": round(pass_rate, 4),
            "checked_samples": float(checked),
            "invalid_count": float(max(0, checked - passed_count)),
        }
        for name, count in sorted(taxonomy.items()):
            metrics[f"contract_issues.{name}"] = float(count)

        error_groups = self._build_error_groups(indices_by_error, counts_by_error=taxonomy)

        return ValidationResult(
            plugin_name=self.name,
            passed=passed,
            params=dict(self.params),
            thresholds={"min_pass_rate": min_pass_rate},
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
            "Each RL sample must have a prompt or messages with a user request.",
            "Schema context must be either a separate schema_context field or inside a helixschema fence.",
            "Reference answer is required because semantic reward and offline eval depend on it.",
        ]

    @staticmethod
    def _sample_issues(sample: Mapping[str, Any]) -> list[str]:
        issues: list[str] = []

        prompt_text = extract_query_text(sample.get(COL_PROMPT))
        schema_text = extract_query_text(sample.get("schema_context"))
        reference_answer = extract_query_text(
            sample.get("reference_answer") or sample.get("expected_answer")
        )

        if COL_MESSAGES in sample and isinstance(sample.get(COL_MESSAGES), list):
            user_text = ""
            assistant_text = ""
            for message in sample[COL_MESSAGES]:
                if not isinstance(message, Mapping):
                    continue
                role = message.get("role")
                content = extract_query_text(message.get("content"))
                if role == "user":
                    user_text = content
                elif role == "assistant":
                    assistant_text = content
            prompt_text = prompt_text or user_text
            schema_text = schema_text or extract_schema_block(user_text)
            reference_answer = reference_answer or assistant_text

        if not prompt_text.strip():
            issues.append("missing_prompt")
        if not schema_text.strip():
            schema_text = extract_schema_block(prompt_text)
            if not schema_text.strip():
                issues.append("missing_schema_context")
        if not reference_answer.strip():
            issues.append("missing_reference_answer")

        return issues


__all__ = ["HelixQLSAPOPromptContractValidator"]
