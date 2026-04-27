"""HelixQL Gold Syntax Backend validation plugin (SFT)."""

from __future__ import annotations

import time
from collections import Counter, defaultdict
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, ClassVar

from community_libs.helixql import (
    extract_query_text,
    extract_schema_block,
    get_compiler,
)
from src.data.validation.base import ValidationPlugin, ValidationResult

if TYPE_CHECKING:
    from datasets import Dataset, IterableDataset


class HelixQLGoldSyntaxBackendValidator(ValidationPlugin):
    """Validate gold HelixQL answers via ``helix compile``."""

    REQUIRED_LIBS = ("helixql",)
    required_fields: ClassVar[list[str]] = []
    supports_streaming = True

    def _validate_contract(self) -> None:
        timeout_seconds = self.params.get("timeout_seconds")
        if timeout_seconds is None or int(timeout_seconds) <= 0:
            raise ValueError(
                "helixql_gold_syntax_backend requires positive params.timeout_seconds"
            )
        max_error_examples = self._max_error_examples()
        if max_error_examples < -1:
            raise ValueError(
                "helixql_gold_syntax_backend requires params.max_error_examples >= -1"
            )

    def _get_compiler(self):
        return get_compiler(timeout_seconds=int(self.params["timeout_seconds"]))

    def validate(
        self,
        dataset: Dataset | IterableDataset,
    ) -> ValidationResult:
        start_time = time.time()
        samples = self._get_samples_for_validation(dataset)
        min_pass_rate = float(self._threshold("min_pass_rate", 1.0))
        compiler = self._get_compiler()

        checked = 0
        passed_count = 0
        errors: list[str] = []
        warnings: list[str] = []
        taxonomy: Counter[str] = Counter()
        indices_by_error: dict[str, list[int]] = defaultdict(list)

        for idx, sample in enumerate(samples):
            if not isinstance(sample, Mapping):
                taxonomy["malformed_sample"] += 1
                if self._should_collect_error_example(len(indices_by_error["malformed_sample"])):
                    indices_by_error["malformed_sample"].append(idx)
                continue
            schema_text, query_text = self._extract_schema_and_query(sample)
            if not schema_text.strip():
                taxonomy["missing_schema"] += 1
                if self._should_collect_error_example(len(indices_by_error["missing_schema"])):
                    indices_by_error["missing_schema"].append(idx)
                checked += 1
                continue
            if not query_text.strip():
                taxonomy["missing_query"] += 1
                if self._should_collect_error_example(len(indices_by_error["missing_query"])):
                    indices_by_error["missing_query"].append(idx)
                checked += 1
                continue

            result = compiler.validate(schema=schema_text, query=query_text)
            checked += 1
            if result.ok:
                passed_count += 1
            else:
                taxonomy[result.error_type] += 1
                if self._should_collect_error_example(len(indices_by_error[result.error_type])):
                    indices_by_error[result.error_type].append(idx)

        pass_rate = self._safe_ratio(passed_count, checked, self.name)
        compiler_error_ratio = 1.0 - pass_rate if checked else 0.0
        passed = pass_rate >= min_pass_rate

        if checked == 0:
            warnings.append("No samples were checked by helixql_gold_syntax_backend")
        elif not passed:
            errors.append(f"pass_rate={pass_rate:.4f} is below threshold {min_pass_rate:.4f}")

        metrics: dict[str, float] = {
            "pass_rate": round(pass_rate, 4),
            "compiler_error_ratio": round(compiler_error_ratio, 4),
            "checked_samples": float(checked),
        }
        for name, count in sorted(taxonomy.items()):
            metrics[f"error_taxonomy.{name}"] = float(count)

        error_groups = self._build_error_groups(indices_by_error, counts_by_error=taxonomy)

        return ValidationResult(
            plugin_name=self.name,
            passed=passed,
            params=dict(self.params),
            thresholds={"min_pass_rate": min_pass_rate},
            metrics=metrics,
            warnings=warnings,
            errors=errors,
            execution_time_ms=(time.time() - start_time) * 1000,
            error_groups=error_groups,
        )

    def get_recommendations(self, result: ValidationResult) -> list[str]:
        if result.passed:
            return []
        return [
            "Fix gold answers that fail the real Helix backend.",
            "Ensure schema context is extracted from the user/question field without loss.",
        ]

    @staticmethod
    def _extract_schema_and_query(sample: Mapping[str, Any]) -> tuple[str, str]:
        if "messages" in sample and isinstance(sample.get("messages"), list):
            user_text = ""
            assistant_text = ""
            for message in sample["messages"]:
                if not isinstance(message, Mapping):
                    continue
                role = message.get("role")
                content = extract_query_text(message.get("content"))
                if role == "user":
                    user_text = content
                elif role == "assistant":
                    assistant_text = content
            return extract_schema_block(user_text), assistant_text

        prompt_text = extract_query_text(sample.get("question") or sample.get("prompt"))
        schema_text = extract_query_text(sample.get("schema_context")) or extract_schema_block(
            prompt_text
        )
        query_text = extract_query_text(
            sample.get("expected_answer") or sample.get("reference_answer") or sample.get("answer")
        )
        return schema_text, query_text


__all__ = ["HelixQLGoldSyntaxBackendValidator"]
