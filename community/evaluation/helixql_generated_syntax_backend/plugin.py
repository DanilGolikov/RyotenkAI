"""HelixQL Generated Syntax Backend evaluator plugin."""

from __future__ import annotations

from collections import Counter
from typing import Any

from community_libs.helixql import (
    extract_query_text,
    extract_schema_block,
    get_compiler,
)
from src.evaluation.plugins.base import EvalResult, EvalSample, EvaluatorPlugin
from src.evaluation.plugins.utils import PluginReportRow, save_plugin_report


class HelixQLGeneratedSyntaxBackendPlugin(EvaluatorPlugin):
    REQUIRED_LIBS = ("helixql",)

    def _validate_contract(self) -> None:
        timeout_seconds = self._param("timeout_seconds")
        if timeout_seconds is None or int(timeout_seconds) <= 0:
            raise ValueError(
                "helixql_generated_syntax_backend requires positive params.timeout_seconds"
            )
        min_valid_ratio = float(self._threshold("min_valid_ratio", 0.8))
        if not (0.0 <= min_valid_ratio <= 1.0):
            raise ValueError(
                "helixql_generated_syntax_backend thresholds.min_valid_ratio must be in [0, 1]"
            )

    def _get_compiler(self):
        return get_compiler(timeout_seconds=int(self.params["timeout_seconds"]))

    def evaluate(self, samples: list[EvalSample]) -> EvalResult:
        if not samples:
            return EvalResult(
                plugin_name=self.name,
                passed=True,
                metrics={"valid_ratio": 1.0},
                sample_count=0,
            )

        min_valid_ratio = float(self._threshold("min_valid_ratio", 0.8))
        compiler = self._get_compiler()

        valid_count = 0
        failed_indices: list[int] = []
        taxonomy: Counter[str] = Counter()
        rows: list[PluginReportRow] = []

        for idx, sample in enumerate(samples):
            schema_text = extract_query_text(
                sample.metadata.get("schema_context")
            ) or extract_schema_block(sample.question)
            query_text = extract_query_text(sample.model_answer)

            if not schema_text.strip():
                ok = False
                error_type = "missing_schema"
            elif not query_text.strip():
                ok = False
                error_type = "missing_query"
            else:
                result = compiler.validate(schema=schema_text, query=query_text)
                ok = result.ok
                error_type = result.error_type

            if ok:
                valid_count += 1
                rows.append(
                    PluginReportRow(
                        idx=idx,
                        question=sample.question,
                        model_answer=sample.model_answer,
                        expected_answer=sample.expected_answer,
                        score=1.0,
                        raw_score=None,
                        note="compile: ok",
                    )
                )
            else:
                taxonomy[error_type] += 1
                failed_indices.append(idx)
                rows.append(
                    PluginReportRow(
                        idx=idx,
                        question=sample.question,
                        model_answer=sample.model_answer,
                        expected_answer=sample.expected_answer,
                        score=0.0,
                        raw_score=None,
                        note=error_type,
                    )
                )

        total = len(samples)
        valid_ratio = valid_count / total
        passed = valid_ratio >= min_valid_ratio

        metrics: dict[str, Any] = {
            "valid_count": valid_count,
            "invalid_count": total - valid_count,
            "total_count": total,
            "valid_ratio": round(valid_ratio, 4),
            "compile_pass_ratio": round(valid_ratio, 4),
        }
        for name, count in sorted(taxonomy.items()):
            metrics[f"error_taxonomy.{name}"] = count

        errors: list[str] = []
        if not passed:
            errors.append(
                f"valid_ratio={valid_ratio:.2%} is below threshold {min_valid_ratio:.2%} "
                f"({valid_count}/{total} passed)"
            )

        eval_result = EvalResult(
            plugin_name=self.name,
            passed=passed,
            metrics=metrics,
            errors=errors,
            recommendations=self.get_recommendations(
                EvalResult(
                    plugin_name=self.name,
                    passed=passed,
                    metrics=metrics,
                    errors=errors,
                    sample_count=total,
                    failed_samples=failed_indices,
                )
            ),
            sample_count=total,
            failed_samples=failed_indices,
        )

        if getattr(self, "_save_report", False):
            save_plugin_report(self.name, rows, eval_result)
        return eval_result

    def get_recommendations(self, result: EvalResult) -> list[str]:
        if result.passed:
            return []
        return [
            "Ensure the model returns only HelixQL without markdown fences or extra text.",
            "Inspect error_taxonomy to see whether the model fails on parse errors or loses schema grounding.",
            "For final scoring, keep a separate compare plugin; use this backend plugin for formal correctness.",
        ]


__all__ = ["HelixQLGeneratedSyntaxBackendPlugin"]
