"""HelixQL Syntax Check evaluator plugin."""

from __future__ import annotations

import re
from typing import Any

from src.evaluation.plugins.base import EvalResult, EvalSample, EvaluatorPlugin
from src.evaluation.plugins.utils import PluginReportRow, save_plugin_report

_QUERY_LINE_RE = re.compile(r"^\s*QUERY\s+", flags=re.MULTILINE)
_QUERY_SIG_RE = re.compile(
    r"^\s*QUERY\s+[A-Za-z_][A-Za-z0-9_]*\s*\(.*?\)\s*=>",
    flags=re.MULTILINE,
)

_DEFAULT_MIN_VALID_RATIO = 0.8


def _check_helixql_syntax(text: str) -> tuple[bool, str]:
    if not text or not text.strip():
        return False, "empty response"

    if not _QUERY_LINE_RE.search(text):
        return False, "missing QUERY keyword"

    if not _QUERY_SIG_RE.search(text):
        return False, "invalid QUERY signature (expected: QUERY name(...) =>)"

    lines = [ln for ln in text.splitlines() if ln.strip()]
    has_return = any(ln.strip().startswith("RETURN") for ln in lines)
    if not has_return:
        return False, "missing RETURN statement"

    open_count = text.count("{")
    close_count = text.count("}")
    if open_count != close_count:
        return False, f"unmatched braces (open={open_count}, close={close_count})"

    return True, ""


class HelixQLSyntaxPlugin(EvaluatorPlugin):
    def _validate_contract(self) -> None:
        ratio = self._threshold("min_valid_ratio", _DEFAULT_MIN_VALID_RATIO)
        if not (0.0 <= float(ratio) <= 1.0):
            raise ValueError(
                f"HelixQLSyntaxPlugin: thresholds.min_valid_ratio must be in [0, 1], got {ratio!r}"
            )

    def evaluate(self, samples: list[EvalSample]) -> EvalResult:
        if not samples:
            return EvalResult(
                plugin_name=self.name,
                passed=True,
                metrics={"valid_count": 0, "total_count": 0, "valid_ratio": 1.0},
                errors=["No samples to evaluate"],
                sample_count=0,
            )

        min_valid_ratio: float = float(self._threshold("min_valid_ratio", _DEFAULT_MIN_VALID_RATIO))

        valid_count = 0
        failed_indices: list[int] = []
        failure_reasons: list[str] = []
        rows: list[PluginReportRow] = []

        for idx, sample in enumerate(samples):
            ok, reason = _check_helixql_syntax(sample.model_answer)
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
                        note="",
                    )
                )
            else:
                failed_indices.append(idx)
                failure_reasons.append(f"[{idx}] {reason}")
                rows.append(
                    PluginReportRow(
                        idx=idx,
                        question=sample.question,
                        model_answer=sample.model_answer,
                        expected_answer=sample.expected_answer,
                        score=0.0,
                        raw_score=None,
                        note=reason,
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
        }

        errors: list[str] = []
        if not passed:
            errors.append(
                f"valid_ratio={valid_ratio:.2%} is below threshold {min_valid_ratio:.2%} "
                f"({valid_count}/{total} samples passed)"
            )

        recommendations = self.get_recommendations(
            EvalResult(
                plugin_name=self.name,
                passed=passed,
                metrics=metrics,
                errors=errors,
                sample_count=total,
                failed_samples=failed_indices,
            )
        )

        result = EvalResult(
            plugin_name=self.name,
            passed=passed,
            metrics=metrics,
            errors=errors,
            recommendations=recommendations,
            sample_count=total,
            failed_samples=failed_indices,
        )

        if getattr(self, "_save_report", False):
            save_plugin_report(self.name, rows, result)

        return result

    def get_recommendations(self, result: EvalResult) -> list[str]:
        if result.passed:
            return []

        recommendations: list[str] = []
        valid_ratio = result.metrics.get("valid_ratio", 0.0)

        if valid_ratio < 0.5:
            recommendations.append(
                "Less than 50% of responses are valid HelixQL. "
                "Consider reviewing training data quality and prompt format."
            )
        elif valid_ratio < 0.8:
            recommendations.append(
                "Model generates valid HelixQL for most samples, but some fail. "
                "Review failed samples for common error patterns."
            )

        recommendations.append(
            "Check that the model is instructed to output only HelixQL queries "
            "and not wrap them in extra text or markdown fences."
        )

        invalid_count = result.metrics.get("invalid_count", 0)
        if invalid_count > 0 and result.failed_samples:
            recommendations.append(
                f"Inspect failed sample indices: {result.failed_samples[:10]}"
                + ("..." if len(result.failed_samples) > 10 else "")
            )

        return recommendations


__all__ = ["HelixQLSyntaxPlugin"]
