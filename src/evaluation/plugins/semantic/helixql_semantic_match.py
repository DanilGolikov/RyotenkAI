from __future__ import annotations

from typing import Any, ClassVar

from src.evaluation.plugins.base import EvalResult, EvalSample, EvaluatorPlugin
from src.evaluation.plugins.registry import EvaluatorPluginRegistry
from src.evaluation.plugins.utils import PluginReportRow, aggregate_scores, save_plugin_report
from src.utils.domains.helixql import semantic_match_details


@EvaluatorPluginRegistry.register
class HelixQLSemanticMatchPlugin(EvaluatorPlugin):
    name = "helixql_semantic_match"
    priority = 20
    requires_expected_answer = True

    MANIFEST: ClassVar[dict[str, Any]] = {
        "description": (
            "Compares the model's HelixQL answer to the expected answer using a "
            "deterministic semantic score (sequence ratio + token jaccard + hard-eval hits)."
        ),
        "category": "semantic",
        "stability": "stable",
        "params_schema": {},
        "thresholds_schema": {
            "min_mean_score": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.7},
        },
        "suggested_params": {},
        "suggested_thresholds": {"min_mean_score": 0.7},
    }

    @classmethod
    def get_description(cls) -> str:
        return "Compares the model's HelixQL answer to expected_answer using a deterministic semantic score"

    def _validate_contract(self) -> None:
        min_mean = float(self._threshold("min_mean_score", 0.7))
        if not (0.0 <= min_mean <= 1.0):
            raise ValueError("helixql_semantic_match thresholds.min_mean_score must be in [0, 1]")

    def evaluate(self, samples: list[EvalSample]) -> EvalResult:
        scores: list[float] = []
        raw_scores: list[int] = []
        failed_indices: list[int] = []
        rows: list[PluginReportRow] = []
        exact_matches = 0
        near_matches = 0
        hard_eval_passes = 0

        for idx, sample in enumerate(samples):
            details = semantic_match_details(
                candidate=sample.model_answer,
                expected=sample.expected_answer or "",
                user_text=sample.question,
            )
            score = float(details["score"])
            raw_score = max(1, min(5, round(score * 5)))
            scores.append(score)
            raw_scores.append(raw_score)

            if bool(details["exact_match"]):
                exact_matches += 1
            if bool(details["near_match"]):
                near_matches += 1
            if bool(details["hard_eval_pass"]):
                hard_eval_passes += 1
            if score < 0.7:
                failed_indices.append(idx)

            note_parts = [
                f"sequence_ratio={details['sequence_ratio']}",
                f"token_jaccard={details['token_jaccard']}",
            ]
            hard_errors = details.get("hard_eval_errors") or []
            if hard_errors:
                note_parts.append(f"hard_eval_errors={','.join(hard_errors)}")

            rows.append(
                PluginReportRow(
                    idx=idx,
                    question=sample.question,
                    model_answer=sample.model_answer,
                    expected_answer=sample.expected_answer,
                    score=score,
                    raw_score=raw_score,
                    note="; ".join(note_parts),
                )
            )

        base_result = aggregate_scores(
            scores=scores,
            raw_scores=raw_scores,
            failed_indices=failed_indices,
            plugin_name=self.name,
            threshold_key="min_mean_score",
            thresholds=self.thresholds,
            recommendations=[],
        )
        metrics: dict[str, Any] = dict(base_result.metrics)
        total = len(samples) if samples else 0
        metrics.update(
            {
                "exact_match_ratio": round(exact_matches / total, 4) if total else 0.0,
                "near_match_ratio": round(near_matches / total, 4) if total else 0.0,
                "hard_eval_pass_ratio": round(hard_eval_passes / total, 4) if total else 0.0,
            }
        )

        result = EvalResult(
            plugin_name=self.name,
            passed=base_result.passed,
            metrics=metrics,
            errors=base_result.errors,
            recommendations=self.get_recommendations(
                EvalResult(
                    plugin_name=self.name,
                    passed=base_result.passed,
                    metrics=metrics,
                    errors=base_result.errors,
                    sample_count=base_result.sample_count,
                    failed_samples=failed_indices,
                )
            ),
            sample_count=base_result.sample_count,
            failed_samples=failed_indices,
        )

        if getattr(self, "_save_report", False):
            save_plugin_report(self.name, rows, result)
        return result

    def get_recommendations(self, result: EvalResult) -> list[str]:
        if result.passed:
            return []
        return [
            "Analyze exact_match_ratio separately from hard_eval_pass_ratio: they answer different questions.",
            "If hard_eval_pass is high but semantic score is low, the model may produce a formally valid but wrong query.",
            "Keep the judge as a second independent semantic scorer, not a replacement for this deterministic comparator.",
        ]


__all__ = ["HelixQLSemanticMatchPlugin"]
