"""
Shared utilities for score-based evaluator plugins.

Provides:
- aggregate_scores()    — build EvalResult from per-sample scores (avoids boilerplate).
- PluginReportRow       — per-sample data row for a plugin Markdown report.
- save_plugin_report()  — write runs/{run}/evaluation/{plugin_name}_report.md.

Usage (aggregate_scores):
    from src.evaluation.plugins.utils import aggregate_scores

    result = aggregate_scores(
        scores=[0.75, 1.0, 0.5],
        raw_scores=[4, 5, 3],
        failed_indices=[2],
        plugin_name="my_plugin",
        threshold_key="min_mean_score",
        thresholds=self.thresholds,
        recommendations=self.get_recommendations(...),
    )

Usage (save_plugin_report):
    from src.evaluation.plugins.utils import PluginReportRow, save_plugin_report

    rows = [
        PluginReportRow(idx=0, question="Q", model_answer="A", expected_answer="E",
                        score=0.75, raw_score=4, note="looks correct"),
        PluginReportRow(idx=1, question="Q2", model_answer="A2", expected_answer="E2",
                        score=None, raw_score=None, note="API error: timeout"),
    ]
    save_plugin_report("cerebras_judge", rows, result)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.evaluation.plugins.base import EvalResult
from src.utils.logger import get_run_log_dir, logger  # top-level import enables monkeypatching in tests

_SCORE_MIN = 1
_SCORE_MAX = 5
_DEFAULT_MIN_MEAN_SCORE = 0.6


def aggregate_scores(
    *,
    scores: list[float],
    raw_scores: list[int],
    failed_indices: list[int],
    plugin_name: str,
    threshold_key: str,
    thresholds: dict[str, Any],
    recommendations: list[str],
) -> EvalResult:
    """
    Build an EvalResult from per-sample normalized scores.

    Args:
        scores:          Normalized [0, 1] scores for each sample.
        raw_scores:      Original integer scores (e.g. 1–5) for distribution metrics.
        failed_indices:  Indices of samples that scored below a per-sample threshold
                         or errored. These appear in EvalResult.failed_samples.
        plugin_name:     Plugin identifier — stored in EvalResult.plugin_name.
        threshold_key:   Name of the threshold in `thresholds` dict (e.g. "min_mean_score").
        thresholds:      Plugin thresholds dict (from config).
        recommendations: Pre-computed recommendation strings (from get_recommendations()).

    Returns:
        EvalResult with metrics: mean_score, p50_score, score_distribution, sample_count.
    """
    if not scores:
        return EvalResult(
            plugin_name=plugin_name,
            passed=True,
            metrics={"mean_score": 0.0, "p50_score": 0.0, "sample_count": 0},
            errors=["No samples to evaluate"],
            sample_count=0,
        )

    mean_score = sum(scores) / len(scores)
    p50_score = _percentile(sorted(scores), 50)

    score_distribution: dict[str, int] = {str(i): 0 for i in range(_SCORE_MIN, _SCORE_MAX + 1)}
    for rs in raw_scores:
        key = str(max(_SCORE_MIN, min(_SCORE_MAX, rs)))
        score_distribution[key] = score_distribution.get(key, 0) + 1

    min_mean: float = float(thresholds.get(threshold_key, _DEFAULT_MIN_MEAN_SCORE))
    passed = mean_score >= min_mean

    errors: list[str] = []
    if not passed:
        errors.append(f"mean_score={mean_score:.3f} is below threshold {min_mean:.3f}")
    if failed_indices:
        errors.append(
            f"{len(failed_indices)} of {len(scores) + len(failed_indices)} samples failed to be judged (API errors)"
        )

    return EvalResult(
        plugin_name=plugin_name,
        passed=passed,
        metrics={
            "mean_score": round(mean_score, 4),
            "p50_score": round(p50_score, 4),
            "score_distribution": score_distribution,
            "sample_count": len(scores),
        },
        errors=errors,
        recommendations=recommendations if not passed else [],
        sample_count=len(scores),
        failed_samples=failed_indices,
    )


def _percentile(sorted_values: list[float], pct: int) -> float:
    """Return the p-th percentile of an already-sorted list."""
    if not sorted_values:
        return 0.0
    idx = (pct / 100) * (len(sorted_values) - 1)
    lo = int(idx)
    hi = lo + 1
    if hi >= len(sorted_values):
        return sorted_values[-1]
    frac = idx - lo
    return sorted_values[lo] + frac * (sorted_values[hi] - sorted_values[lo])


# ---------------------------------------------------------------------------
# Plugin report
# ---------------------------------------------------------------------------


@dataclass
class PluginReportRow:
    """
    Per-sample data row for a plugin Markdown report.

    Filled by the plugin during evaluate() alongside scores collection.
    Passed to save_plugin_report() at the end of evaluate().

    Attributes:
        idx:             Sample index (0-based).
        question:        Input question/task.
        model_answer:    Answer produced by the model under evaluation.
        expected_answer: Ground-truth reference answer (None if not provided).
        score:           Normalized [0, 1] score. None if the sample failed.
        raw_score:       Original integer score (e.g. 1-5). None if the sample failed.
        note:            Reasoning from the judge, error message, or empty string.
    """

    idx: int
    question: str
    model_answer: str
    expected_answer: str | None
    score: float | None
    raw_score: int | None
    note: str


def save_plugin_report(
    plugin_name: str,
    rows: list[PluginReportRow],
    result: EvalResult,
) -> None:
    """
    Write runs/{run}/evaluation/{plugin_name}_report.md.

    Renders each sample as a named section (Task N / Model answer / Expected answer / Score),
    analogous to answers.md format — multi-line content is fully readable without truncation.
    Best-effort: errors are logged but never propagate to the caller.

    Args:
        plugin_name: Plugin identifier (used as filename and report title).
        rows:        Per-sample data collected during evaluate().
        result:      Aggregated EvalResult for the summary header.
    """
    try:
        out_dir = get_run_log_dir() / "evaluation"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{plugin_name}_report.md"

        verdict = "OK" if result.passed else "FAILED"
        lines: list[str] = [f"# {plugin_name} Report\n\n"]
        lines.append(f"**Verdict:** {verdict}\n\n")

        scalar_metrics = {k: v for k, v in result.metrics.items() if isinstance(v, int | float)}
        if scalar_metrics:
            lines.append("**Metrics:** " + " | ".join(f"{k}: {v}" for k, v in scalar_metrics.items()) + "\n\n")

        if result.errors:
            for err in result.errors:
                lines.append(f"> ⚠ {err}\n")
            lines.append("\n")

        lines.append("---\n\n")

        for row in rows:
            num = row.idx + 1

            if row.score is None:
                score_label = "FAILED"
            elif row.raw_score is not None:
                score_label = f"raw: {row.raw_score}/{_SCORE_MAX}"
            else:
                score_label = "OK" if row.score == 1.0 else "FAILED"

            lines.append(f"### Task {num}\n\n")
            lines.append("~~~\n")
            lines.append(row.question)
            lines.append("\n~~~\n\n")

            lines.append("### Model answer\n\n")
            lines.append("~~~\n")
            lines.append(row.model_answer)
            lines.append("\n~~~\n\n")

            if row.expected_answer:
                lines.append("### Expected answer\n\n")
                lines.append("~~~\n")
                lines.append(row.expected_answer)
                lines.append("\n~~~\n\n")

            lines.append(f"**Score:** {score_label}")
            if row.note:
                lines.append(f"  \n**Comment:** {row.note}")
            lines.append("\n\n---\n\n")

        out_path.write_text("".join(lines), encoding="utf-8")

        logger.info(f"[EVAL] Plugin report saved to {out_path}")

    except Exception as exc:
        logger.warning(f"[EVAL] Failed to save {plugin_name}_report.md (non-fatal): {exc}")


__all__ = [
    "PluginReportRow",
    "aggregate_scores",
    "save_plugin_report",
]
