"""CerebrasJudgePlugin — orchestrates CerebrasProvider across a batch of samples."""

from __future__ import annotations

from src.evaluation.plugins.base import EvalResult, EvalSample, EvaluatorPlugin
from src.evaluation.plugins.utils import PluginReportRow, aggregate_scores, save_plugin_report
from src.utils.logger import logger

from .provider import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    CerebrasProvider,
)

_SCORE_MIN = 1
_SCORE_MAX = 5
_SCORE_RANGE = _SCORE_MAX - _SCORE_MIN
_DEFAULT_MAX_SAMPLES = 50


class CerebrasJudgePlugin(EvaluatorPlugin):
    """LLM-as-judge evaluation plugin using the Cerebras API."""

    requires_expected_answer = True
    _secrets: dict[str, str]

    def evaluate(self, samples: list[EvalSample]) -> EvalResult:
        api_key: str = self._secrets["EVAL_CEREBRAS_API_KEY"]

        model: str = str(self.params.get("model", DEFAULT_MODEL))
        temperature: float = float(self.params.get("temperature", DEFAULT_TEMPERATURE))
        max_tokens: int = int(self.params.get("max_tokens", DEFAULT_MAX_TOKENS))
        max_retries: int = int(self.params.get("max_retries", DEFAULT_MAX_RETRIES))
        max_samples: int = int(self.params.get("max_samples", _DEFAULT_MAX_SAMPLES))

        provider = CerebrasProvider(
            api_key=api_key,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries,
        )

        eval_samples = samples[:max_samples]
        if len(samples) > max_samples:
            logger.warning(
                f"[CEREBRAS] Limiting evaluation to {max_samples} of {len(samples)} "
                f"samples (max_samples={max_samples})"
            )

        normalized_scores: list[float] = []
        raw_scores: list[int] = []
        failed_indices: list[int] = []
        rows: list[PluginReportRow] = []

        for idx, sample in enumerate(eval_samples):
            expected = sample.expected_answer or ""
            try:
                response = provider.judge(
                    question=sample.question,
                    expected=expected,
                    model_answer=sample.model_answer,
                )
                raw_score = response.score
                normalized = (raw_score - _SCORE_MIN) / _SCORE_RANGE
                normalized_scores.append(normalized)
                raw_scores.append(raw_score)
                rows.append(
                    PluginReportRow(
                        idx=idx,
                        question=sample.question,
                        model_answer=sample.model_answer,
                        expected_answer=sample.expected_answer,
                        score=normalized,
                        raw_score=raw_score,
                        note=response.reasoning,
                    )
                )
                logger.debug(
                    f"[CEREBRAS] Sample {idx}: score={raw_score}/{_SCORE_MAX}, "
                    f"reasoning={response.reasoning[:80]!r}"
                )
            except RuntimeError as exc:
                logger.warning(f"[CEREBRAS] Failed to judge sample {idx}: {exc}")
                failed_indices.append(idx)
                rows.append(
                    PluginReportRow(
                        idx=idx,
                        question=sample.question,
                        model_answer=sample.model_answer,
                        expected_answer=sample.expected_answer,
                        score=None,
                        raw_score=None,
                        note=f"API error: {exc}",
                    )
                )

        if not normalized_scores:
            result = EvalResult(
                plugin_name=self.name,
                passed=False,
                errors=["All samples failed to be judged by Cerebras API"],
                sample_count=len(eval_samples),
                failed_samples=list(range(len(eval_samples))),
            )
            if getattr(self, "_save_report", False):
                save_plugin_report(self.name, rows, result)
            return result

        recommendations = self.get_recommendations(
            EvalResult(plugin_name=self.name, passed=True, metrics={})
        )

        result = aggregate_scores(
            scores=normalized_scores,
            raw_scores=raw_scores,
            failed_indices=failed_indices,
            plugin_name=self.name,
            threshold_key="min_mean_score",
            thresholds=self.thresholds,
            recommendations=recommendations,
        )
        if getattr(self, "_save_report", False):
            save_plugin_report(self.name, rows, result)
        return result

    def get_recommendations(self, result: EvalResult) -> list[str]:  # noqa: ARG002
        return [
            "Review low-scoring samples in evaluation/answers.md",
            "Consider fine-tuning with more diverse training examples",
            "Check if the model answer format matches expected format",
        ]

    def _validate_contract(self) -> None:
        model = self._param("model", DEFAULT_MODEL)
        if not isinstance(model, str) or not model:
            raise ValueError(
                f"CerebrasJudgePlugin: 'model' param must be a non-empty string, got {model!r}"
            )

        temperature = self._param("temperature", DEFAULT_TEMPERATURE)
        if not isinstance(temperature, int | float) or not (0.0 <= float(temperature) <= 2.0):
            raise ValueError(
                f"CerebrasJudgePlugin: 'temperature' must be in [0.0, 2.0], got {temperature!r}"
            )

        max_samples = self._param("max_samples", _DEFAULT_MAX_SAMPLES)
        if not isinstance(max_samples, int) or max_samples < 1:
            raise ValueError(
                f"CerebrasJudgePlugin: 'max_samples' must be a positive integer, got {max_samples!r}"
            )


__all__ = ["CerebrasJudgePlugin"]
